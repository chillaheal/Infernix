#include "models/qwen_image/image_weights.h"

#include "artifact/schema.h"
#include "ops/common/canonical_math.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix::models::qwen_image {

namespace {

using artifact::ArtifactError;
using artifact::Binder;
using artifact::Directory;
using artifact::Json;
using artifact::ParameterReference;
using artifact::Residency;
using artifact::Shape;

constexpr std::uint64_t kTimeEmbedInputDim = 256; // 2 x 128 sinusoidal widths, the recipe's fixed input

std::uint64_t cfg_integer(const Json& config, const char* key, std::string_view component) {
    if (!config.contains(key) || !config.at(key).is_number_integer()) {
        throw ArtifactError(std::string(component) + " config lacks integer " + key);
    }
    return config.at(key).get<std::uint64_t>();
}

float cfg_number(const Json& config, const char* key, std::string_view component) {
    if (!config.contains(key) || !config.at(key).is_number()) {
        throw ArtifactError(std::string(component) + " config lacks number " + key);
    }
    return config.at(key).get<float>();
}

// Bind one streamed parameter; report the largest single object its parts address. A quantized
// role (exact_format FP8) additionally needs the RowScale layout the linear route reads.
struct Bound {
    ParameterReference ref;
    std::uint64_t max_bytes = 0;
};

Bound bind_streamed(Binder& binder, std::string_view name, Shape shape, QType exact_format) {
    Bound out{binder.parameter(name, std::move(shape), Residency::Streamed, exact_format), 0};
    for (const auto& part : out.ref.binding.parts) {
        const auto& geometry = binder.reader().geometry(part.object);
        if (exact_format == QType::FP8_E4M3FN_ROW_BF16 && geometry.layout != QuantLayout::RowScale) {
            throw ArtifactError(out.ref.name + ": a quantized image role needs the RowScale representation");
        }
        if (geometry.bytes > out.max_bytes) { out.max_bytes = geometry.bytes; }
    }
    return out;
}

// Bind, fold the streaming bound into the component's high-water mark, and yield the reference.
ParameterReference take(Binder& binder, std::uint64_t& high_water, std::string_view name, Shape shape,
                       QType exact_format) {
    Bound bound = bind_streamed(binder, name, std::move(shape), exact_format);
    if (bound.max_bytes > high_water) { high_water = bound.max_bytes; }
    return std::move(bound.ref);
}

// One integer array of a component config. A key the config does not carry yields NO entries (that
// is what "this architecture has no deepstack taps" means); a present key must be an array of
// integers. The caller decides whether an empty result is legal for the route it serves.
std::vector<std::uint64_t> cfg_integer_array(const Json& config, const char* key, std::string_view component) {
    if (!config.contains(key)) { return {}; }
    if (!config.at(key).is_array()) {
        throw ArtifactError(std::string(component) + " config " + key + " is not an array");
    }
    std::vector<std::uint64_t> out;
    for (const Json& value : config.at(key)) {
        if (!value.is_number_integer()) {
            throw ArtifactError(std::string(component) + " config array " + key + " holds a non-integer");
        }
        out.push_back(value.get<std::uint64_t>());
    }
    return out;
}

std::vector<float> cfg_float_array(const Json& config, const char* key, std::string_view component) {
    if (!config.contains(key) || !config.at(key).is_array()) {
        throw ArtifactError(std::string(component) + " config lacks array " + key);
    }
    std::vector<float> out;
    for (const Json& value : config.at(key)) {
        if (!value.is_number()) {
            throw ArtifactError(std::string(component) + " config array " + key + " holds a non-number");
        }
        out.push_back(value.get<float>());
    }
    if (out.empty()) {
        throw ArtifactError(std::string(component) + " config array " + key + " is empty");
    }
    return out;
}

// Read the denoise schedule's constants from the artifact's "scheduler" component through the strict
// reader. The reader names the key it refuses and, for a deviating value, prints what the artifact
// declared; this adds WHICH artifact the refusal concerns so the operator who passed --image-artifact
// reads the reason instead of guessing it. An artifact packed before the converter carried the
// scheduler document has no such component and is refused: no job is ever drawn with a schedule the
// artifact did not declare. It throws while the MaterializationPlan is still being built -- before any
// quiesce, yield, CUDA allocation or partially initialised model state.
SchedulerConstants read_scheduler_constants(Binder& binder) {
    const std::string artifact_name = binder.reader().file_path(0).string();
    const Directory& directory      = binder.reader().directory();
    if (directory.components.find("scheduler") == directory.components.end()) {
        throw ArtifactError(artifact_name + " carries no \"scheduler\" component, so the "
                            "FlowMatchEulerDiscrete schedule this build produces would not be the "
                            "schedule the artifact declares");
    }
    const auto constants = read_scheduler_config(directory.component("scheduler").config);
    if (!constants.has_value()) {
        throw ArtifactError(artifact_name + ": " + constants.error());
    }
    return *constants;
}

// Read the decoder geometry from the artifact's own "vae" component document through the strict reader
// (src/models/qwen_image/artifact_constants.h): the checkpoint's vae/config.json, which the converter
// packed verbatim (tools/convert/qwen_image.py records["vae"] = {"config": ...}). Same rule as the two
// documents above -- no compiled fallback, no coerced JSON type, no default for an absent key, and every
// refusal names the key and prints what the artifact declared, with this naming the artifact the operator
// passed. An artifact packed before the converter carried the document has none and is refused: the
// alternative is decoding through a chain the weights were not trained under.
VaeTopology read_vae_component_topology(Binder& binder) {
    const std::string artifact_name = binder.reader().file_path(0).string();
    const Directory& directory      = binder.reader().directory();
    if (directory.components.find("vae") == directory.components.end()) {
        throw ArtifactError(artifact_name + " carries no \"vae\" component, so the decoder geometry this "
                            "build runs would not be the geometry the artifact declares");
    }
    const auto topology = read_vae_topology(directory.component("vae").config);
    if (!topology.has_value()) {
        throw ArtifactError(artifact_name + ": " + topology.error());
    }
    return *topology;
}

// --- the decode chain's operands, as the artifact's topology shapes them ------------------------------
//
// One bound operand of the production decode route: the recipe's logical name and the shape the
// artifact's own declared topology says that role has. The NAMES are the recipe's contract
// (tools/convert/qwen_image.py maps every VAE source tensor one-to-one as a direct BF16 parameter, so
// the source key is the operand authority this file is allowed to spell); the SHAPES are not literals --
// every channel count below is a field of the topology, which is what lets a VAE of another geometry bind
// (and, if it lies about its geometry, fail) without this file holding a compiled default. The CUDA graph
// runs the same chain from the same fields (image_pipeline.cu), and the VRAM ledger charges for the same
// stages (image_device_need.h): one topology, three consumers, no second copy of a width anywhere.
struct VaeDecoderOperand {
    std::string name;
    Shape shape;
};

// A shape as the operator reads it in a refusal: the role's stored shape, printed exactly.
std::string shape_text(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) { out += ", "; }
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

// The chain's operands in the order the graph consumes them: the latent->decoder projection, conv_in, the
// mid block's resnets and its ONE attention, the up blocks (kQwenImageVaeResnetsPerUpBlock resnets each,
// with a convolution shortcut only where the channels change, and a resampler only where the stage
// upsamples), then norm_out and conv_out at the shallowest width.
//
// A topology this function cannot describe is a topology nothing in this build runs, so it is refused
// with the reason rather than emitting a shape whose arithmetic wrapped: the strict reader only ever
// produces one that passes, and check_vae_topology applies the same rules to whatever it is handed.
[[nodiscard]] std::expected<std::vector<VaeDecoderOperand>, std::string>
vae_decoder_operands(const VaeTopology& topology) {
    // NEVER READ, in each of the artifact-owned fields: a topology nobody read would otherwise bind an
    // all-zero decoder, which is the vacuous pass this plan exists to remove. The fields are named and
    // printed as they stand, so a refusal says which declaration is missing rather than asserting a cause
    // the numbers do not show.
    if (topology.latent_channels == 0 || topology.spatial_downscale == 0 || topology.denorm_elements == 0) {
        return std::unexpected(
            "the VAE topology was never read from the artifact, so the decode chain has no channels to be "
            "bound at (latent_channels " + std::to_string(topology.latent_channels) + ", spatial_downscale " +
            std::to_string(topology.spatial_downscale) + ", denorm_elements " +
            std::to_string(topology.denorm_elements) + ")");
    }
    // The one field that is NOT the artifact's declaration is checked on its own terms: a topology that
    // was read but never stamped carries no resident calibration, and the VRAM ledger would reserve
    // nothing for the weights the interlude keeps resident. An admission taken from a zero allowance would
    // admit anything, so the binding refuses it as a distinct failure rather than calling it unread.
    if (topology.weight_resident_bytes == 0) {
        return std::unexpected(
            "the VAE topology declares its decoder but carries no resident calibration: "
            "weight_resident_bytes is 0, and the family's binding (bind_image_vae) is what stamps it from "
            "kQwenImageWeightResidentBytes, because it -- not the checkpoint -- is what says how many bytes "
            "this build holds resident during a job");
    }
    // The per-channel statistics and the latent plane are ONE fact about the artifact (read_vae_topology
    // verified the declared pair's length against z_dim); they may not disagree by the time a job runs.
    if (topology.denorm_elements != topology.latent_channels) {
        return std::unexpected("the VAE topology denormalizes " + std::to_string(topology.denorm_elements) +
                               " latent channels per plane but declares " +
                               std::to_string(topology.latent_channels) + " as its latent width (z_dim): the "
                               "statistics and the plane are one fact the artifact declares");
    }
    if (topology.mid_stage.channels_out == 0 ||
        topology.mid_stage.channels_in != topology.mid_stage.channels_out) {
        return std::unexpected("the VAE topology's mid block takes " +
                               std::to_string(topology.mid_stage.channels_in) + " channels and writes " +
                               std::to_string(topology.mid_stage.channels_out) + ", but this build's mid "
                               "block runs its resnets at one level width and has no convolution shortcut");
    }
    if (topology.up_stages.size() != kQwenImageVaeUpBlocks) {
        return std::unexpected("the VAE topology declares " + std::to_string(topology.up_stages.size()) +
                               " up blocks but image_pipeline.cu runs " +
                               std::to_string(kQwenImageVaeUpBlocks));
    }
    for (std::size_t b = 0; b < topology.up_stages.size(); ++b) {
        const VaeStage& stage = topology.up_stages[b];
        const std::string at  = "its up block " + std::to_string(b);
        if (stage.channels_in == 0 || stage.channels_out == 0) {
            return std::unexpected("the VAE topology's " + at + " carries " +
                                   std::to_string(stage.channels_in) + " channels in and " +
                                   std::to_string(stage.channels_out) + " out, which is no convolution");
        }
        // The chain has to be one chain: each stage continues its predecessor's output, the first one
        // continues the mid block, and every block except the last doubles the side -- the pattern whose
        // doublings read_vae_topology verified against the artifact's own scale_factor_spatial.
        const std::uint32_t expected_in =
            (b == 0) ? topology.mid_stage.channels_out : topology.up_stages[b - 1].channels_out;
        if (stage.channels_in != expected_in) {
            return std::unexpected("the VAE topology's " + at + " takes " +
                                   std::to_string(stage.channels_in) + " channels in but " +
                                   (b == 0 ? "the mid block writes "
                                           : "up block " + std::to_string(b - 1) + " writes ") +
                                   std::to_string(expected_in) + ": the decoder is one chain, not a set of "
                                   "stages that disagree about their seams");
        }
        const bool expected_doubling = (b + 1 < topology.up_stages.size());
        if (stage.doubles_side != expected_doubling) {
            return std::unexpected("the VAE topology's " + at + (stage.doubles_side ? " upsamples"
                                                                                    : " does not upsample") +
                                   " but image_pipeline.cu's chain " +
                                   (expected_doubling ? "upsamples every block except the last"
                                                      : "runs the last block at the same side"));
        }
    }
    const std::uint64_t final_in = topology.up_stages.back().channels_out;
    if (topology.final_channels <= final_in) {
        return std::unexpected("the VAE topology's final stage carries " +
                               std::to_string(topology.final_channels) + " channels, which leaves " +
                               std::to_string(final_in) + " already used by the last up block and no "
                               "channels for conv_out to write");
    }

    const std::uint64_t latent       = topology.latent_channels;
    const std::uint64_t mid_channels = topology.mid_stage.channels_out;
    const std::uint64_t out_channels = topology.final_channels - final_in;

    std::vector<VaeDecoderOperand> out;
    out.push_back({"vae/post_quant_conv.weight", {latent, latent, 1, 1}});
    out.push_back({"vae/post_quant_conv.bias", {latent}});
    out.push_back({"vae/decoder.conv_in.weight", {mid_channels, latent, 3, 3}});
    out.push_back({"vae/decoder.conv_in.bias", {mid_channels}});

    // The mid block: kQwenImageVaeMidBlockResnets resnets at one width (so never a shortcut) and the
    // route's single spatial attention, whose qkv projection is what image_pipeline.cu's att() splits.
    for (std::uint32_t mid = 0; mid < kQwenImageVaeMidBlockResnets; ++mid) {
        const std::string p = "vae/decoder.mid_block.resnets." + std::to_string(mid) + ".";
        out.push_back({p + "conv1.weight", {mid_channels, mid_channels, 3, 3}});
        out.push_back({p + "conv1.bias", {mid_channels}});
        out.push_back({p + "conv2.weight", {mid_channels, mid_channels, 3, 3}});
        out.push_back({p + "conv2.bias", {mid_channels}});
        out.push_back({p + "norm1.gamma", {mid_channels, 1, 1, 1}});
        out.push_back({p + "norm2.gamma", {mid_channels, 1, 1, 1}});
    }
    const std::string attn = "vae/decoder.mid_block.attentions.0.";
    out.push_back({attn + "to_qkv.weight", {3ULL * mid_channels, mid_channels, 1, 1}});
    out.push_back({attn + "to_qkv.bias", {3ULL * mid_channels}});
    out.push_back({attn + "proj.weight", {mid_channels, mid_channels, 1, 1}});
    out.push_back({attn + "proj.bias", {mid_channels}});
    out.push_back({attn + "norm.gamma", {mid_channels, 1, 1}});

    for (std::size_t b = 0; b < topology.up_stages.size(); ++b) {
        const VaeStage& stage = topology.up_stages[b];
        const std::uint64_t cin  = stage.channels_in;
        const std::uint64_t cout = stage.channels_out;
        const std::string p      = "vae/decoder.up_blocks." + std::to_string(b) + ".";
        for (std::uint32_t r = 0; r < kQwenImageVaeResnetsPerUpBlock; ++r) {
            const std::string rp   = p + "resnets." + std::to_string(r) + ".";
            const std::uint64_t rin = (r == 0) ? cin : cout;
            out.push_back({rp + "conv1.weight", {cout, rin, 3, 3}});
            out.push_back({rp + "conv1.bias", {cout}});
            out.push_back({rp + "conv2.weight", {cout, cout, 3, 3}});
            out.push_back({rp + "conv2.bias", {cout}});
            out.push_back({rp + "norm1.gamma", {rin, 1, 1, 1}});
            out.push_back({rp + "norm2.gamma", {cout, 1, 1, 1}});
            if (r == 0 && cin != cout) {
                out.push_back({rp + "conv_shortcut.weight", {cout, cin, 1, 1}});
                out.push_back({rp + "conv_shortcut.bias", {cout}});
            }
        }
        if (stage.doubles_side) {
            out.push_back({p + "upsampler.resample.1.weight", {cout, cout, 3, 3}});
            out.push_back({p + "upsampler.resample.1.bias", {cout}});
        }
    }

    out.push_back({"vae/decoder.norm_out.gamma", {final_in, 1, 1, 1}});
    out.push_back({"vae/decoder.conv_out.weight", {out_channels, final_in, 3, 3}});
    out.push_back({"vae/decoder.conv_out.bias", {out_channels}});
    return out;
}

} // namespace

VaeDenorm read_vae_denorm(Binder& binder, const VaeTopology& topology) {
    const std::string artifact_name = binder.reader().file_path(0).string();
    const Directory& directory      = binder.reader().directory();
    if (directory.components.find("vae") == directory.components.end()) {
        throw ArtifactError(artifact_name + " carries no \"vae\" component, so the per-channel latent "
                            "statistics the decode denormalizes with cannot be read from the artifact");
    }
    const Json& config = directory.component("vae").config;
    VaeDenorm out{cfg_float_array(config, "latents_mean", "vae"),
                  cfg_float_array(config, "latents_std", "vae")};
    if (out.mean.size() != out.std.size()) {
        throw ArtifactError("vae config latents_mean and latents_std have different lengths");
    }
    // The statistics pair and the latent plane are ONE fact the artifact declares: read_vae_topology
    // verified the declared lengths against the artifact's own "z_dim" before the decoder bound a single
    // weight, and image_pipeline.cu divides every latent plane by these values, so a route handed a pair
    // of another length would either read past it or normalize nothing. Refused naming both numbers
    // rather than drawn with a plane the checkpoint never described.
    if (out.mean.size() != static_cast<std::size_t>(topology.denorm_elements) ||
        topology.denorm_elements != topology.latent_channels) {
        throw ArtifactError(artifact_name + ": the vae config's latents_mean/latents_std pair holds " +
                            std::to_string(out.mean.size()) + " values, but the bound decoder topology "
                            "denormalizes " + std::to_string(topology.denorm_elements) + " latent channels "
                            "per plane (z_dim " + std::to_string(topology.latent_channels) + ")");
    }
    return out;
}

DiTWeights bind_dit(Binder& binder) {
    const Directory& directory = binder.reader().directory();
    const Json& config         = directory.component("dit").config;
    const std::string_view component = "dit";

    DiTWeights out;
    // Refuse the artifact before a single parameter is resolved: the schedule is read first so an
    // unhandled scheme costs no bindings, no device work and no partially initialised state.
    out.scheduler = read_scheduler_constants(binder);
    // The rope axes come from the same "dit" document the dimensions are derived from, and are refused
    // the same way: the reader names the key and the value it refused, this names the artifact the
    // operator passed. An artifact that does not declare the axes this build rotates on is never bound,
    // so the emitter never sees a set it guessed -- and no parameter is resolved before the refusal.
    if (const auto axes = read_dit_rope_axes(config); !axes.has_value()) {
        throw ArtifactError(binder.reader().file_path(0).string() + ": " + axes.error());
    } else {
        out.rope = *axes;
    }
    out.dims.heads      = static_cast<std::uint32_t>(cfg_integer(config, "num_attention_heads", component));
    out.dims.head_dim   = static_cast<std::uint32_t>(cfg_integer(config, "attention_head_dim", component));
    out.dims.layers     = static_cast<std::uint32_t>(cfg_integer(config, "num_layers", component));
    out.dims.context_in = static_cast<std::uint32_t>(cfg_integer(config, "context_in_dim", component));
    out.dims.eps        = cfg_number(config, "eps", component);
    const std::uint64_t mlp_ratio    = cfg_integer(config, "mlp_ratio", component);
    const std::uint64_t in_ch        = cfg_integer(config, "in_channels", component);
    const std::uint64_t out_ch       = cfg_integer(config, "out_channels", component);
    const std::uint64_t patch        = cfg_integer(config, "patch_size", component);
    const std::uint64_t hidden       = static_cast<std::uint64_t>(out.dims.heads) * out.dims.head_dim;
    const std::uint64_t intermediate = hidden * mlp_ratio;
    const std::uint64_t image_in     = in_ch * patch * patch;
    const std::uint64_t image_out    = out_ch * patch * patch;
    out.dims.hidden       = static_cast<std::uint32_t>(hidden);
    out.dims.intermediate = static_cast<std::uint32_t>(intermediate);
    out.dims.image_in     = static_cast<std::uint32_t>(image_in);
    out.dims.image_out    = static_cast<std::uint32_t>(image_out);
    if (hidden == 0 || intermediate == 0 || image_in == 0 || image_out == 0 || out.dims.layers == 0) {
        throw ArtifactError("dit config derives a zero-sized dimension");
    }

    const QType fp8 = QType::FP8_E4M3FN_ROW_BF16;
    out.img_in       = take(binder, out.max_object_bytes, "dit/img_in", {hidden, image_in}, QType::BF16);
    out.proj_out     = take(binder, out.max_object_bytes, "dit/proj_out", {image_out, hidden}, QType::BF16);
    out.modulation   = take(binder, out.max_object_bytes, "dit/modulation", {4 * hidden, hidden}, QType::BF16);
    out.norm_out     = take(binder, out.max_object_bytes, "dit/norm_out", {hidden, hidden}, QType::BF16);
    out.txt_in_in    = take(binder, out.max_object_bytes, "dit/txt_in/in", {hidden, out.dims.context_in}, fp8);
    out.txt_in_out   = take(binder, out.max_object_bytes, "dit/txt_in/out", {hidden, hidden}, fp8);
    out.txt_in_norm  = take(binder, out.max_object_bytes, "dit/txt_in/norm", {out.dims.context_in}, QType::BF16);
    out.time_linear_1 =
        take(binder, out.max_object_bytes, "dit/time_embed/linear_1", {hidden, kTimeEmbedInputDim}, QType::BF16);
    out.time_linear_2 =
        take(binder, out.max_object_bytes, "dit/time_embed/linear_2", {hidden, hidden}, QType::BF16);

    out.blocks.resize(static_cast<std::size_t>(out.dims.layers));
    for (std::uint32_t layer = 0; layer < out.dims.layers; ++layer) {
        const std::string p = "dit/layers/" + std::to_string(layer) + "/";
        auto& block         = out.blocks[static_cast<std::size_t>(layer)];
        std::uint64_t block_high_water = 0;
        auto block_take = [&](ParameterReference& slot, std::string_view suffix, Shape shape, QType format) {
            slot = take(binder, block_high_water, p + std::string(suffix), std::move(shape), format);
        };
        block_take(block.q, "attention/query", {hidden, hidden}, fp8);
        block_take(block.k, "attention/key", {hidden, hidden}, fp8);
        block_take(block.v, "attention/value", {hidden, hidden}, fp8);
        block_take(block.query_norm, "attention/query_norm", {out.dims.head_dim}, QType::BF16);
        block_take(block.key_norm, "attention/key_norm", {out.dims.head_dim}, QType::BF16);
        block_take(block.attention_output, "attention/output", {hidden, hidden}, fp8);
        block_take(block.mlp_gate, "mlp/gate", {intermediate, hidden}, fp8);
        block_take(block.mlp_up, "mlp/up", {intermediate, hidden}, fp8);
        block_take(block.mlp_down, "mlp/down", {hidden, intermediate}, fp8);
        if (block_high_water > out.max_object_bytes) { out.max_object_bytes = block_high_water; }
        if (block_high_water > out.max_block_object_bytes) { out.max_block_object_bytes = block_high_water; }
    }
    return out;
}

TextTowerWeights bind_text_encoder(Binder& binder) {
    const Directory& directory = binder.reader().directory();
    const Json& root           = directory.component("text").config;
    const std::string_view component = "text";
    if (!root.contains("text_config")) { throw ArtifactError("text config lacks text_config"); }
    const Json& config = root.at("text_config");

    TextTowerWeights out;
    out.dims.heads        = static_cast<std::uint32_t>(cfg_integer(config, "num_attention_heads", component));
    out.dims.kv_heads     = static_cast<std::uint32_t>(cfg_integer(config, "num_key_value_heads", component));
    out.dims.head_dim     = static_cast<std::uint32_t>(cfg_integer(config, "head_dim", component));
    out.dims.hidden       = static_cast<std::uint32_t>(cfg_integer(config, "hidden_size", component));
    out.dims.intermediate = static_cast<std::uint32_t>(cfg_integer(config, "intermediate_size", component));
    out.dims.vocab        = static_cast<std::uint32_t>(cfg_integer(config, "vocab_size", component));
    out.dims.layers       = static_cast<std::uint32_t>(cfg_integer(config, "num_hidden_layers", component));
    out.dims.rope_theta   = cfg_number(config, "rope_theta", component);
    out.dims.eps          = cfg_number(config, "rms_norm_eps", component);

    // The M-RoPE split the tower rotates by comes from the artifact, validated the way the LLM's own
    // config parser validates it (models/qwen3_5/config.cpp rope): three positive sections, their sum
    // equal to the head's pair count, and the interleaved walk actually producing those counts. Nothing
    // is defaulted -- a checkpoint whose rope_scaling is absent, or is not the interleaved form this
    // tower's kernel implements, is refused at bind time instead of being rotated by a guessed rule.
    if (!config.contains("rope_scaling")) {
        throw ArtifactError("text config lacks rope_scaling, so its M-RoPE cannot be read");
    }
    const Json& scaling = config.at("rope_scaling");
    if (!scaling.is_object()) { throw ArtifactError("text rope_scaling is not an object"); }
    if (!scaling.contains("mrope_interleaved") || !scaling.at("mrope_interleaved").is_boolean() ||
        !scaling.at("mrope_interleaved").get<bool>()) {
        throw ArtifactError("text M-RoPE is not the interleaved form this tower implements");
    }
    const Json sections = scaling.contains("mrope_section") ? scaling.at("mrope_section") : Json();
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("text M-RoPE requires three sections");
    }
    std::uint64_t section_sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const Json& entry = sections[i];
        // is_number_unsigned, not is_number_integer: a section is a count, so this is the check
        // that rejects a negative one here instead of letting the reader throw on the conversion.
        if (!entry.is_number_unsigned()) {
            throw ArtifactError("text M-RoPE section is not a non-negative integer");
        }
        out.dims.mrope_section[i] = static_cast<std::uint32_t>(entry.get<std::uint64_t>());
        section_sum += out.dims.mrope_section[i];
    }
    out.dims.mrope_interleaved = true;
    const std::uint64_t pairs = out.dims.head_dim / 2ULL;
    if (out.dims.head_dim % 2 != 0 || section_sum != pairs) {
        throw ArtifactError("text M-RoPE sections differ from the head's pair count");
    }
    // Count the assignment the rotation kernel performs back against the declared sections: H owns the
    // pairs i % 3 == 1 below 3 * section[1], W the pairs i % 3 == 2 below 3 * section[2], T the rest.
    // Sections the walk cannot produce would rotate some pairs with the wrong axis' position and every
    // row of the tower would inherit the error.
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < pairs; ++pair) {
        const std::size_t axis = pair % 3 == 1 && pair < 3ULL * out.dims.mrope_section[1]   ? 1
                                 : pair % 3 == 2 && pair < 3ULL * out.dims.mrope_section[2] ? 2
                                                                                            : 0;
        ++counts[axis];
    }
    if (counts != out.dims.mrope_section) {
        throw ArtifactError("text M-RoPE sections exceed the interleaved axis ranges");
    }

    const std::uint64_t hidden = out.dims.hidden;
    const std::uint64_t q      = static_cast<std::uint64_t>(out.dims.heads) * out.dims.head_dim;
    const std::uint64_t kv     = static_cast<std::uint64_t>(out.dims.kv_heads) * out.dims.head_dim;
    const std::uint64_t inter  = out.dims.intermediate;
    if (hidden == 0 || q == 0 || kv == 0 || inter == 0 || out.dims.layers == 0 || out.dims.vocab == 0) {
        throw ArtifactError("text config derives a zero-sized dimension");
    }

    const QType fp8 = QType::FP8_E4M3FN_ROW_BF16;
    out.embed_tokens =
        take(binder, out.max_object_bytes, "text/embed_tokens", {out.dims.vocab, hidden}, fp8);
    out.final_norm = take(binder, out.max_object_bytes, "text/final_norm", {hidden}, QType::BF16);

    out.layers.resize(static_cast<std::size_t>(out.dims.layers));
    for (std::uint32_t layer = 0; layer < out.dims.layers; ++layer) {
        const std::string p = "text/layers/" + std::to_string(layer) + "/";
        auto& row           = out.layers[static_cast<std::size_t>(layer)];
        auto row_take = [&](ParameterReference& slot, std::string_view suffix, Shape shape, QType format) {
            slot = take(binder, out.max_object_bytes, p + std::string(suffix), std::move(shape), format);
        };
        row_take(row.input_norm, "input_norm", {hidden}, QType::BF16);
        row_take(row.post_attention_norm, "post_attention_norm", {hidden}, QType::BF16);
        row_take(row.q, "attention/query", {q, hidden}, fp8);
        row_take(row.k, "attention/key", {kv, hidden}, fp8);
        row_take(row.v, "attention/value", {kv, hidden}, fp8);
        row_take(row.query_norm, "attention/query_norm", {out.dims.head_dim}, QType::BF16);
        row_take(row.key_norm, "attention/key_norm", {out.dims.head_dim}, QType::BF16);
        row_take(row.attention_output, "attention/output", {hidden, q}, fp8);
        row_take(row.mlp_gate, "mlp/gate", {inter, hidden}, fp8);
        row_take(row.mlp_up, "mlp/up", {inter, hidden}, fp8);
        row_take(row.mlp_down, "mlp/down", {hidden, inter}, fp8);
    }
    return out;
}

VisionWeights bind_image_vision(Binder& binder) {
    // The vision tower lives in the text encoder's shards, so its geometry comes from that
    // component's own "vision_config" -- the same object the converter read to declare the names
    // bound here. Every role is direct BF16 (the FP8 recipe assigns only dit/* and text/*), so a
    // quantized vision object in the artifact is a recipe drift and throws in bind_streamed.
    const Directory& directory   = binder.reader().directory();
    const Json& root             = directory.component("text").config;
    const std::string_view component = "text";
    if (!root.contains("vision_config")) { throw ArtifactError("text config lacks vision_config"); }
    const Json& config = root.at("vision_config");

    VisionWeights out;
    out.dims.depth         = static_cast<std::uint32_t>(cfg_integer(config, "depth", component));
    out.dims.hidden        = static_cast<std::uint32_t>(cfg_integer(config, "hidden_size", component));
    out.dims.intermediate  = static_cast<std::uint32_t>(cfg_integer(config, "intermediate_size", component));
    out.dims.heads         = static_cast<std::uint32_t>(cfg_integer(config, "num_heads", component));
    out.dims.patch         = static_cast<std::uint32_t>(cfg_integer(config, "patch_size", component));
    out.dims.temporal      = static_cast<std::uint32_t>(cfg_integer(config, "temporal_patch_size", component));
    out.dims.merge         = static_cast<std::uint32_t>(cfg_integer(config, "spatial_merge_size", component));
    out.dims.position_rows = static_cast<std::uint32_t>(cfg_integer(config, "num_position_embeddings", component));
    out.dims.out_hidden    = static_cast<std::uint32_t>(cfg_integer(config, "out_hidden_size", component));
    for (const std::uint64_t index : cfg_integer_array(config, "deepstack_visual_indexes", component)) {
        if (index >= out.dims.depth) {
            throw ArtifactError("vision config deepstack index is outside the block count");
        }
        out.dims.deepstack_indexes.push_back(static_cast<std::uint32_t>(index));
    }

    const std::uint64_t hidden      = out.dims.hidden;
    const std::uint64_t inter       = out.dims.intermediate;
    const std::uint64_t patch_width = 3ULL * out.dims.temporal * out.dims.patch * out.dims.patch;
    const std::uint64_t merger      = static_cast<std::uint64_t>(out.dims.merge) * out.dims.merge * hidden;
    const auto side                 = static_cast<std::uint32_t>(std::sqrt(double(out.dims.position_rows)));
    out.dims.patch_width      = static_cast<std::uint32_t>(patch_width);
    out.dims.merger_width     = static_cast<std::uint32_t>(merger);
    out.dims.position_grid_side = side;
    if (hidden == 0 || inter == 0 || patch_width == 0 || merger == 0 || out.dims.depth == 0 ||
        out.dims.out_hidden == 0 || out.dims.heads == 0 || hidden % out.dims.heads != 0 ||
        std::uint64_t(side) * side != out.dims.position_rows) {
        throw ArtifactError("vision config derives an unusable dimension");
    }
    out.dims.head_dim = static_cast<std::uint32_t>(hidden / out.dims.heads);

    const QType bf16 = QType::BF16;
    out.patch_embedding =
        take(binder, out.max_object_bytes, "vision/patch_embedding", {hidden, patch_width}, bf16);
    out.patch_embedding_bias =
        take(binder, out.max_object_bytes, "vision/patch_embedding_bias", {hidden}, bf16);
    out.position_embedding =
        take(binder, out.max_object_bytes, "vision/position_embedding", {out.dims.position_rows, hidden},
             bf16);

    out.blocks.resize(static_cast<std::size_t>(out.dims.depth));
    for (std::uint32_t layer = 0; layer < out.dims.depth; ++layer) {
        const std::string p  = "vision/layers/" + std::to_string(layer) + "/";
        auto& row            = out.blocks[static_cast<std::size_t>(layer)];
        auto row_take = [&](ParameterReference& slot, std::string_view suffix, Shape shape) {
            slot = take(binder, out.max_object_bytes, p + std::string(suffix), std::move(shape), bf16);
        };
        row_take(row.norm1_weight, "norm1_weight", {hidden});
        row_take(row.norm1_bias, "norm1_bias", {hidden});
        row_take(row.norm2_weight, "norm2_weight", {hidden});
        row_take(row.norm2_bias, "norm2_bias", {hidden});
        row_take(row.q, "attention/query", {hidden, hidden});
        row_take(row.q_bias, "attention/query_bias", {hidden});
        row_take(row.k, "attention/key", {hidden, hidden});
        row_take(row.k_bias, "attention/key_bias", {hidden});
        row_take(row.v, "attention/value", {hidden, hidden});
        row_take(row.v_bias, "attention/value_bias", {hidden});
        row_take(row.attention_output, "attention/output", {hidden, hidden});
        row_take(row.attention_output_bias, "attention/output_bias", {hidden});
        row_take(row.fc1, "mlp/fc1", {inter, hidden});
        row_take(row.fc1_bias, "mlp/fc1_bias", {inter});
        row_take(row.fc2, "mlp/fc2", {hidden, inter});
        row_take(row.fc2_bias, "mlp/fc2_bias", {hidden});
    }

    // One patch merger at the geometry the recipe declares: fc1 is merger-wide on both variants,
    // fc2 lands on the text tower's width, and only the LayerNorm differs (hidden for the main
    // merger, the merged row for a deepstack one) -- see Qwen3VLVisionPatchMerger.forward.
    const auto bind_merger = [&](VisionMergerWeights& m, const std::string& p, std::uint64_t norm_width) {
        m.norm_weight = take(binder, out.max_object_bytes, p + "norm_weight", {norm_width}, bf16);
        m.norm_bias   = take(binder, out.max_object_bytes, p + "norm_bias", {norm_width}, bf16);
        m.fc1         = take(binder, out.max_object_bytes, p + "fc1", {merger, merger}, bf16);
        m.fc1_bias    = take(binder, out.max_object_bytes, p + "fc1_bias", {merger}, bf16);
        m.fc2         = take(binder, out.max_object_bytes, p + "fc2", {out.dims.out_hidden, merger}, bf16);
        m.fc2_bias    = take(binder, out.max_object_bytes, p + "fc2_bias", {out.dims.out_hidden}, bf16);
    };
    bind_merger(out.merger, "vision/merger/", hidden);
    out.deepstack.resize(out.dims.deepstack_indexes.size());
    for (std::size_t i = 0; i < out.deepstack.size(); ++i) {
        bind_merger(out.deepstack[i], "vision/deepstack/" + std::to_string(i) + "/", merger);
    }
    return out;
}

VaeWeights bind_image_vae(Binder& binder) {
    // The production decode chain's named operands, exactly as the qualified route consumes them
    // (tests/models/qwen4_exp/test_image_interlude_dit_real.cu): every "vae/<source key>" tensor, direct
    // BF16, at the shape the converter stored -- and the shape is DERIVED, from the artifact's own "vae"
    // component config through the strict reader (read_vae_component_topology), never from a compiled
    // channel literal. So an artifact that declares the shipped family's geometry binds the shipped
    // plan, and an artifact that declares another binds that one or is refused: the 1152-channel deepest
    // level halving to 144, the 64-wide latent and the 4-channel conv_out are the artifact's declaration
    // of its decoder, read here and nowhere else. What stays code-owned is what THIS build runs -- the
    // number of up blocks and resnets per block (artifact_constants.h, which the reader verifies the
    // artifact's "num_res_blocks" against) and the 3x3 convolution kernel the ops take. The artifact's
    // up-block time_conv tensors are deliberately NOT bound: dupup3d replaces them.
    VaeWeights out;
    out.topology = read_vae_component_topology(binder);
    // The calibration of this build's resident bytes, not a claim of the checkpoint's: the reader leaves
    // it zero (see the header's authority table) and the owner of the streaming plan -- here -- stamps
    // what it measured. check_vae_topology below refuses a topology that never got the stamp.
    out.topology.weight_resident_bytes = kQwenImageWeightResidentBytes;

    const QType bf16 = QType::BF16;
    const auto add = [&](std::string_view name, Shape shape) {
        out.by_name.insert_or_assign(std::string(name),
                                     take(binder, out.max_object_bytes, name, std::move(shape), bf16));
    };

    const auto operands = vae_decoder_operands(out.topology);
    if (!operands.has_value()) {
        throw ArtifactError(binder.reader().file_path(0).string() + ": " + operands.error());
    }
    // Binding each role AT the topology's shape is already verification: Binder::binding refuses a stored
    // tensor whose coverage or shape differs from what was asked for, so an artifact whose bytes disagree
    // with its own declared config never produces a plan. check_vae_topology re-reads every resolved
    // reference against the same table afterwards, and is the call that says WHICH stage and role it was.
    for (const auto& operand : *operands) { add(operand.name, operand.shape); }

    // #75 reference/edit encode chain (src/models/qwen_image/image_pipeline.cu
    // encode_reference_image): the same named-tensor discipline for the encoder + quant_conv. The
    // stored shapes come verbatim from the artifact's conversion report (104 rows). The encoder
    // downsampler time_conv tensors are deliberately NOT bound: the image path runs the pure-2D
    // branch (fresh feat_cache), where every time_conv is skipped -- same rule as the decoder's.
    //
    // RETAINED COMPILED LITERAL -- not artifact-derived, and deliberately out of this task's scope. The
    // channel numbers below (96/192/384/768, the 128-wide quant_conv/conv_out, the 3*768 attention
    // projection) are this build's table, NOT read from the VaeTopology: the published topology describes
    // the DECODER, which is the one geometry the VRAM ledger and the CUDA decode graph share. The encoder
    // is a different route -- it runs before the DiT, its output is the reference latent, and nothing in
    // the admission model prices it -- so it is left as a literal here rather than silently made to look
    // artifact-owned. The matching marker at the other end of this route is image_pipeline.cu's
    // encode_reference_image guard, which says the same thing where the channel lists are consumed; a
    // grep for "RETAINED COMPILED LITERAL" or for that guard finds both sites. Retiring it means
    // publishing the encoder's geometry as a second artifact-owned topology the binder reads and verifies
    // the same way vae_decoder_operands does, which is a task of its own, not a comment's job.
    add("vae/quant_conv.weight", {128, 128, 1, 1});
    add("vae/quant_conv.bias", {128});
    add("vae/encoder.conv_in.weight", {96, 4, 3, 3});
    add("vae/encoder.conv_in.bias", {96});
    const struct DnCfg { std::uint64_t cin, cout; bool down; } dcfg[5] = {
        {96, 96, true}, {96, 192, true}, {192, 384, true}, {384, 768, true}, {768, 768, false},
    };
    for (int b = 0; b < 5; ++b) {
        const std::string p = "vae/encoder.down_blocks." + std::to_string(b) + ".";
        const auto cin = dcfg[b].cin, cout = dcfg[b].cout;
        for (int r = 0; r < 2; ++r) {
            const std::string rp = p + "resnets." + std::to_string(r) + ".";
            const std::uint64_t rin = r == 0 ? cin : cout;
            add(rp + "conv1.weight", {cout, rin, 3, 3});
            add(rp + "conv1.bias", {cout});
            add(rp + "conv2.weight", {cout, cout, 3, 3});
            add(rp + "conv2.bias", {cout});
            add(rp + "norm1.gamma", {rin, 1, 1, 1});
            add(rp + "norm2.gamma", {cout, 1, 1, 1});
            if (r == 0 && cin != cout) {
                add(rp + "conv_shortcut.weight", {cout, cin, 1, 1});
                add(rp + "conv_shortcut.bias", {cout});
            }
        }
        if (dcfg[b].down) {
            add(p + "downsampler.resample.1.weight", {cout, cout, 3, 3});
            add(p + "downsampler.resample.1.bias", {cout});
        }
    }
    for (int mid = 0; mid < 2; ++mid) {
        const std::string p = "vae/encoder.mid_block.resnets." + std::to_string(mid) + ".";
        add(p + "conv1.weight", {768, 768, 3, 3});
        add(p + "conv1.bias", {768});
        add(p + "conv2.weight", {768, 768, 3, 3});
        add(p + "conv2.bias", {768});
        add(p + "norm1.gamma", {768, 1, 1, 1});
        add(p + "norm2.gamma", {768, 1, 1, 1});
    }
    const std::string eattn = "vae/encoder.mid_block.attentions.0.";
    add(eattn + "to_qkv.weight", {3 * 768, 768, 1, 1});
    add(eattn + "to_qkv.bias", {3 * 768});
    add(eattn + "proj.weight", {768, 768, 1, 1});
    add(eattn + "proj.bias", {768});
    add(eattn + "norm.gamma", {768, 1, 1});
    add("vae/encoder.norm_out.gamma", {768, 1, 1, 1});
    add("vae/encoder.conv_out.weight", {128, 768, 3, 3});
    add("vae/encoder.conv_out.bias", {128});

    // The last gate before the binding hands the plan to the caller: every operand the CUDA decode graph
    // consumes has to carry exactly the shape the artifact's own declared topology says it has, and the
    // topology has to describe the one chain this build runs. A deviation throws here -- which IS the
    // loader gate, before any quiesce, yield, CUDA allocation or partially initialised model state -- so
    // no job is ever decoded through a geometry the artifact did not declare, and none is decoded with a
    // geometry the ledger did not price.
    if (const auto verdict = check_vae_topology(out.topology, out); !verdict.has_value()) {
        throw ArtifactError(binder.reader().file_path(0).string() + ": " + verdict.error());
    }
    return out;
}

std::expected<void, std::string> check_vae_topology(const VaeTopology& topology, const VaeWeights& weights) {
    // vae_decoder_operands is the single place the chain's shape rules live: it refuses a topology that
    // was never read, one whose stages disagree about their seams, and one that does not describe the
    // block/resnet/doubling pattern image_pipeline.cu runs. Its operand list is then the expectation this
    // comparison checks the bound plan against.
    const auto operands = vae_decoder_operands(topology);
    if (!operands.has_value()) { return std::unexpected(operands.error()); }
    for (const auto& operand : *operands) {
        const auto found = weights.by_name.find(operand.name);
        if (found == weights.by_name.end()) {
            // A role the graph consumes that the artifact does not store is a decode that would fail
            // mid-kernel; the binding reads it as a refusal naming the role.
            return std::unexpected("the bound VAE decoder has no \"" + operand.name +
                                   "\", which the decode chain this build runs consumes at " +
                                   shape_text(operand.shape) + ": the artifact's own declared topology says "
                                   "the role exists, so it is not a role this route may skip");
        }
        if (found->second.shape != operand.shape) {
            return std::unexpected("\"" + operand.name + "\" is bound as " + shape_text(found->second.shape) +
                                   " but the artifact's declared topology gives it " +
                                   shape_text(operand.shape) + ": the decoder weights and the geometry the "
                                   "ledger and the graph were built from disagree, so the artifact is "
                                   "refused rather than decoded through a chain its weights were not "
                                   "trained under");
        }
    }
    return {};
}

std::vector<std::byte> dequantize_embed_table(artifact::Reader& reader, const TextTowerWeights& tower) {
    const ParameterReference& ref = tower.embed_tokens;
    if (ref.binding.parts.size() != 1) {
        throw std::runtime_error("image text tower: embed_tokens binds more than one object");
    }
    const auto object = ref.binding.parts.front().object;
    const WeightGeometry& g  = reader.geometry(object);
    const std::size_t vocab  = static_cast<std::size_t>(tower.dims.vocab);
    const std::size_t hidden = static_cast<std::size_t>(tower.dims.hidden);
    if (g.format != QType::FP8_E4M3FN_ROW_BF16 || g.layout != QuantLayout::RowScale ||
        g.code_bytes != vocab * hidden || g.scale_bytes != vocab * 2 || g.scale_offset < g.code_bytes) {
        throw std::runtime_error("image text tower: embed_tokens is not a plain [codes][scales] FP8 plane");
    }
    const std::vector<std::byte> obj = reader.read_object(object);
    if (obj.size() < g.scale_offset + g.scale_bytes) {
        throw std::runtime_error("image text tower: embed_tokens object is shorter than its geometry");
    }
    const auto* codes  = reinterpret_cast<const std::uint8_t*>(obj.data());
    const auto* scales = reinterpret_cast<const std::uint16_t*>(obj.data() + g.scale_offset);
    std::vector<std::byte> table(vocab * hidden * 2);
    auto* rows = reinterpret_cast<std::uint16_t*>(table.data());
    for (std::size_t row = 0; row < vocab; ++row) {
        const float scale = ops::canon::bf16_to_f32(scales[row]);
        for (std::size_t col = 0; col < hidden; ++col) {
            const std::uint8_t byte = codes[row * hidden + col];
            const float value       = ops::canon::e4m3_value(byte & 0x7FU) * ((byte & 0x80U) ? -scale : scale);
            rows[row * hidden + col] = ops::canon::f32_to_bf16_rn(value);
        }
    }
    return table;
}

} // namespace infernix::models::qwen_image
