#include "models/qwen_image/image_weights.h"

#include "artifact/schema.h"

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

} // namespace

VaeDenorm read_vae_denorm(Binder& binder) {
    const Json& config = binder.reader().directory().component("vae").config;
    VaeDenorm out{cfg_float_array(config, "latents_mean", "vae"),
                  cfg_float_array(config, "latents_std", "vae")};
    if (out.mean.size() != out.std.size()) {
        throw ArtifactError("vae config latents_mean and latents_std have different lengths");
    }
    return out;
}

DiTWeights bind_dit(Binder& binder) {
    const Directory& directory = binder.reader().directory();
    const Json& config         = directory.component("dit").config;
    const std::string_view component = "dit";

    DiTWeights out;
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

VaeWeights bind_image_vae(Binder& binder) {
    // The production decode chain's named operands, exactly as the qualified route consumes them
    // (tests/models/qwen4_exp/test_image_interlude_dit_real.cu): every "vae/<source key>" tensor,
    // direct BF16, at the shape the converter stored. The decoder's structural constants (the
    // 1152-channel first level halving to 144, 3x3 convolutions, three resnets per up block) are
    // the architecture of that route, not per-artifact config; a deviation throws at bind time.
    // The artifact's up-block time_conv tensors are deliberately NOT bound: dupup3d replaces them.
    VaeWeights out;
    const QType bf16 = QType::BF16;
    const auto add = [&](std::string_view name, Shape shape) {
        out.by_name.insert_or_assign(std::string(name),
                                     take(binder, out.max_object_bytes, name, std::move(shape), bf16));
    };

    add("vae/post_quant_conv.weight", {64, 64, 1, 1});
    add("vae/post_quant_conv.bias", {64});
    add("vae/decoder.conv_in.weight", {1152, 64, 3, 3});
    add("vae/decoder.conv_in.bias", {1152});
    for (int mid = 0; mid < 2; ++mid) {
        const std::string p = "vae/decoder.mid_block.resnets." + std::to_string(mid) + ".";
        add(p + "conv1.weight", {1152, 1152, 3, 3});
        add(p + "conv1.bias", {1152});
        add(p + "conv2.weight", {1152, 1152, 3, 3});
        add(p + "conv2.bias", {1152});
        add(p + "norm1.gamma", {1152, 1, 1, 1});
        add(p + "norm2.gamma", {1152, 1, 1, 1});
    }
    const std::string attn = "vae/decoder.mid_block.attentions.0.";
    add(attn + "to_qkv.weight", {3 * 1152, 1152, 1, 1});
    add(attn + "to_qkv.bias", {3 * 1152});
    add(attn + "proj.weight", {1152, 1152, 1, 1});
    add(attn + "proj.bias", {1152});
    add(attn + "norm.gamma", {1152, 1, 1});

    const struct UpCfg { std::uint64_t cin, cout; bool up; } cfg[5] = {
        {1152, 1152, true}, {1152, 1152, true}, {1152, 576, true}, {576, 288, true}, {288, 144, false},
    };
    for (int b = 0; b < 5; ++b) {
        const std::string p = "vae/decoder.up_blocks." + std::to_string(b) + ".";
        const auto cin = cfg[b].cin, cout = cfg[b].cout;
        for (int r = 0; r < 3; ++r) {
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
        if (cfg[b].up) {
            add(p + "upsampler.resample.1.weight", {cout, cout, 3, 3});
            add(p + "upsampler.resample.1.bias", {cout});
        }
    }
    add("vae/decoder.norm_out.gamma", {144, 1, 1, 1});
    add("vae/decoder.conv_out.weight", {4, 144, 3, 3});
    add("vae/decoder.conv_out.bias", {4});

    // #75 reference/edit encode chain (src/models/qwen_image/image_pipeline.cu
    // encode_reference_image): the same named-tensor discipline for the encoder + quant_conv. The
    // stored shapes come verbatim from the artifact's conversion report (104 rows). The encoder
    // downsampler time_conv tensors are deliberately NOT bound: the image path runs the pure-2D
    // branch (fresh feat_cache), where every time_conv is skipped -- same rule as the decoder's.
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
    return out;
}

} // namespace infernix::models::qwen_image
