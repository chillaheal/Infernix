#pragma once

// The Qwen-Image-2.1 image lane's model constants are read FROM the `.infernix` artifact and refused
// when the artifact cannot express the scheme the code implements. Nothing here may fall back to a
// compiled value: an artifact that says nothing about the denoise schedule must not be drawn with the
// schedule of another family (docs/maintainer/qwen-image-2-1-variables.md).
//
// Where each parameter's authority lies, so a reader can look it up instead of guessing:
//
//   | parameter                       | authority                              |
//   |---------------------------------|----------------------------------------|
//   | scheduler values and mode flags | the artifact                           |
//   | DiT rope axes                   | the artifact                           |
//   | DiT rope theta                  | the pipeline implementation (code)     |
//   | VAE decoder stages and channels | the artifact                           |
//   | VAE latent width, downscale     | the artifact                           |
//   | VAE per-block resnet count      | the artifact, verified against the graph |
//   | VAE up-block temporal shortcut  | the pipeline implementation (code),      |
//   |                                 | verified against the artifact            |
//   | VAE live-buffer planes          | the pipeline implementation (code)     |
//   | VAE resident calibration        | the pipeline implementation (code)     |
//   | CUDA launch configuration       | the runtime implementation             |
//
// The VAE rows travel together and are the reason the image lane has one decoder geometry rather than
// two: the VRAM ledger (image_device_need.h) and the CUDA decode graph (image_pipeline.cu) both read
// the VaeTopology below, which the binding builds from the artifact's own "vae" component config. The
// two rows marked code-owned are read off THIS build and then verified against whatever the artifact
// declares -- the per-block resnet count because image_pipeline.cu's dup_up() really does run three
// resnets per up block and an artifact that declares another count describes a chain this build does
// not run, the live-buffer planes and the resident calibration because they are measurements of this
// build's allocator and ops, never the checkpoint's claims.
//
// The rope row is deliberately the odd one out: the shipped checkpoint declares no rope theta in
// transformer/config.json at all (verified against the live artifact's own "dit" component), yet the
// supported Diffusers Qwen-Image pipeline rotates with 10000, so naming it an artifact field would be
// an invention. rope.h carries it as kQwenImageRopeTheta, a named implementation default that changes
// only when the supported pipeline implementation is deliberately changed. An artifact that declares a
// "rope_theta" of its own is therefore not a source of the value but a claim about the scheme its weights
// were trained under: read_dit_rope_axes verifies such a claim against kQwenImageRopeTheta and refuses the
// artifact when it disagrees, instead of silently rotating those weights at 10000.
//
// The scheduler document is the converter's verbatim copy of the checkpoint's own
// scheduler/scheduler_config.json (tools/convert/qwen_image.py), packed as the artifact's "scheduler"
// component: the reader adds no field, defaults no field, and coerces no type. A JSON integer is only
// ever an integer (256 ok, 256.0 refused), a boolean only ever a boolean (true ok, 1 refused), a
// string only ever a string, and a float must be finite.
//
// Host-only code: the readers are called while the artifact is bound, before any quiesce, yield, CUDA
// allocation or partially initialised model state.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace infernix::models::qwen_image {

// The six values the resolution-dependent FlowMatchEulerDiscrete schedule is built from, exactly as
// the artifact declares them (an integer stays a JSON integer, so it is a distinct C++ type from the
// shifts and never silently accepts 256.0). Zero here means NEVER READ: the only producer is
// read_scheduler_config, and the image execution refuses an unread set before it allocates anything.
struct SchedulerConstants {
    std::int64_t base_image_seq_len = 0;
    double       base_shift         = 0.0;
    std::int64_t max_image_seq_len  = 0;
    double       max_shift          = 0.0;
    std::int64_t num_train_timesteps = 0;
    double       shift_terminal     = 0.0;
};

// The three axes the DiT rotates the joint sequence on (frame / height / width), exactly as the
// artifact's "dit" component declares them: positive even channel counts summing to attention_head_dim,
// so the emitted table is attention_head_dim / 2 interleaved (cos, sin) pairs wide. Nothing here is
// pinned at the shipped {16, 56, 56} -- the emitter's row width comes from these numbers. Zero here
// means NEVER READ: the only producer is read_dit_rope_axes, and the image execution refuses an unread
// set before it allocates anything. Theta is NOT a member, because the artifact does not declare one.
struct DitRopeAxes {
    std::uint32_t axes[3] = {0, 0, 0};
};

// The decoder chain THIS build runs, as far as more than one translation unit has to agree on it: five
// up blocks whose last one does not upsample (four 2x doublings), and dup_up()'s three resnets per block
// (resnets.0/.1/.2, image_pipeline.cu:329-332). Like the live-buffer planes and the resident calibration
// these are code-owned -- they are what this build's decode graph runs and what the ledger that prices it
// charges for -- and they are published here rather than repeated per file so the strict reader that
// VERIFIES the artifact's own claims (read_vae_topology compares "dim_mult"'s length against the block
// count and "num_res_blocks" + 1 against the resnet count, refusing either deviation) and the binding
// that resolves every operand shape from the resulting topology read the same numbers.
inline constexpr std::size_t   kQwenImageVaeUpBlocks          = 5;
inline constexpr std::uint32_t kQwenImageVaeResnetsPerUpBlock = 3;
// The mid block carries the plain num_res_blocks resnets (the up blocks the +1 form), so its count is not
// a third fact: it is the artifact's own declaration, which read_vae_topology verifies against the number
// above. Derived rather than written again so the two can never be changed apart.
inline constexpr std::uint32_t kQwenImageVaeMidBlockResnets   = kQwenImageVaeResnetsPerUpBlock - 1;
// The dupup3d shortcut factor the pure-2D image decode passes at each up block: 2 where the ENCODER
// temporally downsampled (the checkpoint's own "temperal_downsample", which read_vae_topology verifies
// entry by entry against this table, mirrored because the decoder walks the chain the other way) and 1
// where it did not -- including the last block, which does not upsample at all. This build implements one
// such scheme, so it is code-owned here and VERIFIED against the artifact's claim there: an artifact that
// declares another would have its duplicated shortcut taken from the wrong side of the time axis, which
// is a silent divergence of exactly the class this lane refuses everywhere else.
inline constexpr std::array<bool, kQwenImageVaeUpBlocks> kQwenImageVaeUpBlockTemporal{
    true, true, true, false, false};

// One stage of the artifact's VAE decoder chain, as the binding builds it from the artifact's own
// "vae" component config and as BOTH of its consumers read it: image_pipeline.cu runs these channel
// widths and image_device_need.h charges for exactly the same ones. There is no second copy -- the
// header array and the two channel literals this record replaced are gone, and an artifact that
// declares a chain this build does not run is refused by read_vae_topology rather than drawn with
// another family's geometry.
struct VaeStage {
    std::uint32_t channels_in  = 0; // the channels the bound conv takes in
    std::uint32_t channels_out = 0; // ... and the ones it writes out
    // The code's live-buffer inventory at this stage, in whole [channels_out, side, side] planes: the
    // count is read off image_pipeline.cu, not off the artifact, because it is what THIS build
    // allocates. `att` keeps the qkv buffer (3C), its attention output (C) and the still-live input
    // (C) at the spatial_attn call (:207-222); `dup_up` holds the upsampled input, the duplicated
    // shortcut and the conv output at the DOUBLED resolution (:252-264); `dresnet` holds its input,
    // shortcut and two conv outputs (:161-192). A stage whose inventory changes changes the ledger's
    // peak with it, which is why the number lives with the stage instead of beside it.
    std::uint32_t planes       = 0;
    bool doubles_side          = false; // the stage works at the doubled side after its upsample

    friend bool operator==(const VaeStage&, const VaeStage&) = default;
};

// The image route's decoder geometry, read FROM the artifact's "vae" component config by
// read_vae_topology and verified against every bound decoder weight by check_vae_topology
// (src/models/qwen_image/image_weights.cpp) before the binding returns. Zero in any field means NEVER
// READ: the only producer is read_vae_topology (plus the code-owned calibration the family's binding
// stamps below), and an image job refuses an unread set before it allocates anything.
//
// Nothing here may fall back to a compiled value: an artifact that says nothing about its decoder must
// not be decoded with the decoder of another family, and the ledger must not reserve bytes for planes
// the graph never allocates. The shipped family's values -- the 1152-channel deepest level halving to
// 144, the 64-wide latent, the 16x downscale, the 64 per-channel statistics -- are what the live
// artifact declares and what tests/convert/fixtures/qwen_image_vae_config.json carries; the guards read
// them through this reader rather than pinning them here.
struct VaeTopology {
    std::uint32_t latent_channels   = 0; // the artifact's z_dim: the latent plane the DiT's image_in is
    std::uint32_t spatial_downscale = 0; // the artifact's scale_factor_spatial: pixels per latent row
    VaeStage mid_stage{};                // the deepest level: two resnets and one attention, no doubling
    std::vector<VaeStage> up_stages;     // the decoder's up blocks, deepest first: the graph AND the ledger
    std::uint32_t final_channels    = 0; // the last stage's width plus the artifact's out_channels
    std::uint32_t denorm_elements   = 0; // the per-channel latent mean/std the route normalizes with
    // The calibration ONLY: the bytes the route holds resident beyond the terms the ledger enumerates
    // (the top-level projections, the op scratch, the interlude's floor). It is a measurement of THIS
    // build, so read_vae_topology leaves it zero and the family's binding stamps it from
    // kQwenImageWeightResidentBytes; check_vae_topology refuses an unstamped topology, because an
    // admission decision taken from a zero allowance would refuse nothing at all.
    std::uint64_t weight_resident_bytes = 0;

    friend bool operator==(const VaeTopology&, const VaeTopology&) = default;
};

} // namespace infernix::models::qwen_image

// The readers return std::expected, which the project compiles per target at C++23 (the CUDA lane is
// pinned to C++20, where the standard library does not provide std::expected). The records above are
// plain data and stay visible to every translation unit; the reader declarations appear exactly where
// std::expected exists, which is where they are called -- the host binding path. They are never called
// from a device translation unit.
#if !defined(__CUDACC__)
#    if defined(_MSVC_LANG)
#        if defined(_HAS_CXX23) && _HAS_CXX23
#            define INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS 1
#        endif
#    elif __cplusplus >= 202302L
#        define INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS 1
#    endif
#endif

#ifdef INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
#    include <nlohmann/json.hpp>

#    include <expected>

namespace infernix::models::qwen_image {

// Read and verify the artifact's scheduler document. Refusals name the key and, for a deviating value,
// the value itself, so the operator who passed --image-artifact sees WHY the artifact was refused; the
// caller prefixes which artifact it concerns. Every key the supported path reads must be present with
// the JSON type it promises, and every mode flag is verified, not just the six numbers:
// use_dynamic_shifting == true, time_shift_type == "exponential", use_karras_sigmas == false,
// use_beta_sigmas == false, use_exponential_sigmas == false, invert_sigmas == false, shift == 1.0,
// stochastic_sampling == false. A key the supported path has not verified is an unhandled scheme and is
// refused, because the schedule it would produce is not this schedule.
[[nodiscard]] std::expected<SchedulerConstants, std::string> read_scheduler_config(const nlohmann::json& config);

// Read and verify the DiT rope axes from the artifact's "dit" component config -- the checkpoint's own
// transformer/config.json, which the converter packed verbatim. "attention_head_dim" must be a JSON
// integer and "axes_dims_rope" a JSON array of exactly three positive EVEN JSON integers whose sum is
// that head width; anything else is a refusal naming the key, the offending value, and both numbers.
// Unlike the scheduler document, this one carries the whole transformer architecture, so keys the rope
// rotation does not read are the recipe's parameter contract's business and are not judged here. ONE
// exception, and it is a fail-closed check rather than a source of values: if the artifact DOES declare a
// "rope_theta", it must be a JSON float equal to this build's implementation default (rope.h,
// kQwenImageRopeTheta) or the artifact is refused naming the key, the declared value and the authority it
// was compared against. Absent is the shipped case and stays absent -- theta is never read FROM the
// artifact (the emitter always rotates with the code's constant), but an artifact that declares another
// base describes weights trained under a different rotary scheme, and drawing them at the code's base in
// silence is the divergence this lane refuses everywhere else.
[[nodiscard]] std::expected<DitRopeAxes, std::string> read_dit_rope_axes(const nlohmann::json& config);

// Read and verify the VAE decoder's topology from the artifact's "vae" component config -- the
// checkpoint's own vae/config.json, which the converter packed verbatim
// (tools/convert/qwen_image.py: records["vae"] = {"config": _load_config(root / "vae" / "config.json")}).
// Like the rope axes document this one carries the whole architecture, so keys the decoder route does
// not read are the recipe's parameter contract's business and are not judged here; the strictness that
// does apply is the same: a JSON integer only ever an integer (144 ok, 144.0 refused), a boolean only
// ever a boolean, no default for an absent key, and every refusal names the key and prints the value as
// the artifact declared it.
//
// Which keys, and why each one is the artifact's:
//   "dim_mult"             the five levels' widths, scaled by "decoder_base_dim": the channels every
//                          stage's convolutions carry, and the only place the 1152/576/288/144 widths
//                          come from. Exactly "kVaeUpBlocks" positive JSON integers.
//   "decoder_base_dim"     the shallowest level's width, which is also the width norm_out normalizes
//                          at before the final convolution.
//   "num_res_blocks"       NOT adopted: the graph's dup_up() runs three resnets per up block, so the
//                          artifact's declaration must say two (three = num_res_blocks + 1, the
//                          diffusers chain's convention) or the artifact is refused naming the value.
//   "z_dim"                the latent plane width: the DiT's image_in is the same number, and
//                          post_quant_conv / conv_in take exactly that many channels in.
//   "scale_factor_spatial" the pixels per latent row; verified, not merely adopted, because the number
//                          of upsampling stages the chain declares must produce exactly it (16 == 2^4
//                          for five blocks whose last one does not upsample).
//   "out_channels"         the final convolution's output channels (the RGBA the route hands back),
//                          which with the last stage's width makes the decode's last two planes.
//   "latents_mean"/"latents_std"
//                          the per-channel denormalization pair, whose length the latent width has to
//                          agree with -- an artifact that disagrees with itself is refused naming both.
//   "is_residual"          the graph adds a residual around every resnet and has no branch for an
//                          artifact that declares otherwise, so the declaration is verified.
//   "attn_scales"          must be the empty array: this build places its one attention in the mid
//                          block, and a non-empty list would describe blocks whose attention never runs.
//   "temperal_downsample"  the checkpoint's (misspelled) per-block temporal claim, mirrored by the
//                          graph's dupup3d shortcut factor; the pure-2D image route implements one
//                          scheme, so what the artifact declares is verified against it.
//
// VaeTopology::weight_resident_bytes is deliberately left zero: it is a calibration of this build, not
// a declaration of the checkpoint's.
[[nodiscard]] std::expected<VaeTopology, std::string> read_vae_topology(const nlohmann::json& config);

} // namespace infernix::models::qwen_image
#endif // INFERNIX_IMAGE_READS_ARTIFACT_CONSTANTS
