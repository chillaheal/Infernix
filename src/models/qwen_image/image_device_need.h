#pragma once

// One image job's device-memory need, computed from the buffer inventory the production pipeline
// actually allocates (src/models/qwen_image/image_pipeline.cu) rather than from a scaled measurement.
// The serving layer asks for it before the LLM yields any VRAM, so a job that cannot be served is
// refused with nothing moved. The two properties a scaled area guess cannot have: a reference image
// and true CFG both raise the need, because the pipeline sizes every sequence-shaped buffer by the
// longer of the two joint sequences (image_pipeline.cu:332-342) and carries the reference block ahead
// of the target latent (:328).
//
// The image ledger (image_device_bytes_peak(), image_pipeline.h:93-99) is the oracle for this model:
// the qualification test asserts the predicted total is never below the measured peak for the same
// inputs, and the allowance below is what that test tightens.

#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_weights.h"

#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen_image {

// BF16 is the element type of every image activation, weight upload and intermediate.
inline constexpr std::uint64_t kImageElementBytes = 2;
inline constexpr std::uint64_t kImageMiB          = 1024ULL * 1024ULL;

// The operands and op scratch the inventory above does not enumerate: the DiT's top-level
// projections, the VAE's weights, and whatever workspace the linear/conv/attention ops take per
// call. The starting value was the measured peak this project has -- 6183 MiB for the certified
// 512x512 route (image-spike/blackout_step8d.outer.log:30) -- deliberately conservative: an
// over-estimated need refuses a job that might have run, while an under-estimated one OOMs it. The
// ledger qualification (tests/models/qwen_image/test_image_need_ledger_real.cu) measures the split
// and lowers this to what the route actually holds.
//
// Bundle B fas 2: with the copy-stream prefetch, TWO DiT block slots are resident at once and are
// counted explicitly by `blocks_resident * image_block_bytes` below. The old 6183 MiB allowance was
// set when only one block was resident, so it already carried one block's worth inside this slack
// term. Lowering it by one image_block_bytes (~211 MiB for the shipped dims) keeps the predicted
// total -- and therefore the sizes the serving layer admits -- exactly as §8 measured them, while
// modelling the second resident block honestly instead of double-counting it.
inline constexpr std::uint64_t kImageWeightResidentBytes = 5972ULL * kImageMiB;

// One DiT block resident on device: the seven FP8 RowScale roles (four [hidden,hidden] and three
// [intermediate,hidden], one byte per code) plus the two BF16 head norms. Derived from the bound
// dims, so a recipe change moves it with them.
[[nodiscard]] constexpr std::uint64_t image_block_bytes(const DiTDims& dims) noexcept {
    const std::uint64_t d = dims.hidden, mlp = dims.intermediate;
    return 4ULL * d * d + 3ULL * mlp * d + 2ULL * dims.head_dim * kImageElementBytes;
}

// One decoder stage: the channels its widest planes carry and how many full [C,H,W] planes the code
// holds live at once. The counts are read off image_pipeline.cu: `att` keeps the qkv buffer (3C),
// its attention output (C) and the still-live input (C) at the spatial_attn call (:207-222);
// `dup_up` holds the upsampled input, the duplicated shortcut and the conv output at the DOUBLED
// resolution (:252-264); `dresnet` holds its input, shortcut and two conv outputs (:161-192).
struct VaeStage {
    std::uint32_t channels_in;
    std::uint32_t channels_out;
    std::uint32_t planes;
    bool          doubles_side; // the stage works at the doubled side after its upsample
};

// The decoder's fixed chain as image_pipeline.cu:620-665 runs it (the channels are the recipe's).
inline constexpr VaeStage kVaeMidStage{1152, 1152, 5, false};
inline constexpr VaeStage kVaeUpStages[5] = {
    {1152, 1152, 3, true},
    {1152, 1152, 3, true},
    {1152, 576, 3, true},
    {576, 288, 3, true},
    {288, 144, 3, false},
};

// The last two planes of the decode: the norm_out stage (144 channels) and the 4-channel conv_out,
// both at full pixel resolution.
inline constexpr std::uint64_t kVaeFinalChannels = 144ULL + 4ULL;

struct ImageDeviceNeedRequest {
    ImageGeometry geometry;             // the target image's snapped geometry
    DiTDims       dims;                 // the bound DiT's dimensions
    std::uint32_t text_tokens     = 0;  // prompt rows the joint sequence carries
    std::uint32_t negative_tokens = 0;  // negative-prompt rows (0 without true CFG)
    std::uint32_t ref_tokens      = 0;  // reference-image latent tokens ahead of the target
    std::uint32_t blocks_resident = 1;  // DiT blocks resident at once (the streamed route: one)
};

struct ImageDeviceNeed {
    std::uint64_t dit_activations     = 0; // block workspace + joint/ping-pong/norm buffers at s_max
    std::uint64_t latent_and_velocity = 0; // [ref+target] latent, its hidden, the text stages
    std::uint64_t resident_weights    = 0; // the allowance above plus the resident block(s)
    std::uint64_t vae_peak            = 0; // the widest simultaneous set of decode planes
    std::uint64_t rope_and_tables     = 0; // both rope tables and the shared constants

    [[nodiscard]] std::uint64_t total() const noexcept {
        return dit_activations + latent_and_velocity + resident_weights + vae_peak + rope_and_tables;
    }
};

// The need, term by term as the pipeline allocates. Every sequence-shaped buffer follows
// `s_max = max(text + ref + target, negative + ref + target)`, exactly the `s_max` the job guards
// with, so references and CFG raise the number by construction rather than by a tuned multiplier.
[[nodiscard]] constexpr ImageDeviceNeed image_device_need(const ImageDeviceNeedRequest& req) noexcept {
    const std::uint64_t target = req.geometry.tokens; // hl * wl
    const std::uint64_t all_img = static_cast<std::uint64_t>(req.ref_tokens) + target;
    const std::uint64_t d       = req.dims.hidden;
    const std::uint64_t mlp     = req.dims.intermediate;
    const std::uint64_t cin     = req.dims.image_in;
    const std::uint64_t s_pos   = static_cast<std::uint64_t>(req.text_tokens) + all_img;
    const std::uint64_t s_neg   = static_cast<std::uint64_t>(req.negative_tokens) + all_img;
    const std::uint64_t s_max   = s_neg > s_pos ? s_neg : s_pos;
    const std::uint64_t text_max =
        static_cast<std::uint64_t>(req.negative_tokens > req.text_tokens ? req.negative_tokens
                                                                        : req.text_tokens);

    ImageDeviceNeed need;
    // The per-block scratch is allocated once for the whole loop: thirteen buffers at sD and three at
    // sMlp (image_pipeline.cu:412-420). The per-step buffers add the joint sequence, its ping-pong
    // partner and the two norm stages at sD, and the velocity output at [s_max,Cin] (:434-446).
    need.dit_activations = (13ULL + 4ULL) * s_max * d * kImageElementBytes +
                           3ULL * s_max * mlp * kImageElementBytes +
                           s_max * cin * kImageElementBytes;
    // The [ref+target] latent the loop evolves, its projected image hidden, and both text passes'
    // embed/front-end stages at the longer text length (:426-433, :448-455).
    need.latent_and_velocity =
        all_img * cin * kImageElementBytes + all_img * d * kImageElementBytes +
        3ULL * text_max * d * kImageElementBytes;
    // The interleaved rope tables are S * 64 * 2 floats per pass, held as BF16 on device (:379-397),
    // plus the shared ones/zeros/tnorm constants (:366-376).
    need.rope_and_tables =
        (s_pos + s_neg) * 64ULL * 2ULL * kImageElementBytes + 5ULL * d * kImageElementBytes;
    // The widest decoder stage: `planes` full planes at the stage's working side. Each up-stage
    // doubles the side before its widest planes exist, so the late low-channel stages dominate.
    std::uint64_t hl = req.geometry.hl, wl = req.geometry.wl;
    const auto stage_bytes = [](const VaeStage& stage, std::uint64_t sh, std::uint64_t sw) {
        const std::uint64_t channels =
            stage.channels_in > stage.channels_out ? stage.channels_in : stage.channels_out;
        return static_cast<std::uint64_t>(stage.planes) * channels * sh * sw * kImageElementBytes;
    };
    need.vae_peak = stage_bytes(kVaeMidStage, hl, wl);
    for (const auto& stage : kVaeUpStages) {
        if (stage.doubles_side) {
            hl *= 2;
            wl *= 2;
        }
        const std::uint64_t bytes = stage_bytes(stage, hl, wl);
        if (bytes > need.vae_peak) { need.vae_peak = bytes; }
    }
    const std::uint64_t final_planes = kVaeFinalChannels *
                                       static_cast<std::uint64_t>(req.geometry.height) *
                                       static_cast<std::uint64_t>(req.geometry.width) *
                                       kImageElementBytes;
    if (final_planes > need.vae_peak) { need.vae_peak = final_planes; }

    need.resident_weights = kImageWeightResidentBytes +
                            static_cast<std::uint64_t>(req.blocks_resident) * image_block_bytes(req.dims);
    return need;
}

} // namespace infernix::models::qwen_image
