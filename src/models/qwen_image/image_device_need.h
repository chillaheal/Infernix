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

// How many DiT block slots the shipped route keeps resident at once -- and the ONE number the need
// model, the serving layer and the ledger cert must all agree on. The pipeline's CopyLane double-
// buffers exactly this many slots (a static_assert in image_pipeline.cu pins it to 2), the serving
// layer stamps it onto ImageDeviceNeedRequest::blocks_resident, and the cert requests the same value
// the allowance below was calibrated against. A single symbol so `resident_weights` can never drift
// from the slots actually allocated, nor from the blocks_resident the 3072 MiB allowance measured.
inline constexpr std::uint32_t kImageResidentBlocks = 2;

// The operands and op scratch the inventory above does not enumerate: the DiT's top-level
// projections, the VAE's weights, and whatever workspace the linear/conv/attention ops take per
// call. An over-estimated need refuses a job that might have run; an under-estimated one OOMs it.
//
// Two calibrations set this term:
//   - the ledger qualification (test_image_need_ledger_real.cu) proves predicted >= measured for the
//     certified inputs, and
//   - the stage bench (image-spike/stage_bench_A_full_overlap.err.log) gives the REAL per-step ledger
//     peak a live job reaches -- higher than the cert's, because the cert feeds zero-filled trivial
//     inputs while a served job carries a real prompt/negative/reference. The cert alone would let
//     this drift below what a live job actually holds.
//
// Subtracting the enumerated structural terms (dit_activations + latent_and_velocity + vae_peak +
// rope_and_tables) from the real peak leaves the residual this term must absorb: ~772 MiB at 512,
// ~322 at 768, and ~1180 at 768+ref / 768+CFG (the ref/CFG rows, where the real sequence is longest
// and the cert's trivial inputs understate it most). The 1024 row's structural terms alone already
// exceed its real peak, so its residual is negative. So the largest residual to cover is ~1184 MiB;
// 3072 MiB leaves ~1.9 GiB on top of it for the 512 MiB interlude floor plus whatever the KV pool
// and the LLM restore hold during the interlude. That puts the predicted total at ~1.6-2.5x the real
// peak on every certified row -- down from the ~4x this term carried when it was the whole-measured-
// peak proxy (6183, then 5972 after the second resident block was counted explicitly in bundle B
// fas 2) -- without ever dropping predicted below the measured peak plus the 512 MiB floor.
inline constexpr std::uint64_t kImageWeightResidentBytes = 3072ULL * kImageMiB;

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

// The bound ViT's shape, as far as the ledger needs it: enough to derive the streamed block's roles
// and the eleven live planes of vision_tower.cu rather than hardcode a number. All zero (or hidden
// zero) means no vision lane -- the certified text-to-image route -- and the vision term is then
// exactly zero, byte for byte.
struct VisionLaneDims {
    std::uint32_t hidden      = 0; // ViT hidden width (1152 in the shipped config)
    std::uint32_t intermediate = 0; // the ViT MLP's inner width
    std::uint32_t head_dim    = 0;
    std::uint32_t patch       = 0; // patch size in pixels (16)
    std::uint32_t temporal    = 0; // frames per patch (2)
    std::uint32_t merge       = 0; // spatial_merge_size (2): merge*merge patches make one merged row
    std::uint32_t out_hidden  = 0; // the merger's output width == the text tower's hidden
};

// The worst legal reference plane the ViT can ever be fed: prepare_reference_image resizes every
// reference to the 1024x1024 area target on the 32-grid (reference_image.cpp:21,155-158), so the
// largest patch grid is 1024/16 x 1024/16. The reference's size is independent of the requested
// OUTPUT size, which is why this is a constant of the route and not of the geometry.
inline constexpr std::uint64_t kVisionWorstReferencePatches = 64ULL * 64ULL;

struct ImageDeviceNeedRequest {
    ImageGeometry geometry;             // the target image's snapped geometry
    DiTDims       dims;                 // the bound DiT's dimensions
    std::uint32_t text_tokens     = 0;  // kept prompt rows, the <|image_pad|> rows included
    std::uint32_t negative_tokens = 0;  // negative-prompt rows (0 without true CFG), same form
    std::uint32_t ref_tokens      = 0;  // reference-image latent tokens ahead of the target
    std::uint32_t blocks_resident = kImageResidentBlocks; // the shipped route's resident slots (2)
    // The edit route only. Each vision token is one <|image_pad|> row INSIDE text_tokens and expands
    // to four joint rows, which ref_tokens already counts (ref_tokens == 4 * vision_tokens). The joint
    // sequence is therefore non-pad text + ref + target: passing the VLM length together with
    // ref_tokens would count those rows twice (for a portrait reference, ~1014 rows too many).
    std::uint32_t   vision_tokens = 0;
    VisionLaneDims  vision;             // zero == no vision lane, and the vision term is then zero
};

struct ImageDeviceNeed {
    std::uint64_t dit_activations     = 0; // block workspace + joint/ping-pong/norm buffers at s_max
    std::uint64_t latent_and_velocity = 0; // [ref+target] latent, its hidden, the text stages
    std::uint64_t resident_weights    = 0; // the allowance above plus the resident block(s)
    std::uint64_t vae_peak            = 0; // the widest simultaneous set of decode planes
    std::uint64_t rope_and_tables     = 0; // both rope tables and the shared constants
    std::uint64_t vision_peak         = 0; // the ViT lane: one streamed block + its live set

    [[nodiscard]] std::uint64_t total() const noexcept {
        return dit_activations + latent_and_velocity + resident_weights + vae_peak + rope_and_tables +
               vision_peak;
    }
};

// The need, term by term as the pipeline allocates. Every sequence-shaped buffer follows
// `s_max = max(non_pad_text + ref + target, negative_non_pad_text + ref + target)`, exactly the
// `s_max` the job guards with, so references and CFG raise the number by construction rather than by
// a tuned multiplier. `non_pad_text` is the prompt with its <|image_pad|> rows removed: those rows
// are the reference's vision tokens, and the joint sequence carries each of them four times, already
// counted in `ref`. The text-side stages still size by the FULL kept length (the tower's output
// carries the pad rows), which is what `text_max` is for.
[[nodiscard]] constexpr ImageDeviceNeed image_device_need(const ImageDeviceNeedRequest& req) noexcept {
    const std::uint64_t target = req.geometry.tokens; // hl * wl
    const std::uint64_t all_img = static_cast<std::uint64_t>(req.ref_tokens) + target;
    const std::uint64_t d       = req.dims.hidden;
    const std::uint64_t mlp     = req.dims.intermediate;
    const std::uint64_t cin     = req.dims.image_in;
    const std::uint64_t vision  = static_cast<std::uint64_t>(req.vision_tokens);
    const std::uint64_t joint_pos =
        (static_cast<std::uint64_t>(req.text_tokens) > vision ? static_cast<std::uint64_t>(req.text_tokens) - vision : 0ULL);
    const std::uint64_t joint_neg =
        (static_cast<std::uint64_t>(req.negative_tokens) > vision ? static_cast<std::uint64_t>(req.negative_tokens) - vision : 0ULL);
    const std::uint64_t s_pos   = joint_pos + all_img;
    const std::uint64_t s_neg   = joint_neg + all_img;
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

    // The ViT lane runs inside the same interlude as the DiT loop, before it: peak == one block's
    // roles + the per-item activation + one merger (vision_tower.h:15), and vision_tower.cu:197-219
    // really does hold the whole live set across the block loop -- eleven planes at
    // [hidden, patches], one at [intermediate, patches], the patch input, three int32/float index
    // planes and the merger output. Sized by the worst legal reference (kVisionWorstReferencePatches),
    // not the requested one: the reference's size is the route's, independent of the output size, and
    // the probe has to refuse before the yield rather than OOM the resident LLM. Sequential with the
    // decode, so it is SUMMED like vae_peak, never maxed against it.
    if (req.vision.hidden != 0) {
        const std::uint64_t vh   = req.vision.hidden;
        const std::uint64_t vmlp = req.vision.intermediate;
        const std::uint64_t p    = kVisionWorstReferencePatches;
        const std::uint64_t merge = req.vision.merge != 0 ? req.vision.merge : 1U;
        const std::uint64_t patch_width =
            3ULL * (req.vision.temporal != 0 ? req.vision.temporal : 1U) *
            (req.vision.patch != 0 ? req.vision.patch : 1U) *
            (req.vision.patch != 0 ? req.vision.patch : 1U);
        const std::uint64_t merged = p / (merge * merge);
        need.vision_peak =
            11ULL * p * vh * kImageElementBytes +                             // the eleven live planes
            p * vmlp * kImageElementBytes +                                    // d_up
            p * patch_width * kImageElementBytes +                             // d_patches
            p * (2ULL * 4ULL + 4ULL * 4ULL + 4ULL * 4ULL) +                    // pos / idx / weights
            merged * (req.vision.out_hidden != 0 ? req.vision.out_hidden : vh) *
                kImageElementBytes +                                           // d_out, the merger rows
            (4ULL * vh * vh + 3ULL * vmlp * vh +
             2ULL * req.vision.head_dim * kImageElementBytes);                 // one block's roles
    }

    need.resident_weights = kImageWeightResidentBytes +
                            static_cast<std::uint64_t>(req.blocks_resident) * image_block_bytes(req.dims);
    return need;
}

} // namespace infernix::models::qwen_image
