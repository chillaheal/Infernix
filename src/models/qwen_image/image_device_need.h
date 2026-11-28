#pragma once

// One image job's device-memory need, computed from the buffer inventory the production pipeline
// actually allocates (src/models/qwen_image/image_pipeline.cu) rather than from a scaled measurement.
// The serving layer asks for it before the LLM yields any VRAM, so a job that cannot be served is
// refused with nothing moved. The two properties a scaled area guess cannot have: a reference image
// and true CFG both raise the need, because the pipeline sizes every sequence-shaped buffer by the
// longer of the two joint sequences (image_pipeline.cu:332-342) and carries the reference block ahead
// of the target latent (:328).
//
// The image ledger (image_device_bytes_peak(), image_pipeline.h) is the oracle for this model:
// the qualification test asserts the predicted total is never below the measured peak for the same
// inputs, and the resident calibration in models/qwen_image/image_weights.h
// (kQwenImageWeightResidentBytes, carried in on the request's `vae`) is what that test tightens.

#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_weights.h"

// require_patches_within_vision_ledger below refuses by throwing a std::runtime_error, the same
// fail-closed refusal the vision tower and every other image-lane guard raises.
#include <cstdint>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen_image {

// BF16 is the element type of every image activation, weight upload and intermediate.
inline constexpr std::uint64_t kImageElementBytes = 2;
inline constexpr std::uint64_t kImageMiB          = 1024ULL * 1024ULL;

// How many DiT block slots the shipped route keeps resident at once -- and the ONE number the need
// model, the serving layer and the ledger cert must all agree on. The pipeline's CopyLane double-
// buffers exactly this many slots (a static_assert in image_pipeline.cu pins it to 2), the serving
// layer stamps it onto ImageDeviceNeedRequest::blocks_resident, and the cert requests the same value
// the resident calibration was measured with. A single symbol so `resident_weights` can never drift
// from the slots actually allocated, nor from the blocks_resident that calibration (stamped onto the
// VaeTopology by the binding, and multiplied by this constant below) was taken against.
inline constexpr std::uint32_t kImageResidentBlocks = 2;

// One DiT block resident on device: the seven FP8 RowScale roles (four [hidden,hidden] and three
// [intermediate,hidden], one byte per code) plus the two BF16 head norms. Derived from the bound
// dims, so a recipe change moves it with them.
[[nodiscard]] constexpr std::uint64_t image_block_bytes(const DiTDims& dims) noexcept {
    const std::uint64_t d = dims.hidden, mlp = dims.intermediate;
    return 4ULL * d * d + 3ULL * mlp * d + 2ULL * dims.head_dim * kImageElementBytes;
}

// The decoder chain this ledger prices is NOT a table kept here: its channels, its latent width, its
// final conv_out width and its resident calibration are the artifact's own declaration, read by
// read_vae_topology into the VaeTopology (models/qwen_image/artifact_constants.h) that the family's
// binding stamps and verifies against every bound weight, and handed in on the request below. The CUDA
// decode graph reads the SAME object off the bound VaeWeights (image_pipeline.cu), so the bytes the
// graph allocates and the bytes the ledger reserves can no longer be two different guesses -- and an
// artifact that declares a decoder this build does not run is refused by the binding instead of being
// priced against another family's geometry. The one code-owned field the terms below multiply by,
// VaeStage::planes (how many full [C,H,W] planes the route holds live at a stage), is documented where
// it is set: it is a measurement of image_pipeline.cu's att()/dup_up()/dresnet(), not a claim of the
// checkpoint's.

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

// The worst legal reference plane the ViT can ever be fed -- DERIVED here rather than asserted, because
// it is the number the ledger below BILLS the lane at (image_device_need() multiplies vision_peak by it)
// and the number require_patches_within_vision_ledger enforces, so the two must stay one constant.
//
// How the route makes the plane (prepare_reference_planes, src/serve/reference_image.cpp:320-352): from
// the decoded reference's size (W, H) it computes the two exact targets
//     exact_w = sqrt(kReferenceTargetArea * W / H)   and   exact_h = sqrt(kReferenceTargetArea * H / W)
// whose product is exactly kReferenceTargetArea (1024*1024, reference_image.cpp:21), and rounds EACH
// SIDE INDEPENDENTLY to a multiple of 32 with serve::snap_to_32 (src/serve/openai_images.cpp:75-89 --
// Python round()'s half-to-even at the 32-grid). It then refuses anything whose snapped side fell below
// 32 (reference_image.cpp:350-352), and run_vision_tower is handed the plane as a
// {1, target_h/16, target_w/16} patch grid (patch 16 x spatial_merge 2, src/serve/reference_image.h:56-63).
// Two consequences, and the second is why this is arithmetic rather than a square:
//   * the snap moves a side by at most half the grid (16 px) and it can round BOTH sides UP, so a
//     non-square reference lands ABOVE the target area: a 4:3 reference is 1184x896 = 4144 patches, more
//     than the square case's 4096. That is what the 64x64 claim this header used to carry got wrong, and
//     a guard sitting behind that claim turned an ordinary photograph edit into a refusal;
//   * the widest aspect the route still admits is the one whose SHORT side just clears the gate: its
//     exact short side must be above 16 (at exactly 16 the half-grid tie rounds to the even 0, which the
//     route refuses), so its exact long side is kReferenceTargetArea/16 = 65536 and that snaps to 65536
//     itself -- 65536x32, at the 16-wide patch 4096x2 = 8192 patches, exactly TWICE the square reference.
//     The lane has to be billed for that plane, not for the square one.
//
// The loop below states that arithmetic instead of handing over its answer, and in integers only (no
// floating point, so no rounding mode can move it): for every snapped side the route's own gate admits
// as the SHORT side it takes the longest snapped long side whose (side - half-grid slack) still leaves
// the exact partner inside the target area, and keeps the most patches. Every plane the route can
// produce meets that condition -- each snapped side is within slack of its exact value and the exact
// values multiply to the area -- so what is scanned is a SUPERSET: it can over-bill, never under-bill.
// tests/test_image_device_need.cpp enumerates the REAL route (the exported serve::snap_to_32, and the
// real prepare_reference_planes on encoded references) to prove the constant is reached and never
// exceeded, so it is not a padded guess and a change to the route's snapping is caught there.
//
// The three route constants below are restated here because src/serve owns them and src/models may not
// depend on src/serve; that host suite is the tie back to the route, and the serve boundary refuses a
// plane this header did not bill (reference_image.cpp), so a drift answers the client 400 instead of 500.
inline constexpr std::uint64_t kReferenceTargetAreaPx = 1024ULL * 1024ULL; // reference_image.cpp:21
inline constexpr std::uint64_t kReferenceSnapGridPx   = 32ULL;             // serve::snap_to_32's grid
inline constexpr std::uint64_t kVisionPatchSidePx     = 16ULL;             // the ViT's patch side

[[nodiscard]] inline consteval std::uint64_t worst_reference_plane_patches(
    std::uint64_t target_area_px, std::uint64_t snap_grid_px, std::uint64_t patch_side_px) noexcept {
    const std::uint64_t slack = snap_grid_px / 2ULL; // the snap moves a side by at most half a grid
    std::uint64_t worst = 0ULL;
    // The route refuses a snapped side below the grid, so that is the shortest side to try; past the
    // square the two roles have merely swapped, so no bound beyond the area itself is needed.
    for (std::uint64_t short_px = snap_grid_px; short_px <= target_area_px; short_px += snap_grid_px) {
        const std::uint64_t short_exact = short_px - slack; // what its exact side cannot fall below
        if (short_exact == 0ULL) { continue; }
        // The exact long side is at most area / short_exact, and a snapped side may sit slack above its
        // exact value: the longest legal snapped long side is the last multiple of the grid whose own
        // exact floor still fits.
        const std::uint64_t long_exact = target_area_px / short_exact;
        const std::uint64_t long_px    = ((long_exact + slack) / snap_grid_px) * snap_grid_px;
        if (long_px < snap_grid_px) { break; } // no partner left on this side of the square
        const std::uint64_t patches = (long_px / patch_side_px) * (short_px / patch_side_px);
        if (patches > worst) { worst = patches; }
    }
    return worst;
}

inline constexpr std::uint64_t kVisionWorstReferencePatches =
    worst_reference_plane_patches(kReferenceTargetAreaPx, kReferenceSnapGridPx, kVisionPatchSidePx);

// The same promise the way the boundary that produces the plane reads it, in pixels: src/serve has no
// business knowing the ViT's patch size, and a snapped side is always a multiple of the grid, so the
// division back to patches is exact.
inline constexpr std::uint64_t kVisionWorstReferencePlanePixels =
    kVisionWorstReferencePatches * kVisionPatchSidePx * kVisionPatchSidePx;

// The one thing about this derivation that is not a restatement of its own answer: the snap's slack
// really does lift the worst plane above the square reference, which is the whole of the C1 finding.
// (What it is NOT allowed to do is freeze the number -- tests/test_image_device_need.cpp is what proves
// the derived maximum is reached, and a route change must be re-derived from the route, not tuned here.)
static_assert(kVisionWorstReferencePatches >
                  kReferenceTargetAreaPx / kVisionPatchSidePx / kVisionPatchSidePx,
              "the worst legal reference plane must sit ABOVE the square reference's patch count: the "
              "32-grid snap rounds each side independently and can round both up, so a non-square "
              "reference lands outside kReferenceTargetArea. If this fires, the derivation stopped "
              "modelling the snap.");

// That number is not an estimate the caller may exceed: it is the CEILING over the charge. The ledger
// below bills need.vision_peak for the plane the request DECLARES (ImageDeviceNeedRequest::
// reference_patches -- the widest of the references prepare_reference_planes produced), and the serving
// layer yields on the total it bills: never more than the tower will be handed, never less.
// require_patches_within_vision_ledger keeps both ends honest -- run_vision_tower calls it on the count
// its caller hands it BEFORE a single plane is sized, so a caller handing it a bigger plane than the
// admission billed for would hold live device bytes no decision priced: the joint ceiling admits 20480
// merged rows, which would be 81920 patches, ten times the worst legal reference -- ~3.0 GiB of live
// activations against the ~325 MiB the ceiling itself bills (the per-patch terms at the shipped ViT's
// shape) -- and is refused instead of starving the resident LLM on a stale number. That is the exact
// blindness this branch exists to close, and the blindness was in the reader:
// vision_tower.cu sized every live plane from the CALLER's count and refused only int overflow.
//
// The guard therefore lives beside the charge it protects, so the two cannot drift apart, and
// run_vision_tower calls it on the count its caller hands it BEFORE a single plane is sized
// (src/models/qwen_image/vision_tower.cu). It is a host function on purpose: the boundary is arithmetic,
// so the suite that owns the charge (tests/test_image_device_need.cpp) can pin both ends of it -- the
// derived worst plane admitted, one patch above it refused -- without CUDA in the way, which the production route itself cannot offer a host
// test (the CUDA target is deliberately kept out of host tests, see src/models/qwen_image/sources.cmake).
inline void require_patches_within_vision_ledger(std::uint64_t patches_count) {
    if (patches_count > kVisionWorstReferencePatches) {
        // The message is the derivation, not just the number: whoever hits it must be able to tell a
        // pathological reference (the client's, refused at the serve boundary with a 400 long before
        // here) from a ledger whose snap stopped matching the route's (the tree's, and the reason the
        // square 64x64 this guard used to quote was wrong).
        throw std::runtime_error(
            "vision tower: " + std::to_string(patches_count) + " patches is more than the " +
            std::to_string(kVisionWorstReferencePatches) + " the worst legal reference plane produces -- "
            "the " +
            std::to_string(kReferenceTargetAreaPx) + "-pixel reference target of "
            "src/serve/reference_image.cpp, each side snapped to the " +
            std::to_string(kReferenceSnapGridPx) + "-grid (which rounds UP, and can round both sides up) "
            "at the " +
            std::to_string(kVisionPatchSidePx) + "-wide ViT patch, at the widest aspect the route's own "
            "below-" + std::to_string(kReferenceSnapGridPx) + "-pixel gate still admits (that plane is " +
            std::to_string(kVisionWorstReferencePlanePixels) + " pixels); and that is the number the VRAM "
            "ledger bills the ViT at, so a larger plane would hold live device bytes no admission "
            "decision priced and the route refuses it rather than run the tower under-priced");
    }
}

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
    // The edit route only, and the pair that says what the ViT lane is CHARGED for. The tower runs the
    // references one at a time into the same live set (src/models/qwen_image/vision_tower.cu sizes every
    // plane from the caller's count), so the lane costs the WIDEST of the planes this job will hand it --
    // not the widest the snap could ever produce, which is what the lane used to bill and what made an
    // ordinary photographic edit pay ~160 MiB for bytes no allocation ever occupied. `reference_patches`
    // is that widest plane in patches ((width/16) * (height/16), which is also the reference's own latent
    // token count -- src/serve/image_generation_service.cpp prices ref_tokens the same way), and
    // `reference_lanes` is how many references the tower will run. Both are filled from the SAME
    // prepare_reference_planes answer the job encodes, so the price and the work cannot disagree.
    // require_patches_within_vision_ledger stays the ceiling over both: any plane above
    // kVisionWorstReferencePatches is refused, admitted plane or not.
    std::uint32_t   reference_lanes   = 0;
    std::uint32_t   reference_patches = 0;
    // The decoder geometry the job's OWN artifact declared, read by read_vae_topology and verified against
    // every bound decoder weight by check_vae_topology (image_weights.cpp) -- the same object the CUDA
    // graph decodes from, which is why the ledger has no decoder constants of its own any more. A
    // reference, so the caller cannot hand in a copy that drifted from the bound one, and it has no
    // default: every construction site must say which artifact's topology it is pricing, and a job whose
    // binding never ran cannot be admitted at all.
    const VaeTopology& vae;
};

struct ImageDeviceNeed {
    std::uint64_t dit_activations     = 0; // block workspace + joint/ping-pong/norm buffers at s_max
    std::uint64_t latent_and_velocity = 0; // [ref+target] latent, its hidden, the text stages
    std::uint64_t resident_weights    = 0; // the topology's resident calibration + the resident block(s)
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
    // The widest simultaneous planes of a stage are its OUTPUT-channel buffers: `dup_up` holds the
    // upsampled input, the duplicated shortcut and the conv output -- all `channels_out` wide -- at the
    // doubled side (image_pipeline.cu:335,339,343), which is the stage's high-water and the largest
    // vae term on a big canvas. The previous max(channels_in, channels_out) charged the up-projection's
    // INPUT width to those planes, so on the {576,288} stage it doubled the peak: the measured ledger
    // high-water at 2016 (12295 MiB, image_device_bytes_peak()) sits under the max() figure's vae term
    // alone (13395), and the corrected channels_out figure keeps the total above the measured peak plus
    // the 512 MiB reservation floor on every certified row (min clearance 1217 MiB at 2048).
    const auto stage_bytes = [](const VaeStage& stage, std::uint64_t sh, std::uint64_t sw) {
        return static_cast<std::uint64_t>(stage.planes) * stage.channels_out * sh * sw * kImageElementBytes;
    };
    need.vae_peak = stage_bytes(req.vae.mid_stage, hl, wl);
    for (const auto& stage : req.vae.up_stages) {
        if (stage.doubles_side) {
            hl *= 2;
            wl *= 2;
        }
        const std::uint64_t bytes = stage_bytes(stage, hl, wl);
        if (bytes > need.vae_peak) { need.vae_peak = bytes; }
    }
    const std::uint64_t final_planes = req.vae.final_channels *
                                       static_cast<std::uint64_t>(req.geometry.height) *
                                       static_cast<std::uint64_t>(req.geometry.width) *
                                       kImageElementBytes;
    if (final_planes > need.vae_peak) { need.vae_peak = final_planes; }

    // The ViT lane runs inside the same interlude as the DiT loop, before it: peak == one block's
    // roles + the per-item activation + one merger (vision_tower.h:15), and vision_tower.cu:207-232
    // really does hold the whole live set across the block loop -- eleven planes at
    // [hidden, patches], one at [intermediate, patches], the patch input, three int32/float index
    // planes and the merger output -- every one of them a Buf sized from the CALLER's patch count and
    // freed by RAII before the call returns. So the lane costs the plane this job actually runs:
    // `req.reference_patches`, the widest of the planes prepare_reference_planes produced, because the
    // tower encodes the references one after another into that same live set (a two-reference edit pays
    // the widest, never the sum).
    //
    // It used to be billed at kVisionWorstReferencePatches -- the widest plane the snap could EVER
    // produce -- and that was wrong twice over. It charged a square 1024^2 reference (4096 patches) for
    // the 65536x32 pathological aspect (8192) it will never run, and the yield it spends that on is the
    // one scarce resource on this route: on the deployed engine a 1536x1536 photographic edit was refused
    // with "needs 4915 MiB; the model can yield 5238 MiB with 329 MiB already free and 256 MiB reserved
    // below the headroom" while the gate is need + 512 MiB margin + 256 MiB headroom -- 116 MiB short,
    // with ~160 MiB of the shortfall bought by bytes no allocation would ever occupy.
    // kVisionWorstReferencePatches stays exactly where it was, as the CEILING: run_vision_tower refuses
    // any plane above it (require_patches_within_vision_ledger, called before a single allocation) and
    // check_image_device_need_request refuses an admission that declares one, so the price can never
    // fall under what the tower will be handed -- a serve that ran the tower over a wider plane than it
    // admitted gets a refusal, not an OOM of the resident LLM.
    //
    // Sequential with the decode, so the lane is SUMMED like vae_peak, never maxed against it.
    if (req.vision.hidden != 0) {
        const std::uint64_t vh   = req.vision.hidden;
        const std::uint64_t vmlp = req.vision.intermediate;
        const std::uint64_t p    = req.reference_patches;
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

    need.resident_weights = req.vae.weight_resident_bytes +
                            static_cast<std::uint64_t>(req.blocks_resident) * image_block_bytes(req.dims);
    return need;
}

// The serving layer calls this BEFORE it prices a job and refuses on its words. It returns
// std::optional<std::string> rather than std::expected because this header is compiled by translation
// units pinned to C++20 (src/serve builds with -std:c++20, and the CUDA lane too), which is exactly why
// check_vae_topology -- a C++23-only host gate -- lives in a different header. Same stance as that one:
// an admission that cannot name what it is reserving for is refused, not guessed.
//
// image_device_need() itself stays constexpr and noexcept: pure arithmetic that charges what a request
// declares. A request that declares a lane without saying what runs through it has to be stopped at the
// door instead of being billed zero for a lane that then holds live bytes nobody reserved. Each branch
// names the field it refused, because the caller is another engineer fixing an admission, not a client.
[[nodiscard]] inline std::optional<std::string> check_image_device_need_request(
    const ImageDeviceNeedRequest& req) {
    if (req.vision.hidden == 0) {
        // No lane, so nothing may name references to it: a text-to-image job that declares reference
        // planes has lost its lane dimensions somewhere upstream, and pricing it as text-to-image would
        // reserve nothing for a tower that then runs.
        if (req.reference_lanes != 0 || req.reference_patches != 0) {
            return std::string(
                "the request declares " + std::to_string(req.reference_lanes) + " reference lane(s) at " +
                std::to_string(req.reference_patches) +
                " patches but carries no vision lane (vision.hidden == 0), so the reference tower that "
                "will run has nothing reserved for it: hand the request the bound vision lane's shape");
        }
        return std::nullopt;
    }
    if (req.reference_lanes == 0 || req.reference_patches == 0) {
        return std::string(
            "the request declares a vision lane at " + std::to_string(req.vision.hidden) +
            " hidden but no reference plane (reference_lanes == " + std::to_string(req.reference_lanes) +
            ", reference_patches == " + std::to_string(req.reference_patches) +
            "): the tower's live set is sized from the plane it is handed, so the caller must declare the "
            "widest of the reference planes prepare_reference_planes produced, in patches");
    }
    // The same bound run_vision_tower enforces on the count it is handed, checked here too so an
    // admission never reserves for a plane the lane would refuse at run time.
    if (req.reference_patches > kVisionWorstReferencePatches) {
        return std::string("the request declares a reference plane of " +
                           std::to_string(req.reference_patches) +
                           " patches, above the " + std::to_string(kVisionWorstReferencePatches) +
                           " the worst legal reference plane produces and the ceiling the vision "
                           "tower itself refuses");
    }
    return std::nullopt;
}

} // namespace infernix::models::qwen_image
