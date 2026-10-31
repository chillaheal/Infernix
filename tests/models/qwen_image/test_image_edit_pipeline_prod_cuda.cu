// GPU qualification of the EDIT route's ONE denoising step on the INTERLEAVED joint sequence
// (src/models/qwen_image/image_pipeline.cu run_image_job with ref_grids set): img_in over the packed
// [reference | target] latents, the joint sequence assembled by joint_scatter_u16_kernel from
// build_joint_layout (the reference's VAE tokens land on the expanded <|image_pad|> slots INSIDE the
// text, transformer_qwenimage21.py forward 906-923), the 32-block chain with the target-mask
// modulation and the block-causal image ids (run_block(..., ed.target_mask, ed.image_ids)), the
// masked norm_out (ops::adaln_scale_masked, transformer 1012 + _select_modulation_rows 238-254) and
// proj_out, then the target slice of the velocity.
//
// The reference is torch's own step of the same route, dumped by image-edit/dump_edit_step_oracle.py
// on the same fixtures: the M4 conditioned prompt (tensors_edit_text_tower) and the M3 vision merger
// rows (tensors_edit_vision). One step, one sigma from 1 to 0, true_cfg_scale 1.0 (no CFG),
// use_kv_cache false, seed 7, the "small" case: the solid RGB(20,40,200) reference plane the processor
// resizes to 352x224 (grid [1,14,22], 77 merged rows), so 4 * 77 = 308 reference tokens, a 352x224
// target with 308 tokens and 77 appended target slots, and 21 + 308 + 308 = 637 joint positions.
//
// Four groups of asserts, and the failure mode each one rules out:
//
//  (a) the velocity gate. Same constants and the same floor construction as
//      test_image_edit_prompt_cuda.cu: floor = rel_fro(bf16_oracle, f32) is the reference's OWN BF16
//      error, and a boundary passes at rel_fro(engine, f32) <= max(2 * floor, 1e-2), whole tensor and
//      row by row. Two independent BF16 chains of a 32-block chain sit at the reference's own error
//      apart; any band below it would measure torch, not the engine. INSIDE the DiT the floor has a
//      second, larger term: the artifact's DiT linears are fp8_e4m3fn_row_bf16, so the engine cannot sit
//      at BF16 distance from this reference at all. floor there is MEASURED (fp8_floor: the same walk
//      replayed with the artifact's FP8 weights decoded by the repo's own codec), never chosen, and the
//      band stays 2 * floor at the house 1e-2 -- which is what blackout #27's nine reds asked for and
//      what a garbage joint (rel_fro 1.414) still fails by an order of magnitude. The M4 test's text/image CLASS
//      split has no counterpart in a target-only slice (the reference's rows are discarded by the
//      pipeline's noise_pred[:, -latents.size(1):]), so the class split is applied where it IS
//      meaningful: the reference half and the target half of the packed img_in input are gated apart,
//      because reading the wrong half is exactly the layout defect this route can have and a
//      whole-tensor norm would let 308 wrong rows hide among 308 right ones.
//
//  (b) the reference-slot probe, which the text-to-image certification never had. The joint hidden is
//      internal to dit_velocity and joint_scatter_u16_kernel is file-static, so the probe is taken at
//      the two PUBLIC surfaces that decide it: the engine's own build_joint_layout (which row is
//      scattered where, and which rows read the sampled modulation) and the engine's own img_in weight
//      through the same op the pipeline calls (image_pipeline.cu 695 and 796). For a probed reference
//      slot it asserts BOTH sides: the row is near-equal to img_in(reference latents) and far from the
//      VLM merger row for that slot. A regression that leaves the vision row in the slot -- the shape
//      the route has when the scatter is fed the merger tensor instead of the VAE latents -- fails the
//      near-equal side loudly; a scatter that points at a text row, at the target half, or that forgets
//      the target-mask split fails the index side.
//
//  (c) the exact shape, count and finiteness of what the job returned.
//
//  (d) the #17 ledger: zero live image bytes before the job, after it, and after a FORCED throw. The
//      forced throws are the edit route's own host-side refusals -- a prompt_image_pad_mask of the
//      wrong length, and a mask whose slot count disagrees with ref_grids -- the input class that only
//      exists on this route. A third case pins the boundary the other way: a mask with the same count
//      but a different shape is a legal joint sequence and must NOT be refused.
//
// The text-to-image route is deliberately NOT exercised here: test_image_pipeline_prod_cuda.cu stays
// its certified guard, and this file only ever runs with ref_grids non-empty.
//
// Skips (77) without INFERNIX_TEST_ARTIFACT or INFERNIX_IMAGE_ORACLE, or when either is absent;
// returns 2 with no CUDA device. It runs the real pipeline on the real GPU: it must not overlap a
// live serve.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "models/qwen_image/image_device_util.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/joint_layout.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;   // to_bf16, from_bf16, bits_of, from_bits, view, h2d,
                                                // cuda_ok, Buf, Uploaded, upload_ref
namespace qo  = qioracle;
namespace ops = infernix::ops;

// The gate, copied from test_image_edit_prompt_cuda.cu: the engine may drift at most twice the
// reference's own BF16 error, and a boundary the reference computes exactly gets the 1e-2 floor.
constexpr double kFloorFactor     = 2.0;
constexpr double kGlobalFloor     = 1.0e-2;
constexpr double kSliceFloorShare = 1.0;   // a row may add at most one whole tensor's floor
constexpr double kDistinguishable = 0.5;   // "far": the two candidate rows must differ by more than this

// The M6 small case (dump_edit_step_oracle.py). These are the expectations the dump must agree with,
// not a guess at its contents: every one of them is re-read and checked against the files below.
constexpr std::size_t kKept       = 98;    // prompt rows after the 14-row system prefix
constexpr std::size_t kDropRows   = 14;
constexpr std::size_t kImageSlots = 77;    // <|image_pad|> slots in those rows
constexpr std::size_t kTextRows   = kKept - kImageSlots;                     // 21
constexpr std::size_t kRefHl = 14, kRefWl = 22;                              // 308 reference tokens
constexpr std::size_t kTargetHl = 14, kTargetWl = 22;                         // 308 target tokens
constexpr std::size_t kRefTokens    = kRefHl * kRefWl;
constexpr std::size_t kTargetTokens = kTargetHl * kTargetWl;
constexpr std::size_t kExpanded     = kTextRows + 4 * kImageSlots + kTargetTokens;  // 637
constexpr std::uint32_t kHeightPx = 224, kWidthPx = 352;
constexpr std::size_t kCin = 64;
constexpr std::size_t kProbeSlot = 38, kProbeK = 2;    // dump meta.json "probe"

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : fallback;
}

double diff_sq(const std::vector<float>& a, const std::vector<float>& b, std::size_t offset,
               std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset + i]) - static_cast<double>(b[offset + i]);
        sum += v * v;
    }
    return sum;
}

// The two-offset form: compare one row of `a` against one row of `b` when the rows do not start at the
// same element -- the probe compares joint row r of img_in against merger row r/4.
double diff_sq_rows(const std::vector<float>& a, std::size_t offset_a, const std::vector<float>& b,
                    std::size_t offset_b, std::size_t count) {
    require(offset_a + count <= a.size() && offset_b + count <= b.size(),
            "diff_sq_rows: a row range runs past its tensor");
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset_a + i]) - static_cast<double>(b[offset_b + i]);
        sum += v * v;
    }
    return sum;
}

double norm_sq(const std::vector<float>& a, std::size_t offset, std::size_t count) {
    require(offset + count <= a.size(), "norm_sq: a range runs past its tensor");
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset + i]);
        sum += v * v;
    }
    return sum;
}

// The measured floor of the DiT taps. The artifact's DiT linears are fp8_e4m3fn_row_bf16, so the engine
// cannot sit at BF16 distance from a BF16 reference: the representation alone moves every [637, 4096]
// tap. The floor is therefore MEASURED, not retuned -- it is the distance the same reference walk covers
// when its DiT weights are replaced by the artifact's FP8 ones decoded through the repo's own codec
// (image-edit/dump_edit_step_fp8_shadow.py into tensors_edit_step_fp8, replayed by the same oracle
// script). Returns the FP8 twin's rel_fro to the f32 reference, or -1.0 when the twin is absent, so a
// BF16-only dump keeps the old band and the skip semantics.
double fp8_floor(const std::string& shadow_dir, const std::string& name,
                 const std::vector<float>& f32) {
    std::vector<float> twin;
    try {
        const qo::NpyArray a = qo::read_npy(qo::join(shadow_dir, name));
        if (a.shape.size() != 2) { return -1.0; }
        twin = a.floats();
    } catch (const std::exception&) { return -1.0; }
    if (twin.size() != f32.size()) { return -1.0; }
    const double base = std::sqrt(norm_sq(f32, 0, f32.size()));
    return base > 0.0 ? std::sqrt(diff_sq(twin, f32, 0, f32.size())) / base : 0.0;
}

// One contiguous class of rows of `engine` against the f32 reference, with the reference's own BF16
// error as the floor -- test_image_edit_prompt_cuda.cu's gate_class, on a row range instead of a mask.
// `fatal` gates decide; a non-fatal call only prints and returns the verdict, so a red whole-tensor
// gate no longer hides the per-row evidence behind its own throw (blackout #20 learned this).
bool gate_class(const std::string& tag, const std::string& what, const std::vector<float>& engine,
                const std::vector<float>& f32, const std::vector<float>& bf16, std::size_t D,
                std::size_t first_row, std::size_t rows, double base, bool fatal = true,
                double measured_floor = 0.0) {
    require(rows > 0, tag + ": the " + what + " row class is empty");
    require(first_row * D + rows * D <= engine.size() && first_row * D + rows * D <= f32.size() &&
                first_row * D + rows * D <= bf16.size(),
            tag + ": the " + what + " range runs past one of the tensors");
    const std::size_t off = first_row * D;
    const double engine_err = std::sqrt(diff_sq(engine, f32, off, rows * D)) / base;
    const double floor_err  = std::sqrt(diff_sq(bf16, f32, off, rows * D)) / base;
    const double band = std::max(std::max(kFloorFactor * floor_err, kFloorFactor * measured_floor),
                                 kGlobalFloor);
    std::cout << "  " << tag << " " << what << " rows=" << rows << " eng=" << engine_err
              << " floor=" << floor_err << " fp8floor=" << measured_floor
              << " band=" << band << "\n";
    if (fatal) {
        require(engine_err <= band, tag + ": the " + what +
                                        " drift further than twice the reference's own BF16 error");
    }
    return engine_err <= band;
}

// Read a dumped [rows, cols] float32 array and check its shape before it is trusted.
std::vector<float> read_matrix(const std::string& path, const std::string& what, std::size_t rows,
                               std::size_t cols) {
    const qo::NpyArray a = qo::read_npy(path);
    require(a.shape.size() == 2 && a.shape[0] == rows && a.shape[1] == cols,
            what + " is not [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
    return a.floats();
}

void require_finite(const std::vector<float>& v, const std::string& what) {
    for (float x : v) require(std::isfinite(x), what + " is not finite");
}

} // namespace

int main() {
    const std::string artifact_path = env_or("INFERNIX_TEST_ARTIFACT", "");
    const std::string oracle_path   = env_or("INFERNIX_IMAGE_ORACLE", "");
    if (artifact_path.empty() || oracle_path.empty()) {
        std::cout << "INFERNIX_TEST_ARTIFACT / INFERNIX_IMAGE_ORACLE unset; skipping the edit-step route\n";
        return 77;
    }
    std::error_code ec;
    if (!std::filesystem::exists(artifact_path, ec) || !std::filesystem::exists(oracle_path, ec)) {
        std::cout << "artifact or oracle absent; skipping the edit-step route\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        std::cout << "no CUDA device; the edit-step qualification is runnable\n";
        return 2;
    }
    const std::string step_dir   = qo::join(oracle_path, "tensors_edit_step");
    // The FP8 twin of the same walk (see fp8_floor). Optional: its absence leaves every measured floor
    // at 0 and the gates fall back to the house 1e-2 band.
    const std::string fp8_dir    = qo::join(oracle_path, "tensors_edit_step_fp8");
    const std::string text_dir   = qo::join(oracle_path, "tensors_edit_text_tower");
    const std::string vision_dir = qo::join(oracle_path, "tensors_edit_vision");

    try {
        Reader reader(artifact_path);
        Binder binder(reader);
        DiTWeights dit = bind_dit(binder);
        VaeWeights vae = bind_image_vae(binder);
        const VaeDenorm denorm = read_vae_denorm(binder);
        (void)std::move(binder).finish();
        const std::size_t D = dit.dims.hidden;
        require(D == 4096, "the edit route was authored against hidden 4096, not " + std::to_string(D));
        require(dit.dims.image_in == kCin, "image_in is not the 64 latent channels");
        require(dit.blocks.size() == 32, "the dump was taken with a 32-block transformer");
        require(image_device_bytes_live() == 0, "the ledger holds bytes before anything runs");

        // ---- the dump, with its shapes checked as it is read ----
        const std::vector<float> pe_f32  = read_matrix(qo::join(step_dir, "prompt_embeds_f32.npy"),
                                                       "prompt_embeds_f32", kKept, D);
        const std::vector<float> pe_bf16 = read_matrix(qo::join(step_dir, "prompt_embeds_bf16.npy"),
                                                       "prompt_embeds_bf16", kKept, D);
        const std::vector<float> ref_f32  = read_matrix(qo::join(step_dir, "ref_lat_packed_f32.npy"),
                                                        "ref_lat_packed_f32", kRefTokens, kCin);
        const std::vector<float> ref_bf16 = read_matrix(qo::join(step_dir, "ref_lat_packed_bf16.npy"),
                                                        "ref_lat_packed_bf16", kRefTokens, kCin);
        const std::vector<float> init_f32  = read_matrix(qo::join(step_dir, "target_latents_init_f32.npy"),
                                                         "target_latents_init_f32", kTargetTokens, kCin);
        const std::vector<float> init_bf16 = read_matrix(qo::join(step_dir, "target_latents_init_bf16.npy"),
                                                          "target_latents_init_bf16", kTargetTokens, kCin);
        const std::vector<float> vel_f32  = read_matrix(qo::join(step_dir, "noise_pred_f32.npy"),
                                                        "noise_pred_f32", kTargetTokens, kCin);
        const std::vector<float> vel_bf16 = read_matrix(qo::join(step_dir, "noise_pred_bf16.npy"),
                                                        "noise_pred_bf16", kTargetTokens, kCin);
        const std::vector<float> lat_f32  = read_matrix(qo::join(step_dir, "latents_after_step_f32.npy"),
                                                        "latents_after_step_f32", kTargetTokens, kCin);
        const std::vector<float> lat_bf16 = read_matrix(qo::join(step_dir, "latents_after_step_bf16.npy"),
                                                         "latents_after_step_bf16", kTargetTokens, kCin);
        const std::vector<float> imgin_f32  = read_matrix(qo::join(step_dir, "img_in_ref_f32.npy"),
                                                          "img_in_ref_f32", kRefTokens, D);
        const std::vector<float> imgin_bf16 = read_matrix(qo::join(step_dir, "img_in_ref_bf16.npy"),
                                                          "img_in_ref_bf16", kRefTokens, D);
        const std::vector<float> all_merger =
            read_matrix(qo::join(vision_dir, "small_merger.npy"), "small_merger", kImageSlots, D);
        const qo::NpyArray merger_probe = qo::read_npy(qo::join(step_dir, "merger_probe_row_f32.npy"));
        require(merger_probe.shape.size() == 1 && merger_probe.shape[0] == D,
                "merger_probe_row_f32 is not one [D] row");
        const std::vector<float> merger_row = merger_probe.floats();
        const qo::NpyArray mask_array = qo::read_npy(qo::join(step_dir, "prompt_image_pad_mask.npy"));
        require(mask_array.shape.size() == 1 && mask_array.shape[0] == kKept,
                "the dumped prompt image pad mask is not one byte per kept prompt row");
        const std::vector<std::uint8_t> want_mask = mask_array.bools();
        const qo::NpyArray mean_array = qo::read_npy(qo::join(step_dir, "latents_mean.npy"));
        const qo::NpyArray std_array  = qo::read_npy(qo::join(step_dir, "latents_std.npy"));
        require(mean_array.shape.size() == 1 && mean_array.shape[0] == kCin &&
                    std_array.shape.size() == 1 && std_array.shape[0] == kCin,
                "the dumped VAE denormalization is not the 64 production channels");
        const std::vector<float> mean = mean_array.floats(), sdev = std_array.floats();
        const std::vector<float> time_steps = qo::read_npy(qo::join(step_dir, "timesteps.npy")).floats();
        const std::vector<float> sigmas     = qo::read_npy(qo::join(step_dir, "sigmas.npy")).floats();
        require(time_steps.size() == 1, "the dump carries more than one timestep");
        require(sigmas.size() == 2, "one step needs two sigmas");
        // The single step must run from sigma 1 to sigma 0: the velocity is recovered from the returned
        // latent through new = init + (sigma_next - sigma) * v, and only for that pair is it exactly
        // v = init - new. Anything else and (a) would gate a different step than the one dumped.
        require(std::fabs(sigmas[0] - 1.0) < 1.0e-6 && std::fabs(sigmas[1]) < 1.0e-9,
                "the dumped schedule is not the one step from sigma 1 to sigma 0");

        // The dump must be the M4 conditioning, not a re-derived lookalike: the same bytes the
        // conditioned-tower test already gates, read from tensors_edit_text_tower, and the merger row
        // the qualified M3 vision tower produced. A step oracle that quietly re-encoded the prompt would
        // agree with itself and gate nothing.
        {
            const std::vector<float> m4 =
                read_matrix(qo::join(text_dir, "small_prompt_embeds_f32.npy"),
                            "the M4 small prompt embeds", kKept, D);
            require(m4 == pe_f32, "the step dump's prompt embeds differ from the M4 text-tower dump");
            const qo::NpyArray m4_mask = qo::read_npy(qo::join(text_dir, "small_image_pad_mask.npy"));
            require(m4_mask.shape.size() == 1 && m4_mask.shape[0] == kKept + kDropRows,
                    "the M4 dump's mask is not the full 112-row prompt");
            const std::vector<std::uint8_t> m4_full = m4_mask.bools();
            require(std::equal(want_mask.begin(), want_mask.end(), m4_full.begin() +
                                                              static_cast<std::ptrdiff_t>(kDropRows)),
                    "the step dump's mask is not the M4 mask's kept-row tail");
            for (std::size_t d = 0; d < D; ++d) {
                require(all_merger[kProbeSlot * D + d] == merger_row[d],
                        "the probed merger row is not the M3 merger row for that slot");
            }
        }

        std::size_t slots = 0;
        for (std::uint8_t m : want_mask) slots += (m != 0U ? 1U : 0U);
        require(slots == kImageSlots, "the prompt carries " + std::to_string(slots) +
                                          " image pad slots, not " + std::to_string(kImageSlots));
        require(kImageSlots * 4 == kRefTokens, "the x4 invariant does not hold for this case");
        require(kTargetTokens % 4 == 0, "the target grid is not a whole number of 2x2 slots");
        require_finite(pe_f32, "prompt_embeds");
        require_finite(ref_f32, "ref_lat_packed");
        require_finite(init_f32, "target_latents_init");
        require_finite(vel_f32, "noise_pred");
        require_finite(lat_f32, "latents_after_step");
        require_finite(imgin_f32, "img_in_ref");
        require_finite(all_merger, "small_merger");
        // The reference plane and the noise are different tensors. If they were equal, the probe could
        // not tell "the reference row is in the slot" from "the target row is in the slot".
        require(qo::rel_fro(ref_f32, init_f32) > kDistinguishable,
                "the reference latents and the target noise are not distinguishable");
        // The VAE constants the job carries are the ones torch normalized the reference with, and the
        // ones the artifact's own vae config declares.
        require(mean == denorm.mean && sdev == denorm.std,
                "the dumped VAE normalization differs from the artifact's vae config");
        std::cout << "edit step case: kept=" << kKept << " slots=" << slots << " ref=" << kRefTokens
                  << " target=" << kTargetTokens << " joint=" << kExpanded << " t=" << time_steps[0]
                  << "\n";

        // ---- (b) the reference-slot probe ----
        std::vector<std::uint8_t> joint_mask = want_mask;
        joint_mask.insert(joint_mask.end(), kTargetTokens / 4, 1U);   // append_target_slots, pipeline 749-757
        const std::vector<ImageShape> shapes{
            ImageShape{1, static_cast<long long>(kRefHl), static_cast<long long>(kRefWl)},
            ImageShape{1, static_cast<long long>(kTargetHl), static_cast<long long>(kTargetWl)}};
        const JointLayout layout = build_joint_layout(std::span<const std::uint8_t>(joint_mask),
                                                      std::span<const ImageShape>(shapes));
        require(layout.base_len == static_cast<long long>(kKept + kTargetTokens / 4),
                "the joint layout's base length is not the prompt rows plus the target slots");
        require(layout.expanded_len == static_cast<long long>(kExpanded),
                "the joint layout expanded to " + std::to_string(layout.expanded_len) +
                    " positions, not 637: the x4 expansion is not what the route assembles");
        require(layout.text_tokens == static_cast<long long>(kTextRows),
                "the joint layout does not carry the 21 text rows");
        require(layout.ref_tokens == static_cast<long long>(kRefTokens),
                "the joint layout does not carry the 308 reference tokens");
        require(layout.target_tokens == static_cast<long long>(kTargetTokens),
                "the joint layout does not carry the 308 target tokens");

        // Walk the map. Every one of these is a way the interleaved sequence can be assembled wrong and
        // still produce a plausible-looking tensor of the right size.
        std::size_t probe_pos = 0;
        long long ref_seen = 0, target_seen = 0;
        require(layout.expanded_image_pad_mask.size() == static_cast<std::size_t>(kExpanded) &&
                    layout.scatter_row.size() == static_cast<std::size_t>(kExpanded) &&
                    layout.target_token_mask.size() == static_cast<std::size_t>(kExpanded) &&
                    layout.image_block_ids.size() == static_cast<std::size_t>(kExpanded),
                "the joint layout's per-position arrays are not one per expanded position");
        for (std::size_t s = 0; s < layout.expanded_image_pad_mask.size(); ++s) {
            if (layout.expanded_image_pad_mask[s] == 0U) {
                require(layout.scatter_row[s] == -1,
                        "a text position carries an img_in row (position " + std::to_string(s) + ")");
                require(layout.image_block_ids[s] == -1,
                        "a text position is inside an image block (position " + std::to_string(s) + ")");
                require(layout.target_token_mask[s] == 0,
                        "a text position reads the sampled modulation row (position " +
                            std::to_string(s) + ")");
                continue;
            }
            const long long row = layout.scatter_row[s];
            require(row >= 0 && row < static_cast<long long>(kRefTokens + kTargetTokens),
                    "an image position scatters img_in row " + std::to_string(row) + " (position " +
                        std::to_string(s) + ")");
            if (layout.image_block_ids[s] == 0) {
                // The reference block must NOT read the sampled modulation row: in the reference only
                // the LAST img_shapes block does (build_token_metadata, forward 926, and
                // _select_modulation_rows 238-254; the engine's mirror is adaln_scale_masked at 883-890).
                require(layout.target_token_mask[s] == 0,
                        "a reference position reads the sampled modulation row (position " +
                            std::to_string(s) + ")");
                require(row < static_cast<long long>(kRefTokens),
                        "a reference position scatters a target-half img_in row (position " +
                            std::to_string(s) + ", row " + std::to_string(row) + ")");
                if (ref_seen == static_cast<long long>(kProbeSlot) * 4 +
                                   static_cast<long long>(kProbeK)) {
                    probe_pos = s;
                }
                ++ref_seen;
            } else {
                require(layout.image_block_ids[s] == 1,
                        "an image position names block " + std::to_string(layout.image_block_ids[s]) +
                            " (position " + std::to_string(s) + ")");
                require(layout.target_token_mask[s] == 1,
                        "a target position does not read the sampled modulation row (position " +
                            std::to_string(s) + ")");
                require(row >= static_cast<long long>(kRefTokens),
                        "a target position scatters a reference-half img_in row (position " +
                            std::to_string(s) + ")");
                ++target_seen;
            }
        }
        require(ref_seen == static_cast<long long>(kRefTokens),
                "the reference block holds " + std::to_string(ref_seen) + " positions, not 308");
        require(target_seen == static_cast<long long>(kTargetTokens),
                "the target block holds " + std::to_string(target_seen) + " positions, not 308");
        require(layout.scatter_row[probe_pos] ==
                    static_cast<long long>(kProbeSlot) * 4 + static_cast<long long>(kProbeK),
                "the probed reference slot does not scatter its own img_in row");
        std::cout << "probe: joint position " << probe_pos << " scatters packed img_in row "
                  << layout.scatter_row[probe_pos] << " (reference slot " << kProbeSlot
                  << ", position " << kProbeK << " of its 4)\n";

        // img_in(reference | target) on device: the engine's own weight, the engine's own op, the same
        // packed order the job builds (image_pipeline.cu 695, 725, 742, 796). Scoped so every Buf is
        // back at zero before the ledger is read.
        std::vector<float> imgin_got;
        {
            const std::size_t all_img = kRefTokens + kTargetTokens;
            std::vector<std::uint16_t> packed;
            packed.reserve(all_img * kCin);
            for (float x : ref_bf16) packed.push_back(to_bf16(x));
            for (float x : init_bf16) packed.push_back(to_bf16(x));
            Buf d_packed, d_hs_img;
            Uploaded Wimg = upload_ref(reader, dit.img_in);
            d_packed.alloc(packed.size() * sizeof(std::uint16_t));
            h2d(d_packed.d, packed.data(), packed.size() * sizeof(std::uint16_t));
            d_hs_img.alloc(all_img * D * sizeof(std::uint16_t));
            // out is a Tensor&, so the destination view needs a name -- a temporary would not bind.
            Tensor hs_img = view(d_hs_img.d, static_cast<int>(D), static_cast<int>(all_img));
            ops::linear(view(d_packed.d, static_cast<int>(kCin), static_cast<int>(all_img)), Wimg.w,
                        hs_img, 0);
            cuda_ok(cudaStreamSynchronize(0), "the probe's img_in did not complete");
            std::vector<std::uint16_t> bits(all_img * D);
            cuda_ok(cudaMemcpy(bits.data(), d_hs_img.d, bits.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "img_in D2H");
            imgin_got = from_bits(bits);
        }
        require(imgin_got.size() == (kRefTokens + kTargetTokens) * D,
                "the probe returned the wrong number of rows");
        require_finite(imgin_got, "the probed img_in");
        require(image_device_bytes_live() == 0, "the img_in probe left image device bytes live");

        const double imgin_base = std::sqrt(norm_sq(imgin_f32, 0, imgin_f32.size()));
        require(imgin_base > 0.0, "the dumped img_in reference rows are all zeros");
        // The whole-tensor floor is what a single row is allowed to add on top of its own error -- the
        // M4 test's per-row rule, restated for this tensor. Both floors are 0 here: the reference ran
        // BF16, so the dumped "f32" rows are BF16-exact and carry no floor; the band then rests on
        // kGlobalFloor alone. A scattered wrong row is ~1/sqrt(308) = 0.057 of the whole tensor,
        // 5.7x outside that floor.
        const double imgin_global_floor =
            std::sqrt(diff_sq(imgin_bf16, imgin_f32, 0, imgin_f32.size())) / imgin_base;
        gate_class("img_in", "reference half", imgin_got, imgin_f32, imgin_bf16, D, 0, kRefTokens,
                   imgin_base);
        for (std::size_t row = 0; row < kRefTokens; ++row) {
            const double row_engine = std::sqrt(diff_sq(imgin_got, imgin_f32, row * D, D)) / imgin_base;
            const double row_floor  = std::sqrt(diff_sq(imgin_bf16, imgin_f32, row * D, D)) / imgin_base;
            require(row_engine <= std::max(std::max(kFloorFactor * row_floor,
                                                    kSliceFloorShare * imgin_global_floor),
                                           kGlobalFloor),
                    "img_in reference row " + std::to_string(row) + " drifts beyond twice the "
                    "reference's own plus one whole tensor's floor");
        }
        // The target half is not all zero and not a copy of the reference half: a scatter that reads one
        // half for both blocks would pass a reference-only gate.
        {
            double target_norm = 0.0;
            for (std::size_t row = kRefTokens; row < kRefTokens + kTargetTokens; ++row) {
                target_norm += norm_sq(imgin_got, row * D, D);
            }
            require(std::sqrt(target_norm) > 0.0, "the target half of img_in is all zeros");
            const double between =
                std::sqrt(diff_sq_rows(imgin_got, 0, imgin_got, kRefTokens * D, kRefTokens * D)) /
                imgin_base;
            std::cout << "  img_in reference half vs target half: rel_fro=" << between << "\n";
            require(between > kDistinguishable,
                    "the two halves of the packed img_in input are the same rows");
        }

        // THE probe, both sides: near img_in(reference latents), far from that slot's VLM merger row.
        {
            const std::size_t row = kProbeSlot * 4 + kProbeK;
            const double row_engine = std::sqrt(diff_sq(imgin_got, imgin_f32, row * D, D)) / imgin_base;
            const double row_floor  = std::sqrt(diff_sq(imgin_bf16, imgin_f32, row * D, D)) / imgin_base;
            const double near_band  = std::max(kFloorFactor * row_floor, kGlobalFloor);
            std::cout << "  probe img_in row " << row << " eng=" << row_engine << " floor=" << row_floor
                      << " band=" << near_band << "\n";
            require(row_engine <= near_band,
                    "the probed reference slot is not img_in(reference latents) to within twice the "
                    "reference's own BF16 error");

            const double merger_norm = std::sqrt(norm_sq(merger_row, 0, D));
            require(merger_norm > 0.0, "the dumped merger row for the probed slot is all zeros");
            const double far =
                std::sqrt(diff_sq_rows(imgin_got, row * D, merger_row, 0, D)) / merger_norm;
            std::cout << "  probe img_in row " << row << " vs that slot's merger row: rel_fro=" << far
                      << " (must exceed " << kDistinguishable << ")\n";
            require(far > kDistinguishable,
                    "the probed reference slot is closer to the VLM merger row than to img_in: a vision "
                    "row left in the slot would not be caught");

            // and for every slot, not only the probed one: each reference row must be distinguishable
            // from the merger row of the slot its four positions belong to
            double worst = 1.0e30;
            std::size_t worst_row = 0;
            for (std::size_t slot = 0; slot < kImageSlots; ++slot) {
                const double slot_norm = std::sqrt(norm_sq(all_merger, slot * D, D));
                require(slot_norm > 0.0, "merger row " + std::to_string(slot) + " is all zeros");
                for (std::size_t k = 0; k < 4; ++k) {
                    const std::size_t r = slot * 4 + k;
                    const double d =
                        std::sqrt(diff_sq_rows(imgin_got, r * D, all_merger, slot * D, D)) / slot_norm;
                    if (d < worst) { worst = d; worst_row = r; }
                }
            }
            std::cout << "  min over all 308 reference rows of rel_fro(img_in row, that slot's merger "
                         "row) = " << worst << " (row " << worst_row << ")\n";
            require(worst > kDistinguishable,
                    "reference row " + std::to_string(worst_row) + " is not distinguishable from its "
                    "slot's vision row");
        }

        // ---- (a) the one step, on the real route ----
        ImageJobInputs inputs;
        inputs.prompt_embeds         = bits_of(pe_bf16);
        inputs.init_latent           = bits_of(init_bf16);
        inputs.ref_latents           = bits_of(ref_bf16);
        inputs.ref_grids             = {{static_cast<std::uint32_t>(kRefHl),
                                         static_cast<std::uint32_t>(kRefWl)}};
        inputs.prompt_image_pad_mask = want_mask;
        inputs.timesteps             = time_steps;
        inputs.sigmas                = sigmas;
        inputs.latents_mean          = mean;
        inputs.latents_std           = sdev;
        inputs.height_px             = kHeightPx;
        inputs.width_px              = kWidthPx;
        inputs.true_cfg_scale        = 1.0F;   // no CFG: one forward, as the dump took it
        require(!inputs.ref_grids.empty(), "this test only exercises the edit route");
        require(inputs.prompt_image_pad_mask.size() == kKept,
                "the job's prompt mask must be one byte per kept row");

        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("no image stream"); }
        // Ask the pipeline for its one-off trace of the SHARED inputs (image_pipeline.cu,
        // INFERNIX_IMAGE_TRACE_DIR). #21's velocity error was uniform over all 308 target rows, which is
        // the signature of modulation, of the assembled joint, or of the block chain -- and the blackout
        // window is too expensive to spend guessing which. The trace costs the pipeline nothing when the
        // variable is unset, which is how it always runs in serving.
        const std::filesystem::path trace_dir =
            std::filesystem::temp_directory_path() / "infernix_edit_trace";
        std::error_code tec;
        std::filesystem::remove_all(trace_dir, tec);
        std::filesystem::create_directories(trace_dir, tec);
        require(!tec, "cannot create the trace directory: " + tec.message());
#ifdef _MSC_VER
        require(_putenv_s("INFERNIX_IMAGE_TRACE_DIR", trace_dir.string().c_str()) == 0,
                "cannot publish the trace directory");
#else
        require(setenv("INFERNIX_IMAGE_TRACE_DIR", trace_dir.string().c_str(), 1) == 0,
                "cannot publish the trace directory");
#endif
        GeneratedImage out;
        reset_image_device_peak();
        out = run_image_job(reader, dit, vae, inputs, stream);
        cudaStreamDestroy(stream);

        // (c) what came back
        require(out.height == kHeightPx && out.width == kWidthPx,
                "the job returned " + std::to_string(out.width) + "x" + std::to_string(out.height) +
                    " instead of 352x224");
        require(out.channels.size() == 4ULL * kHeightPx * kWidthPx,
                "the decoded image is not [4, height, width]");
        require(out.latent.size() == kTargetTokens * kCin,
                "the final latent is not the target's [308, 64]");
        for (float x : out.channels) require(std::isfinite(x), "the decoded image is not finite");
        for (std::uint16_t b : out.latent) {
            require(std::isfinite(from_bf16(b)), "the final latent is not finite");
        }
        require(image_device_bytes_live() == 0, "the job left image device bytes live");
        std::cout << "peak image device bytes: "
                  << image_device_bytes_peak() / (1024ULL * 1024ULL) << " MiB\n";

        // (a) the velocity. run_image_job returns only the final latent, and with sigma 1 -> 0 the one
        // scheduler step is new = init - v, so the velocity the DiT produced is recoverable exactly from
        // it. Both sides of that identity are gated: the velocity (the plan's gate) and the latent, so a
        // defect in the DiT cannot hide behind a compensating defect in the sampler.
        const std::vector<float> new_lat = from_bits(out.latent);
        std::vector<float> velocity(new_lat.size());
        for (std::size_t i = 0; i < velocity.size(); ++i) {
            velocity[i] = static_cast<float>(static_cast<double>(init_f32[i]) -
                                             static_cast<double>(new_lat[i]));
        }
        require_finite(velocity, "the recovered velocity");
        const double vel_base = std::sqrt(norm_sq(vel_f32, 0, vel_f32.size()));
        require(vel_base > 0.0, "the dumped velocity is all zeros");
        const double vel_global_floor =
            std::sqrt(diff_sq(vel_bf16, vel_f32, 0, vel_f32.size())) / vel_base;
        std::cout << "  edit velocity whole-tensor floor (the reference's own BF16 error) = "
                  << vel_global_floor << "\n";
        // The velocity and the post-step latent are the DiT's output under the ARTIFACT's FP8 weights, so
        // their floor is the FP8 twin's own distance to the reference (0.018 and 0.021 at this fixture),
        // not the reference's BF16 error, which is 0 because the reference DiT ran BF16.
        const double vel_fp8_floor = std::max(0.0, fp8_floor(fp8_dir, "noise_pred_f32.npy", vel_f32));
        const bool vel_ok = gate_class("edit", "velocity (target slice)", velocity, vel_f32, vel_bf16,
                                       kCin, 0, kTargetTokens, vel_base, false, vel_fp8_floor);
        // Same degeneracy as img_in: the reference DiT ran BF16, so vel_global_floor is 0 and the band
        // rests on the FP8 twin's measured floor. A wrong target row is ~1/sqrt(308) = 0.057 of the
        // tensor -- far outside that band, so the row gate keeps its power.
        // Every red row is PRINTED, not thrown on: uniform drift (wrong attention/mask/modulation --
        // a global mechanism) and a wrong row subset (a layout/scatter defect) demand different
        // fixes, and blackout #20's throw showed only the whole-tensor number.
        bool rows_ok = true;
        std::size_t worst_row = 0;
        double worst = 0.0;
        for (std::size_t row = 0; row < kTargetTokens; ++row) {
            const double row_engine =
                std::sqrt(diff_sq(velocity, vel_f32, row * kCin, kCin)) / vel_base;
            const double row_floor =
                std::sqrt(diff_sq(vel_bf16, vel_f32, row * kCin, kCin)) / vel_base;
            const double row_band =
                std::max(std::max(kFloorFactor * row_floor,
                                  std::max(kSliceFloorShare * vel_global_floor,
                                           kSliceFloorShare * vel_fp8_floor)),
                         kGlobalFloor);
            if (row_engine > row_band) {
                rows_ok = false;
                std::cout << "  edit red velocity row " << row << " eng=" << row_engine
                          << " floor=" << row_floor << " band=" << row_band << "\n";
            }
            if (row_engine > worst) { worst = row_engine; worst_row = row; }
        }
        std::cout << "  edit velocity rows: worst " << worst_row << " eng=" << worst
                  << (rows_ok ? ", all inside" : ", RED rows above") << "\n";
        // Is the whole error a multiplicative factor? An output-head scale, a modulation gain or a
        // normalization constant all leave v_cuda = a * v_ref, which this fit exposes: a* moves off 1 and
        // the residual collapses to the noise floor. A direction-changing bug keeps a big residual at
        // any a, and that distinction decides whether to look at the heads or inside the blocks.
        {
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < velocity.size(); ++i) {
                num += static_cast<double>(velocity[i]) * vel_f32[i];
                den += static_cast<double>(vel_f32[i]) * vel_f32[i];
            }
            const double a = den > 0.0 ? num / den : 0.0;
            double rs = 0.0;
            for (std::size_t i = 0; i < velocity.size(); ++i) {
                const double d = static_cast<double>(velocity[i]) - a * vel_f32[i];
                rs += d * d;
            }
            std::cout << "  edit velocity scalar fit a*=" << a << " residual="
                      << (vel_base > 0.0 ? std::sqrt(rs) / vel_base : 0.0) << "\n";
        }
        const double lat_base = std::sqrt(norm_sq(lat_f32, 0, lat_f32.size()));
        require(lat_base > 0.0, "the dumped post-step latent is all zeros");
        const double lat_fp8_floor =
            std::max(0.0, fp8_floor(fp8_dir, "latents_after_step_f32.npy", lat_f32));
        const bool lat_ok = gate_class("edit", "final latent", new_lat, lat_f32, lat_bf16, kCin, 0,
                                       kTargetTokens, lat_base, false, lat_fp8_floor);

        // ---- the shared inputs, from the trace the job just wrote ----
        // temb carries BOTH modulation rows in the order the engine built them: the sampled t first, then
        // the t=0 row causal_condition appends. Matching it clears the modulation path end to end (nlin
        // and mod are ops::linear of it, and the row SELECTION is the host gate above), including the row
        // ORDER, which no mask gate can see. joint_in is the sequence as block 0 received it -- txt_in's
        // rows and the scattered img_in rows before any block touched them. The taps say where the chain
        // starts to diverge if both of those match. temb and the assembled joint are built from BF16 chains the
// engine shares with the reference, so they keep the house 1e-2 band; every tap INSIDE the DiT is gated
// on the FP8 twin's measured drift instead, because the artifact's weights -- not the engine -- are what
// separates those from a BF16 reference.
        auto read_trace = [&](const std::string& name, std::size_t n) {
            std::vector<float> v;
            std::FILE* h = std::fopen((trace_dir.string() + "\\" + name).c_str(), "rb");
            if (h == nullptr) { return v; }
            v.resize(n);
            const std::size_t got = std::fread(v.data(), sizeof(float), n, h);
            std::fclose(h);
            if (got != n) { v.clear(); }
            return v;
        };
        auto gate_trace = [&](const std::string& what, const std::vector<float>& got,
                              const std::vector<float>& want, double band) {
            if (got.size() != want.size()) {
                std::cout << "  edit " << what << ": NO TRACE (" << got.size() << " of "
                          << want.size() << " floats)\n";
                return false;
            }
            const double base = std::sqrt(norm_sq(want, 0, want.size()));
            const double err  = base > 0.0
                ? std::sqrt(diff_sq(got, want, 0, want.size())) / base : 0.0;
            std::cout << "  edit " << what << " eng=" << err << " band=" << band
                      << (err <= band ? " OK" : " RED") << "\n";
            return err <= band;
        };
        auto rel_rows = [](const std::vector<float>& got, const std::vector<float>& want,
                           std::size_t off, std::size_t n) {
            const double base = std::sqrt(norm_sq(want, off, n));
            return base > 0.0 ? std::sqrt(diff_sq(got, want, off, n)) / base : 0.0;
        };
        // Every [637, 4096] surface is asked twice: whole tensor, and the 308 target rows alone. A wrong
        // SHARED input (the modulation source, rope, txt_in, the assembly) moves both classes together;
        // a causal_condition mechanism -- which modulation row a token reads -- moves only the target
        // rows, and only the target rows can be wrong while the velocity is uniformly wrong.
        auto gate_trace_split = [&](const std::string& what, const std::vector<float>& got,
                                    const std::vector<float>& want, double band) {
            const bool ok = gate_trace(what, got, want, band);
            const std::size_t first = static_cast<std::size_t>(kExpanded - kTargetTokens) * 4096;
            const std::size_t count = static_cast<std::size_t>(kTargetTokens) * 4096;
            if (got.size() != want.size() || want.size() != kExpanded * 4096) { return ok; }
            std::cout << "  edit " << what << " by class: target rows eng="
                      << rel_rows(got, want, first, count)
                      << ", text+reference rows eng=" << rel_rows(got, want, 0, first) << "\n";
            return ok;
        };
        bool trace_ok = true;
        {
            const std::vector<float> temb_want =
                qo::read_npy(qo::join(step_dir, "temb_f32.npy")).floats();
            require(temb_want.size() == 2ULL * 4096, "the dumped temb is not [2, 4096]");
            const std::vector<float> temb_got = read_trace("temb.f32", temb_want.size());
            trace_ok &= gate_trace("temb (sampled row + t=0 row)", temb_got, temb_want, 1.0e-2);
            // The aggregate cannot see a SWAP: two rows of similar norm exchanged sit at a diluted
            // distance, and the row ORDER is exactly what causal_condition asks of the engine -- row 0
            // must be the sampled t (the target reads it), row 1 the t=0 row every non-target reads. So
            // each engine row is matched against its own row AND against the other one: two mirrored
            // reds name a swapped build, one lone red names the timestep chain itself.
            if (temb_got.size() == temb_want.size()) {
                const std::size_t C = 4096;
                for (std::size_t r = 0; r < 2; ++r) {
                    const std::size_t o = 1 - r;
                    const std::vector<float> g(temb_got.begin() + r * C, temb_got.begin() + (r + 1) * C);
                    const std::vector<float> same(temb_want.begin() + r * C, temb_want.begin() + (r + 1) * C);
                    const std::vector<float> other(temb_want.begin() + o * C, temb_want.begin() + (o + 1) * C);
                    std::cout << "  edit temb row " << (r == 0 ? "0 (sampled t)" : "1 (t = 0)")
                              << " eng=" << rel_rows(g, same, 0, C)
                              << " against the other row eng=" << rel_rows(g, other, 0, C) << "\n";
                }
            }
            const std::vector<float> jin_want =
                qo::read_npy(qo::join(step_dir, "joint_in_f32.npy")).floats();
            require(jin_want.size() == kExpanded * 4096, "the dumped joint_in is not [637, 4096]");
            trace_ok &= gate_trace_split("joint as block 0 received it",
                                         read_trace("joint_in.f32", jin_want.size()), jin_want, 1.0e-2);
            // Blackout #23 left the joint's TARGET block unexplained: 1.42 from the dump (the distance
            // two independent tensors sit at) while the reference half was bit-exact. The target half of
            // the packed img_in input had never been gated on EITHER side -- the dump saved only its
            // reference half, and the "img_in reference half" gate above is the TEST's own GEMM of the
            // bytes it packed, not the job's buffer. These three close the gap and separate the three
            // remaining hypotheses: the upload of init_latent into d_lat, the GEMM's rows, the scatter's
            // indices. The job was handed the packed bytes, so its latent trace must be bit-exact; if it
            // is and the joint still differs, the scatter is what lost the rows.
            {
                std::vector<float> tgt_want;
                try {
                    tgt_want = qo::read_npy(qo::join(step_dir, "img_in_target_f32.npy")).floats();
                } catch (const std::exception&) {
                    tgt_want.clear();
                }
                if (tgt_want.size() != kTargetTokens * 4096ULL) {
                    std::cout << "  edit img_in target half: THE DUMP CARRIES NO img_in_target_f32.npy\n";
                    trace_ok = false;
                } else {
                    // rel_rows takes ONE offset for both tensors, and tgt_want holds only the target
                    // half, so the halves are compared with the two-offset form.
                    const double tgt_base = std::sqrt(norm_sq(tgt_want, 0, kTargetTokens * 4096ULL));
                    const double probe_tgt =
                        tgt_base > 0.0
                            ? std::sqrt(diff_sq_rows(imgin_got, kRefTokens * 4096ULL, tgt_want, 0,
                                                     kTargetTokens * 4096ULL)) / tgt_base
                            : 0.0;
                    std::cout << "  edit img_in target half (the test's own GEMM of the packed rows) eng="
                              << probe_tgt << " band=" << kGlobalFloor
                              << (probe_tgt <= kGlobalFloor ? " OK" : " RED") << "\n";
                    trace_ok &= probe_tgt <= kGlobalFloor;
                }

                std::vector<float> packed_want;
                packed_want.reserve((kRefTokens + kTargetTokens) * kCin);
                packed_want.insert(packed_want.end(), ref_bf16.begin(), ref_bf16.end());
                packed_want.insert(packed_want.end(), init_bf16.begin(), init_bf16.end());
                const std::size_t pn = (kRefTokens + kTargetTokens) * kCin;
                const std::vector<float> packed_got = read_trace("latent_packed.f32", pn);
                if (packed_got.size() != pn) {
                    std::cout << "  edit the job's packed latent: NO TRACE\n";
                    trace_ok = false;
                } else {
                    const double pr = rel_rows(packed_got, packed_want, kRefTokens * kCin,
                                               kTargetTokens * kCin);
                    std::cout << "  edit the job's packed latent by class: reference half eng="
                              << rel_rows(packed_got, packed_want, 0, kRefTokens * kCin)
                              << ", target half eng=" << pr << " (both must be 0: the job was handed "
                              << "these bytes)\n";
                    trace_ok &= pr <= 0.0;
                }

                std::vector<float> hs_want = imgin_f32;
                hs_want.insert(hs_want.end(), tgt_want.begin(), tgt_want.end());
                const std::size_t hn = (kRefTokens + kTargetTokens) * 4096ULL;
                const std::vector<float> hs_got = read_trace("hs_img.f32", hn);
                if (hs_got.size() != hn) {
                    std::cout << "  edit the job's img_in output: NO TRACE\n";
                    trace_ok = false;
                } else {
                    const double hr = rel_rows(hs_got, hs_want, kRefTokens * 4096ULL,
                                               kTargetTokens * 4096ULL);
                    std::cout << "  edit the job's img_in output by class: reference half eng="
                              << rel_rows(hs_got, hs_want, 0, kRefTokens * 4096ULL)
                              << ", target half eng=" << hr << " band=" << kGlobalFloor << "\n";
                    trace_ok &= hr <= kGlobalFloor;
                }
            }
            // The other end of the chain: everything above the velocity head can be right and the head
            // (layer_norm + masked adaln_scale, before proj_out) still wrong.
            {
                std::vector<float> nout_want;
                try {
                    nout_want = qo::read_npy(qo::join(step_dir, "norm_out_f32.npy")).floats();
                } catch (const std::exception&) {
                    nout_want.clear();
                }
                if (nout_want.empty()) {
                    std::cout << "  edit velocity head input: the dump carries no norm_out file\n";
                } else {
                    // The engine's DiT runs the artifact's FP8 weights, so the head input's floor is the
                    // FP8 twin's own drift (6.0e-2 here), not the reference's BF16 error.
                    const double twin = fp8_floor(fp8_dir, "norm_out_f32.npy", nout_want);
                    const double band = twin >= 0.0 ? std::max(kFloorFactor * twin, kGlobalFloor) : 1.0e-2;
                    trace_ok &= gate_trace_split("velocity head input (norm_out)",
                                                 read_trace("norm_out.f32", nout_want.size()),
                                                 nout_want, band);
                }
            }
            for (const int tap : {0, 1, 2, 4, 8, 16, 24, 31}) {
                const std::string nm = "block_" + std::to_string(tap) + "_joint_f32.npy";
                std::vector<float> want;
                try {
                    want = qo::read_npy(qo::join(step_dir, nm)).floats();
                } catch (const std::exception&) {
                    want.clear();
                }
                if (want.empty()) {
                    std::cout << "  edit block " << tap << " tap: the dump carries no tap file\n";
                    continue;
                }
                // The drift grows monotonically with depth (4.7e-3 at block 0 to 2.8e-2 at block 31) and
                // the FP8 twin reproduces it tap for tap, so each tap is gated on ITS OWN measured FP8
                // floor. A mechanism defect moves the engine further than the representation does.
                const double twin = fp8_floor(fp8_dir, nm, want);
                const double band = twin >= 0.0 ? std::max(kFloorFactor * twin, kGlobalFloor) : 1.0e-2;
                trace_ok &= gate_trace_split("block " + std::to_string(tap) + " output",
                                             read_trace("block_" + std::to_string(tap) + ".f32",
                                                        want.size()),
                                             want, band);
            }
        }

        require(vel_ok && rows_ok && lat_ok && trace_ok,
                "the velocity (target slice), the final latent or a SHARED INPUT of the DiT step (temb, "
                "the assembled joint, a block output) drifts further than the band allows -- the red rows "
                "and the first diverging shared input are printed above");

        // ---- (d) the ledger's throw path, on the edit route's own host-side refusals ----
        {
            Buf hog;   // the job must not keep the test's own bytes either
            hog.alloc(64ULL * 1024 * 1024);
            require(image_device_bytes_live() == hog.bytes,
                    "the ledger does not hold exactly the test's own bytes between jobs");

            ImageJobInputs short_mask = inputs;
            short_mask.prompt_image_pad_mask.assign(kKept - 1, 1U);   // one row short
            bool refused = false;
            try {
                cudaStream_t s2 = nullptr;
                if (cudaStreamCreate(&s2) != cudaSuccess) { throw std::runtime_error("no image stream"); }
                const GeneratedImage bad = run_image_job(reader, dit, vae, short_mask, s2);
                (void)bad;
                cudaStreamDestroy(s2);
            } catch (const std::exception& error) {
                refused             = true;
                const std::string m = error.what();
                require(m.find("image pad mask") != std::string::npos,
                        "the short mask failed for a reason other than its length: " + m);
            }
            require(refused, "a prompt image pad mask of the wrong length was accepted");
            require(image_device_bytes_live() == hog.bytes,
                    "the short-mask refusal left image device bytes live");

            ImageJobInputs few_slots = inputs;
            few_slots.prompt_image_pad_mask.assign(kKept, 0U);
            // 60 slots, not the fixture's own 77: a mask of exactly kRefTokens/4 ones IS the correct
            // input (the engine appends the target slots itself), so the negative case has to disagree
            // with the reference token count -- 4 * 60 = 240 cannot cover 308 reference tokens.
            for (std::size_t s = 0; s < kRefTokens / 4 - 17; ++s) few_slots.prompt_image_pad_mask[s] = 1U;
            refused = false;
            try {
                cudaStream_t s3 = nullptr;
                if (cudaStreamCreate(&s3) != cudaSuccess) { throw std::runtime_error("no image stream"); }
                const GeneratedImage bad = run_image_job(reader, dit, vae, few_slots, s3);
                (void)bad;
                cudaStreamDestroy(s3);
            } catch (const std::exception& error) {
                refused             = true;
                const std::string m = error.what();
                require(m.find("do not cover the img_shapes blocks") != std::string::npos ||
                            m.find("reference tokens") != std::string::npos,
                        "the slot-count disagreement failed for a reason other than the x4 invariant: " +
                            m);
            }
            require(refused, "an image pad mask that cannot cover ref_grids was accepted");
            require(image_device_bytes_live() == hog.bytes,
                    "the slot-count refusal left image device bytes live");

            // The boundary the other way: the same count in a different place is a legal joint sequence
            // and must NOT be refused. The route's host-side gates are the counts only; this pins that,
            // so a future over-strict check cannot silently reject a real request whose reference image
            // sits at the start of the prompt.
            ImageJobInputs front = inputs;
            front.prompt_image_pad_mask.assign(kKept, 0U);
            for (std::size_t s = 0; s < kImageSlots; ++s) front.prompt_image_pad_mask[s] = 1U;
            cudaStream_t s4 = nullptr;
            if (cudaStreamCreate(&s4) != cudaSuccess) { throw std::runtime_error("no image stream"); }
            const GeneratedImage accepted = run_image_job(reader, dit, vae, front, s4);
            cudaStreamDestroy(s4);
            require(accepted.latent.size() == kTargetTokens * kCin,
                    "the repositioned-mask run did not complete");
            require(image_device_bytes_live() == hog.bytes,
                    "the repositioned-mask run left image device bytes live");
            hog.retire(0);
        }
        require(image_device_bytes_live() == 0, "the ledger holds bytes at the end of the run");
    } catch (const std::exception& error) {
        std::cout << "EDIT_STEP_FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "EDIT_STEP_OK\n";
    return 0;
}
