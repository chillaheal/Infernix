// GPU qualification of the EDIT route's conditioned text tower
// (src/models/qwen_image/text_tower.cu run_conditioned_text_tower): the 36-layer Qwen3-VL forward run
// over an inputs_embeds whose <|image_pad|> rows are the vision tower's merger rows, with the deepstack
// rows added to those same rows after decoder layers 0, 1 and 2.
//
// The reference is the checkpoint's OWN conditioned forward, dumped by
// image-edit/dump_edit_text_oracle.py on the SAME artifact:
//   <case>_prompt_embeds_f32.npy   the exact forward (float32 weights), post-drop,
//   <case>_prompt_embeds_bf16.npy  the same forward in BF16 -- the reference's own dtype error,
//   <case>_image_pad_mask.npy      the model's own mask over the KEPT rows (1 where id == 151655).
// The merger/deepstack rows the scatter consumes are the vision tower's own outputs from the qualified
// M3 dump (tensors_edit_vision), so the two towers are chained exactly as the edit route chains them.
//
// The gate is the M3 one, and for the same measured reason: two independent BF16 chains of a 36-layer
// tower sit ~0.12-0.18 apart on this tensor, so any a-priori band below the reference's own error would
// measure the reference, not the engine. Floor = the distance from the exact forward to the FARTHEST
// certified correct forward over the same rows, computed here from the dumped files:
//
//   {case}_prompt_embeds_bf16.npy   and, when INFERNIX_IMAGE_ORACLE_F32FED points at it,
//   {case}_prompt_embeds_f32fed.npy -- the same BF16 tower fed the FP32 ViT's merger rows and taps
//   (image-edit/dump_text_fp32fed_probe.py, whose BF16-fed pass was verified bit-identical to the file
//   above). Both dumped passes are fed IDENTICAL BF16 picture rows -- dump_edit_text_oracle.py:494-496
//   asserts the difference is 0.0 for each of them -- so their two-pass spread carries the TOWER's dtype
//   error and nothing of the ViT's, while the engine's image rows arrive carrying the ViT (its own merger
//   rows sit 0.1396 from the oracle's, against a checkpoint BF16-vs-FP32 merger scatter of 0.1580).
//   Measured on the image rows, that third view sits 0.429 from the BF16-fed pass and 0.079 is what the
//   gate was dividing by: a floor no forward whose picture came from a different ViT arithmetic could
//   meet. Absent the file, the certified two-view gate is exactly as it was.
//
//   whole tensor: rel_fro(engine, f32) <= max(2 * floor, 1e-2)
//   per row:      ||engine_r - f32_r|| / ||f32|| <= max(2 * that row's own floor share, floor)
//   per class:    the text rows and the image rows are gated SEPARATELY (the plan's Step 2) -- a
//                 whole-tensor norm would let 1014-14 bad image rows hide among the text rows.
//
// Three further asserts carry the parts no tolerance can express: the mask must match the model's own
// byte for byte, the deepstack add must be ON the executed path (the ablated run must differ from the
// stock one), and the conditioned entry must reduce BIT-EXACTLY to run_text_tower when the scatter is
// the identity -- which is the certified t2i route's regression, without a second fixture.
//
// Skips (77) without INFERNIX_TEST_ARTIFACT or INFERNIX_IMAGE_ORACLE, or when either is absent;
// returns 2 with no CUDA device. Runs on the real GPU: it must not overlap a live serve.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen_image/image_device_util.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "models/qwen_image/text_tower.h"
#include "models/qwen_image/vision_tower.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;
namespace qo = qioracle;

constexpr double kFloorFactor     = 2.0;    // the engine may drift at most 2x the reference's own BF16 error
constexpr double kGlobalFloor     = 1.0e-2; // for boundaries the reference computes exactly
// Per-layer cross-precision floor: {case}_layerNN_hidden_f32.npy is the oracle's FP32 tower, so the
// engine's BF16 rows drift up to the certified tower's a-priori tolerance (2e-2,
// test_image_texttower_prod_cuda.cu -- "never shrunk to pass"), not the reference's own rounding.
// A wrong feed (scatter, deepstack add or tap order) lands at ~0.08+ on the pad rows, 4x outside.
constexpr double kTowerAPriori    = 2.0e-2;
// Handoff-gate bands, one per fed row set: the error M3 certified on the SAME plane for the same
// output (max over small/portrait, the VISION_OK gates of blackout #18): deepstack 0/1/2
// (tap08/16/24) 0.0576/0.1587/0.1742, main merger 0.3987. A wrong tap, pad order or run compares
// unrelated draws and lands at rel_fro ~1.4 -- 2.5x to 24x outside. One uniform band cannot work:
// tap08 needs 0.06 while the merger needs 0.42.
constexpr double kHandoffDeepstack[3] = {6.0e-2, 1.6e-1, 2.0e-1};
constexpr double kHandoffMergerBand   = 4.2e-1;
constexpr double kSliceFloorShare = 1.0;    // a row may add at most one whole tensor's floor
constexpr std::size_t kSystemPrefixRows = 14; // len(apply_chat_template(system only)) -- asserted, not used

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : fallback;
}

double diff_sq(const std::vector<float>& a, const std::vector<float>& b, std::size_t offset, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset + i]) - static_cast<double>(b[offset + i]);
        sum += v * v;
    }
    return sum;
}

double norm_sq(const std::vector<float>& a, std::size_t offset, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset + i]);
        sum += v * v;
    }
    return sum;
}

// One class of rows (text or image) of the kept prompt conditioning, against both references.
// `fatal` gates decide; a non-fatal call only prints and returns the verdict, so the global gate no
// longer throws before the per-row and class gates can show WHERE the drift lives (blackout #20).
bool gate_class(const std::string& tag, const std::string& what, const std::vector<float>& engine,
                const std::vector<float>& f32, const std::vector<float>& floor_ref, std::size_t D,
                const std::vector<std::uint8_t>& selected, double base,
                double global_floor = kGlobalFloor, bool fatal = true,
                const std::vector<float>* fed_ref = nullptr) {
    double engine_sq = 0.0, floor_sq = 0.0, fed_sq = 0.0, class_sq = 0.0, dev_dot = 0.0;
    std::size_t rows = 0;
    for (std::size_t row = 0; row < selected.size(); ++row) {
        if (!selected[row]) continue;
        ++rows;
        engine_sq += diff_sq(engine, f32, row * D, D);
        floor_sq  += diff_sq(floor_ref, f32, row * D, D);
        // The third certified view's distance to the SAME anchor, by the SAME formula. A floor built
        // from the distance to one correct forward cannot be the floor for a class whose input the
        // reference family itself varies -- so the floor is how far the EXACT forward is from the
        // farthest correct one, not from the convenient one.
        if (fed_ref != nullptr) { fed_sq += diff_sq(*fed_ref, f32, row * D, D); }
        class_sq  += norm_sq(f32, row * D, D);
        for (std::size_t i = 0; i < D; ++i) {
            const std::size_t o = row * D + i;
            const double e = static_cast<double>(engine[o]) - static_cast<double>(f32[o]);
            const double b = static_cast<double>(floor_ref[o]) - static_cast<double>(f32[o]);
            dev_dot += e * b;
        }
    }
    require(rows > 0, tag + ": the " + what + " row class is empty");
    const double engine_err  = std::sqrt(engine_sq) / base;
    const double floor_tower = std::sqrt(floor_sq) / base;
    const double floor_fed   = (fed_ref != nullptr) ? std::sqrt(fed_sq) / base : 0.0;
    const double floor_err   = std::max(floor_tower, floor_fed);
    const double band        = std::max(kFloorFactor * floor_err, global_floor);
    // The same numbers over THIS class's own norm. `base` is a shared denominator, so a class whose
    // rows carry a larger magnitude (the merger rows do) reads bigger here than it is; the classrel
    // ratio is what is comparable between text and image rows. Diagnostics only -- the band below is
    // unchanged and still decides.
    const double class_nrm  = std::sqrt(class_sq > 0.0 ? class_sq : 1.0);
    // cos = the angle between the engine's deviation and the reference's OWN BF16 deviation on these
    // rows; gain = how far along the reference's deviation the engine's one reaches. Both near (1, >=1)
    // = the same rounding direction, just amplified -- a numerical spread, not a semantic bug. cos near
    // 0 or negative with a high ratio = a mechanism of the engine's own. Printed at every layer and at
    // the end, so the layer where one class's ratio or cosine diverges from the other's is visible.
    const double cos  = (engine_sq > 0.0 && floor_sq > 0.0)
                            ? dev_dot / (std::sqrt(engine_sq) * std::sqrt(floor_sq)) : 0.0;
    const double gain = floor_sq > 0.0 ? dev_dot / floor_sq : 0.0;
    const double class_floor = std::max(std::sqrt(floor_sq), std::sqrt(fed_sq)) / class_nrm;
    std::cout << "  " << tag << " " << what << " rows=" << rows << " eng=" << engine_err
              << " floor=" << floor_err << " band=" << band
              << " floor_tower=" << floor_tower << " floor_fedview=" << floor_fed
              << " classrel eng=" << std::sqrt(engine_sq) / class_nrm
              << " floor=" << class_floor
              << " ratio=" << (floor_err > 0.0 ? engine_err / floor_err : 0.0)
              << " cos=" << cos << " gain=" << gain << "\n";
    if (fatal) {
        require(engine_err <= band,
                tag + ": the " + what + " rows drift further than twice the reference's own");
    }
    return engine_err <= band;
}

} // namespace

int main() {
    const std::string artifact_path = env_or("INFERNIX_TEST_ARTIFACT", "");
    const std::string oracle_path   = env_or("INFERNIX_IMAGE_ORACLE", "");
    if (artifact_path.empty() || oracle_path.empty()) {
        std::cout << "INFERNIX_TEST_ARTIFACT / INFERNIX_IMAGE_ORACLE unset; skipping the conditioned tower\n";
        return 77;
    }
    std::error_code ec;
    if (!std::filesystem::exists(artifact_path, ec) || !std::filesystem::exists(oracle_path, ec)) {
        std::cout << "artifact or oracle absent; skipping the conditioned tower\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        std::cout << "no CUDA device; the conditioned tower qualification is runnable\n";
        return 2;
    }
    const std::string dir         = qo::join(oracle_path, "tensors_edit_text_tower");
    const std::string vision_dir  = qo::join(oracle_path, "tensors_edit_vision");

    try {
        Reader reader(artifact_path);
        Binder binder(reader);
        // One binder for both towers so the caller's finish() sees a single streamed-only plan -- the
        // edit route's shape: the vision tower's rows are what conditions the text tower.
        TextTowerWeights tower = bind_text_encoder(binder);
        VisionWeights vision   = bind_image_vision(binder);
        (void)std::move(binder).finish();
        require(tower.layers.size() == tower.dims.layers && tower.dims.hidden == vision.dims.out_hidden,
                "the text tower and the vision merger do not share a hidden width");
        const std::size_t D = tower.dims.hidden;

        // The production dequantization (the serve route caches this once per process).
        const std::vector<std::byte> embed_table = dequantize_embed_table(reader, tower);
        require(embed_table.size() == 2ULL * tower.dims.vocab * D, "the embed table is not [vocab, hidden]");
        std::cout << "text tower bound: " << tower.layers.size() << " layers, hidden " << D << ", vocab "
                  << tower.dims.vocab << ", embed table " << embed_table.size() / (1024ULL * 1024ULL) << " MiB\n";

        for (const std::string tag : {"small", "portrait"}) {
            // ---- the model's own conditioned input, from the dump ----
            const qo::NpyArray ids_array = qo::read_npy(qo::join(dir, std::string(tag) + "_input_ids.npy"));
            require(ids_array.shape.size() == 1, "input_ids is not a flat id sequence");
            const std::vector<int> ids = ids_array.ints();
            const std::size_t S = ids.size();

            const qo::NpyArray want_embeds = qo::read_npy(qo::join(dir, std::string(tag) + "_prompt_embeds_f32.npy"));
            const qo::NpyArray want_bf16   = qo::read_npy(qo::join(dir, std::string(tag) + "_prompt_embeds_bf16.npy"));
            require(want_embeds.shape.size() == 2 && want_embeds.shape[1] == D, "prompt_embeds is not [rows, D]");
            const std::size_t kept = want_embeds.shape[0];
            // The dropped prefix is what the dump says it is -- derived, then pinned to the system line.
            require(kept < S, "the dumped conditioning is not shorter than the prompt");
            require(S - kept == kSystemPrefixRows, "the dropped prefix is not the system line's row count");
            const std::size_t drop_rows = S - kept;
            const std::vector<float> f32  = want_embeds.floats();
            const std::vector<float> bf16 = want_bf16.floats();
            require(bf16.size() == f32.size(), "the two references differ in size");

            // The third certified view, probed by file (see the header). Empty keeps every gate below
            // byte-for-byte the two-view gate it has always been.
            std::vector<float> fed_view;
            {
                const std::string fed_dir = env_or("INFERNIX_IMAGE_ORACLE_F32FED", "");
                if (!fed_dir.empty()) {
                    const std::string fed_file = qo::join(fed_dir, tag + "_prompt_embeds_f32fed.npy");
                    if (std::filesystem::exists(fed_file, ec)) {
                        const qo::NpyArray fed_array = qo::read_npy(fed_file);
                        require(fed_array.shape.size() == 2 && fed_array.shape[0] == kept &&
                                    fed_array.shape[1] == D,
                                tag + ": the dumped f32fed conditioning is not [kept, D]");
                        fed_view = fed_array.floats();
                        std::cout << "  " << tag << " floor views: bf16 + f32fed (" << fed_file << ")\n";
                    }
                }
            }
            const std::vector<float>* const fed_ptr = fed_view.empty() ? nullptr : &fed_view;

            const qo::NpyArray mask_array = qo::read_npy(qo::join(dir, std::string(tag) + "_image_pad_mask.npy"));
            // The dump carries the mask at the FULL prompt length -- the model is handed the mask before
            // the system prefix is dropped -- while the engine returns the kept rows. The reference is
            // therefore the tail below the system line, and the prefix must be pad-free for that to hold.
            require(mask_array.shape.size() == 1 && mask_array.shape[0] == S,
                    "the dumped mask is not one entry per prompt row");
            const std::vector<std::uint8_t> mask_full = mask_array.bools();
            for (std::size_t s = 0; s < drop_rows; ++s) {
                require(mask_full[s] == 0U, "the dropped system prefix carries image pad tokens");
            }
            const std::vector<std::uint8_t> want_mask(mask_full.begin() + static_cast<std::ptrdiff_t>(drop_rows),
                                                      mask_full.end());
            std::vector<std::uint8_t> is_image(S, 0U);
            for (std::size_t s = 0; s < S; ++s) {
                is_image[s] = (static_cast<std::uint32_t>(ids[s]) == kImagePadTokenId) ? 1U : 0U;
            }
            std::size_t pads = 0;
            for (std::size_t s = drop_rows; s < S; ++s) pads += is_image[s];

            // ---- the vision rows the scatter consumes: the qualified M3 tower, same artifact ----
            const qo::NpyArray grid_array = qo::read_npy(qo::join(vision_dir, std::string(tag) + "_grid.npy"));
            const std::vector<float> grid_values = grid_array.floats();
            require(grid_values.size() == 3, "the vision grid is not [t, h, w]");
            const VisionItemGeometry grid{static_cast<std::uint32_t>(grid_values[0]),
                                          static_cast<std::uint32_t>(grid_values[1]),
                                          static_cast<std::uint32_t>(grid_values[2])};
            const qo::NpyArray pixels = qo::read_npy(qo::join(vision_dir, std::string(tag) + "_patches.npy"));
            require(pixels.shape.size() == 2 && pixels.shape[0] == std::size_t(grid.height) * grid.width &&
                        pixels.shape[1] == vision.dims.patch_width,
                    "the vision pixels are not [patches, patch_width]");
            const std::vector<float> pixel_values = pixels.floats();
            std::vector<std::uint16_t> patch_bits(pixel_values.size());
            for (std::size_t i = 0; i < pixel_values.size(); ++i) { patch_bits[i] = to_bf16(pixel_values[i]); }
            const VisionTowerOutput rows = run_vision_tower(reader, vision, patch_bits, grid);
            const std::size_t merged = rows.merger_rows.size() / D;
            // The x4 invariant seen from the text side: one merger row per pad token, and the model's
            // own mask counts exactly those tokens.
            require(merged == pads, "the merger rows and the image pad tokens disagree: " +
                                        std::to_string(merged) + " rows against " + std::to_string(pads) + " pads");
            std::cout << "case " << tag << ": S=" << S << " kept=" << kept << " drop=" << drop_rows
                      << " pads=" << pads << "\n";

            // The mask must be one contiguous run (a single reference image) and equal the model's own.
            {
                bool opened = false, closed = false;
                for (std::size_t s = 0; s < kept; ++s) {
                    require(want_mask[s] == is_image[drop_rows + s],
                            "the dumped mask disagrees with the ids at kept row " + std::to_string(s));
                    if (want_mask[s]) {
                        require(!closed, "the image pad run is not contiguous: a second run starts at " +
                                             std::to_string(s));
                        opened = true;
                    } else if (opened) {
                        closed = true;
                    }
                }
            }

            // ---- the M-RoPE positions the conditioned rotation must use ----
            // A 1-D arange is what the reference degenerates to when NO image is present, so the
            // certified text-only route can never catch a missing M-RoPE: it is pinned here, against
            // the reference's own [3,S] table, before the forward is judged. The count below is the
            // discriminator -- an engine that rotated with pos[s]=s would print 0.
            const std::vector<std::array<int, 3>> grids{std::array<int, 3>{static_cast<int>(grid.temporal),
                                                                           static_cast<int>(grid.height),
                                                                           static_cast<int>(grid.width)}};
            const qo::NpyArray want_pos  = qo::read_npy(qo::join(dir, tag + "_rope_positions.npy"));
            require(want_pos.shape.size() == 2 && want_pos.shape[0] == 3 && want_pos.shape[1] == S,
                    "the dumped mrope positions are not [3, S]");
            const std::vector<int> want_pos_flat = want_pos.ints();
            const std::vector<int> pos           = build_mrope_positions(ids, grids, 2U);
            require(pos.size() == want_pos_flat.size(), "the engine's mrope positions differ in size");
            std::size_t differing_rows = 0;
            for (std::size_t s = 0; s < S; ++s) {
                require(pos[s] == want_pos_flat[s] && pos[S + s] == want_pos_flat[S + s] &&
                            pos[2ULL * S + s] == want_pos_flat[2ULL * S + s],
                        tag + ": the engine's mrope positions differ from the model's own at row " +
                            std::to_string(s));
                if (s >= drop_rows &&
                    (pos[s] != int(s) || pos[S + s] != int(s) || pos[2ULL * S + s] != int(s))) {
                    ++differing_rows;
                }
            }
            std::cout << "  " << tag << " mrope: " << differing_rows << " of " << kept
                      << " kept rows differ from the 1-D arange\n";
            require(differing_rows == (tag == "small" ? 89ULL : 1026ULL),
                    tag + ": the picture no longer moves the rotary positions it must (" +
                        std::to_string(differing_rows) + ")");

            // ---- the conditioned forward ----
            // The tower is FED the reference's own rows (the fixtures, converted to BF16 -- exact,
            // they are BF16-exact), filled in by the handoff block below: the oracle knows only the
            // reference's rows, and feeding the engine's certified-but-different rows would push the
            // handoff error into every tower gate, so "red at layer 0" could no longer mean "feed or
            // scatter wrong". The engine's own vision output is checked against the reference's rows
            // in that block, in M3's certified bands.
            ConditionedVision conditioned;
            conditioned.grids = grids;
            require(vision.deepstack.size() == 3,
                    "the bound vision tower does not have the three certified deepstack taps");

            // ---- the rows the engine's vision produced, and the rows the tower is FED ----
            // M3 qualified the ViT stage by stage against torch boundary dumps on THIS plane; the
            // three DEEPSTACK mergers are qualified nowhere else in THIS test, and a wrong deepstack
            // row is added three times inside the tower -- enough on its own to be the whole residual.
            // This block checks the engine's own vision output (rows) against the reference's rows in
            // M3's certified per-row-set bands (kHandoffDeepstack / kHandoffMergerBand): a wrong tap,
            // pad order or run compares two unrelated draws of the same distribution and lands at
            // rel_fro ~1.4, 2.5x to 24x outside the band; the engine's differing accumulation lands
            // inside it (0.027 at tap08, 0.065 at tap16, measured in #17/#18). The print says WHICH
            // row set, so a red gate names the feed.
            //
            // Why the M3 bands and not a computed floor: these four fixtures carry no FP32
            // information. Every value in {case}_deepstack_N_f32.npy and {case}_inputs_embeds_f32.npy
            // is exactly BF16-representable (bf16_exact=1.000000), and the deepstack files are
            // bit-identical to M3's own {case}_deepstack_N.npy (rel_fro=0.000000, bf16_bitmatch=1.000000
            // for all six). Rounding the reference to BF16 therefore measures nothing: floor_rows comes
            // out 0 and the usual band collapses to kGlobalFloor, which no BF16 engine can meet --
            // torch's OWN error at this boundary is 2.9% (M3's tap08 floor). The handoff band IS the
            // reference's own error at this boundary, measured by M3 on the same plane.
            {
                auto gate_fed_rows = [&](const std::vector<std::uint16_t>& have, const std::vector<float>& want,
                                         const std::string& what, double band_rows,
                                         const std::vector<float>* twin = nullptr) {
                    require(have.size() == want.size(),
                            tag + ": the engine's " + what + " is not the size the reference dumped");
                    std::vector<std::uint16_t> rounded_bits(want.size());
                    for (std::size_t i = 0; i < want.size(); ++i) { rounded_bits[i] = to_bf16(want[i]); }
                    const std::vector<float> rounded   = from_bits(rounded_bits);
                    const std::vector<float> got       = from_bits(have);
                    const double base_rows  = std::sqrt(norm_sq(want, 0, want.size()));
                    const double floor_rows = std::sqrt(diff_sq(rounded, want, 0, want.size())) / base_rows;
                    const double err        = std::sqrt(diff_sq(got, want, 0, want.size())) / base_rows;
                    // These four fixtures are BF16-exact, so rounding them measures nothing and
                    // floor_rows comes out 0. Where the oracle carries the FP32 ViT's own rows for this
                    // set (INFERNIX_IMAGE_ORACLE_VISION_F32), THAT is the checkpoint's error at this
                    // boundary and the only floor that exists here. Printed either way; the a-priori
                    // band still wins wherever it is larger, so no gate is relaxed by the measurement.
                    double twin_floor = 0.0;
                    if (twin != nullptr && twin->size() == want.size()) {
                        twin_floor = std::sqrt(diff_sq(*twin, want, 0, want.size())) / base_rows;
                    }
                    const double floor_used = std::max(floor_rows, twin_floor);
                    // The handoff band is the floor for a BF16-exact fixture; where the fixture does
                    // carry a floor, the house band applies too -- take the larger, never the smaller.
                    band_rows = std::max(band_rows, std::max(kFloorFactor * floor_used, kGlobalFloor));
                    std::cout << "  " << tag << " " << what << " rel_fro=" << err << " floor=" << floor_rows
                              << " model_scatter=" << twin_floor << " band=" << band_rows << "\n";
                    // WHICH row, not only how much: one wrong fed row contaminates every row downstream
                    // through attention, and it shows up in the aggregate as a class error of about
                    // 1/sqrt(rows) -- which is what blackout #21 measured for the merger feed (0.14 of 77
                    // rows). The per-row share below separates that from real accumulation drift.
                    {
                        const std::size_t n_rows = want.size() / (D > 0 ? D : 1);
                        double rw_worst = 0.0;
                        std::size_t rw_at = 0;
                        for (std::size_t r = 0; r < n_rows; ++r) {
                            const double e = std::sqrt(diff_sq(got, want, r * D, D)) /
                                             (base_rows > 0.0 ? base_rows : 1.0);
                            if (e > rw_worst) { rw_worst = e; rw_at = r; }
                        }
                        std::cout << "  " << tag << " " << what << " worst fed row " << rw_at
                                  << " share=" << rw_worst << " (1/sqrt(rows) = "
                                  << (n_rows > 0 ? 1.0 / std::sqrt(static_cast<double>(n_rows)) : 0.0)
                                  << ")\n";
                    }
                    require(err <= band_rows,
                            tag + ": the engine's " + what + " is not the reference's rows");
                };
                auto bits_of = [](const std::vector<float>& v) {
                    std::vector<std::uint16_t> b(v.size());
                    for (std::size_t i = 0; i < v.size(); ++i) { b[i] = to_bf16(v[i]); }
                    return b;
                };
                // The FP32 ViT's own rows for the same sets, probed by file: empty keeps the gate as it
                // was (print-only floor 0), present gives the handoff a measured model scatter.
                auto fed_twin = [&](const std::string& fname) {
                    const std::string d = env_or("INFERNIX_IMAGE_ORACLE_VISION_F32", "");
                    if (d.empty()) { return std::vector<float>{}; }
                    const std::string f = qo::join(d, fname);
                    if (!std::filesystem::exists(f, ec)) { return std::vector<float>{}; }
                    return qo::read_npy(f).floats();
                };
                for (std::size_t i = 0; i < vision.deepstack.size(); ++i) {
                    const std::string name = tag + "_deepstack_" + std::to_string(i) + "_f32.npy";
                    const qo::NpyArray want_ds = qo::read_npy(qo::join(dir, name));
                    require(want_ds.shape.size() == 2 && want_ds.shape[0] == merged && want_ds.shape[1] == D,
                            tag + ": the dumped " + name + " is not [pads, hidden]");
                    const std::vector<float> ds = want_ds.floats();
                    const std::vector<float> ds_twin =
                        fed_twin(tag + "_deepstack_" + std::to_string(i) + "_f32.npy");
                    gate_fed_rows(rows.deepstack_rows[i], ds, "deepstack rows " + std::to_string(i),
                                  kHandoffDeepstack[i], ds_twin.empty() ? nullptr : &ds_twin);
                    // The tower is fed the reference's own rows: BF16-exact, so this conversion is exact.
                    conditioned.deepstack_rows.push_back(bits_of(ds));
                }
                const qo::NpyArray want_ie = qo::read_npy(qo::join(dir, tag + "_inputs_embeds_f32.npy"));
                require(want_ie.shape.size() == 2 && want_ie.shape[1] == D,
                        tag + ": the dumped inputs_embeds are not [rows, hidden]");
                const std::vector<float> ie = want_ie.floats();
                std::vector<float> want_merger;
                if (want_ie.shape[0] == S) {
                    for (std::size_t s = 0; s < S; ++s) {
                        if (!(ids[s] >= 0 && static_cast<std::uint32_t>(ids[s]) == kImagePadTokenId)) continue;
                        want_merger.insert(want_merger.end(), ie.begin() + s * D, ie.begin() + (s + 1) * D);
                    }
                } else {
                    require(want_ie.shape[0] == kept,
                            tag + ": the dumped inputs_embeds are neither [S, hidden] nor [kept, hidden]");
                    for (std::size_t s = 0; s < kept; ++s) {
                        if (!want_mask[s]) continue;
                        want_merger.insert(want_merger.end(), ie.begin() + s * D, ie.begin() + (s + 1) * D);
                    }
                }
                const std::vector<float> merger_twin = fed_twin(tag + "_merger_f32.npy");
                gate_fed_rows(rows.merger_rows, want_merger, "merger rows", kHandoffMergerBand,
                              merger_twin.empty() ? nullptr : &merger_twin);
                conditioned.merger_rows = bits_of(want_merger);
                require(conditioned.deepstack_rows.size() == vision.deepstack.size(),
                        "the fed deepstack rows are not one per deepstack add");
                for (const auto& add : conditioned.deepstack_rows) {
                    require(add.size() == merged * D, "a deepstack tensor is not [pads, hidden]");
                }
                require(conditioned.merger_rows.size() == merged * D,
                        "the fed merger rows are not [pads, hidden]");
            }

            // The three layers the deepstack adds land on, asked for by pointer: production passes
            // nothing and the forward is unchanged. When the oracle dumped every layer
            // (INFERNIX_TEXT_CURVE=1 in image-edit\dump_edit_text_oracle.py) the whole tower is asked
            // for instead: the three certify the handoff, the rest is the growth curve that separates
            // an image-row defect BORN at layer 0 (a wrong scatter or add) from one that ACCUMULATES
            // alongside the text rows (the BF16 depth of the tower). Probed by file, so an oracle
            // dumped without the curve keeps the certified three-layer gate exactly as it was.
            std::size_t want_layers = conditioned.deepstack_rows.size();
            if (std::filesystem::exists(qo::join(dir, tag + "_layer03_hidden_f32.npy"))) {
                want_layers = tower.layers.size();
            }
            std::vector<std::vector<std::uint16_t>> layer_snapshots(want_layers);
            reset_image_device_peak();
            const ConditionedTextTowerOutput out =
                run_conditioned_text_tower(reader, tower, embed_table, ids, drop_rows, conditioned,
                                           &layer_snapshots);
            require(out.prompt_embeds.size() == kept * D, "the conditioned run returned the wrong row count");
            require(out.image_pad_mask.size() == kept, "the conditioned run returned the wrong mask length");
            for (std::size_t s = 0; s < kept; ++s) {
                require(out.image_pad_mask[s] == want_mask[s],
                        "the engine's mask differs from the model's own at kept row " + std::to_string(s));
            }
            require(image_device_bytes_live() == 0, "the conditioned tower left image device bytes live");

            // ---- where the drift enters: the three layers the deepstack adds land on ----
            // Unlike the fed rows, {case}_layerNN_hidden_f32.npy IS a genuine FP32 tensor
            // (bf16_exact=0.00005): the oracle ran the tower in FP32 while the engine runs BF16, so this
            // is a cross-precision comparison and the band is floored by the certified tower's a-priori
            // tolerance kTowerAPriori, not by the reference's own (tiny) BF16 rounding. The anchor's hooks
            // fire after the decoder layer INCLUDING that layer's deepstack add -- the same point the
            // engine's snapshot is taken -- and they see the FULL sequence, dropped prefix included,
            // hence [S, D]. These three numbers separate "the picture is fed wrong" (red at layer 0, a
            // wrong scatter/add lands ~4x outside kTowerAPriori on the pad rows) from "the tower drifts
            // on its own" (green at 0, red as the layers accumulate). The tower was fed the reference's
            // own rows (above), so a red here is the engine's: scatter, deepstack add or accumulation --
            // never the handoff.
            // The aggregate over ALL S rows admits an image-row-resident defect of exactly the size the
            // final gate calls red: a class error of 0.2129 on 77 of 112 rows is 0.2129/sqrt(77/112) =
            // 0.024 per row, and the band is 2e-2. So the same numbers are printed per class -- an image
            // row that is already off at layer 0 is a feed/add defect, one that grows with the text rows
            // is the BF16 depth of the tower. Non-fatal: the three aggregate gates above still decide.
            std::vector<std::uint8_t> layer_text(S, 0U), layer_image(S, 0U);
            for (std::size_t s = 0; s < kept; ++s) {
                layer_text[drop_rows + s]  = want_mask[s] ? 0U : 1U;
                layer_image[drop_rows + s] = want_mask[s];
            }
            // The aggregate that DECIDES is over the kept rows, because the kept rows are the contract:
            // run_conditioned_text_tower returns rows [drop_rows, S) and nothing else is ever read
            // (text_tower.h). Blackout #41 showed why the full-[S,D] aggregate cannot say anything:
            // row 0 is an attention sink. Its norm is 78x the mean row at layer 6 (197x for portrait)
            // and it holds 99.8% of ||h||^2 from that layer on, so an aggregate over all S rows IS
            // row 0's gate -- and row 0 sits in the dropped system prefix, which production never
            // emits. That is also why the three fatal layers were always green: the sink forms at
            // layer 6 (largest row 1.9x at layer 5). It is printed below as a labelled diagnostic, not
            // dropped, because a 56x-same-direction spread on the sink row is still the engine's.
            std::vector<std::uint8_t> layer_kept(S, 0U);
            for (std::size_t s = drop_rows; s < S; ++s) { layer_kept[s] = 1U; }
            const std::vector<std::uint8_t> all_rows(S, 1U);
            for (std::size_t nl = 0; nl < layer_snapshots.size(); ++nl) {
                const std::string lname =
                    tag + "_layer" + (nl < 10 ? "0" : "") + std::to_string(nl) + "_hidden_f32.npy";
                const qo::NpyArray want_layer = qo::read_npy(qo::join(dir, lname));
                require(want_layer.shape.size() == 2 && want_layer.shape[0] == S && want_layer.shape[1] == D,
                        tag + ": the dumped " + lname + " is not [S, hidden]");
                require(layer_snapshots[nl].size() == S * D,
                        tag + ": the engine's layer " + std::to_string(nl) + " snapshot is not [S, hidden]");
                const std::vector<float> wl = want_layer.floats();
                const std::vector<float> el = from_bits(layer_snapshots[nl]);
                for (std::size_t i = 0; i < el.size(); ++i) {
                    require(std::isfinite(el[i]), tag + ": layer " + std::to_string(nl) +
                                                      " produced a non-finite value");
                }
                // The engine's OWN curve, on disk, when asked for. Blackout #42 localised the remaining
                // red: the image rows acquire a ~0.18 class-relative deviation at layer 16 and hold it
                // flat to 33, and the text rows sit at 0.005 of their own scale until layer 33 and then
                // jump to 0.166 -- both exactly where the ORACLE's own one-layer step is large (image
                // 0.2676 into layer 16, text 0.6647 into layer 34, from image-edit\analyze_text_curve.py).
                // One printed ratio cannot tell a mechanism apart from an engine tensor that simply sits
                // between two of the reference's, so the snapshots themselves go to disk and the same
                // nearest-layer map that located the oracle's sink row runs over them on CPU afterwards
                // (image-edit\analyze_engine_curve.py) -- no second GPU window per question. Raw float32,
                // no npy header: the file is S*D floats and its byte size is the whole contract.
                const std::string curve_dir = env_or("INFERNIX_ENGINE_CURVE_DIR", "");
                if (!curve_dir.empty()) {
                    const std::filesystem::path fname =
                        std::filesystem::path(curve_dir) /
                        (tag + "_engine_layer" + (nl < 10 ? "0" : "") + std::to_string(nl) + ".bin");
                    // Blackout #43 died here with "cannot open the engine curve file ...layer00.bin":
                    // ofstream does not create a missing parent directory, and the gate turned the whole
                    // M4 run red on a diagnostic. The directory is the test's to make.
                    std::error_code ec;
                    std::filesystem::create_directories(curve_dir, ec);
                    require(!ec, "cannot create the engine curve directory " + curve_dir + ": " + ec.message());
                    std::ofstream curve(fname, std::ios::binary | std::ios::trunc);
                    require(static_cast<bool>(curve), "cannot open the engine curve file " + fname.string());
                    curve.write(reinterpret_cast<const char*>(el.data()),
                                static_cast<std::streamsize>(el.size() * sizeof(float)));
                    require(curve.good(), "failed writing the engine curve file " + fname.string());
                }
                std::vector<std::uint16_t> lrounded_bits(wl.size());
                for (std::size_t i = 0; i < wl.size(); ++i) { lrounded_bits[i] = to_bf16(wl[i]); }
                const std::vector<float> lrounded = from_bits(lrounded_bits);
                const double layer_base = std::sqrt(norm_sq(wl, 0, wl.size()));
                // The three layers the adds land on stay FATAL (they are the handoff gate M4 ships
                // on). The curve beyond them is diagnostic: one layer of the other 33 blowing past a
                // band must not hide the shape of the curve, which is the whole reason to ask for it.
                gate_class(tag, "layer " + std::to_string(nl), el, wl, lrounded, D, layer_kept,
                           layer_base, kTowerAPriori, nl < conditioned.deepstack_rows.size());
                // Same denominator, the dropped prefix included: the sink row's own line, kept on
                // purpose so the number that made #41's curve look like a step function stays visible
                // next to the one that decides.
                gate_class(tag, "layer " + std::to_string(nl) + " all rows incl. dropped prefix",
                           el, wl, lrounded, D, all_rows, layer_base, kTowerAPriori, false);
                gate_class(tag, "layer " + std::to_string(nl) + " text rows", el, wl, lrounded, D,
                           layer_text, layer_base, kTowerAPriori, false);
                gate_class(tag, "layer " + std::to_string(nl) + " image rows", el, wl, lrounded, D,
                           layer_image, layer_base, kTowerAPriori, false);
            }

            // ---- the numeric gate ----
            const std::vector<float> engine = from_bits(out.prompt_embeds);
            for (std::size_t i = 0; i < engine.size(); ++i) {
                require(std::isfinite(engine[i]), "the conditioned tower produced a non-finite value");
            }
            const double base        = std::sqrt(norm_sq(f32, 0, f32.size()));
            const double floor_tower = std::sqrt(diff_sq(bf16, f32, 0, f32.size())) / base;
            const double floor_fed   = fed_ptr == nullptr
                                          ? 0.0
                                          : std::sqrt(diff_sq(*fed_ptr, f32, 0, f32.size())) / base;
            const double global_floor = std::max(floor_tower, floor_fed);
            std::cout << "  " << tag << " floor: tower=" << floor_tower << " fedview=" << floor_fed
                      << " decided=" << global_floor << "\n";
            require(base > 0.0, "the reference conditioning is all zeros");
            // Every gate below prints before the run decides. Blackout #20's throw on the global gate
            // hid the per-row evidence: uniform drift across all rows with the deepstack layers green is
            // BF16 accumulation through the amplifier, one red row class is a mechanism defect.
            bool global_ok = true, rows_ok = true;
            {
                const double engine_err = std::sqrt(diff_sq(engine, f32, 0, f32.size())) / base;
                const double band       = std::max(kFloorFactor * global_floor, kGlobalFloor);
                std::cout << "  " << tag << " prompt_embeds eng=" << engine_err << " floor=" << global_floor
                          << " band=" << band << "\n";
                global_ok = engine_err <= band;
            }
            std::vector<std::uint8_t> text_rows(kept), image_rows(kept);
            std::size_t worst_row = 0;
            double worst = 0.0;
            for (std::size_t s = 0; s < kept; ++s) {
                text_rows[s]  = want_mask[s] ? 0U : 1U;
                image_rows[s] = want_mask[s];
                const double row_engine = std::sqrt(diff_sq(engine, f32, s * D, D)) / base;
                const double row_tower  = std::sqrt(diff_sq(bf16, f32, s * D, D)) / base;
                const double row_fed    = fed_ptr == nullptr
                                             ? 0.0
                                             : std::sqrt(diff_sq(*fed_ptr, f32, s * D, D)) / base;
                const double row_floor  = std::max(row_tower, row_fed);
                const double row_band   = std::max(kFloorFactor * row_floor,
                                                   kSliceFloorShare * global_floor);
                if (row_engine > row_band) {
                    rows_ok = false;
                    std::cout << "  " << tag << " red kept row " << s
                              << (want_mask[s] ? " (image)" : " (text)") << " eng=" << row_engine
                              << " floor=" << row_floor << " band=" << row_band << "\n";
                }
                if (row_engine > worst) { worst = row_engine; worst_row = s; }
            }
            std::cout << "  " << tag << " kept rows: worst " << worst_row << " eng=" << worst
                      << (rows_ok ? ", all inside" : ", RED rows above") << "\n";
            const bool text_ok  = gate_class(tag, "text", engine, f32, bf16, D, text_rows, base,
                                             kGlobalFloor, false, fed_ptr);
            const bool image_ok = gate_class(tag, "image", engine, f32, bf16, D, image_rows, base,
                                             kGlobalFloor, false, fed_ptr);

            // ---- the text class, split by whether the picture MOVED the row ----
            // build_mrope_positions advances the running position by max(h/merge, w/merge) per image
            // instead of one per token, so every row AFTER a picture sits below its own index on all
            // three rotary axes while the rows before it keep pos == index: blackout #30's "89 of 98
            // kept rows differ from the 1-D arange" is 77 image rows + 12 shifted text rows, leaving 9
            // text rows at arange. That separates the two leading suspects for free. A rotary defect
            // (how the t/h/w sections are built and applied across head_dim) hits the shifted rows and
            // the vision rows together and must spare the arange rows; an attention defect on vision
            // CONTENT sorts by what a row is, not by where it sits, so the shifted text rows would
            // follow the plain text rows. Same gate, same floor, print-only: the four verdicts above
            // still decide -- no band moved.
            std::vector<std::uint8_t> text_arange(kept, 0U), text_shifted(kept, 0U);
            std::size_t n_arange = 0, n_shifted = 0;
            for (std::size_t s = 0; s < kept; ++s) {
                if (want_mask[s]) continue;
                const std::size_t j = drop_rows + s;
                const bool at_arange = pos[j] == static_cast<int>(j) && pos[S + j] == static_cast<int>(j) &&
                                       pos[2ULL * S + j] == static_cast<int>(j);
                if (at_arange) { text_arange[s] = 1U; ++n_arange; }
                else           { text_shifted[s] = 1U; ++n_shifted; }
            }
            if (n_arange != 0) {
                gate_class(tag, "text rows at arange positions", engine, f32, bf16, D, text_arange, base,
                           kGlobalFloor, false);
            }
            if (n_shifted != 0) {
                gate_class(tag, "text rows the picture shifted", engine, f32, bf16, D, text_shifted, base,
                           kGlobalFloor, false);
            }

            // The yardstick, asked directly. The engine is a BF16 chain; the anchor every gate above is
            // measured against is an FP32 pass with per-layer upcast weights. If the engine sits CLOSE to
            // torch's own BF16 pass while still far from FP32, the drift is what BF16 costs on these rows
            // and the FP32 anchor -- not the engine -- is the odd reference out; far from BOTH means the
            // engine carries a mechanism of its own. Same rows, same denominator, only the reference
            // swapped. Blackout #21/#22 left image rows at 3.1x the reference's own BF16 floor with the
            // deepstack layers green at 0.013 -- this separates accumulation from mechanism for free.
            {
                auto vs_bf16_pass = [&](const std::vector<std::uint8_t>& selected, const char* what) {
                    double sq = 0.0;
                    std::size_t rows = 0;
                    for (std::size_t row = 0; row < selected.size(); ++row) {
                        if (!selected[row]) continue;
                        ++rows;
                        sq += diff_sq(engine, bf16, row * D, D);
                    }
                    std::cout << "  " << tag << " " << what << " rows vs the BF16 pass eng="
                              << (rows ? std::sqrt(sq) / base : 0.0) << "\n";
                };
                vs_bf16_pass(text_rows, "text");
                vs_bf16_pass(image_rows, "image");
                if (n_arange != 0) vs_bf16_pass(text_arange, "text-at-arange");
                if (n_shifted != 0) vs_bf16_pass(text_shifted, "text-shifted-by-picture");
            }

            // ---- is the image-row deviation the SAME one torch's BF16 pass makes, or one of its own? ----
            // Both references now sit ~0.21 from the engine on these rows, so "far from both" no longer
            // separates a mechanism from a third draw of the same rounding. These numbers do:
            //   class-relative err/floor -- each class against its OWN norm, so the merger rows' larger
            //     magnitude stops hiding behind the global denominator (the ratio is what to compare
            //     between the classes: equal ratios = one uniform BF16 cost, a bigger one on image rows
            //     = those rows carry something the tower mishandles);
            //   a* -- the least-squares gain of the engine over the anchor. A scale defect (a doubled or
            //     missing normalization/add) moves it off 1 and leaves a small residual behind;
            //   cos -- the angle between the engine's deviation and torch's OWN deviation on the same
            //     rows. Near 1 = the same direction as the reference's own BF16 error, i.e. precision;
            //     near 0 = an orthogonal mechanism the engine carries by itself.
            // Prints only; the gates above and the final verdict still decide.
            {
                auto diagnose = [&](const std::vector<std::uint8_t>& selected, const char* what) {
                    double ff = 0.0, ef = 0.0, bf = 0.0, dev_dot = 0.0, e_f = 0.0, e_e = 0.0;
                    std::size_t rows = 0;
                    for (std::size_t row = 0; row < selected.size(); ++row) {
                        if (!selected[row]) continue;
                        ++rows;
                        const std::size_t o = row * D;
                        ff += norm_sq(f32, o, D);
                        ef += diff_sq(engine, f32, o, D);
                        bf += diff_sq(bf16, f32, o, D);
                        for (std::size_t i = 0; i < D; ++i) {
                            const double en = static_cast<double>(engine[o + i]);
                            const double f  = static_cast<double>(f32[o + i]);
                            const double b  = static_cast<double>(bf16[o + i]);
                            dev_dot += (en - f) * (b - f);
                            e_f     += en * f;
                            e_e     += en * en;
                        }
                    }
                    require(rows > 0, tag + ": the " + std::string(what) + " class is empty");
                    const double nrm    = std::sqrt(ff > 0.0 ? ff : 1.0);
                    const double a_star = ff > 0.0 ? e_f / ff : 0.0;
                    const double resid  = e_e - 2.0 * a_star * e_f + a_star * a_star * ff;
                    const double cos    = (ef > 0.0 && bf > 0.0)
                                              ? dev_dot / (std::sqrt(ef) * std::sqrt(bf)) : 0.0;
                    std::cout << "  " << tag << " " << what << " class-relative eng="
                              << std::sqrt(ef) / nrm << " floor=" << std::sqrt(bf) / nrm
                              << " ratio=" << (bf > 0.0 ? std::sqrt(ef / bf) : 0.0)
                              << " a*=" << a_star
                              << " resid_after_gain=" << (resid > 0.0 ? std::sqrt(resid) / nrm : 0.0)
                              << " cos_dev=" << cos << "\n";
                };
                diagnose(text_rows, "text");
                diagnose(image_rows, "image");
                // The two subclasses are the decisive pair: text-shifted cos_dev like the image rows
                // (low, ~0.12) with text-at-arange cos_dev like the plain text rows (high, ~0.67) says
                // the deviation travels with the POSITION, not with the vision CONTENT. The reverse --
                // shifted text rows behaving like the unshifted ones -- leaves attention over the
                // vision block as the surviving mechanism.
                if (n_arange != 0) diagnose(text_arange, "text-at-arange");
                if (n_shifted != 0) diagnose(text_shifted, "text-shifted-by-picture");
            }

            // ---- the deepstack add is on the executed path ----
            // A wrong layer index, or an add that silently never runs, only shows up as a delta against
            // the stock run: the band is the reference's own dtype error and a whole BF16 add is smaller
            // than it. Same ids, same merger rows, no deepstack rows -> must NOT be the stock result.
            ConditionedVision ablated;
            ablated.merger_rows = conditioned.merger_rows;  // the SAME fed rows as the main run
            ablated.grids       = grids;
            const ConditionedTextTowerOutput without =
                run_conditioned_text_tower(reader, tower, embed_table, ids, drop_rows, ablated);
            {
                const std::vector<float> ablated_engine = from_bits(without.prompt_embeds);
                double between = 0.0;
                for (std::size_t s = 0; s < kept; ++s) {
                    if (!want_mask[s]) continue;
                    between += diff_sq(engine, ablated_engine, s * D, D);
                }
                const double rel = std::sqrt(between) / base;
                std::cout << "  " << tag << " deepstack moves the image rows by rel_fro=" << rel << "\n";
                require(rel > 1.0e-2, tag + ": the deepstack add changed nothing on the image rows");
            }
            require(global_ok && rows_ok && text_ok && image_ok,
                    tag + ": the conditioned tower drifts further from the exact forward than twice "
                          "the reference's own BF16 error (the red rows are printed above)");

            // ---- the certified t2i route's regression ----
            // M-RoPE makes the conditioned entry differ from the text-only entry on an IMAGE prompt by
            // design, so the certified route is protected where it actually lives: a prompt with no
            // picture at all. Remove the picture's rows and the two entries must agree BIT for BIT --
            // that is the route /v1/images/generations ships on.
            std::vector<int> text_only;
            text_only.reserve(S - pads);
            for (std::size_t s = 0; s < S; ++s) {
                if (!is_image[s]) text_only.push_back(ids[s]);
            }
            require(text_only.size() + pads == S, "the text-only prompt is not the picture's rows removed");
            const ConditionedTextTowerOutput via_conditioned =
                run_conditioned_text_tower(reader, tower, embed_table, text_only, drop_rows, ConditionedVision{});
            const std::vector<std::uint16_t> via_tower =
                run_text_tower(reader, tower, embed_table, text_only, drop_rows);
            require(via_conditioned.prompt_embeds == via_tower,
                    tag + ": the conditioned entry is not bit-identical to run_text_tower without a picture");
            require(via_conditioned.image_pad_mask.size() == kept - pads,
                    tag + ": the picture-free conditioned run returned the wrong mask length");
            for (std::size_t s = 0; s < kept - pads; ++s) {
                require(via_conditioned.image_pad_mask[s] == 0U, tag + ": a picture-free prompt reports a pad row");
            }
            require(image_device_bytes_live() == 0, "the picture-free runs left image device bytes live");
        }

        // ---- the ledger's throw path ----
        // Same seam as the vision tower's: the Windows driver answers cudaMalloc from sysmem, so the
        // lane's own allocation counter is what can refuse mid-run. Fail one halfway through a valid
        // small-case run and require every byte back.
        const auto mib = [](std::uint64_t x) { return x / (1024ULL * 1024ULL); };
        const qo::NpyArray grid_array = qo::read_npy(qo::join(vision_dir, "small_grid.npy"));
        const std::vector<float> grid_values = grid_array.floats();
        const VisionItemGeometry grid{static_cast<std::uint32_t>(grid_values[0]),
                                      static_cast<std::uint32_t>(grid_values[1]),
                                      static_cast<std::uint32_t>(grid_values[2])};
        const qo::NpyArray pixels = qo::read_npy(qo::join(vision_dir, "small_patches.npy"));
        const std::vector<float> pixel_values = pixels.floats();
        std::vector<std::uint16_t> patch_bits(pixel_values.size());
        for (std::size_t i = 0; i < pixel_values.size(); ++i) { patch_bits[i] = to_bf16(pixel_values[i]); }
        const qo::NpyArray ids_array = qo::read_npy(qo::join(dir, "small_input_ids.npy"));
        const std::vector<int> small_ids = ids_array.ints();
        const qo::NpyArray want_embeds = qo::read_npy(qo::join(dir, "small_prompt_embeds_f32.npy"));
        const std::size_t small_drop = small_ids.size() - want_embeds.shape[0];

        Reader counting(artifact_path);
        const VisionTowerOutput counted_rows = run_vision_tower(counting, vision, patch_bits, grid);
        ConditionedVision counted;
        counted.merger_rows    = counted_rows.merger_rows;
        counted.deepstack_rows = counted_rows.deepstack_rows;
        counted.grids          = {std::array<int, 3>{static_cast<int>(grid.temporal),
                                                     static_cast<int>(grid.height), static_cast<int>(grid.width)}};

        g_image_device_alloc_calls      = 0;
        g_image_device_alloc_fail_after = 0;
        const ConditionedTextTowerOutput counted_out =
            run_conditioned_text_tower(counting, tower, embed_table, small_ids, small_drop, counted);
        (void)counted_out;
        require(image_device_bytes_live() == 0, "the counting run left image device bytes live");
        const std::uint64_t calls      = g_image_device_alloc_calls.load();
        const std::uint64_t fail_after = calls / 2ULL;
        require(calls >= 4, "the conditioned tower took too few image lane allocations to fail halfway");
        std::cout << "throw path: small-case peak " << mib(image_device_bytes_peak()) << " MiB, " << calls
                  << " image lane allocations, failing the one after " << fail_after << "\n";
        {
            Buf hog; // the run must not keep the test's own bytes either
            hog.alloc(64ULL * 1024 * 1024);
            Reader starved(artifact_path);
            bool refused = false;
            g_image_device_alloc_calls      = 0;
            g_image_device_alloc_fail_after = fail_after;
            try {
                const ConditionedTextTowerOutput starved_out =
                    run_conditioned_text_tower(starved, tower, embed_table, small_ids, small_drop, counted);
                (void)starved_out;
            } catch (const std::exception& error) {
                refused           = true;
                const std::string message = error.what();
                require(message.find("out of memory") != std::string::npos,
                        "the starved tower failed for a reason other than memory: " + message);
            }
            g_image_device_alloc_fail_after = 0;
            require(refused, "the conditioned tower completed despite a device allocation it could not get");
            require(image_device_bytes_live() == hog.bytes,
                    "the conditioned tower kept image device bytes live after it failed");
            std::cout << "refused an impossible budget and released every byte it had taken\n";
            hog.retire(0);
        }
        require(image_device_bytes_live() == 0, "the conditioned tower left bytes live after a refused run");
    } catch (const std::exception& error) {
        std::cout << "EDIT_PROMPT_FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "EDIT_PROMPT_OK\n";
    return 0;
}
