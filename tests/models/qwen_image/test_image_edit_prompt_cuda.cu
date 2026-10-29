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
// measure the reference, not the engine. Floor = rel_fro(bf16_oracle, f32), computed here from the two
// dumped files:
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
void gate_class(const std::string& tag, const std::string& what, const std::vector<float>& engine,
                const std::vector<float>& f32, const std::vector<float>& floor_ref, std::size_t D,
                const std::vector<std::uint8_t>& selected, double base) {
    double engine_sq = 0.0, floor_sq = 0.0;
    std::size_t rows = 0;
    for (std::size_t row = 0; row < selected.size(); ++row) {
        if (!selected[row]) continue;
        ++rows;
        engine_sq += diff_sq(engine, f32, row * D, D);
        floor_sq += diff_sq(floor_ref, f32, row * D, D);
    }
    require(rows > 0, tag + ": the " + what + " row class is empty");
    const double engine_err = std::sqrt(engine_sq) / base;
    const double floor_err  = std::sqrt(floor_sq) / base;
    const double band       = std::max(kFloorFactor * floor_err, kGlobalFloor);
    std::cout << "  " << tag << " " << what << " rows=" << rows << " eng=" << engine_err
              << " floor=" << floor_err << " band=" << band << "\n";
    require(engine_err <= band, tag + ": the " + what + " rows drift further than twice the reference's own");
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

            // ---- the conditioned forward ----
            ConditionedVision conditioned;
            conditioned.merger_rows    = rows.merger_rows;
            conditioned.deepstack_rows = rows.deepstack_rows;
            require(conditioned.deepstack_rows.size() == vision.deepstack.size(),
                    "the tower returned a different number of deepstack tensors");
            for (const auto& add : conditioned.deepstack_rows) {
                require(add.size() == merged * D, "a deepstack tensor is not [pads, hidden]");
            }
            reset_image_device_peak();
            const ConditionedTextTowerOutput out =
                run_conditioned_text_tower(reader, tower, embed_table, ids, drop_rows, conditioned);
            require(out.prompt_embeds.size() == kept * D, "the conditioned run returned the wrong row count");
            require(out.image_pad_mask.size() == kept, "the conditioned run returned the wrong mask length");
            for (std::size_t s = 0; s < kept; ++s) {
                require(out.image_pad_mask[s] == want_mask[s],
                        "the engine's mask differs from the model's own at kept row " + std::to_string(s));
            }
            require(image_device_bytes_live() == 0, "the conditioned tower left image device bytes live");

            // ---- the numeric gate ----
            const std::vector<float> engine = from_bits(out.prompt_embeds);
            for (std::size_t i = 0; i < engine.size(); ++i) {
                require(std::isfinite(engine[i]), "the conditioned tower produced a non-finite value");
            }
            const double base        = std::sqrt(norm_sq(f32, 0, f32.size()));
            const double global_floor = std::sqrt(diff_sq(bf16, f32, 0, f32.size())) / base;
            require(base > 0.0, "the reference conditioning is all zeros");
            {
                const double engine_err = std::sqrt(diff_sq(engine, f32, 0, f32.size())) / base;
                const double band       = std::max(kFloorFactor * global_floor, kGlobalFloor);
                std::cout << "  " << tag << " prompt_embeds eng=" << engine_err << " floor=" << global_floor
                          << " band=" << band << "\n";
                require(engine_err <= band,
                        tag + ": the conditioned tower drifts further from the exact forward than twice "
                              "the reference's own BF16 error");
            }
            std::vector<std::uint8_t> text_rows(kept), image_rows(kept);
            for (std::size_t s = 0; s < kept; ++s) {
                text_rows[s]  = want_mask[s] ? 0U : 1U;
                image_rows[s] = want_mask[s];
                const double row_engine = std::sqrt(diff_sq(engine, f32, s * D, D)) / base;
                const double row_floor  = std::sqrt(diff_sq(bf16, f32, s * D, D)) / base;
                require(row_engine <= std::max(kFloorFactor * row_floor, kSliceFloorShare * global_floor),
                        tag + ": kept row " + std::to_string(s) + " drifts beyond the reference's own");
            }
            gate_class(tag, "text", engine, f32, bf16, D, text_rows, base);
            gate_class(tag, "image", engine, f32, bf16, D, image_rows, base);

            // ---- the deepstack add is on the executed path ----
            // A wrong layer index, or an add that silently never runs, only shows up as a delta against
            // the stock run: the band is the reference's own dtype error and a whole BF16 add is smaller
            // than it. Same ids, same merger rows, no deepstack rows -> must NOT be the stock result.
            ConditionedVision ablated;
            ablated.merger_rows = rows.merger_rows;
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

            // ---- the certified t2i route's regression ----
            // Scatter the pad tokens' OWN embedding rows and add nothing: the conditioned entry must
            // reduce bit-exactly to run_text_tower, which is the route /v1/images/generations ships on.
            std::vector<std::uint16_t> identity(pads * D);
            const auto* table = reinterpret_cast<const std::uint16_t*>(embed_table.data());
            for (std::size_t s = 0, r = 0; s < S; ++s) {
                if (!is_image[s]) continue;
                std::memcpy(identity.data() + r * D, table + static_cast<std::size_t>(ids[s]) * D,
                            sizeof(std::uint16_t) * D);
                ++r;
            }
            ConditionedVision plain;
            plain.merger_rows = identity;
            const ConditionedTextTowerOutput via_conditioned =
                run_conditioned_text_tower(reader, tower, embed_table, ids, drop_rows, plain);
            const std::vector<std::uint16_t> via_tower = run_text_tower(reader, tower, embed_table, ids, drop_rows);
            require(via_conditioned.prompt_embeds == via_tower,
                    tag + ": the conditioned entry is not bit-identical to run_text_tower on a plain prompt");
            require(via_conditioned.image_pad_mask == want_mask,
                    tag + ": the plain conditioned run disagrees with the model's own mask");
            require(image_device_bytes_live() == 0, "the plain runs left image device bytes live");
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
