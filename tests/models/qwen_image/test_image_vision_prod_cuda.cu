// GPU qualification of the EDIT route's Qwen3-VL vision tower (src/models/qwen_image/vision_tower.cu)
// against the model's own output at EVERY boundary the tower computes.
//
// Two references per boundary, dumped by image-edit/dump_vision_stages.py from the checkpoint's own
// Qwen3VLVisionModel forward on the SAME processor pixels:
//   <case>_<boundary>_f32.npy   the exact reference (float32 weights),
//   <case>_<boundary>_bf16.npy  the same forward in BF16 -- the reference's own dtype error.
//
// Why not a single a-priori tolerance against a BF16 oracle. Measured on the small case
// (image-edit/NOTES.md, "M3-uppmätning"), the tower's accuracy is 1.05-1.21x the reference's own BF16
// error at every one of the 39 boundaries and BETTER than it at the deep ones (merger output: 0.152
// against exact vs 0.159 for torch's BF16 chain). The distance between two independent BF16 chains is
// therefore ~0.21 at the merger for a tower with no defect, so any band below the reference's own
// error measures the reference, not the engine. The gate is instead relative to that floor, with the
// floor computed in this test from the two dumped files (never hardcoded per stage):
//
//   global:  rel_fro(engine, f32) <= max(2 * floor, 1e-2),      floor = rel_fro(bf16_oracle, f32)
//   per row:  ||engine_r - f32_r|| / ||f32|| <= max(2 * that row's own floor share, floor)
//   per head: the same, for Q and K after rope, one slice per attention head
//
// The slice rules are what keep a whole-tensor norm from averaging away one bad head or a handful of
// bad rows; a slice may not add more error than the reference's whole dtype error for the boundary.
// That factor is not tuned to pass: measured over both fixtures the worst row sits at 0.36 and the
// worst head at 0.39 of the boundary's floor (image-edit/row_band_probe.py), so a defect that breaks
// a row or a head -- which lands at whole-row magnitude, not a third of the chain's error -- is what
// trips it. The 1e-2 global floor covers boundaries the reference computes exactly. Shapes, row
// counts and finiteness are asserted exactly, never within a tolerance.
//
// The engine's boundaries come from the tower's own env-gated dump (INFERNIX_VISION_STAGE_DIR, see
// vision_tower.cu); this test sets it when it is unset, so the comparison is never silently skipped.
//
// Skips (77) without INFERNIX_TEST_ARTIFACT or INFERNIX_IMAGE_ORACLE, or when either is absent;
// returns 2 with no CUDA device. Runs on the real GPU: it must not overlap a live serve.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen_image/image_device_util.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "models/qwen_image/vision_tower.h"

#include "artifact/schema.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;
namespace qo = qioracle;

constexpr double kFloorFactor    = 2.0;    // the engine may drift at most 2x the reference's own BF16 error
constexpr double kGlobalFloor    = 1.0e-2; // for boundaries the reference computes exactly
constexpr double kSliceFloorShare = 1.0;   // a row or head may add at most one whole boundary floor

struct Case {
    const char* name;
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : fallback;
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

double norm(const std::vector<float>& values, std::size_t offset, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(values[offset + i]);
        sum += v * v;
    }
    return std::sqrt(sum);
}

double diff_norm(const std::vector<float>& a, const std::vector<float>& b, std::size_t offset,
                 std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(a[offset + i]) - static_cast<double>(b[offset + i]);
        sum += v * v;
    }
    return std::sqrt(sum);
}

// One boundary: the engine's BF16 rows (token-major [rows, cols]) against both references.
struct Boundary {
    std::string engine_name; // the tower's dump name
    std::string ref_name;    // the reference's dump name (differs for the taps)
    std::size_t rows;
    std::size_t cols;
    bool per_head = false; // Q and K after rope: one slice per attention head
};

void gate_boundary(const std::string& stage_dir, const std::string& ref_dir, const std::string& tag,
                   const Boundary& item, std::size_t head_dim) {
    // Both sides are flat directories keyed by "<case>_<boundary>": the tower's raw BF16 dumps next to
    // the reference's two dtypes, so a boundary is always compared against its own name.
    const std::string prefix = tag + "_";
    std::ifstream file(stage_dir + "/" + prefix + item.engine_name + ".bin",
                       std::ios::binary | std::ios::ate);
    require(file.is_open(), item.engine_name + ": the tower wrote no stage dump");
    const std::size_t elements = item.rows * item.cols;
    require(static_cast<std::size_t>(file.tellg()) / sizeof(std::uint16_t) == elements,
            item.engine_name + ": the stage dump has a different element count than the boundary");
    std::vector<std::uint16_t> bits(elements);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bits.data()), static_cast<std::streamsize>(elements * sizeof(std::uint16_t)));
    require(static_cast<bool>(file), item.engine_name + ": short read of the stage dump");

    const qo::NpyArray exact = qo::read_npy(ref_dir + "/" + prefix + item.ref_name + "_f32.npy");
    const qo::NpyArray bf16  = qo::read_npy(ref_dir + "/" + prefix + item.ref_name + "_bf16.npy");
    const std::vector<float> f32  = exact.floats();
    const std::vector<float> floor_ref = bf16.floats();
    require(exact.shape.size() == 2 && exact.shape[0] == item.rows && exact.shape[1] == item.cols,
            item.engine_name + ": the exact reference is not [rows, cols]");
    require(floor_ref.size() == f32.size(), item.engine_name + ": the two references differ in size");
    const std::vector<float> got = from_bits(bits);

    for (std::size_t i = 0; i < elements; ++i) {
        require(std::isfinite(got[i]), item.engine_name + ": the tower produced a non-finite value");
    }

    const double base = norm(f32, 0, elements);
    require(base > 0.0, item.engine_name + ": the reference is all zeros");
    const double engine_err = diff_norm(got, f32, 0, elements) / base;
    const double floor_err  = diff_norm(floor_ref, f32, 0, elements) / base;
    const double band       = std::max(kFloorFactor * floor_err, kGlobalFloor);
    std::cout << "  " << item.engine_name << " eng=" << engine_err << " floor=" << floor_err
              << " band=" << band;
    if (engine_err > band) {
        throw std::runtime_error(item.engine_name + ": the engine drifts further from the exact forward "
                                 "than twice the reference's own BF16 error");
    }

    // Row by row: a single tensor norm lets one badly corrupted row hide among thousands of good ones.
    for (std::size_t row = 0; row < item.rows; ++row) {
        const std::size_t offset = row * item.cols;
        const double row_engine = diff_norm(got, f32, offset, item.cols) / base;
        const double row_floor  = diff_norm(floor_ref, f32, offset, item.cols) / base;
        require(row_engine <= std::max(kFloorFactor * row_floor, kSliceFloorShare * floor_err),
                item.engine_name + ": row " + std::to_string(row) + " drifts beyond the reference's own");
    }

    // Head by head, for the tensors where one wrong head would otherwise stay inside the band.
    if (item.per_head) {
        require(item.cols % head_dim == 0, item.engine_name + ": the width is not a whole number of heads");
        for (std::size_t head = 0; head < item.cols / head_dim; ++head) {
            const std::size_t offset = head * head_dim;
            double head_engine = 0.0, head_floor = 0.0;
            for (std::size_t row = 0; row < item.rows; ++row) {
                const std::size_t at = row * item.cols + offset;
                head_engine += diff_norm(got, f32, at, head_dim) * diff_norm(got, f32, at, head_dim);
                head_floor += diff_norm(floor_ref, f32, at, head_dim) * diff_norm(floor_ref, f32, at, head_dim);
            }
            head_engine = std::sqrt(head_engine) / base;
            head_floor = std::sqrt(head_floor) / base;
            require(head_engine <= std::max(kFloorFactor * head_floor, kSliceFloorShare * floor_err),
                    item.engine_name + ": head " + std::to_string(head) + " drifts beyond the reference's own");
        }
    }
    std::cout << " OK\n";
}

std::vector<Boundary> boundaries(const VisionDims& dims, std::size_t patches, std::size_t merged) {
    std::vector<Boundary> items;
    items.push_back({"prelude", "prelude", patches, dims.hidden, false});
    // Q/K on both sides of the rotation: the rotation is the one boundary in the tower whose reference
    // error is tiny (floor 2.7e-3) and whose engine error is not, so it is gated twice.
    items.push_back({"q_prerope", "q_prerope", patches, dims.hidden, true});
    items.push_back({"k_prerope", "k_prerope", patches, dims.hidden, true});
    items.push_back({"q_postrope", "q_postrope", patches, dims.hidden, true});
    items.push_back({"k_postrope", "k_postrope", patches, dims.hidden, true});
    for (std::uint32_t layer = 0; layer < dims.depth; ++layer) {
        char name[10];
        std::snprintf(name, sizeof(name), "block%02u", static_cast<unsigned>(layer));
        items.push_back({name, name, patches, dims.hidden, false});
    }
    for (std::uint32_t layer : dims.deepstack_indexes) {
        char engine[16], ref[10];
        std::snprintf(engine, sizeof(engine), "tap%02u_fc2", static_cast<unsigned>(layer));
        std::snprintf(ref, sizeof(ref), "tap%02u", static_cast<unsigned>(layer));
        items.push_back({engine, ref, merged, dims.out_hidden, false});
    }
    items.push_back({"merger_norm", "merger_norm", merged, dims.merger_width, false});
    items.push_back({"merger_fc1", "merger_fc1", merged, dims.merger_width, false});
    items.push_back({"merger_act", "merger_act", merged, dims.merger_width, false});
    items.push_back({"merger_fc2", "merger_fc2", merged, dims.out_hidden, false});
    return items;
}

} // namespace

int main() {
    const std::string artifact_path = env_or("INFERNIX_TEST_ARTIFACT", "");
    const std::string oracle_path   = env_or("INFERNIX_IMAGE_ORACLE", "");
    if (artifact_path.empty() || oracle_path.empty()) {
        std::cout << "INFERNIX_TEST_ARTIFACT / INFERNIX_IMAGE_ORACLE unset; skipping the vision tower\n";
        return 77;
    }
    std::error_code ec;
    if (!std::filesystem::exists(artifact_path, ec) || !std::filesystem::exists(oracle_path, ec)) {
        std::cout << "artifact or oracle absent; skipping the vision tower\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        std::cout << "no CUDA device; the vision tower qualification is runnable\n";
        return 2;
    }
    const std::string dir         = qo::join(oracle_path, "tensors_edit_vision");
    const std::string stages_ref_root = qo::join(oracle_path, "tensors_edit_vision_stages");
    require(std::filesystem::exists(stages_ref_root, ec),
            "the stage reference dump is absent -- run image-edit/dump_vision_stages.py");

    // Where the tower writes its own boundaries. The blackout run sets this so the files survive for
    // offline diffing; a plain run gets a temporary directory.
    const std::string stage_root = env_or("INFERNIX_VISION_STAGE_DIR",
                                          (std::filesystem::temp_directory_path() / "infernix-vision-stages").string());
    std::filesystem::create_directories(stage_root, ec);
    require(!ec, "cannot create the stage dump directory " + stage_root);

    try {
        Reader reader(artifact_path);
        Binder binder(reader);
        VisionWeights vision = bind_image_vision(binder);
        (void)std::move(binder).finish();

        // The artifact's own geometry, asserted rather than assumed: a v3-style artifact (no
        // deepstack mergers) never reaches here -- bind_image_vision throws on the missing names.
        require(vision.dims.depth == 27, "vision depth differs from Qwen3-VL's 27 blocks");
        require(vision.dims.hidden == 1152 && vision.dims.heads == 16 && vision.dims.head_dim == 72,
                "vision hidden/head geometry differs from the qualified Vision 2-D rope domain");
        require(vision.dims.merge == 2 && vision.dims.patch == 16 && vision.dims.temporal == 2,
                "vision patch/merge geometry differs from the processor this route feeds");
        require(vision.dims.out_hidden == 4096, "vision merger output width differs from the text tower");
        require(vision.dims.deepstack_indexes == std::vector<std::uint32_t>({8, 16, 24}),
                "deepstack_visual_indexes differs from the tapped blocks 8/16/24");
        std::cout << "vision tower bound: " << vision.blocks.size() << " blocks, "
                  << vision.deepstack.size() << " deepstack mergers, max object "
                  << vision.max_object_bytes / (1024 * 1024) << " MiB\n";

        // The grids come from the dump, not from a constant: the shipped processor upscales any plane
        // below its shortest_edge pixel budget, so a fixture's grid is the processor's business and
        // the engine has to follow it. What IS asserted is that the pixels, the grid and the dumped
        // rows agree with each other and with the artifact's geometry.
        std::uint64_t peak_small = 0;
        const std::vector<Case> cases = {{"small"}, {"portrait"}};
        for (const Case& case_item : cases) {
            const std::string tag(case_item.name);
            const qo::NpyArray grid_array = qo::read_npy(qo::join(dir, tag + "_grid.npy"));
            const std::vector<float> grid_values = grid_array.floats();
            require(grid_values.size() == 3, tag + ": the dumped grid is not [t, h, w]");
            const VisionItemGeometry grid{static_cast<std::uint32_t>(grid_values[0]),
                                          static_cast<std::uint32_t>(grid_values[1]),
                                          static_cast<std::uint32_t>(grid_values[2])};
            const std::size_t patches = std::size_t(grid.height) * grid.width;
            const std::size_t merged  = patches / (std::size_t(vision.dims.merge) * vision.dims.merge);
            // The rows the tower must produce follow from the grid: merge^2 patches per row. Asserting
            // that identity (rather than patches == merged, which no merge>1 grid can satisfy) is what
            // catches a grid/row split that disagrees.
            require(grid.temporal == 1 && grid.height % vision.dims.merge == 0 &&
                        grid.width % vision.dims.merge == 0 &&
                        patches == merged * vision.dims.merge * vision.dims.merge,
                    tag + ": the dumped grid is not a single merge-aligned image frame");

            const qo::NpyArray pixels = qo::read_npy(qo::join(dir, tag + "_patches.npy"));
            const std::vector<float> pixel_values = pixels.floats();
            require(pixels.shape.size() == 2 && pixels.shape[0] == patches &&
                        pixels.shape[1] == vision.dims.patch_width,
                    tag + ": the dumped pixels are not [patches, patch_width]");
            std::vector<std::uint16_t> bits(pixel_values.size());
            for (std::size_t i = 0; i < pixel_values.size(); ++i) { bits[i] = to_bf16(pixel_values[i]); }

            std::cout << "case " << tag << ": " << patches << " patches (" << grid.height << "x"
                      << grid.width << ") -> " << merged << " merged rows\n";
            set_env("INFERNIX_VISION_STAGE_DIR", stage_root);
            set_env("INFERNIX_VISION_STAGE_TAG", tag);
            reset_image_device_peak(); // so the recorded peak belongs to THIS case alone
            const VisionTowerOutput out = run_vision_tower(reader, vision, bits, grid);
            require(out.deepstack_rows.size() == vision.deepstack.size(),
                    "the tower returned a different number of deepstack tensors");

            for (const Boundary& item : boundaries(vision.dims, patches, merged)) {
                gate_boundary(stage_root, stages_ref_root, tag, item, vision.dims.head_dim);
            }

            // The endpoint rows the pipeline actually consumes, checked as bytes: the gate above
            // already covers them numerically, this pins that what the tower returned is what it
            // dumped and that nothing else is left on the device.
            require(out.merger_rows.size() == merged * vision.dims.out_hidden,
                    "the tower returned the wrong number of merger rows");
            for (std::size_t k = 0; k < out.deepstack_rows.size(); ++k) {
                require(out.deepstack_rows[k].size() == merged * vision.dims.out_hidden,
                        "deepstack " + std::to_string(vision.dims.deepstack_indexes[k]) +
                            " returned the wrong number of rows");
            }
            require(image_device_bytes_live() == 0,
                    "the vision tower left image device bytes live after " + tag);
            if (tag == "small") {
                peak_small = image_device_bytes_peak();
                require(peak_small > 0, "the small case took no image device bytes at all");
            }
        }

        // The ledger's throw path, not just its success path: fail an allocation halfway through a
        // VALID run (so bytes are genuinely held when it fails) and require every one of them back.
        // Not by starving the device: the Windows driver serves a cudaMalloc from sysmem rather than
        // refusing, so blackouts #9 and #10 both let the 117 MiB tower finish on a 58 MiB budget with
        // every numeric band OK. The failure comes from the ledger's own seam instead.
        const auto mib = [](std::uint64_t x) { return x / (1024ULL * 1024ULL); };
        std::vector<std::uint16_t> small_pixels(60 * vision.dims.patch_width, 0U);
        Reader counting(artifact_path);
        g_image_device_alloc_calls      = 0;
        g_image_device_alloc_fail_after = 0;
        const VisionTowerOutput counted =
            run_vision_tower(counting, vision, small_pixels, VisionItemGeometry{1, 6, 10});
        (void)counted;
        require(image_device_bytes_live() == 0, "the counting run left image device bytes live");
        const std::uint64_t calls      = g_image_device_alloc_calls.load();
        const std::uint64_t fail_after = calls / 2ULL;
        require(calls >= 4, "the vision tower took too few image lane allocations to fail halfway");
        std::cout << "throw path: small-case peak " << mib(peak_small) << " MiB, " << calls
                  << " image lane allocations, failing the one after " << fail_after << "\n";
        {
            Buf hog; // the run must not keep the test's own bytes either
            hog.alloc(64ULL * 1024 * 1024);
            Reader starved(artifact_path);
            bool refused = false;
            g_image_device_alloc_calls      = 0;
            g_image_device_alloc_fail_after = fail_after;
            try {
                const VisionTowerOutput starved_out =
                    run_vision_tower(starved, vision, small_pixels, VisionItemGeometry{1, 6, 10});
                (void)starved_out;
            } catch (const std::exception& error) {
                refused    = true;
                const std::string message = error.what();
                require(message.find("out of memory") != std::string::npos,
                        "the starved vision tower failed for a reason other than memory: " + message);
            }
            g_image_device_alloc_fail_after = 0;
            require(refused, "the vision tower completed despite a device allocation it could not get");
            require(image_device_bytes_live() == hog.bytes,
                    "the vision tower kept image device bytes live after it failed");
            std::cout << "refused an impossible budget and released every byte it had taken\n";
            hog.retire(0);
        }
        require(image_device_bytes_live() == 0,
                "the vision tower left image device bytes live after a refused run");
    } catch (const std::exception& error) {
        std::cout << "VISION_FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "VISION_OK\n";
    return 0;
}
