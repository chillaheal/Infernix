#include "models/qwen_image/vision_tower.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/softmax_attention.h"
#include "infernix/ops/vision_pos_embed.h"
#include "models/qwen_image/image_device_util.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen_image {
namespace {

using infernix::DType;
using infernix::Tensor;
namespace artifact = infernix::artifact;
namespace ops      = infernix::ops;

// Qwen3-VL's visual rotary base is not a config field (the checkpoint's vision_config has no
// theta), and models/qwen3_5/execution/vision_tower.cpp runs the same constant for the LLM lane.
constexpr float kVisionRopeTheta = 10'000.0F;
// Qwen3VLVisionPatchMerger and the vision blocks both build nn.LayerNorm(eps=1e-6).
constexpr float kLayerNormEps = 1.0e-6F;

// The rope ids and the bilinear position-table taps of one item, generated the way
// models/qwen3_5/program/vision_control.cpp generates them for the LLM lane (ported: this lane has
// no PreparedPromptData, and the edit route's grid comes from its own snapped plane). Storage
// contract, both of which the ops read as-is:
//   position_ids  I32 [patches, 2] with the AXIS as the slow dimension -- axis 0 is the patch row,
//                 axis 1 the column, which is what ops::rope's Vision 2-D form reads.
//   table_*       the four bilinear corners of one patch consecutive, i.e. Tensor [4, patches]
//                 with the corner as the fast dimension.
struct VisionGeometry {
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> table_indices;
    std::vector<float>        table_weights;
};

VisionGeometry vision_geometry(std::uint32_t temporal, std::uint32_t height, std::uint32_t width,
                              std::uint32_t merge, std::uint32_t position_side) {
    const auto t = static_cast<std::int32_t>(temporal), h = static_cast<std::int32_t>(height),
               w = static_cast<std::int32_t>(width), m = static_cast<std::int32_t>(merge),
               side = static_cast<std::int32_t>(position_side);
    const std::size_t patches = std::size_t(t) * std::size_t(h) * std::size_t(w);
    VisionGeometry out;
    out.position_ids.resize(patches * 2);
    out.table_indices.resize(patches * 4);
    out.table_weights.resize(patches * 4);
    // A grid of one patch maps to the table's coordinate 0 (an empty interpolation), as in the
    // LLM lane's coordinate().
    const auto coordinate = [&](std::int32_t index, std::int32_t size) {
        return size <= 1
                   ? 0.0F
                   : static_cast<float>(index) * static_cast<float>(side - 1) /
                         static_cast<float>(size - 1);
    };
    std::size_t cursor = 0;
    for (std::int32_t frame = 0; frame < t; ++frame) {
        for (std::int32_t block_y = 0; block_y < h / m; ++block_y) {
            for (std::int32_t block_x = 0; block_x < w / m; ++block_x) {
                for (std::int32_t inner_y = 0; inner_y < m; ++inner_y) {
                    for (std::int32_t inner_x = 0; inner_x < m; ++inner_x) {
                        const std::int32_t y = block_y * m + inner_y;
                        const std::int32_t x = block_x * m + inner_x;
                        out.position_ids[cursor]          = y;
                        out.position_ids[patches + cursor] = x;
                        const float yf = coordinate(y, h), xf = coordinate(x, w);
                        const auto y0 = static_cast<std::int32_t>(yf), x0 = static_cast<std::int32_t>(xf);
                        const std::int32_t y1 = std::min(y0 + 1, side - 1);
                        const std::int32_t x1 = std::min(x0 + 1, side - 1);
                        const float wy = yf - static_cast<float>(y0), wx = xf - static_cast<float>(x0);
                        out.table_indices[cursor * 4 + 0] = y0 * side + x0;
                        out.table_indices[cursor * 4 + 1] = y0 * side + x1;
                        out.table_indices[cursor * 4 + 2] = y1 * side + x0;
                        out.table_indices[cursor * 4 + 3] = y1 * side + x1;
                        out.table_weights[cursor * 4 + 0] = (1.0F - wy) * (1.0F - wx);
                        out.table_weights[cursor * 4 + 1] = (1.0F - wy) * wx;
                        out.table_weights[cursor * 4 + 2] = wy * (1.0F - wx);
                        out.table_weights[cursor * 4 + 3] = wy * wx;
                        ++cursor;
                    }
                }
            }
        }
    }
    return out;
}

// Boundary dump for the milestone gate. With INFERNIX_VISION_STAGE_DIR set, every boundary the torch
// reference dumps (image-edit/dump_vision_stages.py) is written there as raw BF16, named
// <INFERNIX_VISION_STAGE_TAG>_<name>.bin, in the reference's own [rows, cols] row-major order: the
// tower's column-major [features, patches] buffers read as that order unchanged, which is the same
// reason the production merger rows can be compared token-major. Unset, the sink is one branch per
// boundary and nothing is written.
class StageSink {
  public:
    explicit StageSink(std::string stage_tag)
        : dir(env_or_empty("INFERNIX_VISION_STAGE_DIR")), tag(std::move(stage_tag)) {}

    void write(const std::string& name, void* device, std::size_t elements) const {
        if (dir.empty() || device == nullptr) { return; }
        std::vector<std::uint16_t> host(elements);
        cuda_ok(cudaMemcpy(host.data(), device, elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                "stage dump readback");
        const std::string path = dir + "/" + tag + "_" + name + ".bin";
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) { throw std::runtime_error("stage dump: cannot write " + path); }
        const std::size_t written = std::fwrite(host.data(), sizeof(std::uint16_t), elements, file);
        std::fclose(file);
        if (written != elements) { throw std::runtime_error("stage dump: short write " + path); }
    }

    // tap08_, tap16_, tap24_ -- the reference names its tap files the same way.
    std::string tap_name(std::uint32_t layer) const {
        char suffix[8];
        std::snprintf(suffix, sizeof(suffix), "%02u", static_cast<unsigned>(layer));
        return std::string("tap") + suffix + "_";
    }

  private:
    static std::string env_or_empty(const char* name) {
        const char* value = std::getenv(name);
        return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string();
    }

    std::string dir;
    std::string tag;
};

} // namespace

VisionTowerOutput run_vision_tower(artifact::Reader& reader, const VisionWeights& weights,
                                  const std::vector<std::uint16_t>& patches, VisionItemGeometry grid) {
    ImageStreamScope legacy_scope(nullptr); // every consumer of these rows runs on the default stream
    const VisionDims& dims = weights.dims;
    if (dims.hidden == 0 || dims.intermediate == 0 || dims.heads == 0 || dims.head_dim == 0 ||
        dims.patch_width == 0 || dims.merger_width == 0 || dims.out_hidden == 0 || dims.depth == 0 ||
        dims.merge == 0 || dims.position_grid_side == 0 || weights.blocks.size() != dims.depth ||
        weights.deepstack.size() != dims.deepstack_indexes.size()) {
        throw std::runtime_error("vision tower: dims are unusable or its weights differ from the config");
    }
    if (grid.temporal == 0 || grid.height == 0 || grid.width == 0 || grid.height % dims.merge != 0 ||
        grid.width % dims.merge != 0) {
        throw std::runtime_error("vision tower: the grid must be positive and merge-aligned");
    }
    const std::size_t patches_count = std::size_t(grid.temporal) * grid.height * grid.width;
    // Every dimension below reaches an op as an int; the arithmetic guards here keep a hostile grid
    // from wrapping into a smaller (and silently wrong) allocation.
    if (patches_count > std::size_t(std::numeric_limits<int>::max()) / (dims.hidden + 8)) {
        throw std::runtime_error("vision tower: the grid addresses more than an int of elements");
    }
    const std::size_t merged_count  = patches_count / (std::size_t(dims.merge) * dims.merge);
    if (patches.size() != patches_count * dims.patch_width) {
        throw std::runtime_error("vision tower: the patch rows do not match the grid and patch width");
    }

    const int H  = static_cast<int>(dims.hidden);
    const int I  = static_cast<int>(dims.intermediate);
    const int HQ = static_cast<int>(dims.heads);
    const int D  = static_cast<int>(dims.head_dim);
    const int PW = static_cast<int>(dims.patch_width);
    const int MW = static_cast<int>(dims.merger_width);
    const int OUT = static_cast<int>(dims.out_hidden);
    const int Pi  = static_cast<int>(patches_count);
    const int Vi  = static_cast<int>(merged_count);
    // One attention segment per temporal frame: a single image is one segment, i.e. full (non-causal)
    // attention over all of the item's patches, which is what the tower's blocks do.
    const auto segment_length = static_cast<std::int32_t>(grid.height) * static_cast<int>(grid.width);

    const VisionGeometry geometry =
        vision_geometry(grid.temporal, grid.height, grid.width, dims.merge, dims.position_grid_side);
    const char* stage_tag = std::getenv("INFERNIX_VISION_STAGE_TAG");
    const StageSink stages(stage_tag != nullptr ? std::string(stage_tag) : std::string("case"));
    const std::size_t elements_P = patches_count * dims.hidden; // one [patches, hidden] plane
    const std::size_t elements_M = merged_count * dims.merger_width;

    // ---- every device byte of the tower lives in a Buf: RAII frees it before any return ----
    const std::size_t bytes_HP = patches_count * dims.hidden * 2;   // one [hidden, patches] plane
    const std::size_t bytes_IP = patches_count * dims.intermediate * 2;
    Buf d_patches, d_x, d_pos, d_idx, d_weights, d_ln, d_q, d_k, d_v, d_attended, d_projected, d_up,
        d_down, d_tap, d_normalized, d_hidden, d_out;
    d_patches.alloc(patches_count * dims.patch_width * 2);
    h2d(d_patches.d, patches.data(), patches_count * dims.patch_width * 2);
    d_x.alloc(bytes_HP);
    d_pos.alloc(patches_count * 2 * sizeof(std::int32_t));
    h2d(d_pos.d, geometry.position_ids.data(), patches_count * 2 * sizeof(std::int32_t));
    d_idx.alloc(patches_count * 4 * sizeof(std::int32_t));
    h2d(d_idx.d, geometry.table_indices.data(), patches_count * 4 * sizeof(std::int32_t));
    d_weights.alloc(patches_count * 4 * sizeof(float));
    h2d(d_weights.d, geometry.table_weights.data(), patches_count * 4 * sizeof(float));
    d_ln.alloc(bytes_HP);
    d_q.alloc(bytes_HP);
    d_k.alloc(bytes_HP);
    d_v.alloc(bytes_HP);
    d_attended.alloc(bytes_HP);
    d_projected.alloc(bytes_HP);
    d_up.alloc(bytes_IP);
    d_down.alloc(bytes_HP);
    d_tap.alloc(bytes_HP);
    d_normalized.alloc(bytes_HP); // == merged_count * merger_width * 2
    d_hidden.alloc(bytes_HP);
    d_out.alloc(merged_count * dims.out_hidden * 2);

    const Tensor posv(d_pos.d, DType::I32, {Pi, 2});
    const Tensor idxv(d_idx.d, DType::I32, {4, Pi});
    const Tensor wgtv(d_weights.d, DType::FP32, {4, Pi});
    DeviceContext device;
    const auto execution = device.execution_view().on_stream(nullptr);

    VisionTowerOutput out;
    out.deepstack_rows.resize(weights.deepstack.size());

    // ---- the prelude: patch projection + bias + bilinear position embedding ----
    {
        Uploaded w_embed = upload_ref(reader, weights.patch_embedding);
        Uploaded w_bias  = upload_ref(reader, weights.patch_embedding_bias);
        Uploaded w_table = upload_ref(reader, weights.position_embedding);
        Tensor xv    = view(d_x.d, H, Pi);
        Tensor xin   = view(d_patches.d, PW, Pi);
        ops::linear(xin, w_embed.w, xv, 0);
        ops::add_bias(view(w_bias.buf.d, H, 1), xv, 0);
        // The artifact stores the table as [rows, hidden] row-major, so a [hidden, rows] view is the
        // zero-copy reading of it (the same reasoning as the LLM lane's vision_embed).
        Tensor table = view(w_table.buf.d, H, static_cast<int>(dims.position_rows));
        ops::vision_pos_embed_add(table, idxv, wgtv, xv, 0);
        cuda_ok(cudaGetLastError(), "vision tower prelude launch");
        stages.write("prelude", d_x.d, elements_P);
    }

    // One patch merger on the residual in `source`. Qwen3VLVisionPatchMerger.forward normalizes the
    // MERGED (merger_width) row only when use_postshuffle_norm is set -- which is true of every
    // deepstack merger and false of the main one -- so the flag decides which shape the LayerNorm
    // sees; the merge itself is a pure view of the same bytes either way, because the processor
    // flattened the patches in merge-block order. fc1 stays merger-wide and fc2 lands on
    // out_hidden. Output: [out_hidden, merged] at d_out or the caller's deepstack slot.
    const auto run_merger = [&](void* source, const VisionMergerWeights& merger, bool postshuffle,
                                void* destination, const std::string& dump_prefix) {
        Uploaded n_weight = upload_ref(reader, merger.norm_weight);
        Uploaded n_bias   = upload_ref(reader, merger.norm_bias);
        Uploaded f1       = upload_ref(reader, merger.fc1);
        Uploaded f1_bias  = upload_ref(reader, merger.fc1_bias);
        Uploaded f2       = upload_ref(reader, merger.fc2);
        Uploaded f2_bias  = upload_ref(reader, merger.fc2_bias);
        Tensor normalized;
        if (postshuffle) {
            Tensor merged_view = view(source, MW, Vi);
            normalized = view(d_normalized.d, MW, Vi);
            ops::layer_norm(merged_view, view(n_weight.buf.d, MW, 1), view(n_bias.buf.d, MW, 1),
                            kLayerNormEps, normalized, 0);
        } else {
            Tensor patch_view = view(source, H, Pi);
            Tensor patch_norm = view(d_normalized.d, H, Pi);
            ops::layer_norm(patch_view, view(n_weight.buf.d, H, 1), view(n_bias.buf.d, H, 1),
                            kLayerNormEps, patch_norm, 0);
            normalized = view(d_normalized.d, MW, Vi); // the same bytes, now merge rows
        }
        stages.write(dump_prefix + "norm", d_normalized.d, elements_M);
        Tensor hidden = view(d_hidden.d, MW, Vi);
        ops::linear(normalized, f1.w, hidden, 0);
        ops::add_bias(view(f1_bias.buf.d, MW, 1), hidden, 0);
        stages.write(dump_prefix + "fc1", d_hidden.d, elements_M);
        ops::gelu(hidden, ops::GeluMode::Exact, 0);
        stages.write(dump_prefix + "act", d_hidden.d, elements_M);
        Tensor result = view(destination, OUT, Vi);
        ops::linear(hidden, f2.w, result, 0);
        ops::add_bias(view(f2_bias.buf.d, OUT, 1), result, 0);
        cuda_ok(cudaGetLastError(), "vision tower merger launch");
        stages.write(dump_prefix + "fc2", destination, merged_count * dims.out_hidden);
    };

    // ---- the blocks, streamed one at a time ----
    for (std::uint32_t layer = 0; layer < dims.depth; ++layer) {
        const VisionBlockWeights& block = weights.blocks[static_cast<std::size_t>(layer)];
        Uploaded n1  = upload_ref(reader, block.norm1_weight);
        Uploaded n1b = upload_ref(reader, block.norm1_bias);
        Uploaded n2  = upload_ref(reader, block.norm2_weight);
        Uploaded n2b = upload_ref(reader, block.norm2_bias);
        Uploaded wq  = upload_ref(reader, block.q);
        Uploaded wqb = upload_ref(reader, block.q_bias);
        Uploaded wk  = upload_ref(reader, block.k);
        Uploaded wkb = upload_ref(reader, block.k_bias);
        Uploaded wv  = upload_ref(reader, block.v);
        Uploaded wvb = upload_ref(reader, block.v_bias);
        Uploaded wo  = upload_ref(reader, block.attention_output);
        Uploaded wob = upload_ref(reader, block.attention_output_bias);
        Uploaded wf1 = upload_ref(reader, block.fc1);
        Uploaded wf1b = upload_ref(reader, block.fc1_bias);
        Uploaded wf2 = upload_ref(reader, block.fc2);
        Uploaded wf2b = upload_ref(reader, block.fc2_bias);

        Tensor x   = view(d_x.d, H, Pi);
        Tensor h   = view(d_ln.d, H, Pi);
        ops::layer_norm(x, view(n1.buf.d, H, 1), view(n1b.buf.d, H, 1), kLayerNormEps, h, 0);
        Tensor q = view(d_q.d, H, Pi), k = view(d_k.d, H, Pi), v = view(d_v.d, H, Pi);
        ops::linear(h, wq.w, q, 0);
        ops::linear(h, wk.w, k, 0);
        ops::linear(h, wv.w, v, 0);
        ops::add_bias(view(wqb.buf.d, H, 1), q, 0);
        ops::add_bias(view(wkb.buf.d, H, 1), k, 0);
        ops::add_bias(view(wvb.buf.d, H, 1), v, 0);
        Tensor qh = view(d_q.d, D, HQ, Pi), kh = view(d_k.d, D, HQ, Pi), vh = view(d_v.d, D, HQ, Pi);
        Tensor attended = view(d_attended.d, D, HQ, Pi);
        if (layer == 0) {
            // The same boundary one step earlier: with q_postrope at 6.7x its own BF16 floor while the
            // blocks behind it sit at 1.15x, only the pair of dumps says whether the projection or the
            // rotation is what drifts.
            stages.write("q_prerope", d_q.d, elements_P);
            stages.write("k_prerope", d_k.d, elements_P);
        }
        ops::rope(posv, D, kVisionRopeTheta, qh, kh, execution);
        if (layer == 0) {
            // The reference dumps block 0's Q/K right after the rotation: the boundary rope can get
            // wrong, and the measured floors say the chain is accurate to ~3e-3 this early.
            stages.write("q_postrope", d_q.d, elements_P);
            stages.write("k_postrope", d_k.d, elements_P);
        }
        ops::packed_softmax_attention(qh, kh, vh, {D, HQ, HQ},
                                      static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))),
                                      segment_length, attended, 0);
        Tensor attended_rows = view(d_attended.d, H, Pi);
        Tensor projected     = view(d_projected.d, H, Pi);
        ops::linear(attended_rows, wo.w, projected, 0);
        ops::add_bias(view(wob.buf.d, H, 1), projected, 0);
        ops::residual_add(projected, x, 0);

        // d_ln holds the attention block's now-dead LayerNorm output, so the MLP normalizes into it
        // again -- the same reuse the LLM lane's vision_layer makes.
        Tensor h2 = view(d_ln.d, H, Pi);
        ops::layer_norm(x, view(n2.buf.d, H, 1), view(n2b.buf.d, H, 1), kLayerNormEps, h2, 0);
        Tensor up = view(d_up.d, I, Pi);
        ops::linear(h2, wf1.w, up, 0);
        ops::add_bias(view(wf1b.buf.d, I, 1), up, 0);
        ops::gelu(up, ops::GeluMode::Tanh, 0);
        Tensor down = view(d_down.d, H, Pi);
        ops::linear(up, wf2.w, down, 0);
        ops::add_bias(view(wf2b.buf.d, H, 1), down, 0);
        ops::residual_add(down, x, 0);
        cuda_ok(cudaGetLastError(), "vision tower block launch");
        {
            char name[10];
            std::snprintf(name, sizeof(name), "block%02u", static_cast<unsigned>(layer));
            stages.write(name, d_x.d, elements_P);
        }

        // The deepstack tap: Qwen3VLVisionModel.forward pushes this block's output through the
        // merger of the SAME position in deepstack_visual_indexes when the block index is one of
        // them -- after the block, not before it. The residual keeps going untouched, so the tapped
        // plane is copied first.
        const auto tap = std::find(dims.deepstack_indexes.begin(), dims.deepstack_indexes.end(), layer);
        if (tap != dims.deepstack_indexes.end()) {
            const std::size_t which = static_cast<std::size_t>(
                std::distance(dims.deepstack_indexes.begin(), tap));
            cuda_ok(cudaMemcpyAsync(d_tap.d, d_x.d, bytes_HP, cudaMemcpyDeviceToDevice, 0),
                    "D2D (vision tower deepstack tap)");
            run_merger(d_tap.d, weights.deepstack[which], /*postshuffle=*/true, d_out.d,
                       stages.tap_name(layer));
            auto& rows = out.deepstack_rows[which];
            rows.resize(merged_count * dims.out_hidden);
            cuda_ok(cudaMemcpy(rows.data(), d_out.d, rows.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "D2H (vision tower deepstack rows)");
        }
    }

    // The main merger sees the last block's residual (deepstack never modifies it).
    run_merger(d_x.d, weights.merger, /*postshuffle=*/false, d_out.d, "merger_");
    out.merger_rows.resize(merged_count * dims.out_hidden);
    cuda_ok(cudaMemcpy(out.merger_rows.data(), d_out.d, out.merger_rows.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost),
            "D2H (vision tower merger rows)");
    return out;
}

} // namespace infernix::models::qwen_image
