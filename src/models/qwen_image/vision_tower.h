#pragma once

// The Qwen3-VL vision tower for the EDIT route: one reference image's processor patches in, the
// merger rows the text tower scatters onto its <|image_pad|> slots out, plus the deepstack features
// the text tower's first decoder layers add back onto those rows.
//
// This is NOT models/qwen3_5/execution/vision_tower.*: that tower belongs to the LLM lane (a
// separate library), is driven by a VisionConfig parser that rejects the members this checkpoint's
// vision_config carries (out_hidden_size, deepstack_visual_indexes), and its pass has one output
// per item and one merger -- no tap after an intermediate block, which is exactly what deepstack
// needs. What IS shared is the op layer: the block math below is the same ops::layer_norm,
// ops::linear, ops::rope (Vision 2-D), ops::packed_softmax_attention, ops::gelu and
// ops::residual_add sequence the qualified LLM-lane vision path runs, written in this lane's
// streamed-upload idiom (image_device_util.h) so the tower fits inside the interlude's yielded
// VRAM: peak == one block's roles + the per-item activation + one merger.
//
// Like run_text_tower, everything runs on the legacy default stream with synchronous transfers, and
// all device memory is an image_device_util.h Buf, so image_device_bytes_live() accounts it and
// reaches 0 again on the throw path as well as the success path.

#include "models/qwen_image/image_weights.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::artifact { class Reader; }

namespace infernix::models::qwen_image {

// The processor's grid for one image: patch counts (NOT pixels), so a 1248x832 plane at patch 16 is
// {1, 52, 78}. temporal is 1 for an image.
struct VisionItemGeometry {
    std::uint32_t temporal = 1;
    std::uint32_t height   = 0;
    std::uint32_t width    = 0;
};

struct VisionTowerOutput {
    // The main merger's rows: [merged_count, out_hidden] token-major BF16 bits -- one row per
    // <|image_pad|> slot of the edit template.
    std::vector<std::uint16_t> merger_rows;
    // One entry per VisionDims::deepstack_indexes entry, in that order, same [merged_count,
    // out_hidden] layout: the vector decoder layer (index + 1) of the text tower adds onto the
    // image rows. Empty when the artifact declares no taps.
    std::vector<std::vector<std::uint16_t>> deepstack_rows;
};

// Runs the tower over one item. `patches` is the processor's pixel_values for that item as BF16
// bits in ITS flattening order -- merge-block major, then the merge's inner row and column -- which
// is what makes the merger's plain 4-patch view correct. rows == patches * patch_width.
// Bad geometry, a patch vector of the wrong length or a CUDA failure throw std::runtime_error.
[[nodiscard]] VisionTowerOutput run_vision_tower(artifact::Reader& reader, const VisionWeights& weights,
                                                 const std::vector<std::uint16_t>& patches,
                                                 VisionItemGeometry grid);

} // namespace infernix::models::qwen_image
