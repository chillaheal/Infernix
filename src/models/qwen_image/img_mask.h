#pragma once

// Qwen-Image-2.1 DiT joint-sequence img_mask for one encode batch
// (docs/maintainer/qwen-image-2-1-variables.md, gap #4). Host-only, device-free.
//
// The DiT packs the joint sequence as [text tokens][condition-image slots][target-image slots]; the
// `img_mask` marks which joint positions are image (1) vs text (0) for the joint attention/packing. The
// reference builds it as a row-concat (image-spike/oracle/dump_pipeline.py:124):
//     img_mask = cat(image_pad_mask_base, ones(batch, target_slots), dim=1)
// where image_pad_mask_base is the text encoder's image_pad mask -- all-zero for a text-only prompt
// (width == text_len), and carrying the N condition images' image_pad slots (1s) when images are given --
// and target_slots = (Hl*Wl)//4 is the 2x2-PACKED target image-token count (64 at 256x256, 256 at
// 512x512). So the SAME rule generalizes from text-only to N condition images: the base just widens with
// the condition image_pad slots; the target slots are always appended as ones. Length = base_len +
// target_slots. This replaces the fixed img_mask.npy the in-process interlude reads today.
//
// Note the PACKED target grid (base + target_slots == joint length, 75 at 256x256) is the img_mask /
// hidden-packing view; the rope table (build_joint_rope) indexes the same sequence on its three position
// axes -- the two are the same joint sequence from two sides, not two lengths.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::models::qwen_image {

// Append `target_slots` ones to each item's image_pad_mask_base row. image_pad_mask_base is
// [batch, base_len] row-major; the result is [batch, base_len + target_slots] row-major (uint8). A
// negative base_len or target_slots is treated as 0. The base is copied verbatim (never reinterpreted).
[[nodiscard]] std::vector<std::uint8_t> build_img_mask(const std::uint8_t* image_pad_mask_base,
                                                       int batch, int base_len, int target_slots);

} // namespace infernix::models::qwen_image
