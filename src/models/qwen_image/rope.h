#pragma once

// Qwen-Image-2.1 DiT joint-sequence rotary positions for one text + target-image block
// (docs/maintainer/qwen-image-2-1-variables.md, gap #4). Host-only, device-free: the CUDA execution
// uploads the returned table and feeds it to ops::rope_c64 per attention call.
//
// This reproduces QwenImage21Rope.forward for the text-only case (image == None): a joint sequence of
// `text_prefix` text tokens followed by an H-by-W grid of packed image tokens, S = text_prefix + H*W.
// Positions are built on three axes (frame / height / width, widths {16,56,56}, theta 10000):
//   - text tokens: all three axes equal the token's frame index (0 .. text_prefix-1);
//   - image tokens: frame is the constant text_prefix, height/width are the row-major grid centered
//     at -(H - H/2) / -(W - W/2).
// Each axis emits cos/sin pairs; the row is the 32 (16+8+8 ... -> 16/2 + 56/2 + 56/2 = 64) angles
// stored interleaved as (re, im), so the row is [64*2] and the table is [S, 64*2]. This is the exact
// layout the fixed-geometry oracle dump uses (tensors_full_depth/block_rotary.npy at (8,8,8),
// tensors_pipeline/rot_inter.npy at (11,16,16)), so the same table generalizes across every
// (text_prefix, H, W) instead of being read from a per-geometry .npy.
//
// #75: condition (reference) images are covered by build_joint_rope_blocks below -- the exact
// multi-block position loop of QwenImage21Rope.forward for the serving mask layout
// [text..., cond_1 tokens, ..., cond_N tokens, target tokens]: the text run keeps its frame advance,
// every image block freezes the frame at the current position and advances it by max(h, w) after
// itself (so block k's frame = text_prefix + sum_{i<k} max(h_i, w_i)), and each block's height/
// width grid is centered on zero independently of its position in the sequence.

#include <cstddef>
#include <utility>
#include <vector>

namespace infernix::models::qwen_image {

// Build the joint-sequence rotary table for a text-prefix + single H-by-W image grid.
// Returns S*64*2 floats, row-major [S, 64, 2] with (cos, sin) interleaved, S = text_prefix + H*W.
[[nodiscard]] std::vector<float> build_joint_rope(int text_prefix, int H, int W);

// The multi-block generalization: text + N condition-image grids (each (h_i, w_i) in latent tokens,
// the vae_height/16 x vae_width/16 grids the pipeline passes as img_shapes) + the H-by-W target
// grid. S = text_prefix + sum_i h_i*w_i + H*W.
[[nodiscard]] std::vector<float> build_joint_rope_blocks(
    int text_prefix, const std::vector<std::pair<int, int>>& cond_grids, int H, int W);

} // namespace infernix::models::qwen_image
