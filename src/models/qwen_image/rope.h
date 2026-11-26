#pragma once

// Qwen-Image-2.1 DiT joint-sequence rotary positions for one text + target-image block
// (docs/maintainer/qwen-image-2-1-variables.md, gap #4). Host-only, device-free: the CUDA execution
// uploads the returned table and feeds it to ops::rope_c64 per attention call.
//
// This reproduces QwenImage21Rope.forward for the text-only case (image == None): a joint sequence of
// `text_prefix` text tokens followed by an H-by-W grid of packed image tokens, S = text_prefix + H*W.
// Positions are built on three axes (frame / height / width) whose channel counts the ARTIFACT declares
// -- `const DitRopeAxes&`, read from the artifact's "dit" component by read_dit_rope_axes in
// artifact_constants.{h,cpp}; nothing here is pinned at the shipped {16, 56, 56}:
//   - text tokens: all three axes equal the token's frame index (0 .. text_prefix-1);
//   - image tokens: frame is the constant text_prefix, height/width are the row-major grid centered
//     at -(H - H/2) / -(W - W/2).
// Each axis of `a` channels emits a/2 angles -- angles the caller's theta rotates (kQwenImageRopeTheta
// below), stored interleaved as (re, im), so a row is rope_row_floats(axes) = axes[0] + axes[1] +
// axes[2] floats and the table is [S, rope_row_floats(axes)]. For the shipped {16, 56, 56} that is
// 16/2 + 56/2 + 56/2 = 64 angles in 128 floats, the exact layout the fixed-geometry oracle dump uses
// (tensors_full_depth/block_rotary.npy at (8,8,8), tensors_pipeline/rot_inter.npy at (11,16,16)), so the
// same table generalizes across every (text_prefix, H, W) instead of being read from a per-geometry
// .npy -- and across an architecture that declares other axes.
//
// #75: condition (reference) images are covered by build_joint_rope_blocks below -- the exact
// multi-block position loop of QwenImage21Rope.forward for the serving mask layout
// [text..., cond_1 tokens, ..., cond_N tokens, target tokens]: the text run keeps its frame advance,
// every image block freezes the frame at the current position and advances it by max(h, w) after
// itself (so block k's frame = text_prefix + sum_{i<k} max(h_i, w_i)), and each block's height/
// width grid is centered on zero independently of its position in the sequence.

#include "models/qwen_image/artifact_constants.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace infernix::models::qwen_image {

// The rotary base the angle exponents are taken from: angle = position * theta^(-2*col/axis).
//
// PROVENANCE -- this is NOT an artifact value, and it is deliberately not read from the artifact. The
// shipped checkpoint declares no rope theta anywhere the image lane can see it: not in the "dit"
// component's transformer/config.json (verified against the live artifact's own metadata) and not in the
// scheduler document, and the converter packs both verbatim. What the supported Diffusers Qwen-Image
// pipeline does -- QwenImage21Rope, which this module reproduces and whose dumped rotary tables
// infernix_image_rope_prod_test compares against bit for bit -- rotates with base 10000. So 10000 is
// this build's PIPELINE IMPLEMENTATION DEFAULT, an authority of the code (see the table in
// artifact_constants.h), and not a model constant the artifact is asked about; inventing an artifact key
// for it would only make the artifact look like the source of a value it never declared. It changes when
// the supported pipeline implementation is deliberately changed, in this one line, together with the
// rope oracle that certifies the table.
//
// An artifact that declares a "rope_theta" of its own is VERIFIED against this constant and refused when
// it disagrees (read_dit_rope_axes, artifact_constants.cpp) -- verified, never adopted: this line remains
// the only thing the emitter rotates with, so a foreign declaration cannot bend the table, only refuse the
// binding. That is what closes the hole where an artifact trained at another base would silently have been
// rotated at 10000.
constexpr float kQwenImageRopeTheta = 10000.0F;

// Floats in one emitted rope row: (cos, sin) interleaved, so two per complex pair and one pair per two
// channels of each axis -- exactly the head width the axes add up to (read_dit_rope_axes refuses a
// document whose sum is not attention_head_dim). The shipped {16, 56, 56} gives 128.
[[nodiscard]] constexpr std::size_t rope_row_floats(const DitRopeAxes& axes) noexcept {
    return static_cast<std::size_t>(axes.axes[0]) + static_cast<std::size_t>(axes.axes[1]) +
           static_cast<std::size_t>(axes.axes[2]);
}

// Build the joint-sequence rotary table for a text-prefix + single H-by-W image grid.
// Returns S*rope_row_floats(axes) floats, row-major [S, rope_row_floats/2, 2] with (cos, sin)
// interleaved, S = text_prefix + H*W.
[[nodiscard]] std::vector<float> build_joint_rope(const DitRopeAxes& axes, int text_prefix, int H, int W);

// The multi-block generalization: text + N condition-image grids (each (h_i, w_i) in latent tokens,
// the vae_height/16 x vae_width/16 grids the pipeline passes as img_shapes) + the H-by-W target
// grid. S = text_prefix + sum_i h_i*w_i + H*W.
[[nodiscard]] std::vector<float> build_joint_rope_blocks(
    const DitRopeAxes& axes, int text_prefix, const std::vector<std::pair<int, int>>& cond_grids,
    int H, int W);

// The two steps above are built from, exposed for the edit route's INTERLEAVED layout
// (joint_layout.{h,cpp}), where text runs sit between image blocks rather than in front of all of them.
// Sharing them is what makes the edit route's text-only case bit-identical to build_joint_rope: one
// copy of the formula, so the two routes cannot drift apart -- and one place the artifact's axes are
// applied.
//
// Emit the [rows, rope_row_floats/2, 2] (cos,sin)-interleaved table from per-row positions;
// frame/height/width must have the same length (that length is the row count).
[[nodiscard]] std::vector<float> emit_rope_rows(const DitRopeAxes& axes,
                                                const std::vector<long long>& frame,
                                                const std::vector<long long>& height,
                                                const std::vector<long long>& width);
// Fill one image block's rows: `frame` frozen at frame_value, `height`/`width` the row-major h-by-w
// grid centred at -(h - h/2) / -(w - w/2), starting at block_start. Axes-independent, so it takes none.
void fill_rope_grid(std::vector<long long>& frame, std::vector<long long>& height,
                    std::vector<long long>& width, long long block_start, long long h, long long w,
                    long long frame_value);

} // namespace infernix::models::qwen_image
