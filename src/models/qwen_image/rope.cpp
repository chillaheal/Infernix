// Qwen-Image-2.1 DiT joint-sequence rotary positions (see rope.h). Reproduces QwenImage21Rope.forward
// for the text-only case at any (text_prefix, H, W); verified against the two diffusers-dumped rotary
// references at (8,8,8) and (11,16,16) and by grid invariants in infernix_image_rope_prod_test.
// build_joint_rope_blocks extends it to the reference-image mask [text, cond_1..cond_N, target]
// exactly as the transformer's per-block loop advances `position` by max(h, w) between blocks.

#include "models/qwen_image/rope.h"

#include <algorithm>
#include <cmath>

namespace infernix::models::qwen_image {

// rope_params(index, dim) = polar(1, outer(index, 1/theta^(arange(0,dim,2)/dim))). Each freqs row
// stores angle = index_value * theta^(-2*col/dim); the angle depends only on the integer position.
// Public since the edit route (joint_layout.cpp) walks interleaved positions and shares this emitter.
std::vector<float> emit_rope_rows(const std::vector<long long>& frame, const std::vector<long long>& height,
                                  const std::vector<long long>& width) {
    const std::size_t S = frame.size();
    const std::size_t axes[3] = {16, 56, 56};
    auto lookup = [&](std::size_t axis, long long v, std::size_t col, float& re, float& im) {
        const std::size_t dim = axes[axis];
        const double exponent = static_cast<double>(2 * col) / static_cast<double>(dim);
        const double angle = static_cast<double>(v) * std::pow(10000.0, -exponent);
        re = static_cast<float>(std::cos(angle));
        im = static_cast<float>(std::sin(angle));
    };
    std::vector<float> rot(S * 64 * 2, 0.0F);
    for (std::size_t s = 0; s < S; ++s) {
        float* o = rot.data() + s * 64 * 2;
        float re, im;
        std::size_t col = 0;
        for (std::size_t j = 0; j < axes[0] / 2; ++j, ++col) { lookup(0, frame[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
        for (std::size_t j = 0; j < axes[1] / 2; ++j, ++col) { lookup(1, height[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
        for (std::size_t j = 0; j < axes[2] / 2; ++j, ++col) { lookup(2, width[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
    }
    return rot;
}

void fill_rope_grid(std::vector<long long>& frame, std::vector<long long>& height, std::vector<long long>& width,
                    long long block_start, long long h, long long w, long long frame_value) {
    for (long long i = 0; i < h * w; ++i) frame[block_start + i] = frame_value;
    for (long long hh = 0; hh < h; ++hh) {
        for (long long ww = 0; ww < w; ++ww) {
            const long long idx = block_start + hh * w + ww;
            height[idx] = -(h - h / 2) + hh;
            width[idx]  = -(w - w / 2) + ww;
        }
    }
}

std::vector<float> build_joint_rope_blocks(int text_prefix, const std::vector<std::pair<int, int>>& cond_grids,
                                           int H, int W) {
    const long long tp = text_prefix, h = H, w = W;
    long long cond = 0;
    for (const auto& g : cond_grids) cond += static_cast<long long>(g.first) * g.second;
    const long long S = tp + cond + h * w;

    // Build frame / height / width indices exactly as QwenImage21Rope.forward over the mask
    // [False*tp, True*(cond + h*w)]: one text run, then every img_shapes block back to back.
    std::vector<long long> frame(S), height(S), width(S);
    long long cursor = 0, position = 0;
    for (long long i = 0; i < tp; ++i) frame[cursor + i] = position + i;  // text: frame advances per token
    position += tp;
    cursor   += tp;
    for (const auto& g : cond_grids) {
        const long long ch = g.first, cw = g.second;
        fill_rope_grid(frame, height, width, cursor, ch, cw, position);
        cursor += ch * cw;
        position += std::max(ch, cw); // the transformer's per-block frame advance
    }
    fill_rope_grid(frame, height, width, cursor, h, w, position);

    for (long long s = 0; s < tp; ++s) {  // text: all three axes = frame index (image rows overwritten)
        height[s] = frame[s];
        width[s]  = frame[s];
    }
    return emit_rope_rows(frame, height, width);
}

std::vector<float> build_joint_rope(int text_prefix, int H, int W) {
    return build_joint_rope_blocks(text_prefix, {}, H, W);
}

} // namespace infernix::models::qwen_image
