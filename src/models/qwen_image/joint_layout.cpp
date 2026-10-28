// The M5 joint index map (see joint_layout.h). Two passes: the structure walk (repeats, expansion,
// scatter rows, block ids, target mask) and the rope walk over the segments that walk produced. The
// rope rows come from rope.{h,cpp}'s own emitter, so this module adds no rotary formula of its own.

#include "models/qwen_image/joint_layout.h"

#include "models/qwen_image/rope.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen_image {

namespace {

// One contiguous stretch of the expanded sequence: `len` text rows, or `len` rows of image block
// `block`. The rope walk advances per segment, which is where the interleaving lives.
struct Segment {
    bool image   = false;
    long long block = -1;
    long long start = 0;
    long long len   = 0;
};

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("joint layout: " + what); }

} // namespace

JointLayout build_joint_layout(std::span<const std::uint8_t> image_pad_mask,
                               std::span<const ImageShape> img_shapes) {
    if (img_shapes.empty()) { fail("img_shapes must name at least the target grid"); }

    long long block_tokens = 0;
    for (const ImageShape& s : img_shapes) {
        if (s.frames <= 0 || s.height <= 0 || s.width <= 0) { fail("an img_shapes entry has a non-positive extent"); }
        if (s.frames != 1) { fail("a temporal extent != 1 has no rope grid in this route"); }
        block_tokens += s.tokens();
    }

    long long image_slots = 0;
    for (const std::uint8_t m : image_pad_mask) { image_slots += (m != 0 ? 1 : 0); }
    if (image_slots * 4 != block_tokens) {
        fail("image slots x4 (" + std::to_string(image_slots * 4) +
             ") do not cover the img_shapes blocks (" + std::to_string(block_tokens) +
             " tokens): the vision and VAE planes disagree about the reference grid");
    }

    JointLayout out;
    out.base_len = static_cast<long long>(image_pad_mask.size());
    out.repeats.reserve(image_pad_mask.size());

    std::vector<Segment> segments;
    long long expanded     = 0;  // next expanded position
    long long block        = 0;  // current img_shapes entry
    long long remaining    = img_shapes[0].tokens();
    long long packed_row   = 0;  // next row of the packed [ref | target] img_in input
    Segment cur;

    auto close = [&segments, &cur] {
        if (cur.len > 0) { segments.push_back(cur); }
        cur = Segment{};
    };

    for (const std::uint8_t m : image_pad_mask) {
        if (m == 0) {  // text row: one position, no scatter, no block
            out.repeats.push_back(1);
            out.expanded_image_pad_mask.push_back(0);
            out.scatter_row.push_back(-1);
            out.target_token_mask.push_back(0);
            out.image_block_ids.push_back(-1);
            if (cur.image || cur.len == 0) { close(); cur = {/*image*/ false, -1, expanded, 1}; }
            else { cur.len += 1; }
            out.text_tokens += 1;
            expanded += 1;
            continue;
        }

        // image slot: four positions, each taking the next block's next img_in row.
        out.repeats.push_back(4);
        for (int k = 0; k < 4; ++k) {
            while (remaining == 0) {  // the previous block is full; the mask's ones run on
                if (block + 1 >= static_cast<long long>(img_shapes.size())) {
                    fail("the mask carries more image tokens than img_shapes describes");
                }
                ++block;
                remaining = img_shapes[static_cast<std::size_t>(block)].tokens();
            }
            --remaining;
            const bool is_target = block + 1 == static_cast<long long>(img_shapes.size());
            out.expanded_image_pad_mask.push_back(1);
            out.scatter_row.push_back(packed_row++);
            out.target_token_mask.push_back(is_target ? 1 : 0);
            out.image_block_ids.push_back(block);
            if (cur.image && cur.block == block) { cur.len += 1; }
            else { close(); cur = {/*image*/ true, block, expanded, 1}; }
            expanded += 1;
        }
    }
    close();

    out.expanded_len  = expanded;
    out.target_tokens = img_shapes.back().tokens();
    out.ref_tokens    = block_tokens - out.target_tokens;
    if (packed_row != block_tokens) { fail("internal: scatter rows != block tokens"); }

    // The rope walk, in mask order: text advances the shared position per token on all three axes; an
    // image block freezes the frame at the current position, centres its grid, and advances by max(h,w).
    std::vector<long long> frame(static_cast<std::size_t>(expanded)), height(static_cast<std::size_t>(expanded)),
        width(static_cast<std::size_t>(expanded));
    long long position = 0;
    for (const Segment& seg : segments) {
        if (!seg.image) {
            for (long long i = 0; i < seg.len; ++i) {
                frame[static_cast<std::size_t>(seg.start + i)] = position + i;
                height[static_cast<std::size_t>(seg.start + i)] = position + i;  // rope.cpp: text rows
                width[static_cast<std::size_t>(seg.start + i)]  = position + i;  // take all three axes
            }
            position += seg.len;
            continue;
        }
        const ImageShape& s = img_shapes[static_cast<std::size_t>(seg.block)];
        fill_rope_grid(frame, height, width, seg.start, s.height, s.width, position);
        position += std::max(s.height, s.width);  // the transformer's per-block frame advance
    }
    out.rope = emit_rope_rows(frame, height, width);
    return out;
}

} // namespace infernix::models::qwen_image
