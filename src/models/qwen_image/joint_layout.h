#pragma once

// Instruction-guided image edit (M5): the host joint-sequence index map for one edit step.
//
// The DiT's joint sequence for an edit is NOT [text][ref][target]: the reference's VAE tokens land ON
// the expanded <|image_pad|> slots inside the text sequence (transformer_qwenimage21.py forward
// 847-959). Given the img_mask row -- the text encoder's image_pad_mask_base (1 at every 151655 slot)
// concatenated with target_slots = (Hl*Wl)//4 packed target ones, exactly what build_img_mask
// produces -- and the img_shapes blocks (one per reference image, then the target), this module is the
// whole index arithmetic of that forward, device-free:
//
//   repeats      = where(mask, 4, 1)              (each vision token expands x4)
//   joint        = repeat_interleave([txt rows, zeros(target_slots)], repeats)
//   joint[expanded_image_pad_mask] = img_in(packed [ref | target] latents)   <- scatter_row
//   modulation   = target_token_mask               (build_token_metadata: the LAST block)
//   attention    = block-causal over image_block_ids
//   rope         = the interleaved walk of QwenImage21Rope.forward
//
// So a text row is a single position, every image slot becomes four positions, and the img_in rows
// scatter onto them in block order (ref image 1, ref image 2, ..., target). The rope table uses
// rope.{h,cpp}'s own emitter (emit_rope_rows / fill_rope_grid), so the text-only case is bit-identical
// to build_joint_rope -- the certified text-to-image assembly does not move when this takes over.
//
// The rope walk in mask order: a text run advances the shared position by one per token on all three
// axes; an image block FREEZES the frame at the current position, centres its own height/width grid on
// zero, and advances the position by max(h, w) after itself. Two images with no text between them stay
// two blocks with two frame advances (img_shapes, not mask runs, define blocks).

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen_image {

// One img_shapes entry: (frames, height, width) in PACKED latent tokens (vae_height/16, vae_width/16).
// The edit route serves single-frame grids; frames != 1 is refused (the rope grid has no temporal axis
// in this route -- see rope.h), so tokens() == height*width in practice.
struct ImageShape {
    long long frames = 1;
    long long height = 0;
    long long width  = 0;
    [[nodiscard]] long long tokens() const { return frames * height * width; }
};

// The per-step index map. Every vector is indexed by POSITION IN THE EXPANDED joint sequence unless
// noted; expanded_len is their length.
struct JointLayout {
    long long base_len     = 0;  // the mask's length (VLM sequence + packed target slots)
    long long expanded_len = 0;  // text rows + 4 * image slots
    long long text_tokens  = 0;  // expanded rows that are text
    long long ref_tokens   = 0;  // tokens of every block but the last (== 4 * vision tokens)
    long long target_tokens = 0; // tokens of the last block (== the sampled grid H*W)

    std::vector<int> repeats;                  // indexed by BASE position: 4 on an image slot, else 1
    std::vector<std::uint8_t> expanded_image_pad_mask;  // 1 where an img_in row lands
    std::vector<long long> scatter_row;        // packed img_in row for that position, -1 on text rows
    std::vector<std::uint8_t> target_token_mask;        // 1 on the last block only (the modulation split)
    std::vector<long long> image_block_ids;    // index into img_shapes, -1 on text rows
    std::vector<float> rope;                   // expanded_len * 64 * 2, (cos,sin) interleaved -- the
                                               // same [S,64,2] layout ops::rope_c64 uploads today
};

// Build the map. image_pad_mask is ONE row of build_img_mask's output (base + target_slots ones, uint8
// 0/1; any non-zero is an image slot). img_shapes names every block in sequence order, the target LAST.
// Throws std::invalid_argument when: img_shapes is empty or an entry is degenerate; a block declares a
// temporal extent != 1; or the x4 invariant fails -- 4 * (image slots in the mask) must equal
// sum(img_shapes tokens), i.e. 4 * vision_tokens == ref_tokens and 4 * target_slots == target_tokens.
// That invariant is the M1 geometry gate re-asserted at runtime; a mismatch means the reference plane
// was resized differently by the vision and VAE paths, and the caller must fail the request (host-side,
// before any VRAM yield) rather than assemble a sequence whose slots cannot all be filled.
[[nodiscard]] JointLayout build_joint_layout(std::span<const std::uint8_t> image_pad_mask,
                                             std::span<const ImageShape> img_shapes);

} // namespace infernix::models::qwen_image
