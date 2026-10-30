#pragma once

// The Qwen3-VL text tower as ONE reusable production entry: the qualified 36-layer forward
// (tests/models/qwen_image/test_image_texttower_prod_cuda.cu) composed from the production ops,
// fed by the artifact's streamed text-tower weights (bind_text_encoder) and the prompt's templated
// token ids, producing the DiT's prompt conditioning. Merge authority: that device-qualified test,
// with its per-layer resident uploads replaced by per-layer streaming (the run_image_job pattern)
// so the whole tower runs inside the interlude's yielded VRAM: peak == one layer's roles + the
// per-job activation scratch.
//
// The forward consumes the FULL templated id sequence (causal attention makes the kept rows depend
// on the dropped prefix) and returns the pre-final-RMSNorm hidden rows [drop_rows, S) as token-
// major BF16 bits -- the model's own prompt conditioning proven by the test's composition bridge
// into build_prompt_embeds. The tower's final_norm is therefore NOT applied here (the certified
// route conditions the DiT on the pre-norm hidden).
//
// Like the qualified test, all ops run on the legacy default stream and every transfer is a
// synchronous cudaMemcpy, which the interlude's blocking image stream is ordered against. This
// module allocates device memory ONLY through image_device_util.h's Buf, so the #17 ledger
// (image_pipeline.h image_device_bytes_live) accounts its bytes too.

#include "models/qwen_image/image_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::artifact { class Reader; }

namespace infernix::models::qwen_image {

// Runs the tower for one prompt. `embed_table` is [vocab, hidden] row-major BF16 bits: the caller
// dequantizes the artifact's FP8 RowScale text/embed_tokens ONCE per process and caches it (the
// gather runs per job, the dequantization must not). `token_ids` are the templated ids in sequence
// order; rows [0, drop_rows) are
// computed but dropped from the result. Bad inputs (dims, id range, table size, drop_rows) or a
// CUDA failure throw std::runtime_error. Returns [(S - drop_rows) * hidden] token-major BF16 bits.
[[nodiscard]] std::vector<std::uint16_t> run_text_tower(artifact::Reader& reader, const TextTowerWeights& tower,
                                                        const std::vector<std::byte>& embed_table,
                                                        const std::vector<int>& token_ids,
                                                        std::size_t drop_rows);

// The <|image_pad|> token id of the Qwen3-VL vocabulary (the slot every vision row is written to).
inline constexpr std::uint32_t kImagePadTokenId = 151655;

// The image side of one conditioned prompt, normally filled straight from a VisionTowerOutput:
// `merger_rows` replaces the embedding rows at the pad tokens (in pad order), and `deepstack_rows[i]`
// is ADDED to the hidden rows at those same tokens after decoder layer i -- the reference adds the
// deepstack embeds to text decoder layers 0..len-1 by list position, not by the vision tap index
// (diffusers-reference modeling_qwen3_vl.py:847-852). An empty `deepstack_rows` is the ablated
// forward (the M2 comparison's zeroed-adds variant), not an error.
struct ConditionedVision {
    std::vector<std::uint16_t> merger_rows;                  // [pad_count * hidden] BF16 bits
    std::vector<std::vector<std::uint16_t>> deepstack_rows;  // each [pad_count * hidden] BF16 bits
    // M-RoPE needs the picture's geometry: each maximal run of pad tokens is ONE image, whose PRE-merge
    // patch grid (t, h, w) -- the same numbers as VisionItemGeometry, in patch units -- fixes the three
    // rotary axes of those rows. Grids are in pad order; a run with no grid behind it throws.
    std::vector<std::array<int, 3>> grids;
    std::uint32_t merge = 2U; // vision_config.spatial_merge_size
};

// The Qwen3-VL text tower's M-RoPE positions for one templated prompt: [3*S] axis-major int32
// (axis*S + s), exactly the [3,S] the reference feeds its rotary embedding (get_rope_index,
// modeling_qwen3_vl.py:939-1030, with get_vision_position_ids 887-937). A maximal run of
// kImagePadTokenId rows is one image and consumes t*(h/merge)*(w/merge) rows while advancing the
// running position by ONLY max(h/merge, w/merge) -- every row after a picture therefore sits below
// its own index (the reference's mrope_position_delta). Text rows, <|vision_start|> and
// <|vision_end|> included, advance all three axes by 1 per token. Positions are computed on the FULL
// ids -- the caller must NOT renumber after dropping the system prefix, which is why this takes the
// whole sequence and the tower slices its own output rows. `grids` covers the runs in order and
// `merge` must divide every h and w; anything else (a run without a grid, a non-divisible grid, a
// grid whose row count disagrees with the run) throws before any allocation.
[[nodiscard]] std::vector<int> build_mrope_positions(const std::vector<int>& token_ids,
                                                     const std::vector<std::array<int, 3>>& grids,
                                                     std::uint32_t merge);

struct ConditionedTextTowerOutput {
    std::vector<std::uint16_t> prompt_embeds;  // [(S - drop_rows) * hidden] BF16 bits, pre-final-norm
    std::vector<std::uint8_t> image_pad_mask;  // [S - drop_rows], 1 where the kept id == kImagePadTokenId
};

// The same tower, run over an inputs_embeds whose pad rows are the vision merger rows. Rows [0,
// drop_rows) are computed and dropped from BOTH fields (the mask is sliced exactly like the hidden
// rows -- vision_start/vision_end sit outside the dropped prefix). Every pad token must be covered by
// merger_rows and the deepstack rows, in pad order; a mismatch throws before any allocation.
//
// `layer_snapshots`, when given, receives the FULL [S, hidden] BF16 rows (the dropped prefix
// included, exactly as the anchor's hooks see them) after decoder layers 0..size-1 -- so sizing it to
// the deepstack adds gives the layers the adds land on, and sizing it to the tower gives every layer.
// It is the M4 test's per-layer gate; production passes nothing, and the forward itself is unchanged
// by the request. Any other size throws.
[[nodiscard]] ConditionedTextTowerOutput
run_conditioned_text_tower(artifact::Reader& reader, const TextTowerWeights& tower,
                           const std::vector<std::byte>& embed_table, const std::vector<int>& token_ids,
                           std::size_t drop_rows, const ConditionedVision& vision,
                           std::vector<std::vector<std::uint16_t>>* layer_snapshots = nullptr);

} // namespace infernix::models::qwen_image
