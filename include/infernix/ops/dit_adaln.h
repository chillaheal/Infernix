#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Apply a AdaLN-Zero modulation scale to an activation.
 *
 * Math / indexing:
 *   row(s)  = (s < text_prefix) ? 1 : 0
 *   out[d,r] = x[d,r] * (1 + mod[col_off + d, row(s)])
 * computed in FP32 from the BF16-represented operands and written back as BF16. The modulation is a
 * per-token selected slice of a stored [M,2] table (row 1 for the strictly-causal text prefix, row 0
 * for the target suffix); col_off selects which of the modulation's equal-width channel chunks feeds
 * this scale. This is the scale half of the AdaLN-Zero block modulation; the gate half is
 * adaln_gate_add.
 *
 * Logical shapes:
 *   x and out are same-shaped contiguous BF16 channel-first [D,S,1,1] with the D channels fastest.
 *   mod is contiguous BF16 [M,2,1,1] with M a multiple-of-or-at-least col_off + D. S is 1..20480,
 *   D is 1..65536, col_off is 0..M-D, text_prefix is 0..S.
 *
 * Supported domain:
 *   BF16 storage, FP32 arithmetic, contiguous and 4-byte-aligned operands.
 *
 * Numeric:
 *   The independent oracle evaluates the formula naively in FP64 from the BF16-represented inputs;
 *   the BF16 output is promoted for comparison. Output storage rounding belongs to the Op.
 *
 * Effects:
 *   Writes all of out. out must not overlap x or mod; x and mod are unchanged.
 *
 * Workspace:
 *   None.
 */
void adaln_scale(const Tensor& x, const Tensor& mod, std::int32_t col_off, std::int32_t text_prefix,
                 Tensor& out, cudaStream_t stream);

/**
 * Add a modulation-gated branch onto a residual with a tanh gate.
 *
 * Math / indexing:
 *   row(s)  = (s < text_prefix) ? 1 : 0
 *   out[d,r] = residual[d,r] + tanh(mod[col_off + d, row(s)]) * branch[d,r]
 * computed in FP32 from the BF16-represented operands and written back as BF16. This is the gated
 * residual half of the AdaLN-Zero block modulation (the attention and MLP output residuals).
 *
 * Logical shapes:
 *   residual, branch and out are same-shaped contiguous BF16 channel-first [D,S,1,1]. mod is
 *   contiguous BF16 [M,2,1,1]. S is 1..20480, D is 1..65536, col_off is 0..M-D, text_prefix is 0..S.
 *
 * Supported domain:
 *   BF16 storage, FP32 arithmetic (tanh in FP32), contiguous and 4-byte-aligned operands.
 *
 * Numeric:
 *   The independent oracle evaluates the formula naively in FP64 from the BF16-represented inputs;
 *   the BF16 output is promoted for comparison.
 *
 * Effects:
 *   Writes all of out. out must not overlap residual, branch or mod; those are unchanged.
 *
 * Workspace:
 *   None.
 */
void adaln_gate_add(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                    std::int32_t col_off, std::int32_t text_prefix, Tensor& out,
                    cudaStream_t stream);

/**
 * The same two modulations with the row selected by a per-token mask instead of one boundary.
 *
 * Math / indexing:
 *   row(s)  = target_mask[s] ? 0 : 1        // 0 = the sampled timestep, 1 = the t = 0 embedding
 * and otherwise the identical formulas. The edit route needs this because its non-target set is not
 * a prefix: the REFERENCE image rows sit after the text and still read the t = 0 row
 * (causal_condition), while only the target block reads the sampled t. Passing a mask that is set
 * exactly on a suffix is the prefix rule, so the two forms agree bit-for-bit on every sequence the
 * prefix form can express -- which is what lets the certified text-to-image path keep its own calls.
 *
 * Logical shapes:
 *   as above, with target_mask a contiguous device U8 [S,1,1,1] (one byte per token, the same S as
 *   the activations). Non-zero bytes select row 0; the Op never reads outside those S bytes.
 *
 * Effects / Workspace:
 *   Same as the prefix forms; target_mask is read only and must not overlap out.
 */
void adaln_scale_masked(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                        const Tensor& target_mask, Tensor& out, cudaStream_t stream);
void adaln_gate_add_masked(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                           std::int32_t col_off, const Tensor& target_mask, Tensor& out,
                           cudaStream_t stream);

}  // namespace infernix::ops
