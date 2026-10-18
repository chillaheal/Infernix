#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Apply the Qwen-Image-2.1 DiT complex rotary embedding (use_real = false) in place.
 *
 * The coefficients are a precomputed positional table, not derived here: the DiT builds a 3-axis
 * complex cis table once for the joint text-prefix / image-suffix sequence and every block reuses
 * it. The table is rot BF16 [128,1,S,1], where the leading 128 channels are 64 interleaved
 * (real, imag) pairs for the 64 complex dimensions of a 128-wide head. For each head and token,
 * reading the complex pair as x = (x[2j], x[2j+1]) and the coefficient as cis = (rot[2j], rot[2j+1]),
 * the complete mathematical operation is
 *
 *   out[2j]   = x[2j]   * cis.real - x[2j+1] * cis.imag
 *   out[2j+1] = x[2j]   * cis.imag + x[2j+1] * cis.real
 *
 * with all arithmetic in FP32 and the pair written back as BF16. The Qwen-Image DiT uses full
 * (non-GQA) attention, so q and k share the same 32-head layout and the same table.
 *
 * The pair form takes q BF16 [128,32,S,1] and k BF16 [128,32,S,1], both overwritten in place; the
 * single form takes one x BF16 [128,heads,S,1] with heads in 1..40. S is 1..16384. head_dim is fixed
 * at 128 (64 complex pairs); a caller dispatches here only for a 128-wide head. x/q/k must not
 * overlap rot or, in the pair form, each other; rot stays unchanged. All tensors are contiguous and
 * 4-byte aligned. The independent oracle evaluates the formula naively in FP64 from the BF16
 * represented inputs and final BF16 outputs are promoted for comparison. The Op owns no workspace or
 * persistent state.
 */
void rope_c64(const Tensor& rot, Tensor& q, Tensor& k, cudaStream_t stream);
void rope_c64(const Tensor& rot, Tensor& x, cudaStream_t stream);

} // namespace infernix::ops
