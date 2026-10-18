#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace infernix::ops {

/**
 * 3x3, pad 1, valid cross-correlation (torch Conv2d) over a channel-first feature map, with a
 * per-output-channel bias and an optional same-shaped residual:
 *
 *   ideal[oc,i,j] = b[oc]
 *     + sum_{ic} sum_{u,v in 0..2} x[ic, i-1+u, j-1+v] * w[oc, ic, u, v]   (zero-padded at borders)
 *     + (residual != nullptr ? residual[oc,i,j] : 0)
 *
 * `x` is contiguous BF16 `[Cin,H,W]` (W fastest); `w` is contiguous BF16 `[Cout,Cin,3,3]` in torch
 * layout; `b` is contiguous BF16 `[Cout]`; `out` is contiguous BF16 `[Cout,H,W]`; `residual` (optional)
 * is contiguous BF16 `[Cout,H,W]`. The spatial extent is unchanged (pad 1 with k 3). The oracle
 * evaluates `ideal` in FP64 from the represented inputs; the BF16 out is promoted and compared
 * against it. `out` must not overlap any input. No workspace or persistent state.
 */
void conv2d(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor* residual, Tensor& out,
            cudaStream_t stream);

/**
 * Applies SiLU elementwise in place. For z = x[i]:
 *
 *   ideal[i] = z / (1 + exp(-z)).
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor. The oracle evaluates `ideal` in FP64 from the
 * represented input; the updated BF16 x is promoted and compared directly. Private kernel arithmetic
 * is implementation-defined. The Op mutates only x and uses no workspace.
 */
void silu(Tensor& x, cudaStream_t stream);

/**
 * Elementwise add of two same-shaped tensors:
 *
 *   ideal[i] = a[i] + b[i].
 *
 * `a`, `b` and `out` are contiguous BF16 tensors of identical shape (any rank). The oracle evaluates
 * `ideal` in FP64 from the represented inputs; the BF16 out is promoted and compared. `out` must not
 * overlap `a` or `b`. No workspace.
 */
void add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

/**
 * Nearest-neighbour 2x spatial upsample (torch nn.Upsample(scale_factor=2, mode='nearest-exact') on an
 * integer 2x scale): each input pixel is replicated to a 2x2 block.
 *
 *   ideal[c, 2*i+a, 2*j+b] = x[c, i, j],  a,b in {0,1}.
 *
 * `x` is contiguous BF16 `[C,H,W]`; `out` is contiguous BF16 `[C,2H,2W]`. The oracle evaluates `ideal`
 * exactly from the represented input; the BF16 out is promoted and compared. `out` must not overlap
 * `x`. No workspace.
 */
void upsample_nearest_2x(const Tensor& x, Tensor& out, cudaStream_t stream);

/**
 * QwenImage DupUp3D residual upsample for a single first-chunk frame: a spatial 2x upsample combined
 * with a channel mix (repeat_interleave over the flattened [Cout,factor_t,2,2] slots). With
 * factor = factor_t*4, repeats = Cout*factor/Cin, tt = factor_t-1:
 *
 *   ideal[oc, 2*h+a, 2*w+b] = x[cin, h, w],  cin = (oc*factor + tt*4 + a*2 + b) / repeats.
 *
 * `x` is contiguous BF16 `[Cin,H,W]`; `out` is contiguous BF16 `[Cout,2H,2W]`; `factor_t` is 1 or 2.
 * The oracle evaluates `ideal` exactly from the represented input; the BF16 out is promoted and
 * compared. `out` must not overlap `x`. No workspace.
 */
void dupup3d(const Tensor& x, std::int32_t factor_t, Tensor& out, cudaStream_t stream);

/**
 * Single-head, FULL (non-causal) self-attention over N spatial positions with head dimension C and
 * attention scale:
 *
 *   S[n,m] = scale * sum_c q[c,n] * k[c,m]
 *   P[n,:] = softmax_m(S[n,:])                     (max-subtracted)
 *   ideal[c,n] = sum_m P[n,m] * v[c,m] + (residual ? residual[c,n] : 0)
 *
 * `q`, `k`, `v` and `out` are contiguous BF16 `[C,N]` (N fastest); `residual` (optional) is contiguous
 * BF16 `[C,N]`. `scale` is positive and finite. N must be at most 16384 (a block-resident softmax
 * row of N scores in dynamic shared memory; the score and rescale loops tile over N with blockDim
 * threads, so any N up to the bound is covered). The
 * oracle evaluates `ideal` in FP64 from the represented inputs; the BF16 out is promoted and compared.
 * `out` must not overlap any input. No workspace or persistent state.
 */
void spatial_attn(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* residual, float scale,
                  Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
