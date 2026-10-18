#include "ops/dit_adaln/launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// Channel-first [D,S,1,1] activation, D fastest. Element i maps to channel d = i % D, token
// s = i / D. The modulation row for token s is 1 for the strictly-causal text prefix (s < prefix)
// and 0 for the target suffix; the modulation column is col_off + d, read from a [M,2] table whose
// row stride is M.
__global__ __launch_bounds__(256) void adaln_scale_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ mod,
    __nv_bfloat16* __restrict__ out, int dim, int mod_stride, int col_off, int text_prefix,
    long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int d = static_cast<int>(i % dim);
    const long long s = i / dim;
    const int row = (s < text_prefix) ? 1 : 0;
    const float xv = __bfloat162float(x[i]);
    const float mv = __bfloat162float(mod[col_off + d + static_cast<long long>(row) * mod_stride]);
    out[i] = __float2bfloat16_rn(xv * (1.0F + mv));
}

__global__ __launch_bounds__(256) void adaln_gate_add_kernel(
    const __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ branch,
    const __nv_bfloat16* __restrict__ mod, __nv_bfloat16* __restrict__ out, int dim, int mod_stride,
    int col_off, int text_prefix, long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int d = static_cast<int>(i % dim);
    const long long s = i / dim;
    const int row = (s < text_prefix) ? 1 : 0;
    const float mv = __bfloat162float(mod[col_off + d + static_cast<long long>(row) * mod_stride]);
    const float rv = __bfloat162float(residual[i]);
    const float bv = __bfloat162float(branch[i]);
    out[i] = __float2bfloat16_rn(rv + tanhf(mv) * bv);
}

}  // namespace

void adaln_scale_launch(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                        std::int32_t text_prefix, Tensor* out, std::int32_t tokens,
                        cudaStream_t stream) {
    const long long total = static_cast<long long>(x.ne[0]) * tokens;
    const int threads = 256;
    const unsigned grid = static_cast<unsigned>((total + threads - 1) / threads);
    adaln_scale_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(mod.data),
        static_cast<__nv_bfloat16*>(out->data), x.ne[0], mod.ne[0], col_off, text_prefix, total);
    CUDA_CHECK(cudaGetLastError());
}

void adaln_gate_add_launch(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                           std::int32_t col_off, std::int32_t text_prefix, Tensor* out,
                           std::int32_t tokens, cudaStream_t stream) {
    const long long total = static_cast<long long>(residual.ne[0]) * tokens;
    const int threads = 256;
    const unsigned grid = static_cast<unsigned>((total + threads - 1) / threads);
    adaln_gate_add_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(branch.data),
        static_cast<const __nv_bfloat16*>(mod.data), static_cast<__nv_bfloat16*>(out->data),
        residual.ne[0], mod.ne[0], col_off, text_prefix, total);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace infernix::ops::detail
