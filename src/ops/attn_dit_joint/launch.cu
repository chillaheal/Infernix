#include "ops/attn_dit_joint/launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// Channel-first head layout [head_dim, heads, S, 1]: a head row is the 128 channels of one
// (head, token), head_dim fastest. head_dim is fixed at 128. A CTA handles QB=64 query rows of one
// head; the four threads that share a row hold disjoint quarters of its channels. K and V stream in
// shared tiles of KB=64. The joint block-causal mask is applied per (query, key) and the row is
// online-softmaxed, so no full score matrix is ever materialised.
constexpr int kDim   = 128;
constexpr int kQB    = 64;
constexpr int kKB    = 64;
constexpr int kDims  = 32;              // channels per thread (kDim / 4 lanes share a row)
constexpr int kBlock = kQB * 4;         // 256 threads

__global__ __launch_bounds__(kBlock) void attn_dit_joint_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int heads, int tokens,
    int text_prefix) {
    __shared__ __nv_bfloat16 sK[kKB][kDim];
    __shared__ __nv_bfloat16 sV[kKB][kDim];

    const int t = threadIdx.x;
    const int r = t >> 2;                       // query row within the block, 0..QB-1
    const int g = t & 3;                        // quarter lane of the row, 0..3
    const int gd = g * kDims;                   // channel base this thread owns
    const int head = blockIdx.y;
    const int s = blockIdx.x * kQB + r;         // global query token
    const bool row_valid = (s < tokens);

    const float scale = rsqrtf(static_cast<float>(kDim));
    const float neg_inf = -INFINITY;

    float qreg[kDims];
    if (row_valid) {
        const int qbase = head + heads * s;
        for (int i = 0; i < kDims; ++i) qreg[i] = __bfloat162float(__ldg(&q[(gd + i) + kDim * qbase]));
    } else {
        for (int i = 0; i < kDims; ++i) qreg[i] = 0.0f;
    }

    float acc[kDims];
    for (int i = 0; i < kDims; ++i) acc[i] = 0.0f;
    float m = neg_inf;   // running row max (identical across the row's four lanes)
    float l = 0.0f;      // running row softmax denominator

    for (int kb = 0; kb < tokens; kb += kKB) {
        const int jend = (kKB < tokens - kb) ? kKB : (tokens - kb);

        for (int idx = t; idx < jend * kDim; idx += kBlock) {
            const int jl = idx / kDim;
            const int d  = idx % kDim;
            const int kk = kb + jl;
            const int off = head + heads * kk;
            sK[jl][d] = __ldg(&k[d + kDim * off]);
            sV[jl][d] = __ldg(&v[d + kDim * off]);
        }
        __syncthreads();

        for (int j = 0; j < jend; ++j) {
            float partial = 0.0f;
            for (int i = 0; i < kDims; ++i) partial += qreg[i] * __bfloat162float(sK[j][gd + i]);
            float dot = partial;
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2);

            const int kk = kb + j;
            const bool allow = (s >= text_prefix) || (kk <= s);
            const float score = allow ? dot * scale : neg_inf;

            const float new_m = (m > score) ? m : score;
            if (new_m == neg_inf) continue;   // whole tile masked for this row
            const float alpha = (m == neg_inf) ? 0.0f : expf(m - new_m);
            const float p = (score == neg_inf) ? 0.0f : expf(score - new_m);
            l = l * alpha + p;
            for (int i = 0; i < kDims; ++i) acc[i] = acc[i] * alpha + p * __bfloat162float(sV[j][gd + i]);
            m = new_m;
        }
        __syncthreads();
    }

    if (row_valid) {
        const float inv = 1.0f / l;
        const int obase = head + heads * s;
        for (int i = 0; i < kDims; ++i)
            out[(gd + i) + kDim * obase] = __float2bfloat16_rn(acc[i] * inv);
    }
}

} // namespace

void attn_dit_joint_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                           std::int32_t text_prefix, Tensor* out, std::int32_t heads,
                           std::int32_t tokens, cudaStream_t stream) {
    const unsigned grid_x = static_cast<unsigned>((tokens + kQB - 1) / kQB);
    attn_dit_joint_kernel<<<dim3(grid_x, static_cast<unsigned>(heads)), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<__nv_bfloat16*>(out->data),
        static_cast<int>(heads), static_cast<int>(tokens), static_cast<int>(text_prefix));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
