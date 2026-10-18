#include "ops/vae_decoder/launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kThreads = 256;

// One thread per output pixel (oc, oh, ow); zero-padded 3x3 cross-correlation, float accumulator,
// per-channel bias plus optional residual added in the epilogue.
__global__ __launch_bounds__(256) void vae_conv2d_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ w,
                                                         const __nv_bfloat16* __restrict__ b,
                                                         const __nv_bfloat16* __restrict__ residual,
                                                         __nv_bfloat16* __restrict__ out, int Cin, int H,
                                                         int W, int Cout) {
    const int total = Cout * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow  = idx % W;
    const int rem = idx / W;
    const int oh  = rem % H;
    const int oc  = rem / H;
    float acc = __bfloat162float(b[oc]);
    for (int ci = 0; ci < Cin; ++ci) {
        const __nv_bfloat16* xr = x + ci * H * W;
        const __nv_bfloat16* wr = w + (oc * Cin + ci) * 9;
        for (int u = 0; u < 3; ++u) {
            const int hy = oh - 1 + u;
            if (hy < 0 || hy >= H) continue;
            for (int v = 0; v < 3; ++v) {
                const int hx = ow - 1 + v;
                if (hx < 0 || hx >= W) continue;
                acc += __bfloat162float(xr[hy * W + hx]) * __bfloat162float(wr[u * 3 + v]);
            }
        }
    }
    if (residual != nullptr) acc += __bfloat162float(residual[oc * H * W + oh * W + ow]);
    out[oc * H * W + oh * W + ow] = __float2bfloat16_rn(acc);
}

__global__ __launch_bounds__(256) void vae_silu_kernel(__nv_bfloat16* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float z = __bfloat162float(x[i]);
    x[i] = __float2bfloat16_rn(z / (1.0F + expf(-z)));
}

// Elementwise add: out[i] = a[i] + b[i], float-promoted and rounded to BF16.
__global__ __launch_bounds__(256) void vae_add_kernel(const __nv_bfloat16* __restrict__ a,
                                                      const __nv_bfloat16* __restrict__ b,
                                                      __nv_bfloat16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __bfloat162float(a[i]) + __bfloat162float(b[i]);
    out[i] = __float2bfloat16_rn(x);
}

// Replicate each input pixel to a 2x2 block (nearest-exact on a 2x integer scale).
__global__ __launch_bounds__(256) void vae_upsample2x_kernel(const __nv_bfloat16* __restrict__ x,
                                                             __nv_bfloat16* __restrict__ out, int C,
                                                             int H, int W) {
    const int OW = 2 * W, OH = 2 * H;
    const int total = C * OH * OW;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow  = idx % OW;
    const int rem = idx / OW;
    const int oh  = rem % OH;
    const int c   = rem / OH;
    out[idx] = x[c * H * W + (oh >> 1) * W + (ow >> 1)];
}

// DupUp3D residual shortcut: spatial 2x + channel mix, cin = (oc*factor + tt*4 + a*2 + b)/repeats.
__global__ __launch_bounds__(256) void vae_dupup3d_kernel(const __nv_bfloat16* __restrict__ x,
                                                          __nv_bfloat16* __restrict__ out, int Cin,
                                                          int Cout, int H, int W, int factor_t) {
    const int OH = 2 * H, OW = 2 * W;
    const int total = Cout * OH * OW;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow  = idx % OW;
    const int rem = idx / OW;
    const int oh  = rem % OH;
    const int oc  = rem / OH;
    const int a   = oh & 1, b = ow & 1;
    const int h   = oh >> 1, w = ow >> 1;
    const int factor = factor_t * 4;
    const int repeats = Cout * factor / Cin;
    const int tt = factor_t - 1;
    const int cin = (oc * factor + tt * 4 + a * 2 + b) / repeats;
    out[idx] = x[cin * H * W + h * W + w];
}

// Single-head full attention over N positions, head dim C. One block per query row n; a block-resident
// softmax row of N scores lives in dynamic shared memory (N <= 16384). V is the value plane [C,N].
__global__ __launch_bounds__(256) void vae_spatial_attn_kernel(const __nv_bfloat16* __restrict__ q,
                                                               const __nv_bfloat16* __restrict__ k,
                                                               const __nv_bfloat16* __restrict__ v,
                                                               const __nv_bfloat16* __restrict__ residual,
                                                               __nv_bfloat16* __restrict__ out, int C,
                                                               int N, float scale) {
    extern __shared__ float s[];  // [N] block-resident softmax row, tiled by the score/rescale loops
    __shared__ float s_mx;
    __shared__ float s_sum;
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    for (int m = tid; m < N; m += blockDim.x) {
        float acc = 0.0F;
        for (int c = 0; c < C; ++c) {
            acc += __bfloat162float(q[c * N + n]) * __bfloat162float(k[c * N + m]);
        }
        s[m] = acc * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30F;
        for (int m = 0; m < N; ++m) mx = fmaxf(mx, s[m]);
        float sum = 0.0F;
        for (int m = 0; m < N; ++m) sum += expf(s[m] - mx);
        s_mx = mx;
        s_sum = sum;
    }
    __syncthreads();
    for (int m = tid; m < N; m += blockDim.x) s[m] = expf(s[m] - s_mx) / s_sum;  // s now holds P[n,:]
    __syncthreads();
    for (int c = tid; c < C; c += blockDim.x) {
        float acc = 0.0F;
        for (int m = 0; m < N; ++m) acc += s[m] * __bfloat162float(v[c * N + m]);
        if (residual != nullptr) acc += __bfloat162float(residual[c * N + n]);
        out[c * N + n] = __float2bfloat16_rn(acc);
    }
}

} // namespace

void vae_conv2d_launch(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor* residual,
                       const Tensor& out, cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2], Cout = w.ne[0];
    const int total = Cout * H * W;
    const unsigned grid = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    vae_conv2d_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(w.data),
        static_cast<const __nv_bfloat16*>(b.data),
        static_cast<const __nv_bfloat16*>(residual ? residual->data : nullptr),
        static_cast<__nv_bfloat16*>(out.data), Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

void vae_silu_launch(Tensor& x, cudaStream_t stream) {
    const int n = static_cast<int>(x.numel());
    const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    vae_silu_kernel<<<grid, kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void vae_add_launch(const Tensor& a, const Tensor& b, const Tensor& out, cudaStream_t stream) {
    const int n = static_cast<int>(a.numel());
    const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    vae_add_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(a.data),
                                                  static_cast<const __nv_bfloat16*>(b.data),
                                                  static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void vae_upsample2x_launch(const Tensor& x, const Tensor& out, cudaStream_t stream) {
    const int C = x.ne[0], H = x.ne[1], W = x.ne[2];
    const int total = C * 2 * H * 2 * W;
    const unsigned grid = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    vae_upsample2x_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                         static_cast<__nv_bfloat16*>(out.data), C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

void vae_dupup3d_launch(const Tensor& x, std::int32_t factor_t, const Tensor& out, cudaStream_t stream) {
    const int Cin = x.ne[0], Cout = out.ne[0], H = x.ne[1], W = x.ne[2];
    const int total = Cout * 2 * H * 2 * W;
    const unsigned grid = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    vae_dupup3d_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      static_cast<__nv_bfloat16*>(out.data), Cin, Cout, H, W,
                                                      factor_t);
    CUDA_CHECK(cudaGetLastError());
}

void vae_spatial_attn_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* residual,
                             const Tensor& out, float scale, cudaStream_t stream) {
    const int C = q.ne[0], N = q.ne[1];
    vae_spatial_attn_kernel<<<N, kThreads, N * sizeof(float), stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const __nv_bfloat16*>(residual ? residual->data : nullptr),
        static_cast<__nv_bfloat16*>(out.data), C, N, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
