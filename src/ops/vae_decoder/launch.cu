#include "ops/vae_decoder/launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
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

// One thread per output pixel (oc, oh, ow): 3x3 stride-2 cross-correlation over the right/bottom
// zero-padded plane (window starts at the pixel; terms past H/W are the zero pad), float accumulator.
__global__ __launch_bounds__(256) void vae_conv2d_stride2_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  const __nv_bfloat16* __restrict__ b,
                                                                  __nv_bfloat16* __restrict__ out, int Cin,
                                                                  int H, int W, int Cout, int H2, int W2) {
    const int total = Cout * H2 * W2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow  = idx % W2;
    const int rem = idx / W2;
    const int oh  = rem % H2;
    const int oc  = rem / H2;
    float acc = __bfloat162float(b[oc]);
    for (int ci = 0; ci < Cin; ++ci) {
        const __nv_bfloat16* xr = x + ci * H * W;
        const __nv_bfloat16* wr = w + (oc * Cin + ci) * 9;
        for (int u = 0; u < 3; ++u) {
            const int hy = 2 * oh + u;
            if (hy >= H) continue;
            for (int v = 0; v < 3; ++v) {
                const int hx = 2 * ow + v;
                if (hx >= W) continue;
                acc += __bfloat162float(xr[hy * W + hx]) * __bfloat162float(wr[u * 3 + v]);
            }
        }
    }
    out[idx] = __float2bfloat16_rn(acc);
}

// AvgDown3D image shortcut: one thread per output pixel (c', h', w'); sums the channel-group x
// 2x2 spatial terms (ft >= 1 terms are the temporal front zero pad) in a float accumulator.
__global__ __launch_bounds__(256) void vae_avg_down_kernel(const __nv_bfloat16* __restrict__ x,
                                                           __nv_bfloat16* __restrict__ out, int Cin,
                                                           int Cout, int H, int W, int factor_t) {
    const int H2 = H / 2, W2 = W / 2;
    const int total = Cout * H2 * W2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow  = idx % W2;
    const int rem = idx / W2;
    const int oh  = rem % H2;
    const int oc  = rem / H2;
    const int factor = factor_t * 4;
    const int group  = Cin * factor / Cout;
    float acc = 0.0F;
    for (int g = 0; g < group; ++g) {
        const int k = oc * group + g;
        const int c  = k / factor;
        const int r  = k % factor;
        const int ft = r >> 2;
        if (ft < factor_t - 1) continue; // T=1 pads factor_t-1 zero frames in FRONT, so the real frame
                                          // is the LAST temporal slot, not the first
        const int a = (r >> 1) & 1, b = r & 1;
        acc += __bfloat162float(x[c * H * W + (2 * oh + a) * W + 2 * ow + b]);
    }
    out[idx] = __float2bfloat16_rn(acc / static_cast<float>(group));
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

// Single-head full attention over N positions, head dim C. One block per query row n. K and V stream
// through shared in kThreads-wide column tiles under an online softmax that rescales the running
// max/sum/accumulator when the tile max grows, so block memory is O(C + kThreads) instead of O(N) and
// N reaches a full 2048x2048 latent grid (65536). V is the value plane [C,N].
__global__ __launch_bounds__(256) void vae_spatial_attn_kernel(const __nv_bfloat16* __restrict__ q,
                                                               const __nv_bfloat16* __restrict__ k,
                                                               const __nv_bfloat16* __restrict__ v,
                                                               const __nv_bfloat16* __restrict__ residual,
                                                               __nv_bfloat16* __restrict__ out, int C,
                                                               int N, float scale) {
    extern __shared__ float sm[];
    float* qcol = sm;        // [C] query column for position n, staged once
    float* acc  = qcol + C;  // [C] online value accumulator, weights relative to s_mx
    float* p    = acc + C;   // [kThreads] tile scores exp(score - s_mx)
    __shared__ float s_mx;
    __shared__ float s_sum;
    __shared__ float red[8];
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    for (int c = tid; c < C; c += blockDim.x) {
        qcol[c] = __bfloat162float(q[c * N + n]);
        acc[c]  = 0.0F;
    }
    if (tid == 0) {
        s_mx  = -INFINITY;
        s_sum = 0.0F;
    }
    __syncthreads();
    for (int m0 = 0; m0 < N; m0 += kThreads) {
        const int tn = kThreads < N - m0 ? kThreads : N - m0;
        for (int t = tid; t < tn; t += blockDim.x) {
            const int m = m0 + t;
            float score = 0.0F;
            for (int c = 0; c < C; ++c) score += qcol[c] * __bfloat162float(k[c * N + m]);
            p[t] = score * scale;
        }
        __syncthreads();
        float tmax = -INFINITY;
        for (int t = tid; t < tn; t += blockDim.x) tmax = fmaxf(tmax, p[t]);
        for (int off = 16; off > 0; off >>= 1)
            tmax = fmaxf(tmax, __shfl_down_sync(0xffffffffU, tmax, off));
        if (lane == 0) red[wid] = tmax;
        __syncthreads();
        if (tid == 0) {
            float bm = red[0];
            for (int w = 1; w < 8; ++w) bm = fmaxf(bm, red[w]);
            const float new_mx = fmaxf(s_mx, bm);
            red[0] = expf(s_mx - new_mx);  // rescale factor for sum/acc (0.0F on the first tile)
            s_sum *= red[0];
            s_mx  = new_mx;
        }
        __syncthreads();
        const float mx = s_mx;
        const float r  = red[0];
        // Deterministic tile sum (partials -> shuffle -> fixed-order block combine): a shared
        // float atomicAdd reorders with the warp schedule, which drifted the running sum LSB-wise
        // between a quiet device and the live coexisting process.
        float tsum = 0.0F;
        for (int t = tid; t < tn; t += blockDim.x) {
            const float e = expf(p[t] - mx);
            p[t] = e;
            tsum += e;
        }
        for (int off = 16; off > 0; off >>= 1) tsum += __shfl_down_sync(0xffffffffU, tsum, off);
        __syncthreads(); // every thread read r (red[0]) before the block-sum reuses red
        if (lane == 0) red[wid] = tsum;
        __syncthreads();
        if (tid == 0) {
            float ts = red[0];
            for (int w = 1; w < 8; ++w) ts += red[w];
            s_sum += ts;
        }
        for (int c = tid; c < C; c += blockDim.x) {
            const __nv_bfloat16* vr = v + static_cast<std::size_t>(c) * N + m0;
            float a = acc[c] * r;
            for (int t = 0; t < tn; ++t) a += p[t] * __bfloat162float(vr[t]);
            acc[c] = a;
        }
        __syncthreads();
    }
    const float inv = 1.0F / s_sum;
    for (int c = tid; c < C; c += blockDim.x) {
        float a = acc[c] * inv;
        if (residual != nullptr) a += __bfloat162float(residual[c * N + n]);
        out[c * N + n] = __float2bfloat16_rn(a);
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

void vae_conv2d_stride2_launch(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor& out,
                               cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2], Cout = w.ne[0];
    const int H2 = (H - 2) / 2 + 1, W2 = (W - 2) / 2 + 1;
    const int total = Cout * H2 * W2;
    const unsigned grid = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    vae_conv2d_stride2_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(w.data),
        static_cast<const __nv_bfloat16*>(b.data), static_cast<__nv_bfloat16*>(out.data), Cin, H, W,
        Cout, H2, W2);
    CUDA_CHECK(cudaGetLastError());
}

void vae_avg_down_launch(const Tensor& x, std::int32_t factor_t, const Tensor& out, cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2], Cout = out.ne[0];
    const int total = Cout * (H / 2) * (W / 2);
    const unsigned grid = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    vae_avg_down_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                       static_cast<__nv_bfloat16*>(out.data), Cin, Cout, H,
                                                       W, factor_t);
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
    const std::size_t smem = static_cast<std::size_t>(2 * C + kThreads) * sizeof(float);
    static const bool attr_ok = cudaFuncSetAttribute(vae_spatial_attn_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     48 * 1024) == cudaSuccess;
    (void)attr_ok;  // a failed opt-in still surfaces as the launch error below
    vae_spatial_attn_kernel<<<N, kThreads, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const __nv_bfloat16*>(residual ? residual->data : nullptr),
        static_cast<__nv_bfloat16*>(out.data), C, N, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
