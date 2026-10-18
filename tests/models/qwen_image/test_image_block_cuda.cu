// GPU numeric oracle for one Qwen-Image-2.1 DiT block, self-contained naive CUDA (double accumulators)
// mirroring qioracle::block_forward exactly. Same block math as the CPU block oracle, executed on the
// device and qualified per-stage against a host reference recomputed with the shared header primitives,
// and end-to-end against the authoritative block_out.npy. Per-stage rel errors are logged so a single
// blackout run localizes any divergent kernel (the agent is offline during the run).
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors/*.npy and .../tensors_block_weights/*.npy.
// Returns 77 (CTest skip) when the oracle data is absent; returns 2 (runnable) when no CUDA device.

#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace qo = qioracle;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__,   \
                        __LINE__, cudaGetErrorString(err_));                \
            std::fflush(stdout);                                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

// ---- naive FP32 kernels with double accumulators (mirror the CPU oracle) ----

// C[M,N] = sum_k A[M,K] * W[N,K]  (x @ W^T).
__global__ void k_matmul_xwt(const float* A, const float* W, float* C, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    double acc = 0.0;
    const float* arow = A + (size_t)m * K;
    const float* wrow = W + (size_t)n * K;
    for (int k = 0; k < K; ++k) acc += (double)arow[k] * (double)wrow[k];
    C[(size_t)m * N + n] = (float)acc;
}

// LayerNorm(affine=False) over rows of length d: (x-mean)/sqrt(biased var + eps). One thread per row.
__global__ void k_layernorm(const float* x, float* y, int rows, int d, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * d;
    float* yr = y + (size_t)r * d;
    double mean = 0.0;
    for (int i = 0; i < d; ++i) mean += xr[i];
    mean /= (double)d;
    double var = 0.0;
    for (int i = 0; i < d; ++i) { const double t = (double)xr[i] - mean; var += t * t; }
    var /= (double)d;
    const float inv = 1.0F / sqrtf((float)var + eps);
    for (int i = 0; i < d; ++i) yr[i] = (float)(((double)xr[i] - mean) * (double)inv);
}

// scale-only AdaLN: out[s,d] = x[s,d] * (1 + mod[row][col_off+d]), row = tmask[s]?0:1.
__global__ void k_select_scale(const float* x, const float* mod, const int* tmask, float* out,
                               int S, int D, int col_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * D) return;
    const int s = i / D, dcol = i % D;
    const int row = tmask[s];
    out[i] = x[i] * (1.0F + mod[(size_t)row * (4 * D) + col_off + dcol]);
}

// gated residual: out[s,d] = xin[s,d] + tanh(mod[row][col_off+d]) * branch[s,d].
__global__ void k_gate_add(const float* xin, const float* mod, const int* tmask, const float* branch,
                           float* out, int S, int D, int col_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * D) return;
    const int s = i / D, dcol = i % D;
    const int row = tmask[s];
    const float g = mod[(size_t)row * (4 * D) + col_off + dcol];
    out[i] = xin[i] + tanhf(g) * branch[i];
}

// per-head RMSNorm over (rows) head-rows of head_dim: x*rsqrt(mean(x^2)+eps)*w. One thread per head-row.
__global__ void k_rmsnorm_head(float* qh, const float* w, int rows, int d, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* xr = qh + (size_t)r * d;
    double ss = 0.0;
    for (int i = 0; i < d; ++i) ss += (double)xr[i] * (double)xr[i];
    const float rr = 1.0F / sqrtf((float)(ss / (double)d) + eps);
    for (int i = 0; i < d; ++i) xr[i] = xr[i] * rr * w[i];
}

// 3-axis complex RoPE (use_real=False): pairs (2j,2j+1) * cis[token,j]. One thread per (s*heads+h) row.
__global__ void k_rope(float* qh, const float* rot, int S, int heads, int head_dim) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= S * heads) return;
    const int s = r / heads;
    const int rope_dim = head_dim / 2;
    float* base = qh + (size_t)r * head_dim;
    const float* c = rot + (size_t)s * rope_dim * 2;
    for (int j = 0; j < rope_dim; ++j) {
        const float zr = base[2 * j], zi = base[2 * j + 1];
        const float cr = c[2 * j], ci = c[2 * j + 1];
        base[2 * j] = zr * cr - zi * ci;
        base[2 * j + 1] = zr * ci + zi * cr;
    }
}

// block-causal attention. One thread per (s,head): text prefix strictly causal, target attends [0,S).
__global__ void k_attention(const float* q, const float* k, const float* v, float* attn,
                            int S, int heads, int head_dim, int prefix_len) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= S * heads) return;
    const int s = r / heads, h = r % heads;
    const bool is_text = (s < prefix_len);
    const int kv_hi = is_text ? prefix_len : S;
    const float qscale = 1.0F / sqrtf((float)head_dim);
    const float* qs = q + (size_t)r * head_dim;
    float scores[512];
    float acc[256];
    float maxv = -INFINITY;
    for (int kk = 0; kk < kv_hi; ++kk) {
        const float* ks = k + ((size_t)kk * heads + h) * head_dim;
        double dot = 0.0;
        for (int dd = 0; dd < head_dim; ++dd) dot += (double)qs[dd] * (double)ks[dd];
        const float sc = (float)dot * qscale;
        scores[kk] = sc;
        if (!(is_text && kk > s) && sc > maxv) maxv = sc;
    }
    double sum = 0.0;
    for (int kk = 0; kk < kv_hi; ++kk) {
        if (is_text && kk > s) { scores[kk] = 0.0F; continue; }
        const float e = expf(scores[kk] - maxv);
        scores[kk] = e;
        sum += (double)e;
    }
    const float inv = (float)(1.0 / sum);
    for (int dd = 0; dd < head_dim; ++dd) acc[dd] = 0.0F;
    for (int kk = 0; kk < kv_hi; ++kk) {
        const float p = scores[kk] * inv;
        if (p == 0.0F) continue;
        const float* vs = v + ((size_t)kk * heads + h) * head_dim;
        for (int dd = 0; dd < head_dim; ++dd) acc[dd] += p * vs[dd];
    }
    float* od = attn + (size_t)r * head_dim;
    for (int dd = 0; dd < head_dim; ++dd) od[dd] = acc[dd];
}

__device__ __forceinline__ float d_silu(float x) { return x / (1.0F + expf(-x)); }

__global__ void k_silu_mul(const float* gate, const float* proj, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = d_silu(gate[i]) * proj[i];
}

// ---- host stage reference (mirrors qioracle::block_forward, keeping every intermediate) ----
struct HostStages {
    std::vector<float> ln1, mod1, q, k, v, attn, attn_out, x1, ln2, mod2, proj, gate, hidden, mlp_out, out;
};

static void host_block_stages(const std::vector<float>& xin, const std::vector<float>& mod,
                              const std::vector<float>& rot, const std::vector<std::uint8_t>& tmask,
                              const qo::BlockWeights& w, std::size_t S, std::size_t D, std::size_t heads,
                              std::size_t prefix_len, float eps, HostStages& H) {
    const std::size_t head_dim = D / heads;
    const std::size_t mlp = w.mlp_proj->size() / D;
    auto select_scale = [&](std::size_t col_off, const std::vector<float>& normed) {
        std::vector<float> o(normed.size());
        for (std::size_t s = 0; s < S; ++s) {
            const std::size_t row = tmask[s] ? 0 : 1;
            const float* m = mod.data() + row * (4 * D) + col_off;
            const float* x = normed.data() + s * D;
            float* p = o.data() + s * D;
            for (std::size_t d = 0; d < D; ++d) p[d] = x[d] * (1.0F + m[d]);
        }
        return o;
    };
    auto gate_of = [&](std::size_t col_off, std::size_t s) {
        const std::size_t row = tmask[s] ? 0 : 1;
        return mod.data() + row * (4 * D) + col_off;
    };
    qo::layernorm(xin, H.ln1, S, D, eps);
    H.mod1 = select_scale(0, H.ln1);
    qo::matmul_xwt(H.mod1, *w.to_q, H.q, S, D, D);
    qo::matmul_xwt(H.mod1, *w.to_k, H.k, S, D, D);
    qo::matmul_xwt(H.mod1, *w.to_v, H.v, S, D, D);
    qo::rmsnorm_head(H.q, S * heads, head_dim, *w.norm_q, eps);
    qo::rmsnorm_head(H.k, S * heads, head_dim, *w.norm_k, eps);
    qo::apply_rope_c64(H.q, S, heads, head_dim, rot);
    qo::apply_rope_c64(H.k, S, heads, head_dim, rot);

    const float qscale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    H.attn.assign(S * D, 0.0F);
    std::vector<float> scores(S), probs(S), acc(head_dim);
    for (std::size_t s = 0; s < S; ++s) {
        const bool is_text = (s < prefix_len);
        const std::size_t kv_hi = is_text ? prefix_len : S;
        for (std::size_t h = 0; h < heads; ++h) {
            const float* qs = H.q.data() + (s * heads + h) * head_dim;
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                const float* ks = H.k.data() + (kk * heads + h) * head_dim;
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) dot += (double)qs[d] * (double)ks[d];
                scores[kk] = (float)dot * qscale;
            }
            probs.assign(S, 0.0F);
            float maxv = -std::numeric_limits<float>::infinity();
            for (std::size_t kk = 0; kk < kv_hi; ++kk)
                if (!(is_text && kk > s) && scores[kk] > maxv) maxv = scores[kk];
            double sum = 0.0;
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (is_text && kk > s) continue;
                const double e = std::exp((double)scores[kk] - (double)maxv);
                probs[kk] = (float)e; sum += e;
            }
            const float inv = (float)(1.0 / sum);
            for (std::size_t kk = 0; kk < kv_hi; ++kk) probs[kk] *= inv;
            std::fill(acc.begin(), acc.end(), 0.0F);
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (probs[kk] == 0.0F) continue;
                const float* vs = H.v.data() + (kk * heads + h) * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d) acc[d] += probs[kk] * vs[d];
            }
            float* od = H.attn.data() + (s * heads + h) * head_dim;
            for (std::size_t d = 0; d < head_dim; ++d) od[d] = acc[d];
        }
    }
    qo::matmul_xwt(H.attn, *w.to_out, H.attn_out, S, D, D);
    H.x1.assign(xin.size(), 0.0F);
    for (std::size_t s = 0; s < S; ++s) {
        const float* g = gate_of(D, s);
        for (std::size_t d = 0; d < D; ++d) H.x1[s * D + d] = xin[s * D + d] + std::tanh(g[d]) * H.attn_out[s * D + d];
    }
    qo::layernorm(H.x1, H.ln2, S, D, eps);
    H.mod2 = select_scale(2 * D, H.ln2);
    qo::matmul_xwt(H.mod2, *w.mlp_proj, H.proj, S, mlp, D);
    qo::matmul_xwt(H.mod2, *w.mlp_gate, H.gate, S, mlp, D);
    H.hidden.resize(H.proj.size());
    for (std::size_t i = 0; i < H.hidden.size(); ++i) H.hidden[i] = qo::silu(H.gate[i]) * H.proj[i];
    qo::matmul_xwt(H.hidden, *w.mlp_out, H.mlp_out, S, D, mlp);
    H.out.assign(xin.size(), 0.0F);
    for (std::size_t s = 0; s < S; ++s) {
        const float* g = gate_of(3 * D, s);
        for (std::size_t d = 0; d < D; ++d) H.out[s * D + d] = H.x1[s * D + d] + std::tanh(g[d]) * H.mlp_out[s * D + d];
    }
}

static int run() {
    const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (env == nullptr || std::string(env).empty()) {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string oracle(env);
    const std::string T = qo::join(oracle, "tensors");
    const std::string WD = qo::join(oracle, "tensors_block_weights");

    qo::NpyArray block_in = qo::read_npy(qo::join(T, "block_in.npy"));
    qo::NpyArray block_out = qo::read_npy(qo::join(T, "block_out.npy"));
    qo::NpyArray modulation = qo::read_npy(qo::join(T, "block_modulation.npy"));
    qo::NpyArray rotary = qo::read_npy(qo::join(T, "block_rotary.npy"));
    qo::NpyArray target_mask = qo::read_npy(qo::join(T, "block_target_mask.npy"));

    qo::require(block_in.shape.size() == 3 && block_in.shape[0] == 1, "block_in not [1,S,D]");
    qo::require(block_out.shape == block_in.shape, "block_out shape mismatch");
    const std::size_t S = block_in.shape[1], D = block_in.shape[2];
    const std::size_t heads = 32, head_dim = D / heads;
    qo::require(head_dim == 128, "head_dim must be 128");
    qo::require(S <= 512, "S exceeds kernel local array");
    qo::require(head_dim <= 256, "head_dim exceeds kernel local array");
    qo::require(modulation.shape[0] == 2 && modulation.shape[1] == 4 * D, "modulation not [2,4D]");

    std::vector<std::uint8_t> tmask = target_mask.bools();
    std::size_t prefix_len = S;
    for (std::size_t i = 0; i < S; ++i) if (tmask[i]) { prefix_len = i; break; }
    qo::require(prefix_len < S, "no target tokens");

    auto loadW = [&](const std::string& name) { return qo::read_npy(qo::join(WD, name)).floats(); };
    const std::vector<float> w_q = loadW("blk0_attn_to_q_weight.npy");
    const std::vector<float> w_k = loadW("blk0_attn_to_k_weight.npy");
    const std::vector<float> w_v = loadW("blk0_attn_to_v_weight.npy");
    const std::vector<float> w_out = loadW("blk0_attn_to_out_0_weight.npy");
    const std::vector<float> w_nq = loadW("blk0_attn_norm_q_weight.npy");
    const std::vector<float> w_nk = loadW("blk0_attn_norm_k_weight.npy");
    const std::vector<float> w_mlp_proj = loadW("blk0_img_mlp_proj_weight.npy");
    const std::vector<float> w_mlp_gate = loadW("blk0_img_mlp_gate_layer_weight.npy");
    const std::vector<float> w_mlp_out = loadW("blk0_img_mlp_out_weight.npy");
    const std::size_t mlp = w_mlp_proj.size() / D;

    const std::vector<float> xin = block_in.floats();
    const std::vector<float> mod = modulation.floats();
    const std::vector<float> rot = rotary.floats();  // [S,64,2] interleaved
    const std::vector<float> ref = block_out.floats();
    const float eps = 1e-6F;

    // host reference
    qo::BlockWeights bw{&w_q, &w_k, &w_v, &w_out, &w_nq, &w_nk, &w_mlp_proj, &w_mlp_gate, &w_mlp_out};
    HostStages H;
    host_block_stages(xin, mod, rot, tmask, bw, S, D, heads, prefix_len, eps, H);
    std::printf("host_block_vs_blockout rel_fro=%.3e\n", qo::rel_fro(H.out, ref));

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    // device buffers
    const size_t sD = S * D, sMlp = S * mlp;
    float *dxin,*dln1,*dmod1,*dq,*dk,*dv,*dattn,*dattn_out,*dx1,*dln2,*dmod2,
          *dproj,*dgate,*dhidden,*dmlp_out,*dout;
    float *d_wq,*d_wk,*d_wv,*d_wout,*d_wnq,*d_wnk,*d_wproj,*d_wgate,*d_wmlpout,*d_mod,*d_rot;
    int* d_tmask;
    CUDA_CHECK(cudaMalloc(&dxin, sD * 4));
    CUDA_CHECK(cudaMalloc(&dln1, sD * 4)); CUDA_CHECK(cudaMalloc(&dmod1, sD * 4));
    CUDA_CHECK(cudaMalloc(&dq, sD * 4)); CUDA_CHECK(cudaMalloc(&dk, sD * 4)); CUDA_CHECK(cudaMalloc(&dv, sD * 4));
    CUDA_CHECK(cudaMalloc(&dattn, sD * 4)); CUDA_CHECK(cudaMalloc(&dattn_out, sD * 4));
    CUDA_CHECK(cudaMalloc(&dx1, sD * 4)); CUDA_CHECK(cudaMalloc(&dln2, sD * 4)); CUDA_CHECK(cudaMalloc(&dmod2, sD * 4));
    CUDA_CHECK(cudaMalloc(&dproj, sMlp * 4)); CUDA_CHECK(cudaMalloc(&dgate, sMlp * 4));
    CUDA_CHECK(cudaMalloc(&dhidden, sMlp * 4)); CUDA_CHECK(cudaMalloc(&dmlp_out, sD * 4));
    CUDA_CHECK(cudaMalloc(&dout, sD * 4));
    CUDA_CHECK(cudaMalloc(&d_wq, w_q.size() * 4)); CUDA_CHECK(cudaMalloc(&d_wk, w_k.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_wv, w_v.size() * 4)); CUDA_CHECK(cudaMalloc(&d_wout, w_out.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_wnq, w_nq.size() * 4)); CUDA_CHECK(cudaMalloc(&d_wnk, w_nk.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_wproj, w_mlp_proj.size() * 4)); CUDA_CHECK(cudaMalloc(&d_wgate, w_mlp_gate.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_wmlpout, w_mlp_out.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_mod, mod.size() * 4)); CUDA_CHECK(cudaMalloc(&d_rot, rot.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_tmask, S * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dxin, xin.data(), sD * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mod, mod.data(), mod.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rot, rot.data(), rot.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wq, w_q.data(), w_q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wk, w_k.data(), w_k.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wv, w_v.data(), w_v.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wout, w_out.data(), w_out.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wnq, w_nq.data(), w_nq.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wnk, w_nk.data(), w_nk.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wproj, w_mlp_proj.data(), w_mlp_proj.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgate, w_mlp_gate.data(), w_mlp_gate.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wmlpout, w_mlp_out.data(), w_mlp_out.size() * 4, cudaMemcpyHostToDevice));
    {
        std::vector<int> tm(S);
        for (std::size_t i = 0; i < S; ++i) tm[i] = tmask[i] ? 0 : 1;  // device row = tmask?0:1
        CUDA_CHECK(cudaMemcpy(d_tmask, tm.data(), S * sizeof(int), cudaMemcpyHostToDevice));
    }

    const dim3 mb(32, 8);
    auto matmul = [&](const float* A, const float* W, float* C, int M, int N, int K) {
        dim3 grid((N + mb.x - 1) / mb.x, (M + mb.y - 1) / mb.y);
        k_matmul_xwt<<<grid, mb>>>(A, W, C, M, N, K);
    };
    const int eblocks = (int)((sD + 255) / 256);
    const int mlpblocks = (int)((sMlp + 255) / 256);
    const int rope_threads = 128;
    const int rope_blocks = (int)((S * heads + rope_threads - 1) / rope_threads);

    k_layernorm<<<(int)S, 1>>>(dxin, dln1, (int)S, (int)D, eps);
    k_select_scale<<<eblocks, 256>>>(dln1, d_mod, d_tmask, dmod1, (int)S, (int)D, 0);
    matmul(dmod1, d_wq, dq, (int)S, (int)D, (int)D);
    matmul(dmod1, d_wk, dk, (int)S, (int)D, (int)D);
    matmul(dmod1, d_wv, dv, (int)S, (int)D, (int)D);
    k_rmsnorm_head<<<(int)(S * heads), 1>>>(dq, d_wnq, (int)(S * heads), (int)head_dim, eps);
    k_rmsnorm_head<<<(int)(S * heads), 1>>>(dk, d_wnk, (int)(S * heads), (int)head_dim, eps);
    k_rope<<<rope_blocks, rope_threads>>>(dq, d_rot, (int)S, (int)heads, (int)head_dim);
    k_rope<<<rope_blocks, rope_threads>>>(dk, d_rot, (int)S, (int)heads, (int)head_dim);
    k_attention<<<rope_blocks, rope_threads>>>(dq, dk, dv, dattn, (int)S, (int)heads, (int)head_dim, (int)prefix_len);
    matmul(dattn, d_wout, dattn_out, (int)S, (int)D, (int)D);
    k_gate_add<<<eblocks, 256>>>(dxin, d_mod, d_tmask, dattn_out, dx1, (int)S, (int)D, (int)D);
    k_layernorm<<<(int)S, 1>>>(dx1, dln2, (int)S, (int)D, eps);
    k_select_scale<<<eblocks, 256>>>(dln2, d_mod, d_tmask, dmod2, (int)S, (int)D, (int)(2 * D));
    matmul(dmod2, d_wproj, dproj, (int)S, (int)mlp, (int)D);
    matmul(dmod2, d_wgate, dgate, (int)S, (int)mlp, (int)D);
    k_silu_mul<<<mlpblocks, 256>>>(dgate, dproj, dhidden, (int)sMlp);
    matmul(dhidden, d_wmlpout, dmlp_out, (int)S, (int)D, (int)mlp);
    k_gate_add<<<eblocks, 256>>>(dx1, d_mod, d_tmask, dmlp_out, dout, (int)S, (int)D, (int)(3 * D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // download + per-stage compare vs host reference
    auto dl = [&](float* d, size_t n) { std::vector<float> h(n); cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost); return h; };
    struct Row { const char* name; float* d; size_t n; const std::vector<float>* ref; };
    Row rows[] = {
        {"ln1", dln1, sD, &H.ln1}, {"mod1", dmod1, sD, &H.mod1},
        {"q_rope", dq, sD, &H.q}, {"k_rope", dk, sD, &H.k}, {"v", dv, sD, &H.v},
        {"attn", dattn, sD, &H.attn}, {"attn_out", dattn_out, sD, &H.attn_out}, {"x1", dx1, sD, &H.x1},
        {"ln2", dln2, sD, &H.ln2}, {"mod2", dmod2, sD, &H.mod2},
        {"proj", dproj, sMlp, &H.proj}, {"gate", dgate, sMlp, &H.gate}, {"hidden", dhidden, sMlp, &H.hidden},
        {"mlp_out", dmlp_out, sD, &H.mlp_out}, {"block_out", dout, sD, &H.out},
    };
    bool ok = true;
    for (auto& r : rows) {
        std::vector<float> got = dl(r.d, r.n);
        const double e = qo::rel_fro(got, *r.ref);
        const bool s_ok = (e < 1e-3) && std::isfinite(e);
        ok = ok && s_ok;
        std::printf("stage %-9s rel_fro=%.3e %s\n", r.name, e, s_ok ? "OK" : "FAIL");
    }
    std::vector<float> got_out = dl(dout, sD);
    const double e_ref = qo::rel_fro(got_out, ref);
    const bool ref_ok = (e_ref < 1e-3) && std::isfinite(e_ref);
    ok = ok && ref_ok;
    std::printf("final vs block_out.npy rel_fro=%.3e %s\n", e_ref, ref_ok ? "OK" : "FAIL");
    std::fflush(stdout);

    // free
    cudaFree(dxin); cudaFree(dln1); cudaFree(dmod1); cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(dattn); cudaFree(dattn_out); cudaFree(dx1); cudaFree(dln2); cudaFree(dmod2);
    cudaFree(dproj); cudaFree(dgate); cudaFree(dhidden); cudaFree(dmlp_out); cudaFree(dout);
    cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_wout); cudaFree(d_wnq); cudaFree(d_wnk);
    cudaFree(d_wproj); cudaFree(d_wgate); cudaFree(d_wmlpout); cudaFree(d_mod); cudaFree(d_rot); cudaFree(d_tmask);

    std::printf(ok ? "IMAGE_BLOCK_CUDA_TEST: PASS\n" : "IMAGE_BLOCK_CUDA_TEST: FAIL\n");
    return ok ? 0 : 1;
}

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_BLOCK_CUDA_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
