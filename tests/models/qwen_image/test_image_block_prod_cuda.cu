// GPU qualification of one Qwen-Image-2.1 DiT block composed from the PRODUCTION ops (repo op layer),
// not the self-contained naive scaffolding in test_image_block_cuda.cu. Every device transformation is
// a repo op on the op-development contract:
//   layer_norm(affine=False via ones/zeros)  ->  adaln_scale(0)
//   ops::linear (BF16 A16, general route) to_q/to_k/to_v
//   rmsnorm(unit_offset=false) per head      ->  rope_c64(single form)
//   attn_dit_joint(block-causal)             ->  ops::linear to_out
//   adaln_gate_add(D)                        ->  layer_norm -> adaln_scale(2D)
//   ops::linear mlp_proj/mlp_gate            ->  silu_mul -> ops::linear mlp_out
//   adaln_gate_add(3D)
//
// Qualification oracle: qioracle::block_forward (naive FP64-accumulation float32 math) evaluated on the
// SAME BF16-represented operands the device receives, so the only difference is the production kernels'
// compute/storage rounding -- not weight representation. Every stage is compared per-stage so a single
// blackout run localizes a divergent op (the agent is offline during the run). The naive FP32 block in
// test_image_block_cuda.cu already proved this composition matches block_out.npy at 1.0e-06; this test
// proves the same composition when each stage is a production op.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors/*.npy and .../tensors_block_weights/*.npy.
// Returns 77 (CTest skip) when the oracle data is absent; returns 2 (runnable) when no CUDA device.

#include "infernix/ops/linear.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/rope_c64.h"
#include "infernix/ops/attn_dit_joint.h"
#include "infernix/ops/dit_adaln.h"

#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qo = qioracle;
using infernix::Tensor;
using infernix::DType;
using infernix::Weight;
namespace ops = infernix::ops;  // the production op layer (infernix::ops)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__,   \
                        __LINE__, cudaGetErrorString(err_));               \
            std::fflush(stdout);                                           \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

// Host BF16 codec (no device header needed for the reference path).
std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((x >> 16) | 0x0040U);
    const std::uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}
float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}
std::vector<std::uint16_t> bits_of(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}
// The float value a BF16 operand actually represents.
std::vector<float> represent(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}

// A contiguous BF16 [n,k] weight uploaded to the device, wrapped as a registered Weight.
struct BfWeight {
    int n = 0, k = 0;
    std::vector<std::uint16_t> bits;
    void* d = nullptr;
    Weight w{};

    BfWeight(const std::vector<float>& floats, int rows, int cols) : n(rows), k(cols) {
        bits = bits_of(floats);
        cudaMalloc(&d, bits.size() * sizeof(std::uint16_t));
        cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
        w           = Weight{};
        w.payload   = d;
        w.payload_bytes = bits.size() * sizeof(std::uint16_t);
        w.qtype     = infernix::QType::BF16;
        w.shape[0]  = n;   w.shape[1] = k;
        w.padded_shape[0] = n; w.padded_shape[1] = k;
        w.ndim      = 2;
        w.qdata     = d;
        w.n         = n;   w.k = k;
        w.layout    = infernix::QuantLayout::Contiguous;
    }
    ~BfWeight() { if (d) cudaFree(d); }
    BfWeight(const BfWeight&) = delete;
    BfWeight& operator=(const BfWeight&) = delete;
};

// Upload a BF16 vector to the device; caller frees.
void* upload(const std::vector<float>& v) {
    const auto bits = bits_of(v);
    void* d = nullptr;
    cudaMalloc(&d, bits.size() * sizeof(std::uint16_t));
    cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}

// Download a BF16 device buffer and decode to float.
std::vector<float> download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}

Tensor view(void* p, int a, int b, int c = 1, int dd = 1) {
    return Tensor(p, DType::BF16, {a, b, c, dd});
}

// Host reference stage set (mirrors qioracle::block_forward, keeping every intermediate), evaluated on
// the BF16-represented operands.
struct HostStages {
    std::vector<float> ln1, mod1, q, k, v, attn, attn_out, x1, ln2, mod2, proj, gate, hidden, mlp_out, out;
};

void host_block_stages(const std::vector<float>& xin, const std::vector<float>& mod,
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

int run() {
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
    qo::require(modulation.shape[0] == 2 && modulation.shape[1] == 4 * D, "modulation not [2,4D]");

    std::vector<std::uint8_t> tmask = target_mask.bools();
    std::size_t prefix_len = S;
    for (std::size_t i = 0; i < S; ++i) if (tmask[i]) { prefix_len = i; break; }
    qo::require(prefix_len < S, "no target tokens");

    auto loadW = [&](const std::string& name) { return qo::read_npy(qo::join(WD, name)).floats(); };
    const std::vector<float> w_q = represent(loadW("blk0_attn_to_q_weight.npy"));
    const std::vector<float> w_k = represent(loadW("blk0_attn_to_k_weight.npy"));
    const std::vector<float> w_v = represent(loadW("blk0_attn_to_v_weight.npy"));
    const std::vector<float> w_out = represent(loadW("blk0_attn_to_out_0_weight.npy"));
    const std::vector<float> w_nq = represent(loadW("blk0_attn_norm_q_weight.npy"));
    const std::vector<float> w_nk = represent(loadW("blk0_attn_norm_k_weight.npy"));
    const std::vector<float> w_mlp_proj = represent(loadW("blk0_img_mlp_proj_weight.npy"));
    const std::vector<float> w_mlp_gate = represent(loadW("blk0_img_mlp_gate_layer_weight.npy"));
    const std::vector<float> w_mlp_out = represent(loadW("blk0_img_mlp_out_weight.npy"));
    const std::size_t mlp = w_mlp_proj.size() / D;

    const std::vector<float> xin = represent(block_in.floats());
    const std::vector<float> mod = represent(modulation.floats());
    const std::vector<float> rot = represent(rotary.floats());  // [S,64,2] interleaved
    const std::vector<float> ref = block_out.floats();          // diffusers FP32 reference
    const float eps = 1e-6F;

    qo::BlockWeights bw{&w_q, &w_k, &w_v, &w_out, &w_nq, &w_nk, &w_mlp_proj, &w_mlp_gate, &w_mlp_out};
    HostStages H;
    host_block_stages(xin, mod, rot, tmask, bw, S, D, heads, prefix_len, eps, H);
    std::printf("oracle_vs_block_out rel_fro=%.3e\n", qo::rel_fro(H.out, ref));

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    // Device weights (registered BF16 [n,k] routes) and constant norm weight/bias vectors.
    const BfWeight Wq(w_q, (int)D, (int)D), Wk(w_k, (int)D, (int)D), Wv(w_v, (int)D, (int)D);
    const BfWeight Wout(w_out, (int)D, (int)D);
    const BfWeight Wproj(w_mlp_proj, (int)mlp, (int)D), Wgate(w_mlp_gate, (int)mlp, (int)D);
    const BfWeight Wmlpout(w_mlp_out, (int)D, (int)mlp);
    void* d_nq = upload(w_nq);
    void* d_nk = upload(w_nk);
    const std::vector<float> ones(D, 1.0F), zeros(D, 0.0F);
    void* d_ones = upload(ones), *d_zeros = upload(zeros);

    const std::size_t sD = S * D, sMlp = S * mlp;
    void *dxin = upload(xin), *dmod = upload(mod), *drot = upload(rot);
    void *dln1, *dmod1, *dq, *dk, *dv, *dqr, *dkr, *dattn, *dattn_out, *dx1, *dln2, *dmod2,
         *dproj, *dgate, *dhidden, *dmlp_out, *dout;
    CUDA_CHECK(cudaMalloc(&dln1, sD * 2)); CUDA_CHECK(cudaMalloc(&dmod1, sD * 2));
    CUDA_CHECK(cudaMalloc(&dq, sD * 2)); CUDA_CHECK(cudaMalloc(&dk, sD * 2)); CUDA_CHECK(cudaMalloc(&dv, sD * 2));
    CUDA_CHECK(cudaMalloc(&dqr, sD * 2)); CUDA_CHECK(cudaMalloc(&dkr, sD * 2));
    CUDA_CHECK(cudaMalloc(&dattn, sD * 2)); CUDA_CHECK(cudaMalloc(&dattn_out, sD * 2));
    CUDA_CHECK(cudaMalloc(&dx1, sD * 2)); CUDA_CHECK(cudaMalloc(&dln2, sD * 2)); CUDA_CHECK(cudaMalloc(&dmod2, sD * 2));
    CUDA_CHECK(cudaMalloc(&dproj, sMlp * 2)); CUDA_CHECK(cudaMalloc(&dgate, sMlp * 2));
    CUDA_CHECK(cudaMalloc(&dhidden, sMlp * 2)); CUDA_CHECK(cudaMalloc(&dmlp_out, sD * 2));
    CUDA_CHECK(cudaMalloc(&dout, sD * 2));

    const int Si = (int)S, Di = (int)D, Hi = (int)heads, Mi = (int)mlp;
    const int tp = (int)prefix_len;
    Tensor nq = view(d_nq, (int)head_dim, 1, 1), nk = view(d_nk, (int)head_dim, 1, 1);
    Tensor onev = view(d_ones, Di, 1, 1), zerov = view(d_zeros, Di, 1, 1);
    Tensor modv = view(dmod, 4 * Di, 2);   // [4D,2] table, row via text_prefix (s<tp -> row1=text)
    Tensor rotv = view(drot, 128, 1, Si);  // [128,1,S] interleaved cis pairs

    // ---- block forward through production ops ----
    Tensor x_in = view(dxin, Di, Si);
    Tensor ln1 = view(dln1, Di, Si), mod1 = view(dmod1, Di, Si);
    ops::layer_norm(x_in, onev, zerov, eps, ln1, 0);
    ops::adaln_scale(ln1, modv, 0, tp, mod1, 0);

    Tensor q = view(dq, Di, Si), k = view(dk, Di, Si), v = view(dv, Di, Si);
    ops::linear(mod1, Wq.w, q, 0);
    ops::linear(mod1, Wk.w, k, 0);
    ops::linear(mod1, Wv.w, v, 0);

    // Per-head RMSNorm over the head_dim fastest axis, then complex RoPE, both on [128,32,S] views.
    Tensor qh = view(dq, 128, Hi * Si), kh = view(dk, 128, Hi * Si);
    Tensor qhr = view(dqr, 128, Hi * Si), khr = view(dkr, 128, Hi * Si);
    ops::rmsnorm(qh, nq, eps, /*unit_offset=*/false, qhr, 0);
    ops::rmsnorm(kh, nk, eps, /*unit_offset=*/false, khr, 0);
    Tensor q3 = view(dqr, 128, Hi, Si), k3 = view(dkr, 128, Hi, Si), v3 = view(dv, 128, Hi, Si);
    ops::rope_c64(rotv, q3, 0);
    ops::rope_c64(rotv, k3, 0);

    Tensor attn3 = view(dattn, 128, Hi, Si);
    ops::attn_dit_joint(q3, k3, v3, tp, attn3, 0);
    Tensor attn2 = view(dattn, Di, Si), attn_out = view(dattn_out, Di, Si);
    ops::linear(attn2, Wout.w, attn_out, 0);

    Tensor x1 = view(dx1, Di, Si);
    ops::adaln_gate_add(x_in, attn_out, modv, (int)D, tp, x1, 0);
    Tensor ln2 = view(dln2, Di, Si), mod2 = view(dmod2, Di, Si);
    ops::layer_norm(x1, onev, zerov, eps, ln2, 0);
    ops::adaln_scale(ln2, modv, (int)(2 * D), tp, mod2, 0);

    Tensor proj = view(dproj, Mi, Si), gate = view(dgate, Mi, Si), hidden = view(dhidden, Mi, Si);
    ops::linear(mod2, Wproj.w, proj, 0);
    ops::linear(mod2, Wgate.w, gate, 0);
    ops::silu_mul(gate, proj, hidden, 0);
    Tensor mlp_out = view(dmlp_out, Di, Si);
    ops::linear(hidden, Wmlpout.w, mlp_out, 0);
    Tensor out = view(dout, Di, Si);
    ops::adaln_gate_add(x1, mlp_out, modv, (int)(3 * D), tp, out, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- per-stage qualification vs the BF16-represented oracle ----
    struct Row { const char* name; void* d; size_t n; const std::vector<float>* ref; };
    Row rows[] = {
        {"ln1", dln1, sD, &H.ln1}, {"mod1", dmod1, sD, &H.mod1},
        {"q_rope", dqr, sD, &H.q}, {"k_rope", dkr, sD, &H.k}, {"v", dv, sD, &H.v},
        {"attn", dattn, sD, &H.attn}, {"attn_out", dattn_out, sD, &H.attn_out}, {"x1", dx1, sD, &H.x1},
        {"ln2", dln2, sD, &H.ln2}, {"mod2", dmod2, sD, &H.mod2},
        {"proj", dproj, sMlp, &H.proj}, {"gate", dgate, sMlp, &H.gate}, {"hidden", dhidden, sMlp, &H.hidden},
        {"mlp_out", dmlp_out, sD, &H.mlp_out}, {"block_out", dout, sD, &H.out},
    };
    const double tol = 2e-2;  // BF16 compute through 6 matmuls + attention + LN + rope + silu + gates.
    bool ok = true;
    for (auto& r : rows) {
        std::vector<float> got = download(r.d, r.n);
        const double e = qo::rel_fro(got, *r.ref);
        const bool s_ok = (e < tol) && std::isfinite(e);
        ok = ok && s_ok;
        std::printf("stage %-9s rel_fro=%.3e %s\n", r.name, e, s_ok ? "OK" : "FAIL");
    }
    std::vector<float> got_out = download(dout, sD);
    const double e_repr = qo::rel_fro(got_out, H.out);  // production vs BF16-represented oracle
    const double e_ref = qo::rel_fro(got_out, ref);     // production vs diffusers FP32 block_out.npy
    std::printf("final vs BF16-oracle rel_fro=%.3e (tol %.0e)\n", e_repr, tol);
    std::printf("final vs block_out.npy rel_fro=%.3e\n", e_ref);
    std::fflush(stdout);

    cudaFree(d_nq); cudaFree(d_nk); cudaFree(d_ones); cudaFree(d_zeros);
    cudaFree(dxin); cudaFree(dmod); cudaFree(drot);
    cudaFree(dln1); cudaFree(dmod1); cudaFree(dq); cudaFree(dk); cudaFree(dv);
    cudaFree(dqr); cudaFree(dkr); cudaFree(dattn); cudaFree(dattn_out);
    cudaFree(dx1); cudaFree(dln2); cudaFree(dmod2); cudaFree(dproj); cudaFree(dgate);
    cudaFree(dhidden); cudaFree(dmlp_out); cudaFree(dout);

    std::printf(ok ? "IMAGE_BLOCK_PROD_TEST: PASS\n" : "IMAGE_BLOCK_PROD_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_BLOCK_PROD_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
