// GPU depth qualification of the Qwen-Image-2.1 DiT: the full 32-block chain, every block's seven
// linear projections served as row-scaled FP8 E4M3 weights with BF16 activations (W8A16), through
// the production ops. This is the 32-layer composition gate for the FP8 decision: a single-block
// Frobenius pass (test_image_block_fp8_prod_cuda.cu) is necessary but not sufficient -- what matters
// over a stack is whether block l's FP8 error propagates into a bad regime for block l+1. Two
// measurements are taken per block, both streamed (load block l's real weights, quantize, run, free):
//
//   LOCAL      -- feed block l the FP32 reference input (block_in_all[l]); the output vs
//                 block_out_all[l] is the pure per-block FP8 quantization error on real weights at
//                 depth l (tests FP8 stability across all 32 real blocks; l=0 reproduces the
//                 single-block test).
//   ACCUMULATED-- feed block l the running FP8 hidden state; the output vs block_out_all[l] is the
//                 total error of the FP8 trajectory vs the FP32 trajectory at depth l (the depth
//                 curve / error-transfer gate).
//
// The reference trajectory (block_in_all / block_out_all / sample_full) was produced by the
// authoritative diffusers model (streaming single-block forward, image-spike/oracle/dump_full_depth.py)
// at tiny resolution (T=72) but with all 32 real blocks, D=4096, 32 heads, RoPE, AdaLN, and the
// block-causal attention. The acceptance tolerance is fixed a priori at 4.0e-2 and is applied to
// BOTH the per-block LOCAL error (at every depth) and the deepest ACCUMULATED error; it is never
// narrowed to pass. The full per-block curve is printed regardless so a divergent block is visible.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors_full_depth/*.npy and .../tensors_full_depth_weights/*.npy.
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
#include "ops/common/canonical_math.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qo = qioracle;
using infernix::Tensor;
using infernix::DType;
using infernix::Weight;
namespace ops = infernix::ops;

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
std::vector<float> represent(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}

bool g_alloc_ok = true;
#define CUDA_CHECK_NOMASK(call)                                              \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            g_alloc_ok = false;                                              \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__,     \
                        __LINE__, cudaGetErrorString(err_));                 \
        }                                                                    \
    } while (0)

// Row-scaled FP8 E4M3 [n,k] weight uploaded as one device buffer:
//   [n*k E4M3 codes][pad to 256][n BF16 row scales]. Identical to the single-block FP8 test.
struct FpWeight {
    int n = 0, k = 0;
    std::uint64_t scale_off = 0, total = 0;
    std::vector<float> recon;  // dequantized, for per-projection disclosure
    void* d = nullptr;
    Weight w{};

    FpWeight(const std::vector<float>& W, int rows, int cols) : n(rows), k(cols) {
        const std::uint64_t code_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
        scale_off = (code_bytes + 255ULL) & ~255ULL;
        total = scale_off + static_cast<std::uint64_t>(n) * 2ULL;
        std::vector<std::uint8_t> buf(total, 0);
        recon.assign(static_cast<std::size_t>(code_bytes), 0.0F);
        auto* base = buf.data();
        for (int i = 0; i < n; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
            float amax = 0.0F;
            for (int j = 0; j < k; ++j) amax = std::fmax(amax, std::fabs(W[row + j]));
            float scale = (amax > 0.0F) ? from_bf16(to_bf16(amax / 448.0F)) : 1.0F;
            if (!(scale > 0.0F)) scale = 1.0F;
            const std::uint16_t sh = to_bf16(scale);
            std::memcpy(base + scale_off + static_cast<std::size_t>(i) * 2, &sh, sizeof(sh));
            for (int j = 0; j < k; ++j) {
                const float x = W[row + j] / scale;
                const std::uint8_t mag = ops::canon::e4m3_rn_satfinite(std::fabs(x));
                const std::uint8_t sgn = std::signbit(x) ? 0x80U : 0x00U;
                const std::uint8_t code = static_cast<std::uint8_t>(mag | sgn);
                buf[row + static_cast<std::size_t>(j)] = code;
                const float val = ops::canon::e4m3_value(code & 0x7FU) * (sgn ? -1.0F : 1.0F) * scale;
                recon[row + static_cast<std::size_t>(j)] = val;
            }
        }
        CUDA_CHECK_NOMASK(cudaMalloc(&d, static_cast<std::size_t>(total)));
        CUDA_CHECK_NOMASK(cudaMemcpy(d, base, static_cast<std::size_t>(total), cudaMemcpyHostToDevice));

        w = Weight{};
        w.payload         = d;
        w.payload_bytes   = total;
        w.qtype           = infernix::QType::FP8_E4M3FN_ROW_BF16;
        w.shape[0] = n; w.shape[1] = k; w.shape[2] = 1; w.shape[3] = 1;
        w.padded_shape[0] = n; w.padded_shape[1] = k; w.padded_shape[2] = 1; w.padded_shape[3] = 1;
        w.ndim = 2;
        w.qdata         = d;
        w.qhigh         = nullptr;
        w.high_plane_bytes = 0;
        w.scales        = static_cast<std::uint8_t*>(d) + scale_off;
        w.n = n; w.k = k; w.group = k; w.group_size = static_cast<std::uint32_t>(k);
        w.layout = infernix::QuantLayout::RowScale;
        w.scale_dtype = DType::BF16;
        w.scale_ne[0] = n; w.scale_ne[1] = 1; w.scale_ne[2] = 1; w.scale_ne[3] = 1;
        const std::int64_t ss = static_cast<std::int64_t>(n) * 2;
        w.scale_nb[0] = 2; w.scale_nb[1] = ss; w.scale_nb[2] = ss; w.scale_nb[3] = ss;
    }
    ~FpWeight() { if (d) cudaFree(d); }
    FpWeight(const FpWeight&) = delete;
    FpWeight& operator=(const FpWeight&) = delete;
};

void* upload(const std::vector<float>& v) {
    const auto bits = bits_of(v);
    void* d = nullptr;
    cudaMalloc(&d, bits.size() * sizeof(std::uint16_t));
    cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}
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

// Per-block FP8 weights (pointers to locally-constructed FpWeight) + per-block norm_q/norm_k (BF16).
struct BWeights {
    const FpWeight* Wq; const FpWeight* Wk; const FpWeight* Wv; const FpWeight* Wout;
    const FpWeight* Wproj; const FpWeight* Wgate; const FpWeight* Wmlpout;
    void* d_nq; void* d_nk;
};
// Shared across all blocks (identical for every block per the model forward).
struct Shared { void* dmod; void* drot; void* d_ones; void* d_zeros; };
// Internal scratch for one block run (allocated once, reused every block).
struct WS {
    void *dln1, *dmod1, *dq, *dk, *dv, *dqr, *dkr, *dattn, *dattn_out, *dx1, *dln2, *dmod2,
         *dproj, *dgate, *dhidden, *dmlp_out, *dout, *dloc_in, *drun;
};

// One DiT block forward through production ops (FP8 linears, BF16 everything else). Identical op
// sequence and view shapes to test_image_block_fp8_prod_cuda.cu; din/dout are the running hidden.
void run_block_fp8(void* din, void* dout, const BWeights& B, const Shared& S, WS& ws,
                   int Si, int Di, int Hi, int Mi, int tp, float eps) {
    Tensor nq = view(B.d_nq, 128, 1, 1), nk = view(B.d_nk, 128, 1, 1);
    Tensor onev = view(S.d_ones, Di, 1, 1), zerov = view(S.d_zeros, Di, 1, 1);
    Tensor modv = view(S.dmod, 4 * Di, 2);   // [4D,2] table, row via text_prefix (s<tp -> text row)
    Tensor rotv = view(S.drot, 128, 1, Si);  // [128,1,S] interleaved cis pairs

    Tensor x_in = view(din, Di, Si);
    Tensor ln1 = view(ws.dln1, Di, Si), mod1 = view(ws.dmod1, Di, Si);
    ops::layer_norm(x_in, onev, zerov, eps, ln1, 0);
    ops::adaln_scale(ln1, modv, 0, tp, mod1, 0);

    Tensor q = view(ws.dq, Di, Si), k = view(ws.dk, Di, Si), v = view(ws.dv, Di, Si);
    ops::linear(mod1, B.Wq->w, q, 0);
    ops::linear(mod1, B.Wk->w, k, 0);
    ops::linear(mod1, B.Wv->w, v, 0);

    Tensor qh = view(ws.dq, 128, Hi * Si), kh = view(ws.dk, 128, Hi * Si);
    Tensor qhr = view(ws.dqr, 128, Hi * Si), khr = view(ws.dkr, 128, Hi * Si);
    ops::rmsnorm(qh, nq, eps, /*unit_offset=*/false, qhr, 0);
    ops::rmsnorm(kh, nk, eps, /*unit_offset=*/false, khr, 0);
    Tensor q3 = view(ws.dqr, 128, Hi, Si), k3 = view(ws.dkr, 128, Hi, Si), v3 = view(ws.dv, 128, Hi, Si);
    ops::rope_c64(rotv, q3, 0);
    ops::rope_c64(rotv, k3, 0);

    Tensor attn3 = view(ws.dattn, 128, Hi, Si);
    ops::attn_dit_joint(q3, k3, v3, tp, attn3, 0);
    Tensor attn2 = view(ws.dattn, Di, Si), attn_out = view(ws.dattn_out, Di, Si);
    ops::linear(attn2, B.Wout->w, attn_out, 0);

    Tensor x1 = view(ws.dx1, Di, Si);
    ops::adaln_gate_add(x_in, attn_out, modv, Di, tp, x1, 0);
    Tensor ln2 = view(ws.dln2, Di, Si), mod2 = view(ws.dmod2, Di, Si);
    ops::layer_norm(x1, onev, zerov, eps, ln2, 0);
    ops::adaln_scale(ln2, modv, 2 * Di, tp, mod2, 0);

    Tensor proj = view(ws.dproj, Mi, Si), gate = view(ws.dgate, Mi, Si), hidden = view(ws.dhidden, Mi, Si);
    ops::linear(mod2, B.Wproj->w, proj, 0);
    ops::linear(mod2, B.Wgate->w, gate, 0);
    ops::silu_mul(gate, proj, hidden, 0);
    Tensor mlp_out = view(ws.dmlp_out, Di, Si);
    ops::linear(hidden, B.Wmlpout->w, mlp_out, 0);
    Tensor out = view(dout, Di, Si);
    ops::adaln_gate_add(x1, mlp_out, modv, 3 * Di, tp, out, 0);
}

std::string fwl(int l, const char* role) {
    std::string s = "blk"; s += std::to_string(l); s += "_"; s += role; s += ".npy";
    return s;
}

int run() {
    const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (env == nullptr || std::string(env).empty()) {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string oracle(env);
    const std::string TD = qo::join(oracle, "tensors_full_depth");
    const std::string WD = qo::join(oracle, "tensors_full_depth_weights");

    qo::NpyArray in_all = qo::read_npy(qo::join(TD, "block_in_all.npy"));
    qo::NpyArray out_all = qo::read_npy(qo::join(TD, "block_out_all.npy"));
    qo::NpyArray modulation = qo::read_npy(qo::join(TD, "block_modulation.npy"));
    qo::NpyArray rotary = qo::read_npy(qo::join(TD, "block_rotary.npy"));
    qo::NpyArray target_mask = qo::read_npy(qo::join(TD, "block_target_mask.npy"));

    qo::require(in_all.shape.size() == 4 && in_all.shape[1] == 1, "block_in_all not [L,1,S,D]");
    qo::require(out_all.shape == in_all.shape, "block_in_all/block_out_all shape mismatch");
    const std::size_t L = in_all.shape[0];
    const std::size_t S = in_all.shape[2], D = in_all.shape[3];
    const std::size_t heads = 32, head_dim = D / heads;
    qo::require(head_dim == 128, "head_dim must be 128");
    qo::require(modulation.shape[0] == 2 && modulation.shape[1] == 4 * D, "modulation not [2,4D]");

    const std::vector<std::uint8_t> tmask = target_mask.bools();
    std::size_t prefix_len = S;
    for (std::size_t i = 0; i < S; ++i) if (tmask[i]) { prefix_len = i; break; }
    qo::require(prefix_len < S, "no target tokens");

    const std::vector<float> in_flat = in_all.floats();    // [L, S, D] contiguous (b=1)
    const std::vector<float> out_flat = out_all.floats();
    const std::size_t sD = S * D;
    qo::require(in_flat.size() == L * sD, "block_in_all flat size mismatch");

    const std::vector<float> mod = represent(modulation.floats());
    const std::vector<float> rot = represent(rotary.floats());  // [72,64] complex64 -> 2*S*64 reals

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    const std::vector<float> ones(D, 1.0F), zeros(D, 0.0F);
    const Shared Sd{upload(mod), upload(rot), upload(ones), upload(zeros)};

    const std::size_t mlp0 = 12288;  // mlp dim per block (constant across blocks); re-checked per block
    const std::size_t sMlp = S * mlp0;
    WS ws{};
    void** wp[] = {&ws.dln1, &ws.dmod1, &ws.dq, &ws.dk, &ws.dv, &ws.dqr, &ws.dkr,
                  &ws.dattn, &ws.dattn_out, &ws.dx1, &ws.dln2, &ws.dmod2,
                  &ws.dmlp_out, &ws.dout, &ws.dloc_in, &ws.drun};
    for (void** p : wp) { *p = nullptr; CUDA_CHECK(cudaMalloc(p, sD * 2)); }
    for (void** p : {&ws.dproj, &ws.dgate, &ws.dhidden}) { *p = nullptr; CUDA_CHECK(cudaMalloc(p, sMlp * 2)); }

    const int Si = (int)S, Di = (int)D, Hi = (int)heads, tp = (int)prefix_len;
    const float eps = 1e-6F;

    // seed the accumulated running hidden with block0's reference input (BF16).
    {
        std::vector<float> seed(in_flat.begin(), in_flat.begin() + sD);
        const auto bits = bits_of(seed);
        CUDA_CHECK(cudaMemcpy(ws.drun, bits.data(), sD * 2, cudaMemcpyHostToDevice));
    }

    const double tol = 4e-2;  // fixed a priori; applied to per-block LOCAL (every depth) and deepest ACCUMULATED.
    std::vector<double> local_err(L, 0.0), accum_err(L, 0.0);
    double local_max = 0.0, accum_max = 0.0, local_worst_l = 0.0, accum_worst_l = 0.0;

    for (std::size_t l = 0; l < L; ++l) {
        auto loadW = [&](const char* role) {
            return qo::read_npy(qo::join(WD, fwl((int)l, role))).floats();
        };
        const std::vector<float> w_q = represent(loadW("attn_to_q_weight"));
        const std::vector<float> w_k = represent(loadW("attn_to_k_weight"));
        const std::vector<float> w_v = represent(loadW("attn_to_v_weight"));
        const std::vector<float> w_out = represent(loadW("attn_to_out_0_weight"));
        const std::vector<float> w_nq = represent(loadW("attn_norm_q_weight"));
        const std::vector<float> w_nk = represent(loadW("attn_norm_k_weight"));
        const std::vector<float> w_proj = represent(loadW("img_mlp_proj_weight"));
        const std::vector<float> w_gate = represent(loadW("img_mlp_gate_layer_weight"));
        const std::vector<float> w_mlpout = represent(loadW("img_mlp_out_weight"));
        const std::size_t mlp = w_proj.size() / D;
        qo::require(mlp == mlp0, "mlp dim changed across blocks");

        const FpWeight Wq(w_q, Di, Di), Wk(w_k, Di, Di), Wv(w_v, Di, Di);
        const FpWeight Wout(w_out, Di, Di);
        const FpWeight Wproj(w_proj, (int)mlp, Di), Wgate(w_gate, (int)mlp, Di);
        const FpWeight Wmlpout(w_mlpout, Di, (int)mlp);
        if (!g_alloc_ok) { std::printf("ALLOC_FAILURE block %zu\n", l); return 1; }
        const std::vector<float> wnq = w_nq, wnk = w_nk;
        void* d_nq = upload(wnq), *d_nk = upload(wnk);
        const BWeights B{&Wq, &Wk, &Wv, &Wout, &Wproj, &Wgate, &Wmlpout, d_nq, d_nk};

        const std::vector<float> ref_slice(out_flat.begin() + l * sD, out_flat.begin() + (l + 1) * sD);

        // LOCAL: block l on the FP32 reference input block_in_all[l].
        {
            std::vector<float> sin(in_flat.begin() + l * sD, in_flat.begin() + (l + 1) * sD);
            const auto bits = bits_of(sin);
            CUDA_CHECK(cudaMemcpy(ws.dloc_in, bits.data(), sD * 2, cudaMemcpyHostToDevice));
            run_block_fp8(ws.dloc_in, ws.dout, B, Sd, ws, Si, Di, Hi, (int)mlp, tp, eps);
            const std::vector<float> got = download(ws.dout, sD);
            local_err[l] = qo::rel_fro(got, ref_slice);
            if (local_err[l] > local_max) { local_max = local_err[l]; local_worst_l = (double)l; }
        }
        // ACCUMULATED: block l on the running FP8 hidden state.
        {
            run_block_fp8(ws.drun, ws.dout, B, Sd, ws, Si, Di, Hi, (int)mlp, tp, eps);
            const std::vector<float> got = download(ws.dout, sD);
            accum_err[l] = qo::rel_fro(got, ref_slice);
            if (accum_err[l] > accum_max) { accum_max = accum_err[l]; accum_worst_l = (double)l; }
            const auto bits = bits_of(got);
            CUDA_CHECK(cudaMemcpy(ws.drun, bits.data(), sD * 2, cudaMemcpyHostToDevice));
        }
        cudaFree(d_nq); cudaFree(d_nk);
        CUDA_CHECK(cudaGetLastError());
    }

    std::printf("depth curve (LOCAL = per-block FP8 on FP32 input; ACCUM = FP8 trajectory) :\n");
    for (std::size_t l = 0; l < L; ++l) {
        std::printf("  l=%2zu  local=%.3e %s   accum=%.3e %s\n",
                    l, local_err[l], (local_err[l] < tol ? "OK" : "OVER"),
                    accum_err[l], (accum_err[l] < tol ? "OK" : "OVER"));
    }
    std::printf("LOCAL  max=%.3e @l=%zu (l=0 self-check vs single-block ~2.7e-2: %.3e)\n",
                local_max, (std::size_t)local_worst_l, local_err[0]);
    std::printf("ACCUM  max=%.3e @l=%zu (deepest trajectory error)\n", accum_max, (std::size_t)accum_worst_l);

    bool ok = std::isfinite(local_max) && std::isfinite(accum_max);
    for (std::size_t l = 0; l < L; ++l)
        ok = ok && std::isfinite(local_err[l]) && std::isfinite(accum_err[l]);
    ok = ok && (local_max < tol) && (accum_max < tol);
    std::printf("gate tol=%.0e: local_max<tol=%s  accum_max<tol=%s\n",
                tol, (local_max < tol ? "yes" : "NO"), (accum_max < tol ? "yes" : "NO"));

    for (void** p : wp) if (*p) cudaFree(*p);
    for (void** p : {&ws.dproj, &ws.dgate, &ws.dhidden}) if (*p) cudaFree(*p);
    cudaFree(Sd.dmod); cudaFree(Sd.drot); cudaFree(Sd.d_ones); cudaFree(Sd.d_zeros);

    std::printf(ok ? "IMAGE_DEPTH_FP8_PROD_TEST: PASS\n" : "IMAGE_DEPTH_FP8_PROD_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_DEPTH_FP8_PROD_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
