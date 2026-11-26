// GPU qualification of the Qwen-Image-2.1 DiT TOP LEVEL composed from the PRODUCTION ops (repo op
// layer), i.e. everything on the outside of the transformer blocks that the block/depth tests already
// qualify:
//   img_in          : ops::linear (BF16 A16, general route)          [4096,64]
//   txt_in          : rmsnorm(unit_offset=true = ZeroCenterRMSNorm)
//                     -> ops::linear in_layer -> gelu(Tanh) -> ops::linear out_layer
//   timestep embed  : sinusoidal (host) -> ops::linear linear_1 -> silu -> ops::linear linear_2
//   global mod      : silu(temb) -> ops::linear modulation.1 [16384,4096]
//   norm_out        : ops::linear norm_out.linear (on SiLU(temb)) -> layer_norm(affine=False)
//                     -> adaln_scale (scale-only AdaLayerNormContinuous, row-select via target mask)
//   proj_out        : ops::linear [64,4096] -> sample (velocity prediction)
//
// The transformer blocks are NOT re-run here: the norm_out/proj_out tail is fed the oracle block-0
// output (block_out.npy), and the block itself is independently qualified (test_image_block_prod_cuda
// and the full-depth FP8 test). So this isolates and qualifies the top-level production ops.
//
// Qualification oracle: the naive FP32 (double-accumulation) qo:: primitives from
// qwen_image_oracle_ref.h evaluated on the SAME BF16-represented operands the device receives, so the
// only difference is the production kernels' compute/storage rounding -- not weight representation.
// Every stage is compared per-stage vs the BF16-represented host oracle (gated) and the diffusers FP32
// .npy (reported) so a single blackout run localizes a divergent op.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors/*.npy and .../tensors_toplevel_weights/*.npy.
// Returns 77 (CTest skip) when the oracle data is absent; returns 2 (runnable) when no CUDA device.

#include "infernix/ops/linear.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/gelu.h"
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

// Host BF16 codec (matches the block prod test).
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

int run() {
    const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (env == nullptr || std::string(env).empty()) {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string oracle(env);
    const std::string T  = qo::join(oracle, "tensors");
    const std::string TW = qo::join(oracle, "tensors_toplevel_weights");

    auto F  = [&](const std::string& dir, const std::string& name) { return qo::read_npy(qo::join(dir, name)).floats(); };
    auto LW = [&](const std::string& name) { return qo::read_npy(qo::join(TW, name)).floats(); };

    // ---- constants (from tensors/manifest.json) ----
    const std::size_t D = 4096, heads = 32;
    const std::size_t text_prefix = 8, H = 8, W = 8, img_tokens = 64;
    const std::size_t S = text_prefix + H * W;  // 72
    const std::size_t td = 256;
    const double time_factor = 1000.0;
    const float eps = 1e-6F;

    // ---- inputs (BF16-represented) ----
    const std::vector<float> in_hidden = represent(F(T, "in_hidden_states.npy"));        // [64,64]
    const std::vector<float> in_enc    = represent(F(T, "in_encoder_hidden_states.npy")); // [8,4096]
    const std::vector<float> in_t_repr = represent(F(T, "in_timestep.npy"));             // [1]

    // ---- toplevel weights (BF16-represented) ----
    const std::vector<float> w_img_in = represent(LW("toplevel_img_in_weight.npy"));                    // [4096,64]
    const std::vector<float> w_tnorm  = represent(LW("toplevel_txt_in_text_norm_weight.npy"));           // [4096]
    const std::vector<float> w_in     = represent(LW("toplevel_txt_in_in_layer_weight.npy"));            // [4096,4096]
    const std::vector<float> w_outl   = represent(LW("toplevel_txt_in_out_layer_weight.npy"));           // [4096,4096]
    const std::vector<float> w_tl1    = represent(LW("toplevel_time_text_embed_timestep_embedder_linear_1_weight.npy")); // [4096,256]
    const std::vector<float> w_tl2    = represent(LW("toplevel_time_text_embed_timestep_embedder_linear_2_weight.npy")); // [4096,4096]
    const std::vector<float> w_mod    = represent(LW("toplevel_modulation_1_weight.npy"));               // [16384,4096]
    const std::vector<float> w_nolin  = represent(LW("toplevel_norm_out_linear_weight.npy"));            // [4096,4096]
    const std::vector<float> w_proj   = represent(LW("toplevel_proj_out_weight.npy"));                   // [64,4096]
    const std::vector<float> block_out_rep = represent(F(T, "block_out.npy"));                           // [72,4096]

    // diffusers FP32 references (for the reported diagnostic only).
    const std::vector<float> ref_img_in = F(T, "img_in.npy");
    const std::vector<float> ref_txt_in = F(T, "txt_in.npy");
    const std::vector<float> ref_temb = F(T, "temb.npy");
    const std::vector<float> ref_mod = F(T, "modulation.npy");
    const std::vector<float> ref_norm_out = F(T, "norm_out.npy");
    const std::vector<float> ref_sample = F(T, "sample.npy");

    // ---- host reference stages (BF16-represented operands, FP32/double math) ----
    std::vector<float> hs_img;
    qo::matmul_xwt(in_hidden, w_img_in, hs_img, img_tokens, D, 64);

    std::vector<float> tn, tmid, txt_out;
    qo::zerocenter_rmsnorm(in_enc, tn, w_tnorm, eps);
    qo::matmul_xwt(tn, w_in, tmid, text_prefix, D, D);
    for (float& x : tmid) x = qo::gelu_tanh(x);
    qo::matmul_xwt(tmid, w_outl, txt_out, text_prefix, D, D);

    std::vector<float> tvec = {in_t_repr[0], 0.0F};
    std::vector<float> tproj = qo::temporal_timesteps(tvec, td, time_factor);  // [2,256]
    std::vector<float> temb_a, temb;
    qo::matmul_xwt(tproj, w_tl1, temb_a, 2, D, td);
    for (float& x : temb_a) x = qo::silu(x);
    qo::matmul_xwt(temb_a, w_tl2, temb, 2, D, D);

    std::vector<float> silu_temb(temb.size());
    for (std::size_t i = 0; i < temb.size(); ++i) silu_temb[i] = qo::silu(temb[i]);
    std::vector<float> modulation;
    qo::matmul_xwt(silu_temb, w_mod, modulation, 2, 4 * D, D);

    std::vector<float> nlin;
    qo::matmul_xwt(silu_temb, w_nolin, nlin, 2, D, D);
    std::vector<std::uint8_t> tmask(S);
    for (std::size_t s = 0; s < S; ++s) tmask[s] = (s >= text_prefix) ? 1 : 0;
    std::vector<float> scale_sel, normed, norm_outv;
    qo::layernorm(block_out_rep, normed, S, D, eps);
    scale_sel.resize(S * D);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t row = tmask[s] ? 0 : 1;
        for (std::size_t d = 0; d < D; ++d) scale_sel[s * D + d] = nlin[row * D + d];
    }
    norm_outv.resize(S * D);
    for (std::size_t i = 0; i < S * D; ++i) norm_outv[i] = normed[i] * (1.0F + scale_sel[i]);

    std::vector<float> sample;
    qo::matmul_xwt(norm_outv, w_proj, sample, S, 64, D);

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    // ---- device weights ----
    const BfWeight Wimg(w_img_in, (int)D, 64);
    const BfWeight Winl(w_in, (int)D, (int)D);
    const BfWeight Woutl(w_outl, (int)D, (int)D);
    const BfWeight Wtl1(w_tl1, (int)D, (int)td);
    const BfWeight Wtl2(w_tl2, (int)D, (int)D);
    const BfWeight Wmod(w_mod, (int)(4 * D), (int)D);
    const BfWeight Wnolin(w_nolin, (int)D, (int)D);
    const BfWeight Wproj(w_proj, 64, (int)D);

    // ---- device inputs / constants ----
    void* d_in_hidden = upload(in_hidden);
    void* d_in_enc = upload(in_enc);
    void* d_tnorm = upload(w_tnorm);
    void* d_tproj = upload(tproj);
    void* d_block_out = upload(block_out_rep);
    const std::vector<float> ones4096(D, 1.0F), zeros4096(D, 0.0F), ones8192(2 * D, 1.0F);
    void* d_ones4096 = upload(ones4096), *d_zeros4096 = upload(zeros4096), *d_ones8192 = upload(ones8192);

    // ---- device intermediate buffers ----
    void *d_hs_img, *d_tn, *d_tmid, *d_txt_out, *d_temb_a, *d_temb_a_silu, *d_temb, *d_temb_silu,
         *d_modulation, *d_nlin, *d_ln_out, *d_norm_out, *d_sample;
    const std::size_t sD = S * D;              // 72*4096
    const std::size_t s8 = text_prefix * D;    // 8*4096
    const std::size_t s2 = 2 * D;              // 2*4096
    const std::size_t sImg = img_tokens * D;   // 64*4096
    const std::size_t sMod = 2 * (4 * D);      // 2*16384
    const std::size_t sProj = S * 64;          // 72*64
    CUDA_CHECK(cudaMalloc(&d_hs_img, sImg * 2));
    CUDA_CHECK(cudaMalloc(&d_tn, s8 * 2));
    CUDA_CHECK(cudaMalloc(&d_tmid, s8 * 2));
    CUDA_CHECK(cudaMalloc(&d_txt_out, s8 * 2));
    CUDA_CHECK(cudaMalloc(&d_temb_a, s2 * 2));
    CUDA_CHECK(cudaMalloc(&d_temb_a_silu, s2 * 2));
    CUDA_CHECK(cudaMalloc(&d_temb, s2 * 2));
    CUDA_CHECK(cudaMalloc(&d_temb_silu, s2 * 2));
    CUDA_CHECK(cudaMalloc(&d_modulation, sMod * 2));
    CUDA_CHECK(cudaMalloc(&d_nlin, s2 * 2));
    CUDA_CHECK(cudaMalloc(&d_ln_out, sD * 2));
    CUDA_CHECK(cudaMalloc(&d_norm_out, sD * 2));
    CUDA_CHECK(cudaMalloc(&d_sample, sProj * 2));

    const int Di = (int)D, Si = (int)S, ti = (int)td, tp = (int)text_prefix;

    // ---- top-level forward through production ops ----
    // img_in: x=[K=64,T=64], W=[N=4096,K=64] -> out [N=4096,T=64]
    Tensor hs_img_v = view(d_hs_img, Di, 64);
    ops::linear(view(d_in_hidden, 64, 64), Wimg.w, hs_img_v, 0);

    // txt_in: ZeroCenterRMSNorm (unit_offset=true) -> in_layer -> gelu(Tanh) -> out_layer
    Tensor enc_v = view(d_in_enc, Di, (int)text_prefix), tn_v = view(d_tn, Di, (int)text_prefix);
    ops::rmsnorm(enc_v, view(d_tnorm, Di, 1), eps, /*unit_offset=*/true, tn_v, 0);
    Tensor tmid_v = view(d_tmid, Di, (int)text_prefix);
    ops::linear(tn_v, Winl.w, tmid_v, 0);
    ops::gelu(tmid_v, ops::GeluMode::Tanh, 0);
    Tensor txt_v = view(d_txt_out, Di, (int)text_prefix);
    ops::linear(tmid_v, Woutl.w, txt_v, 0);

    // timestep embed: sinusoidal [256,2] -> linear_1 -> silu -> linear_2 -> temb [4096,2]
    Tensor tproj_v = view(d_tproj, ti, 2), temb_a_v = view(d_temb_a, Di, 2);
    ops::linear(tproj_v, Wtl1.w, temb_a_v, 0);
    Tensor temb_a_silu_v = view(d_temb_a_silu, Di, 2);
    ops::silu_mul(temb_a_v, view(d_ones8192, Di, 2), temb_a_silu_v, 0);
    Tensor temb_v = view(d_temb, Di, 2);
    ops::linear(temb_a_silu_v, Wtl2.w, temb_v, 0);

    // global modulation: silu(temb) -> linear [16384,4096]
    Tensor temb_silu_v = view(d_temb_silu, Di, 2);
    ops::silu_mul(temb_v, view(d_ones8192, Di, 2), temb_silu_v, 0);
    Tensor mod_v = view(d_modulation, 4 * Di, 2);
    ops::linear(temb_silu_v, Wmod.w, mod_v, 0);

    // norm_out: norm_out.linear(SiLU(temb)) -> layer_norm(affine=False) -> adaln_scale (scale-only)
    Tensor nlin_v = view(d_nlin, Di, 2);
    ops::linear(temb_silu_v, Wnolin.w, nlin_v, 0);
    Tensor bo_v = view(d_block_out, Di, Si), ln_v = view(d_ln_out, Di, Si);
    ops::layer_norm(bo_v, view(d_ones4096, Di, 1), view(d_zeros4096, Di, 1), eps, ln_v, 0);
    Tensor norm_out_v = view(d_norm_out, Di, Si);
    ops::adaln_scale(ln_v, nlin_v, 0, tp, norm_out_v, 0);

    // proj_out: sample [N=64,T=72]
    Tensor sample_v = view(d_sample, 64, Si);
    ops::linear(norm_out_v, Wproj.w, sample_v, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- per-stage qualification: gated vs the BF16-represented host oracle; reported vs the .npy ----
    struct Row { const char* name; void* d; std::size_t n; const std::vector<float>* host; const std::vector<float>* ref; };
    Row rows[] = {
        {"img_in",   d_hs_img,   sImg, &hs_img,     &ref_img_in},
        {"txt_in",   d_txt_out,  s8,   &txt_out,    &ref_txt_in},
        {"temb",     d_temb,     s2,   &temb,       &ref_temb},
        {"modulation", d_modulation, sMod, &modulation, &ref_mod},
        {"norm_out", d_norm_out, sD,   &norm_outv,  &ref_norm_out},
        {"sample",   d_sample,   sProj,&sample,     &ref_sample},
    };
    const double tol = 2e-2;  // a-priori: BF16 compute through <=2 matmuls + act/norm per stage.
    bool ok = true;
    for (auto& r : rows) {
        std::vector<float> got = download(r.d, r.n);
        const double e_host = qo::rel_fro(got, *r.host);
        const double e_ref  = qo::rel_fro(got, *r.ref);
        const bool s_ok = (e_host < tol) && std::isfinite(e_host);
        ok = ok && s_ok;
        std::printf("stage %-11s vs_host=%.3e vs_npy=%.3e tol=%.0e %s\n", r.name, e_host, e_ref, tol, s_ok ? "OK" : "FAIL");
    }
    std::fflush(stdout);

    cudaFree(d_in_hidden); cudaFree(d_in_enc); cudaFree(d_tnorm); cudaFree(d_tproj); cudaFree(d_block_out);
    cudaFree(d_ones4096); cudaFree(d_zeros4096); cudaFree(d_ones8192);
    cudaFree(d_hs_img); cudaFree(d_tn); cudaFree(d_tmid); cudaFree(d_txt_out);
    cudaFree(d_temb_a); cudaFree(d_temb_a_silu); cudaFree(d_temb); cudaFree(d_temb_silu);
    cudaFree(d_modulation); cudaFree(d_nlin); cudaFree(d_ln_out); cudaFree(d_norm_out); cudaFree(d_sample);

    std::printf(ok ? "IMAGE_DIT_TOPLEVEL_PROD_TEST: PASS (S=%zu D=%zu prefix=%zu)\n" : "IMAGE_DIT_TOPLEVEL_PROD_TEST: FAIL\n", S, D, text_prefix);
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_DIT_TOPLEVEL_PROD_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
