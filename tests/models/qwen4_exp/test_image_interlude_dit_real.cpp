// C4 (LLM<->image coexistence): the FULL in-process DiT forward on the LLM Program's YIELDED VRAM.
//
// The previous increment (test_image_interlude_op_real.cpp) proved one real image op -- a single
// [4096,4096] FP8 projection -- runs in the 8 GiB the LLM hands over while it sits Quiesced/Leased,
// in-process, and that the LLM resumes undisturbed. This is the next link: the WHOLE one-step DiT
// forward -- the timestep embedding, img_in + txt_in front-end, the joint sequence through all 32
// production blocks, and norm_out + proj_out -- composed exactly as the qualified BF16 pipeline test
// (tests/models/qwen_image/test_image_pipeline_prod_cuda.cu), but with the block weights STREAMED
// one block at a time so the resident image working set peaks at ~one block (~0.9 GiB) and fits inside
// the 8 GiB yield while the LLM's own VRAM stays mapped and idle.
//
// Streaming is the crux of the coexistence claim: the full image weights are ~14 GiB resident (they
// would not sit beside the LLM), but streamed they peak at ~one block's ~436 MiB + activations, so the
// image job lives entirely in the freed region and never displaces the LLM. The toplevel/front-end
// weights (~272 MiB) are small enough to stay resident for the one step.
//
// Self-verifying against the fresh diffusers reference: the step-0 velocity noise_pred (the image
// tokens of the [S,64] output) is checked against noise_pred_step0.npy, so a wrong streamed weight, a
// geometry/layout defect, or a wrong op order lands O(1..10) while a correct assembly lands O(1e-2..1e-1).
// The coexistence gate is the LLM: post-interlude ids == baseline.
//
// Reads $INFERNIX_IMAGE_ORACLE/{tensors_toplevel_weights,tensors_full_depth_weights}/*.npy for the
// model weights and $INFERNIX_IMAGE_ORACLE_PIPELINE (default <oracle>/tensors_pipeline) for the
// prompt/seed/geometry references. Needs both INFERNIX_QWEN4_ARTIFACT (the LLM to yield from) and
// INFERNIX_IMAGE_ORACLE. Skips (77) without them; returns 2 with no CUDA device.
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   INFERNIX_IMAGE_ORACLE=.../oracle [INFERNIX_IMAGE_ORACLE_PIPELINE=.../tensors_pipeline] \
//   infernix_qwen4_exp_image_interlude_dit_real_test

#include "infernix/engine.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/dit_adaln.h"
#include "infernix/ops/rope_c64.h"
#include "infernix/ops/attn_dit_joint.h"

#include "core/tensor.h"
#include "core/weight.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "runtime/expert_offload/vram_monitor.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "models/qwen_image/dit_block.h"

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

namespace {

using T = std::vector<float>;

constexpr std::uint64_t kGiB = 1ULL << 30;

// ---- Host BF16 codec (identical to the pipeline test) ----
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
std::vector<std::uint16_t> bits_of(const T& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}
T represent(const T& v) {
    T out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}
Tensor view(void* p, int a, int b, int c = 1, int dd = 1) { return Tensor(p, DType::BF16, {a, b, c, dd}); }

T download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    if (cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy D2H failed");
    T out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}

// A device buffer the image job owns: allocated in the freed region after the yield, freed (RAII) at
// scope exit -- always before release_image_vram restores the LLM's pools.
struct Buf {
    void* d = nullptr;
    std::size_t bytes = 0;
    void alloc(std::size_t b) {
        bytes = b;
        if (cudaMalloc(&d, b) != cudaSuccess) throw std::runtime_error("cudaMalloc failed (image buffer)");
    }
    void h2d(const void* host, std::size_t b) {
        if (bytes != b) bytes = b;
        if (cudaMemcpy(d, host, b, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D failed (image buffer)");
    }
    ~Buf() { if (d) cudaFree(d); }
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

// A contiguous BF16 [n,k] weight uploaded to the freed device region, wrapped as a registered Weight.
// The destructor frees the device buffer, so a streamed per-block weight dies at the end of its block.
struct WeightBuf {
    int n = 0, k = 0;
    std::size_t bytes = 0;
    void* d = nullptr;
    Weight w{};
    void init(const T& floats, int rows, int cols) {
        n = rows; k = cols;
        const auto bits = bits_of(floats);
        bytes = bits.size() * sizeof(std::uint16_t);
        if (cudaMalloc(&d, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed (image weight)");
        if (cudaMemcpy(d, bits.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("H2D failed (image weight)");
        w             = Weight{};
        w.payload     = d;
        w.payload_bytes = bytes;
        w.qtype       = infernix::QType::BF16;
        w.shape[0]    = n;   w.shape[1] = k;
        w.padded_shape[0] = n; w.padded_shape[1] = k;
        w.ndim        = 2;
        w.qdata       = d;
        w.n           = n;   w.k = k;
        w.layout      = infernix::QuantLayout::Contiguous;
    }
    ~WeightBuf() { if (d) cudaFree(d); }
    WeightBuf() = default;
    WeightBuf(const WeightBuf&) = delete;
    WeightBuf& operator=(const WeightBuf&) = delete;
};

// DiT per-block scratch + one block forward: shared module (verbatim move from this test).
using WS = infernix::models::qwen_image::DiTWorkspace;
using infernix::models::qwen_image::run_block;

std::string bwf(int l, const char* role) {
    std::string s = "blk"; s += std::to_string(l); s += "_"; s += role; s += ".npy";
    return s;
}

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a long story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

std::uint64_t device_free_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return free_b;
}

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* qwen4 = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    const char* oracle = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (qwen4 == nullptr || oracle == nullptr || std::string(oracle).empty()) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT and INFERNIX_IMAGE_ORACLE\n");
        return 77;
    }
    int result = 1;
    try {
        auto real = infernix::open_vram_budget_source(0);
        infernix::testing::set_vram_budget_source([&] { return real->query(); });
        infernix::runtime::expert_offload::testing::set_vram_grow_delay(3600.0);

        infernix::EngineOptions options;
        options.artifact_path         = qwen4;
        options.context_cache.enabled = false;
        options.enable_vision         = false;
        options.max_context           = 4096;
        options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes      = 8 * kGiB;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(infernix::models::qwen4_exp::testing::has_image_yield_program(), "a Program is loaded to yield from");

        const auto baseline = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(baseline.size() == 200, "baseline generated 200 tokens");

        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
            std::printf("NO_CUDA_DEVICE\n");
            return 2;
        }

        // ---- reference + weight dirs ----
        const std::string oroot(oracle);
        std::string TP = qo::join(oroot, "tensors_pipeline");
        if (const char* pe_pipeline = std::getenv("INFERNIX_IMAGE_ORACLE_PIPELINE"))
            if (*pe_pipeline) TP = pe_pipeline;
        const std::string TW = qo::join(oroot, "tensors_toplevel_weights");
        const std::string WD = qo::join(oroot, "tensors_full_depth_weights");

        // ---- geometry derived from the references (any 16-multiple size, e.g. 256 or 512) ----
        const std::size_t D = 4096, heads = 32, td = 256, mlp0 = 12288;
        const float eps = 1e-6F;
        qo::NpyArray pe_a = qo::read_npy(qo::join(TP, "prompt_embeds.npy"));
        qo::require(pe_a.shape.size() == 3 && pe_a.shape[0] == 1 && pe_a.shape[2] == D,
                    "prompt_embeds not [1,?,4096]");
        const std::size_t tp = pe_a.shape[1];
        const T prompt_embeds = represent(pe_a.floats());        // [tp,4096] token-major
        qo::NpyArray img_a = qo::read_npy(qo::join(TP, "image.npy"));
        qo::require(img_a.shape.size() == 4 && img_a.shape[0] == 1 && img_a.shape[1] == 4 &&
                    (int)img_a.shape[2] % 16 == 0 && (int)img_a.shape[3] % 16 == 0,
                    "image.npy not [1,4,H,W] (H,W multiples of 16)");
        const int H = (int)img_a.shape[2], W = (int)img_a.shape[3];
        const std::size_t Hl = (std::size_t)H / 16, Wl = (std::size_t)W / 16;
        const std::size_t img_tokens = Hl * Wl;
        const std::size_t S = tp + img_tokens;
        qo::NpyArray li_a = qo::read_npy(qo::join(TP, "latents_init.npy"));
        qo::require(li_a.shape.size() == 3 && li_a.shape[1] == (int)img_tokens && li_a.shape[2] == 64,
                    "latents_init not [1,img_tokens,64]");
        const T lat_init = represent(li_a.floats());             // [img_tokens,64]
        const T timesteps = qo::read_npy(qo::join(TP, "timesteps.npy")).floats();
        const T sigmas = qo::read_npy(qo::join(TP, "sigmas.npy")).floats();
        const T noise_ref0 = qo::read_npy(qo::join(TP, "noise_pred_step0.npy")).floats();
        qo::require(timesteps.size() == 40 && sigmas.size() == 41, "timesteps/sigmas len mismatch");
        qo::require(noise_ref0.size() == img_tokens * 64, "noise_pred_step0 size");
        std::printf("geometry tp=%zu Hl=%zu Wl=%zu img_tokens=%zu S=%zu (image %dx%d)\n",
                    tp, Hl, Wl, img_tokens, S, H, W);

        const double time_factor = 1000.0;
        const int Di = (int)D, Si = (int)S, Hi = (int)heads, Mi = (int)mlp0, tp_i = (int)tp;
        const std::size_t sD = S * D;
        const std::size_t sMlp = S * mlp0;

        const std::uint64_t free_full = device_free_bytes();

        // Interlude: close the door, drain every LLM stream, THEN yield its physical VRAM.
        const std::uint64_t epoch = infernix::models::qwen4_exp::testing::begin_image_quiesce();
        check(epoch != 0, "begin_image_quiesce returned an execution epoch");
        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 2, "the Program is Quiesced");

        std::uintptr_t lease = 0;
        std::uint64_t freed = 0;
        try {
            freed = infernix::models::qwen4_exp::testing::yield_image_vram(8 * kGiB, 0, lease);
            check(freed > 0, "the yield freed " + std::to_string(freed >> 20) + " MiB toward the image job");
            check(lease != 0, "the yield returned a lease to restore");
            const std::uint64_t free_after_yield = device_free_bytes();
            check(free_after_yield - free_full > kGiB / 2, "driver free VRAM rose across the yield (the unmap is physical)");

            {
                // ---- all image device memory lives inside this scope, freed at scope exit (before release) ----

                // Shared device constants. d_ones/d_zeros back the [D,1] layer_norm views AND the [D,2]
                // time-embedding views, so they hold 2*D.
                Buf d_ones, d_zeros, d_tnorm, drot;
                const T ones(2 * D, 1.0F), zeros(2 * D, 0.0F);
                const T w_tnorm = represent(qo::read_npy(qo::join(TW, "toplevel_txt_in_text_norm_weight.npy")).floats());
                const std::vector<float> rot = qo::build_rope(tp, Hl, Wl);
                d_ones.alloc(2 * D * 2); d_ones.h2d(bits_of(ones).data(), ones.size() * 2);
                d_zeros.alloc(2 * D * 2); d_zeros.h2d(bits_of(zeros).data(), zeros.size() * 2);
                d_tnorm.alloc(w_tnorm.size() * 2); d_tnorm.h2d(bits_of(w_tnorm).data(), w_tnorm.size() * 2);
                drot.alloc(rot.size() * 2); drot.h2d(bits_of(rot).data(), rot.size() * 2);

                // Toplevel/front-end weights (~272 MiB): resident for the one step.
                WeightBuf Wimg, Winl, Woutl, Wtl1, Wtl2, Wmod, Wnolin, Wproj;
                Wimg.init(represent(qo::read_npy(qo::join(TW, "toplevel_img_in_weight.npy")).floats()), Di, 64);
                Winl.init(represent(qo::read_npy(qo::join(TW, "toplevel_txt_in_in_layer_weight.npy")).floats()), Di, Di);
                Woutl.init(represent(qo::read_npy(qo::join(TW, "toplevel_txt_in_out_layer_weight.npy")).floats()), Di, Di);
                Wtl1.init(represent(qo::read_npy(qo::join(TW, "toplevel_time_text_embed_timestep_embedder_linear_1_weight.npy")).floats()), Di, (int)td);
                Wtl2.init(represent(qo::read_npy(qo::join(TW, "toplevel_time_text_embed_timestep_embedder_linear_2_weight.npy")).floats()), Di, Di);
                Wmod.init(represent(qo::read_npy(qo::join(TW, "toplevel_modulation_1_weight.npy")).floats()), 4 * Di, Di);
                Wnolin.init(represent(qo::read_npy(qo::join(TW, "toplevel_norm_out_linear_weight.npy")).floats()), Di, Di);
                Wproj.init(represent(qo::read_npy(qo::join(TW, "toplevel_proj_out_weight.npy")).floats()), 64, Di);

                // DiT per-block scratch (allocated once, reused every block).
                WS ws{};
                Buf b_ln1, b_mod1, b_dq, b_dk, b_dv, b_dqr, b_dkr, b_dattn, b_dattn_out, b_dx1, b_dln2, b_dmod2, b_dmlp_out;
                for (Buf* b : {&b_ln1, &b_mod1, &b_dq, &b_dk, &b_dv, &b_dqr, &b_dkr, &b_dattn, &b_dattn_out,
                               &b_dx1, &b_dln2, &b_dmod2, &b_dmlp_out}) b->alloc(sD * 2);
                ws.dln1 = b_ln1.d; ws.dmod1 = b_mod1.d; ws.dq = b_dq.d; ws.dk = b_dk.d; ws.dv = b_dv.d;
                ws.dqr = b_dqr.d; ws.dkr = b_dkr.d; ws.dattn = b_dattn.d; ws.dattn_out = b_dattn_out.d;
                ws.dx1 = b_dx1.d; ws.dln2 = b_dln2.d; ws.dmod2 = b_dmod2.d; ws.dmlp_out = b_dmlp_out.d;
                Buf b_proj, b_gate, b_hidden;
                for (Buf* b : {&b_proj, &b_gate, &b_hidden}) b->alloc(sMlp * 2);
                ws.dproj = b_proj.d; ws.dgate = b_gate.d; ws.dhidden = b_hidden.d;

                // Per-step buffers.
                Buf d_latent, d_hs_img, d_txt, d_txt2, d_joint, d_hb;
                Buf d_tproj, d_temb_a, d_temb_a_silu, d_temb, d_temb_silu, d_mod, d_nlin;
                Buf d_ln, d_norm_out, d_sample;
                d_latent.alloc(img_tokens * 64 * 2);
                d_hs_img.alloc(img_tokens * D * 2);
                d_txt.alloc(tp * D * 2);
                d_txt2.alloc(tp * D * 2);
                d_joint.alloc(sD * 2);
                d_hb.alloc(sD * 2);
                d_tproj.alloc(2 * td * 2);
                const std::size_t s2 = 2 * D;
                d_temb_a.alloc(s2 * 2);
                d_temb_a_silu.alloc(s2 * 2);
                d_temb.alloc(s2 * 2);
                d_temb_silu.alloc(s2 * 2);
                d_mod.alloc(4 * D * 2 * 2);
                d_nlin.alloc(s2 * 2);
                d_ln.alloc(sD * 2);
                d_norm_out.alloc(sD * 2);
                d_sample.alloc(S * 64 * 2);
                { const auto bits = bits_of(lat_init); d_latent.h2d(bits.data(), img_tokens * 64 * 2); }
                Buf d_pe;
                d_pe.alloc(tp * D * 2);
                { const auto bits = bits_of(prompt_embeds); d_pe.h2d(bits.data(), tp * D * 2); }

                // ---- one full step (i = 0): front-end + 32 streamed blocks + norm_out + proj_out ----
                const float model_input_t = (float)(timesteps[0] / 1000.0);
                const std::vector<float> tproj = qo::temporal_timesteps({model_input_t, 0.0F}, td, time_factor);
                { const auto bits = bits_of(tproj); d_tproj.h2d(bits.data(), tproj.size() * 2); }

                Tensor tproj_v = view(d_tproj.d, (int)td, 2), temb_a_v = view(d_temb_a.d, Di, 2);
                ops::linear(tproj_v, Wtl1.w, temb_a_v, 0);
                Tensor temb_a_silu_v = view(d_temb_a_silu.d, Di, 2);
                ops::silu_mul(temb_a_v, view(d_ones.d, Di, 2), temb_a_silu_v, 0);
                Tensor temb_v = view(d_temb.d, Di, 2);
                ops::linear(temb_a_silu_v, Wtl2.w, temb_v, 0);
                Tensor temb_silu_v = view(d_temb_silu.d, Di, 2);
                ops::silu_mul(temb_v, view(d_ones.d, Di, 2), temb_silu_v, 0);
                Tensor mod_v = view(d_mod.d, 4 * Di, 2);
                ops::linear(temb_silu_v, Wmod.w, mod_v, 0);
                Tensor nlin_v = view(d_nlin.d, Di, 2);
                ops::linear(temb_silu_v, Wnolin.w, nlin_v, 0);

                // img_in: latent [64,img_tokens] -> hs_img [D,img_tokens]
                Tensor hs_img_v = view(d_hs_img.d, Di, (int)img_tokens);
                ops::linear(view(d_latent.d, 64, (int)img_tokens), Wimg.w, hs_img_v, 0);

                // txt_in: prompt_embeds -> ZeroCenterRMSNorm -> in_layer -> gelu(Tanh) -> out_layer.
                // Separate in/out buffers (tn_v, tn2_v): an in-place GEMM is run-to-run non-deterministic (#33).
                Tensor pe_v = view(d_pe.d, Di, tp_i), tn_v = view(d_txt.d, Di, tp_i), tn2_v = view(d_txt2.d, Di, tp_i);
                ops::rmsnorm(pe_v, view(d_tnorm.d, Di, 1), eps, /*unit_offset=*/true, tn_v, 0);
                ops::linear(tn_v, Winl.w, tn2_v, 0);
                ops::gelu(tn2_v, ops::GeluMode::Tanh, 0);
                ops::linear(tn2_v, Woutl.w, tn_v, 0);

                // assemble the joint [text(tp) + image(img_tokens)] token-major buffer.
                {
                    const std::size_t tpB = tp * D * 2, imgB = img_tokens * D * 2;
                    if (cudaMemcpy(d_joint.d, d_txt.d, tpB, cudaMemcpyDeviceToDevice) != cudaSuccess) throw std::runtime_error("joint memcpy txt");
                    if (cudaMemcpy((char*)d_joint.d + tpB, d_hs_img.d, imgB, cudaMemcpyDeviceToDevice) != cudaSuccess) throw std::runtime_error("joint memcpy img");
                }

                // 32-block chain: d_joint in, ping-pong with d_hb; after 32 (even) blocks the result is
                // back in d_joint. Each block's 7 weights + 2 norms are STREAMED (read, H2D, used, freed).
                void* hin = d_joint.d, *hout = d_hb.d;
                const std::uint64_t free_at_peak_base = device_free_bytes();
                for (int l = 0; l < 32; ++l) {
                    WeightBuf Wq, Wk, Wv, Wout, Wproj_b, Wgate, Wmlpout;
                    Wq.init(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_q_weight"))).floats()), Di, Di);
                    Wk.init(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_k_weight"))).floats()), Di, Di);
                    Wv.init(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_v_weight"))).floats()), Di, Di);
                    Wout.init(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_out_0_weight"))).floats()), Di, Di);
                    Wproj_b.init(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_proj_weight"))).floats()), Mi, Di);
                    Wgate.init(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_gate_layer_weight"))).floats()), Mi, Di);
                    Wmlpout.init(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_out_weight"))).floats()), Di, Mi);
                    std::vector<std::uint16_t> nq_b = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_norm_q_weight"))).floats()));
                    std::vector<std::uint16_t> nk_b = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_norm_k_weight"))).floats()));
                    void* d_nq = nullptr, *d_nk = nullptr;
                    if (cudaMalloc(&d_nq, nq_b.size() * 2) != cudaSuccess || cudaMalloc(&d_nk, nk_b.size() * 2) != cudaSuccess)
                        throw std::runtime_error("cudaMalloc failed (block norm)");
                    cudaMemcpy(d_nq, nq_b.data(), nq_b.size() * 2, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_nk, nk_b.data(), nk_b.size() * 2, cudaMemcpyHostToDevice);

                    if (l == 0) {
                        const std::uint64_t free_peak = device_free_bytes();
                        const std::uint64_t peak_used = free_at_peak_base > free_peak ? free_at_peak_base - free_peak : 0;
                        std::printf("  peak device working set at block 0: %llu MiB (of %llu MiB yielded)\n",
                                    static_cast<unsigned long long>(peak_used >> 20),
                                    static_cast<unsigned long long>(freed >> 20));
                        check(peak_used <= 8 * kGiB, "the image working set stays inside the yielded region");
                    }

                    run_block(hin, hout, Wq.w, Wk.w, Wv.w, Wout.w, Wproj_b.w, Wgate.w, Wmlpout.w,
                              d_nq, d_nk, d_mod.d, drot.d, d_ones.d, d_zeros.d, ws,
                              Si, Di, Hi, Mi, tp_i, eps);
                    cudaFree(d_nq);
                    cudaFree(d_nk);
                    std::swap(hin, hout);
                }
                void* h_final = hin;

                // norm_out: layer_norm -> adaln_scale -> proj_out.
                Tensor hfinal_v = view(h_final, Di, Si), ln_v = view(d_ln.d, Di, Si);
                ops::layer_norm(hfinal_v, view(d_ones.d, Di, 1), view(d_zeros.d, Di, 1), eps, ln_v, 0);
                Tensor norm_out_v = view(d_norm_out.d, Di, Si);
                ops::adaln_scale(ln_v, nlin_v, 0, tp_i, norm_out_v, 0);
                Tensor sample_v = view(d_sample.d, 64, Si);
                ops::linear(norm_out_v, Wproj.w, sample_v, 0);
                if (cudaStreamSynchronize(0) != cudaSuccess) throw std::runtime_error("DiT step did not complete");

                // noise_pred = the image tokens (last img_tokens) of the [S,64] velocity, token-major.
                const std::size_t noise_off = tp * 64 * 2;
                const T noise_f = download((char*)d_sample.d + noise_off, img_tokens * 64);
                const double rel_fro = qo::rel_fro(noise_f, noise_ref0);
                double na2 = 0, nb2 = 0, dot = 0;
                for (std::size_t p = 0; p < noise_f.size(); ++p) {
                    na2 += (double)noise_f[p] * (double)noise_f[p];
                    nb2 += (double)noise_ref0[p] * (double)noise_ref0[p];
                    dot += (double)noise_f[p] * (double)noise_ref0[p];
                }
                const double c = dot / (nb2 + 1e-30);
                double resid2 = 0;
                for (std::size_t p = 0; p < noise_f.size(); ++p) {
                    const double r = (double)noise_f[p] - c * (double)noise_ref0[p];
                    resid2 += r * r;
                }
                std::printf("  step-0 noise_pred rel_fro=%.4e  ||a||/||b||=%.4f  bestfit_c=%.4f  residual_rel_fro=%.4e\n",
                            rel_fro, std::sqrt(na2 / (nb2 + 1e-30)), c, std::sqrt(resid2 / (nb2 + 1e-30)));
                check(nb2 > 0.0, "the noise_pred reference is non-degenerate");
                check(rel_fro < 0.1, "the streamed DiT step matches the diffusers step-0 velocity (rel_fro < 0.1)");
            } // all image device memory freed here

            infernix::models::qwen4_exp::testing::release_image_vram(lease);  // give the LLM its VRAM back
        } catch (...) {
            infernix::models::qwen4_exp::testing::release_image_vram(lease);
            infernix::models::qwen4_exp::testing::end_image_quiesce();
            throw;
        }
        infernix::models::qwen4_exp::testing::end_image_quiesce();
        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 0, "the Program returned to Running");
        check(infernix::models::qwen4_exp::testing::image_interlude_submit_open(), "the LLM submit door reopened");
        const auto after = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(after == baseline, "LLM generation is undisturbed after the in-process DiT forward");

        infernix::testing::set_vram_budget_source({});
        result = failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        try { infernix::models::qwen4_exp::testing::end_image_quiesce(); } catch (...) {}
        return 1;
    }
    std::printf(result == 0 ? "qwen4_exp image-interlude DiT checks passed\n" : "FAIL: %d checks failed\n", failures);
    return result;
}
