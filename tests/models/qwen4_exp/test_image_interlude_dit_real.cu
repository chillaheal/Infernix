// C4 (LLM<->image coexistence): the FULL in-process image pipeline on the LLM Program's YIELDED
// VRAM -- the timestep embedding + img_in/txt_in front-end + all 32 production blocks, run for the
// whole 40-step FlowMatchEuler denoise loop, then the VAE decode of the final latent to the image.
//
// The previous increments proved, in-process, that one real image op (#41) and then the whole
// ONE-STEP DiT forward (#42) run in the 8 GiB the LLM hands over while it sits Quiesced/Leased, and
// that the LLM resumes undisturbed. This is the final link: the WHOLE image job -- 40 flow-match
// steps of the 32-block DiT plus the VAE decode -- composed exactly as the qualified BF16 pipeline
// test (tests/models/qwen_image/test_image_pipeline_prod_cuda.cu), but with the block weights
// STREAMED one block at a time so the resident image working set peaks at ~one block (~0.9 GiB) and
// fits inside the 8 GiB yield while the LLM's own VRAM stays mapped and idle. ONE lease covers the
// whole image; every image buffer is RAII-freed before release_image_vram restores the LLM's pools.
//
// Streaming is the crux of the coexistence claim: the full image weights are ~14 GiB resident (they
// would not sit beside the LLM). Across 40 steps a block is needed 40 times, so the 32 blocks'
// weights are cached ONCE in host RAM (~14 GiB, well under the 63.7 GiB host) and H2D-streamed
// per-block per-step into the freed region (no 40x disk I/O). The toplevel/front-end weights
// (~272 MiB) stay resident for the whole loop.
//
// Self-verifying against the fresh diffusers reference: the step-0 velocity noise_pred is checked
// against noise_pred_step0.npy (a wrong streamed weight / layout / op-order lands O(1..10)); the
// FINAL LATENT is checked against latents_final.npy (rel_fro < 0.5) -- independent of the VAE, so a
// VAE defect is isolable from a DiT/sampler assembly defect; and the FINAL IMAGE is checked against
// image.npy (rel_fro < 0.3). The coexistence gate is the LLM: post-interlude ids == baseline.
//
// Reads $INFERNIX_IMAGE_ORACLE/{tensors_toplevel_weights,tensors_full_depth_weights,
// tensors_vae_weights}/*.npy for the model weights and $INFERNIX_IMAGE_ORACLE_PIPELINE (default
// <oracle>/tensors_pipeline) for the prompt/seed/geometry references. Needs both
// INFERNIX_QWEN4_ARTIFACT (the LLM to yield from) and INFERNIX_IMAGE_ORACLE. Skips (77) without
// them; returns 2 with no CUDA device. This is a .cu (not .cpp) because the VAE decode needs the
// channel<->token transpose kernel (no production transpose op exists).
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
#include "infernix/ops/add_bias.h"
#include "infernix/ops/vae_decoder.h"

#include "core/tensor.h"
#include "core/weight.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "models/qwen_image/dit_block.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

// Device transpose: in [rows,cols] row-major -> out [cols,rows] row-major. Bridges the channel-major
// [C,P] layout of the VAE conv/attn ops and the token-major [P,C] layout that ops::linear reads/writes.
__global__ void transpose_u16_kernel(const std::uint16_t* __restrict__ in, std::uint16_t* __restrict__ out,
                                     int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    out[c * rows + r] = in[r * cols + c];
}

// A device buffer the image job owns: allocated in the freed region after the yield, freed (RAII) at
// scope exit -- always before release_image_vram restores the LLM's pools. Movable so the VAE decode
// can collect its many scratch buffers in a vector (all freed at scope end).
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
    Buf(Buf&& o) noexcept : d(o.d), bytes(o.bytes) { o.d = nullptr; o.bytes = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) { if (d) cudaFree(d); d = o.d; bytes = o.bytes; o.d = nullptr; o.bytes = 0; }
        return *this;
    }
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
    // From already-represented host BF16 bits (the block host cache): malloc + H2D only.
    void init_from_bits(const std::vector<std::uint16_t>& bits, int rows, int cols) {
        n = rows; k = cols;
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

// One DiT block's 7 weights + 2 norms, held in host RAM as represented BF16 bits. The 32 blocks are
// cached once (~14 GiB host) and H2D-streamed per-block per-step into the freed region, so the 40-step
// loop never re-reads the weights from disk.
struct BlockHost {
    std::vector<std::uint16_t> Wq, Wk, Wv, Wout, Wproj, Wgate, Wmlpout, nq, nk;
};

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
        infernix::models::qwen4_exp::testing::set_vram_grow_delay(3600.0);

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
        const std::string TW   = qo::join(oroot, "tensors_toplevel_weights");
        const std::string WD   = qo::join(oroot, "tensors_full_depth_weights");
        const std::string WVAE = qo::join(oroot, "tensors_vae_weights");

        // ---- geometry derived from the references (any 16-multiple size, e.g. 256 or 512) ----
        const std::size_t D = 4096, heads = 32, td = 256, mlp0 = 12288;
        const float eps = 1e-6F;
        const double time_factor = 1000.0;
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
        const T img_ref = img_a.floats();                        // [1,4,H,W]
        qo::NpyArray li_a = qo::read_npy(qo::join(TP, "latents_init.npy"));
        qo::require(li_a.shape.size() == 3 && li_a.shape[1] == (int)img_tokens && li_a.shape[2] == 64,
                    "latents_init not [1,img_tokens,64]");
        const T lat_init = represent(li_a.floats());             // [img_tokens,64]
        const T timesteps = qo::read_npy(qo::join(TP, "timesteps.npy")).floats();
        const T sigmas = qo::read_npy(qo::join(TP, "sigmas.npy")).floats();
        const T noise_ref0 = qo::read_npy(qo::join(TP, "noise_pred_step0.npy")).floats();
        const T lat_final_ref = qo::read_npy(qo::join(TP, "latents_final.npy")).floats();
        const T lat_mean = qo::read_npy(qo::join(TP, "latents_mean.npy")).floats();
        const T lat_std = qo::read_npy(qo::join(TP, "latents_std.npy")).floats();
        qo::require(timesteps.size() == 40 && sigmas.size() == 41, "timesteps/sigmas len mismatch");
        qo::require(noise_ref0.size() == img_tokens * 64, "noise_pred_step0 size");
        qo::require(lat_final_ref.size() == img_tokens * 64 && lat_mean.size() == 64 && lat_std.size() == 64,
                    "VAE reference sizes");
        std::printf("geometry tp=%zu Hl=%zu Wl=%zu img_tokens=%zu S=%zu (image %dx%d)\n",
                    tp, Hl, Wl, img_tokens, S, H, W);

        const int Di = (int)D, Si = (int)S, Hi = (int)heads, Mi = (int)mlp0, tp_i = (int)tp;
        const std::size_t sD = S * D;
        const std::size_t sMlp = S * mlp0;

        // ---- cache all 32 blocks' weights in host RAM (represented BF16) once (~14 GiB) ----
        // Host memory is not yielded by the LLM (only VRAM is), so holding this is fine; it makes the
        // 40-step loop H2D-stream per-block per-step without 40x disk reads.
        std::vector<BlockHost> blocks(32);
        for (int l = 0; l < 32; ++l) {
            blocks[l].Wq      = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_q_weight"))).floats()));
            blocks[l].Wk      = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_k_weight"))).floats()));
            blocks[l].Wv      = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_v_weight"))).floats()));
            blocks[l].Wout    = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_to_out_0_weight"))).floats()));
            blocks[l].Wproj   = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_proj_weight"))).floats()));
            blocks[l].Wgate   = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_gate_layer_weight"))).floats()));
            blocks[l].Wmlpout = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "img_mlp_out_weight"))).floats()));
            blocks[l].nq      = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_norm_q_weight"))).floats()));
            blocks[l].nk      = bits_of(represent(qo::read_npy(qo::join(WD, bwf(l, "attn_norm_k_weight"))).floats()));
        }
        std::printf("cached 32 blocks' weights in host RAM (streamed H2D per-block per-step)\n");

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

                // Toplevel/front-end weights (~272 MiB): resident for the whole loop.
                WeightBuf Wimg, Winl, Woutl, Wtl1, Wtl2, Wmod, Wnolin, Wproj;
                // init_from_bits (from represent'd host floats): read + represent + bits_of, then H2D.
                const auto initW = [&](WeightBuf& wb, const std::string& name, int rows, int cols) {
                    wb.init_from_bits(bits_of(represent(qo::read_npy(qo::join(TW, name)).floats())), rows, cols);
                };
                initW(Wimg, "toplevel_img_in_weight.npy", Di, 64);
                initW(Winl, "toplevel_txt_in_in_layer_weight.npy", Di, Di);
                initW(Woutl, "toplevel_txt_in_out_layer_weight.npy", Di, Di);
                initW(Wtl1, "toplevel_time_text_embed_timestep_embedder_linear_1_weight.npy", Di, (int)td);
                initW(Wtl2, "toplevel_time_text_embed_timestep_embedder_linear_2_weight.npy", Di, Di);
                initW(Wmod, "toplevel_modulation_1_weight.npy", 4 * Di, Di);
                initW(Wnolin, "toplevel_norm_out_linear_weight.npy", Di, Di);
                initW(Wproj, "toplevel_proj_out_weight.npy", 64, Di);

                // DiT per-block scratch (allocated once, reused every block and every step).
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

                // Diagnostic gate: INFERNIX_FULL_SKIP_DIT=1 skips the 40-step denoise and seeds the
                // final latent directly from latents_final.npy so the VAE decode runs in isolation.
                const bool skip_dit = std::getenv("INFERNIX_FULL_SKIP_DIT") != nullptr
                                      && *std::getenv("INFERNIX_FULL_SKIP_DIT") != '\0';

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
                if (skip_dit) { const auto bits = bits_of(lat_final_ref); d_latent.h2d(bits.data(), img_tokens * 64 * 2); }
                Buf d_pe;
                d_pe.alloc(tp * D * 2);
                { const auto bits = bits_of(prompt_embeds); d_pe.h2d(bits.data(), tp * D * 2); }

                // txt_in front-end, computed ONCE (it depends only on prompt_embeds, not on t or the
                // latent): ZeroCenterRMSNorm -> in_layer -> gelu(Tanh) -> out_layer, result in d_txt.
                // Separate in/out buffers (tn_v, tn2_v): an in-place GEMM is run-to-run non-deterministic (#33).
                {
                    Tensor pe_v = view(d_pe.d, Di, tp_i), tn_v = view(d_txt.d, Di, tp_i), tn2_v = view(d_txt2.d, Di, tp_i);
                    ops::rmsnorm(pe_v, view(d_tnorm.d, Di, 1), eps, /*unit_offset=*/true, tn_v, 0);
                    ops::linear(tn_v, Winl.w, tn2_v, 0);
                    ops::gelu(tn2_v, ops::GeluMode::Tanh, 0);
                    ops::linear(tn2_v, Woutl.w, tn_v, 0);
                }

                // ---- the 40-step flow-match denoise loop ----
                const std::size_t n_steps = skip_dit ? 0 : timesteps.size();
                for (std::size_t i = 0; i < n_steps; ++i) {
                    const double sigma = sigmas[i], sigma_next = sigmas[i + 1];
                    const float model_input_t = (float)(timesteps[i] / 1000.0);

                    // timestep embedding -> temb -> modulation + nlin (all depend on t, recomputed each step)
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

                    // img_in: current latent [64,img_tokens] -> hs_img [D,img_tokens]
                    Tensor hs_img_v = view(d_hs_img.d, Di, (int)img_tokens);
                    ops::linear(view(d_latent.d, 64, (int)img_tokens), Wimg.w, hs_img_v, 0);

                    // assemble the joint [text(tp) + image(img_tokens)] token-major buffer (txt_in is
                    // stable in d_txt; only hs_img changes per step).
                    {
                        const std::size_t tpB = tp * D * 2, imgB = img_tokens * D * 2;
                        if (cudaMemcpy(d_joint.d, d_txt.d, tpB, cudaMemcpyDeviceToDevice) != cudaSuccess) throw std::runtime_error("joint memcpy txt");
                        if (cudaMemcpy((char*)d_joint.d + tpB, d_hs_img.d, imgB, cudaMemcpyDeviceToDevice) != cudaSuccess) throw std::runtime_error("joint memcpy img");
                    }

                    // 32-block chain: d_joint in, ping-pong with d_hb; after 32 (even) blocks the result
                    // is back in d_joint. Each block's 7 weights + 2 norms are STREAMED from the host
                    // cache (H2D, used, freed) so the resident set peaks at ~one block.
                    void* hin = d_joint.d, *hout = d_hb.d;
                    const std::uint64_t free_at_peak_base = device_free_bytes();
                    for (int l = 0; l < 32; ++l) {
                        WeightBuf Wq, Wk, Wv, Wout, Wproj_b, Wgate, Wmlpout;
                        Wq.init_from_bits(blocks[l].Wq, Di, Di);
                        Wk.init_from_bits(blocks[l].Wk, Di, Di);
                        Wv.init_from_bits(blocks[l].Wv, Di, Di);
                        Wout.init_from_bits(blocks[l].Wout, Di, Di);
                        Wproj_b.init_from_bits(blocks[l].Wproj, Mi, Di);
                        Wgate.init_from_bits(blocks[l].Wgate, Mi, Di);
                        Wmlpout.init_from_bits(blocks[l].Wmlpout, Di, Mi);
                        void* d_nq = nullptr, *d_nk = nullptr;
                        if (cudaMalloc(&d_nq, blocks[l].nq.size() * 2) != cudaSuccess ||
                            cudaMalloc(&d_nk, blocks[l].nk.size() * 2) != cudaSuccess)
                            throw std::runtime_error("cudaMalloc failed (block norm)");
                        cudaMemcpy(d_nq, blocks[l].nq.data(), blocks[l].nq.size() * 2, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_nk, blocks[l].nk.data(), blocks[l].nk.size() * 2, cudaMemcpyHostToDevice);

                        if (i == 0 && l == 0) {
                            const std::uint64_t free_peak = device_free_bytes();
                            const std::uint64_t peak_used = free_at_peak_base > free_peak ? free_at_peak_base - free_peak : 0;
                            std::printf("  peak device working set at step 0 block 0: %llu MiB (of %llu MiB yielded)\n",
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

                    // norm_out: layer_norm(affine=False) -> adaln_scale(scale-only) -> proj_out.
                    Tensor hfinal_v = view(h_final, Di, Si), ln_v = view(d_ln.d, Di, Si);
                    ops::layer_norm(hfinal_v, view(d_ones.d, Di, 1), view(d_zeros.d, Di, 1), eps, ln_v, 0);
                    Tensor norm_out_v = view(d_norm_out.d, Di, Si);
                    ops::adaln_scale(ln_v, nlin_v, 0, tp_i, norm_out_v, 0);
                    Tensor sample_v = view(d_sample.d, 64, Si);
                    ops::linear(norm_out_v, Wproj.w, sample_v, 0);
                    if (cudaStreamSynchronize(0) != cudaSuccess) throw std::runtime_error("DiT step did not complete");

                    // noise_pred = the image tokens (last img_tokens) of the [S,64] velocity, token-major.
                    const std::size_t noise_off = tp * 64 * 2;  // skip the tp text tokens
                    const T noise_f = download((char*)d_sample.d + noise_off, img_tokens * 64);
                    if (i == 0) {  // step-0 localization gate
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
                        check(rel_fro < 0.1, "the streamed DiT step-0 matches the diffusers step-0 velocity (rel_fro < 0.1)");
                    }
                    // scheduler: new_lat = lat + (sigma_next - sigma) * noise, rounded to bf16, back to
                    // d_latent (the reference upcasts to f32, steps, casts back to bf16).
                    const T lat_f = download(d_latent.d, img_tokens * 64);
                    T new_lat(img_tokens * 64);
                    const double ds = (sigma_next - sigma);
                    for (std::size_t p = 0; p < new_lat.size(); ++p)
                        new_lat[p] = (float)((double)lat_f[p] + ds * (double)noise_f[p]);
                    { const auto bits = bits_of(new_lat); d_latent.h2d(bits.data(), img_tokens * 64 * 2); }
                    if (cudaStreamSynchronize(0) != cudaSuccess) throw std::runtime_error("CUDA sync failed (interlude)");
                    if ((i % 5) == 0 || i == n_steps - 1)
                        std::printf("  step %2zu/%zu  t=%8.4f\n", i, n_steps, (double)timesteps[i]);
                }

                if (cudaStreamSynchronize(0) != cudaSuccess) throw std::runtime_error("CUDA sync failed (interlude)");

                // ---- PRIMARY GATE: final latent vs the reference final latent ----
                const T lat_got = download(d_latent.d, img_tokens * 64);
                const double e_lat = qo::rel_fro(lat_got, lat_final_ref);
                std::printf("FINAL_LATENT rel_fro=%.3e (vs latents_final.npy) tol=5e-1 %s\n",
                            e_lat, (e_lat < 0.5 ? "OK" : "FAIL"));
                check(std::isfinite(e_lat) && (skip_dit || e_lat < 0.5), "the 40-step denoise final latent matches the reference (rel_fro < 0.5)");

                // ---- SECONDARY GATE: VAE decode of the final latent -> image ----
                // unpack [img_tokens,64] token-major (token h*Wl+w, feature c) -> [64,Hl,Wl] channel-major,
                // denormalize with the VAE latents mean/std.
                T vae_in(64 * Hl * Wl);
                for (std::size_t c = 0; c < 64; ++c)
                    for (std::size_t h = 0; h < Hl; ++h)
                        for (std::size_t w = 0; w < Wl; ++w) {
                            const double u = (double)lat_got[(h * Wl + w) * 64 + c];
                            vae_in[c * (Hl * Wl) + h * Wl + w] = (float)(u * (double)lat_std[c] + (double)lat_mean[c]);
                        }
                Buf d_vae;
                d_vae.alloc(vae_in.size() * 2);
                { const auto bits = bits_of(vae_in); d_vae.h2d(bits.data(), vae_in.size() * 2); }

                // VAE weights (host cache).
                std::map<std::string, T> WM;
                const auto loadv = [&](const std::string& name) -> const T& {
                    auto it = WM.find(name);
                    if (it != WM.end()) return it->second;
                    return WM.emplace(name, represent(qo::read_npy(qo::join(WVAE, name + ".npy")).floats())).first->second;
                };
                const std::string mid0 = "decoder_mid_block_resnets_0_";
                const std::string mid1 = "decoder_mid_block_resnets_1_";
                const std::string attnp = "decoder_mid_block_attentions_0_";
                const auto load_mid = [&](const std::string& pre) {
                    loadv(pre + "conv1_weight"); loadv(pre + "conv1_bias");
                    loadv(pre + "conv2_weight"); loadv(pre + "conv2_bias");
                    loadv(pre + "norm1_gamma");  loadv(pre + "norm2_gamma");
                };
                loadv("post_quant_conv_weight"); loadv("post_quant_conv_bias");
                loadv("decoder_conv_in_weight"); loadv("decoder_conv_in_bias");
                load_mid(mid0); load_mid(mid1);
                loadv(attnp + "to_qkv_weight"); loadv(attnp + "to_qkv_bias");
                loadv(attnp + "proj_weight");   loadv(attnp + "proj_bias");
                loadv(attnp + "norm_gamma");
                const struct UpCfg { std::size_t cin, cout; bool temporal, up; } cfg[5] = {
                    {1152, 1152, true, true}, {1152, 1152, true, true}, {1152, 576, true, true},
                    {576, 288, false, true},   {288, 144, false, false},
                };
                for (int b = 0; b < 5; ++b) {
                    std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
                    for (int r = 0; r < 3; ++r) {
                        std::string rp = pre + "resnets_" + std::to_string(r) + "_";
                        loadv(rp + "conv1_weight"); loadv(rp + "conv1_bias");
                        loadv(rp + "conv2_weight"); loadv(rp + "conv2_bias");
                        loadv(rp + "norm1_gamma");  loadv(rp + "norm2_gamma");
                        if (r == 0 && cfg[b].cin != cfg[b].cout) {
                            loadv(rp + "conv_shortcut_weight"); loadv(rp + "conv_shortcut_bias");
                        }
                    }
                    if (cfg[b].up) { loadv(pre + "upsampler_resample_1_weight"); loadv(pre + "upsampler_resample_1_bias"); }
                }
                loadv("decoder_norm_out_gamma");
                loadv("decoder_conv_out_weight"); loadv("decoder_conv_out_bias");

                // VAE device decode (all buffers RAII-freed via the va vector at scope end).
                std::vector<Buf> va;
                const auto alloc = [&](std::size_t bytes) -> void* {
                    Buf b; b.alloc(bytes);
                    void* p = b.d;
                    va.push_back(std::move(b));
                    return p;
                };
                const auto xT = [&](void* in, void* out, int rows, int cols) {
                    const int total = rows * cols;
                    transpose_u16_kernel<<<(total + 255) / 256, 256, 0, 0>>>(
                        reinterpret_cast<const std::uint16_t*>(in), reinterpret_cast<std::uint16_t*>(out), rows, cols);
                    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("VAE transpose failed");
                };
                std::map<std::string, void*> dt;
                std::map<std::string, Weight> dlin;
                const auto upl = [&](void* d, const std::string& name) {
                    const auto bits = bits_of(WM.at(name));
                    if (cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)
                        throw std::runtime_error("VAE weight H2D failed");
                };
                const auto d3 = [&](const std::string& name, int a, int b, int c, int dd) -> Tensor {
                    auto it = dt.find(name);
                    if (it == dt.end()) {
                        void* d = alloc(WM.at(name).size() * 2);
                        upl(d, name);
                        it = dt.emplace(name, d).first;
                    }
                    return Tensor(it->second, DType::BF16, {a, b, c, dd});
                };
                const auto dl = [&](const std::string& name, int n, int k) -> Weight {
                    auto it = dlin.find(name);
                    if (it == dlin.end()) {
                        void* d = alloc(WM.at(name).size() * 2);
                        upl(d, name);
                        Weight w{};
                        w.payload = d; w.payload_bytes = WM.at(name).size() * 2;
                        w.qtype = infernix::QType::BF16;
                        w.shape[0] = n; w.shape[1] = k; w.padded_shape[0] = n; w.padded_shape[1] = k;
                        w.ndim = 2; w.qdata = d; w.n = n; w.k = k; w.layout = infernix::QuantLayout::Contiguous;
                        it = dlin.emplace(name, w).first;
                    }
                    return it->second;
                };
                const auto drms = [&](void* x_dev, int C, int P, const std::string& gname) -> void* {
                    const std::size_t tot = (std::size_t)C * P;
                    void* pm = alloc(tot * 2);
                    xT(x_dev, pm, C, P);
                    void* po = alloc(tot * 2);
                    Tensor o_pm = view(pm, C, P), o_po = view(po, C, P);
                    ops::rmsnorm(o_pm, d3(gname, C, 1, 1, 1), 1e-12F, false, o_po, 0);
                    void* chan = alloc(tot * 2);
                    xT(po, chan, P, C);
                    return chan;
                };
                const auto dresnet = [&](void* x_dev, int Cin, int Cout, int Hs, int Ws, const std::string& pre,
                                         bool has_sc) -> void* {
                    const int HW = Hs * Ws;
                    Tensor xin = view(x_dev, Cin, Hs, Ws);
                    Tensor res_buf; Tensor* res = &xin; void* short_dev = nullptr;
                    if (has_sc) {
                        void* xin_tok = alloc((std::size_t)Cin * HW * 2);
                        xT(x_dev, xin_tok, Cin, HW);
                        void* short_tok = alloc((std::size_t)Cout * HW * 2);
                        Tensor o_short = view(short_tok, Cout, HW);
                        ops::linear(view(xin_tok, Cin, HW), dl(pre + "conv_shortcut_weight", Cout, Cin), o_short, 0);
                        ops::add_bias(d3(pre + "conv_shortcut_bias", Cout, 1, 1, 1), o_short, 0);
                        short_dev = alloc((std::size_t)Cout * HW * 2);
                        xT(short_tok, short_dev, HW, Cout);
                        res_buf = Tensor(short_dev, DType::BF16, {Cout, Hs, Ws});
                        res = &res_buf;
                    }
                    void* n1 = drms(x_dev, Cin, HW, pre + "norm1_gamma");
                    Tensor o1 = view(n1, Cin, HW);
                    ops::silu(o1, 0);
                    void* t1 = alloc((std::size_t)Cout * Hs * Ws * 2);
                    Tensor c1 = view(t1, Cout, Hs, Ws);
                    ops::conv2d(view(n1, Cin, Hs, Ws), d3(pre + "conv1_weight", Cout, Cin, 3, 3),
                                d3(pre + "conv1_bias", Cout, 1, 1, 1), nullptr, c1, 0);
                    void* n2 = drms(t1, Cout, HW, pre + "norm2_gamma");
                    Tensor o2 = view(n2, Cout, HW);
                    ops::silu(o2, 0);
                    void* tfin = alloc((std::size_t)Cout * Hs * Ws * 2);
                    Tensor cf = view(tfin, Cout, Hs, Ws);
                    // x must be the 3-dim [Cout,H,W] channel-major view (conv2d derives H,W from x);
                    // the 2-dim o2 view above is only for the in-place silu.
                    ops::conv2d(view(n2, Cout, Hs, Ws), d3(pre + "conv2_weight", Cout, Cout, 3, 3),
                                d3(pre + "conv2_bias", Cout, 1, 1, 1), res, cf, 0);
                    return tfin;
                };
                const auto dattn = [&](void* x_dev, int Hs, int Ws) -> void* {
                    const int C = 1152, HW = Hs * Ws;
                    Tensor xin = view(x_dev, C, HW);
                    void* nrm = drms(x_dev, C, HW, attnp + "norm_gamma");
                    void* on_tok = alloc((std::size_t)C * HW * 2);
                    xT(nrm, on_tok, C, HW);
                    void* qkv_tok = alloc(3 * (std::size_t)C * HW * 2);
                    Tensor o_qkv = view(qkv_tok, 3 * C, HW);
                    ops::linear(view(on_tok, C, HW), dl(attnp + "to_qkv_weight", 3 * C, C), o_qkv, 0);
                    ops::add_bias(d3(attnp + "to_qkv_bias", 3 * C, 1, 1, 1), o_qkv, 0);
                    void* qkv = alloc(3 * (std::size_t)C * HW * 2);
                    xT(qkv_tok, qkv, HW, 3 * C);
                    char* qb = reinterpret_cast<char*>(qkv);
                    const Tensor q = view(qb, C, HW);
                    const Tensor k = view(qb + (C * HW) * 2, C, HW);
                    const Tensor v = view(qb + (2 * C * HW) * 2, C, HW);
                    void* att = alloc((std::size_t)C * HW * 2);
                    Tensor ot = view(att, C, HW);
                    const float scale = 1.0F / std::sqrt(1152.0F);
                    ops::spatial_attn(q, k, v, nullptr, scale, ot, 0);
                    void* ot_tok = alloc((std::size_t)C * HW * 2);
                    xT(att, ot_tok, C, HW);
                    void* prj_tok = alloc((std::size_t)C * HW * 2);
                    Tensor o_prj = view(prj_tok, C, HW);
                    ops::linear(view(ot_tok, C, HW), dl(attnp + "proj_weight", C, C), o_prj, 0);
                    ops::add_bias(d3(attnp + "proj_bias", C, 1, 1, 1), o_prj, 0);
                    void* prj = alloc((std::size_t)C * HW * 2);
                    xT(prj_tok, prj, HW, C);
                    Tensor op = view(prj, C, HW);
                    void* xn = alloc((std::size_t)C * HW * 2);
                    Tensor oxn = view(xn, C, HW);
                    ops::add(op, xin, oxn, 0);
                    return xn;
                };
                const auto dup_up = [&](void* x_dev, int Cin, int Cout, int Hs, int Ws, bool temporal, bool up,
                                        const std::string& pre) -> void* {
                    void* r0 = dresnet(x_dev, Cin, Cout, Hs, Ws, pre + "resnets_0_", Cin != Cout);
                    void* r1 = dresnet(r0, Cout, Cout, Hs, Ws, pre + "resnets_1_", false);
                    void* r2 = dresnet(r1, Cout, Cout, Hs, Ws, pre + "resnets_2_", false);
                    if (up) {
                        const int H0 = Hs, W0 = Ws, HH = 2 * Hs, WW = 2 * Ws;
                        void* xup = alloc((std::size_t)Cout * HH * WW * 2);
                        Tensor o_up = view(xup, Cout, HH, WW);
                        ops::upsample_nearest_2x(view(r2, Cout, H0, W0), o_up, 0);
                        void* sc = alloc((std::size_t)Cout * HH * WW * 2);
                        Tensor o_sc = view(sc, Cout, HH, WW);
                        ops::dupup3d(view(x_dev, Cin, H0, W0), temporal ? 2 : 1, o_sc, 0);
                        void* tfin = alloc((std::size_t)Cout * HH * WW * 2);
                        Tensor sc_buf = Tensor(sc, DType::BF16, {Cout, HH, WW});
                        Tensor o_tf = view(tfin, Cout, HH, WW);
                        ops::conv2d(o_up, d3(pre + "upsampler_resample_1_weight", Cout, Cout, 3, 3),
                                    d3(pre + "upsampler_resample_1_bias", Cout, 1, 1, 1), &sc_buf, o_tf, 0);
                        return tfin;
                    }
                    return r2;
                };

                // run the decode: [64,Hl,Wl] -> post_quant -> conv_in -> mid -> 5 up -> norm_out -> conv_out.
                int Hh = (int)Hl, Ww = (int)Wl;
                void* vx = d_vae.d;  // [64,Hl,Wl] channel-major
                {
                    void* xt = alloc(64 * (Hl * Wl) * 2);
                    xT(vx, xt, 64, (int)(Hl * Wl));
                    void* x1t = alloc(64 * (Hl * Wl) * 2);
                    Tensor o_xt = view(x1t, 64, (int)(Hl * Wl));
                    ops::linear(view(xt, 64, (int)(Hl * Wl)), dl("post_quant_conv_weight", 64, 64), o_xt, 0);
                    ops::add_bias(d3("post_quant_conv_bias", 64, 1, 1, 1), o_xt, 0);
                    void* x1 = alloc(64 * (Hl * Wl) * 2);
                    xT(x1t, x1, (int)(Hl * Wl), 64);
                    vx = x1;
                    void* x2 = alloc((std::size_t)1152 * Hl * Wl * 2);
                    Tensor o_x2 = view(x2, 1152, Hl, Wl);
                    ops::conv2d(view(x1, 64, Hl, Wl), d3("decoder_conv_in_weight", 1152, 64, 3, 3),
                                d3("decoder_conv_in_bias", 1152, 1, 1, 1), nullptr, o_x2, 0);
                    vx = x2;
                    vx = dresnet(vx, 1152, 1152, Hl, Wl, mid0, false);
                    vx = dattn(vx, Hl, Wl);
                    vx = dresnet(vx, 1152, 1152, Hl, Wl, mid1, false);
                    for (int b = 0; b < 5; ++b) {
                        const std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
                        vx = dup_up(vx, (int)cfg[b].cin, (int)cfg[b].cout, Hh, Ww, cfg[b].temporal, cfg[b].up, pre);
                        if (cfg[b].up) { Hh *= 2; Ww *= 2; }
                    }
                    void* n = drms(vx, 144, Hh * Ww, "decoder_norm_out_gamma");
                    Tensor o_n = view(n, 144, Hh * Ww);
                    ops::silu(o_n, 0);
                    void* fin = alloc(4 * (std::size_t)Hh * Ww * 2);
                    Tensor o_fin = view(fin, 4, Hh, Ww);
                    ops::conv2d(view(n, 144, Hh, Ww), d3("decoder_conv_out_weight", 4, 144, 3, 3),
                                d3("decoder_conv_out_bias", 4, 1, 1, 1), nullptr, o_fin, 0);
                    if (cudaStreamSynchronize(0) != cudaSuccess) throw std::runtime_error("VAE decode did not complete");
                    const std::size_t fout = 4 * (std::size_t)Hh * Ww;
                    T dev_final = download(fin, fout);
                    for (auto& v : dev_final) v = std::max(-1.0F, std::min(1.0F, v));
                    const double e_img = qo::rel_fro(dev_final, img_ref);
                    double mn = 1e30, mx = -1e30;
                    for (auto v : dev_final) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
                    std::printf("FINAL_IMAGE  rel_fro=%.3e (vs image.npy) tol=3e-1 %s  (Hh=%d Ww=%d) range [%.3f,%.3f]\n",
                                e_img, (e_img < 0.3 ? "OK" : "FAIL"), Hh, Ww, mn, mx);
                    check(std::isfinite(e_img) && e_img < 0.3, "the VAE-decoded image matches the reference (rel_fro < 0.3)");
                    // dump the generated image (channels 0,1,2 -> RGB) as a P6 PPM so it is viewable.
                    {
                        const std::size_t HW = (std::size_t)Hh * Ww;
                        const char* ppm_env = std::getenv("INFERNIX_IMAGE_OUT_PPM");
                        const char* ppath = (ppm_env && *ppm_env)
                                                ? ppm_env
                                                : "C:/Users/Micke/Documents/Kodprojekt/Qwen Flash Next/image-spike/interlude_full_mine.ppm";
                        FILE* pf = fopen(ppath, "wb");
                        if (pf) {
                            std::fprintf(pf, "P6\n%d %d\n255\n", Ww, Hh);
                            std::vector<unsigned char> px(HW * 3);
                            for (std::size_t h = 0; h < Hh; ++h)
                                for (std::size_t w = 0; w < Ww; ++w) {
                                    const std::size_t idx = h * Ww + w;
                                    for (int c = 0; c < 3; ++c) {
                                        double v = dev_final[(std::size_t)c * HW + idx];
                                        v = std::max(-1.0, std::min(1.0, v));
                                        int u = (int)std::lround(255.0 * 0.5 * (v + 1.0));
                                        px[idx * 3 + c] = (unsigned char)u;
                                    }
                                }
                            fwrite(px.data(), 1, px.size(), pf);
                            fclose(pf);
                            std::printf("WROTE_IMAGE %s (%dx%d)\n", ppath, Ww, Hh);
                        } else {
                            std::printf("IMAGE_WRITE_SKIP %s\n", ppath);
                        }
                    }
                }
            } // all image device memory (DiT + VAE) freed here, before the release

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
        check(after == baseline, "LLM generation is undisturbed after the in-process image job");

        infernix::testing::set_vram_budget_source({});
        result = failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        try { infernix::models::qwen4_exp::testing::end_image_quiesce(); } catch (...) {}
        return 1;
    }
    std::printf(result == 0 ? "qwen4_exp image-interlude full-pipeline checks passed\n" : "FAIL: %d checks failed\n", failures);
    return result;
}
