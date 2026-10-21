// GPU end-to-end assembly of the Qwen-Image-2.1 image pipeline: the qualified text-tower output
// (prompt_embeds) + the qualified DiT (top-level ops + 32 production blocks) + the qualified VAE
// decoder, composed with the FlowMatchEulerDiscrete 40-step sampler, all in BF16, producing the
// FIRST image on device and checking it against a fresh full-pipeline diffusers reference.
//
//   prompt_embeds (fixed) --+
//   init latent (seed)     --+--> [ DiT(t_i) x40  : img_in + txt_in + 32 blocks + norm_out + proj_out
//                                + scheduler.step ] --> final latent  --(PRIMARY GATE vs latents_final)
//   final latent --> VAE decode (post_quant + conv_in + mid + 5 up + conv_out) --> image
//                                                                  (SECONDARY GATE vs image.npy)
//
// Every op is a production op already independently qualified (top-level: #24/#29; 32 blocks: #29;
// attention #22, RoPE #21, AdaLN #23; VAE #31). This test's job is the ASSEMBLY: threading the joint
// [text(11) + image(256)] = 267-token sequence through 32 blocks for 40 flow-match steps, the
// scheduler upcast/cast, and the latent->VAE handoff. The fresh reference (tensors_pipeline/) was
// dumped by image-spike/oracle/dump_pipeline.py running QwenImage21Pipeline.__call__ line-for-line in
// BF16 on this same artifact.
//
// Geometry: 256x256 image -> latent 16x16 -> 256 image tokens; text_len = 11; joint S = 267. D = 4096,
// 32 heads, head_dim 128, time dim 256, mlp 12288.
//
// Tolerances (a-priori, fixed before the first run, never narrowed to pass):
//   final latent vs latents_final.npy : rel-Frobenius < 0.5  (BF16-C++ vs BF16-diffusers, 40 steps x
//       32 blocks accumulated; the DiT 32-block single-step depth error was ~4e-2, 40-step accumulation
//       through a 32-block chain is expected O(0.1..0.5); a real assembly/layout defect is O(1..10))
//   final image  vs image.npy          : rel-Frobenius < 0.3  (the VAE is an additional ~125-op deep
//       BF16 chain on top; the image-level bound is looser than the latent bound by design)
// The final-latent gate is independent of the VAE, so a VAE op defect is isolable from a DiT/sampler
// assembly defect.
//
// Reads $INFERNIX_IMAGE_ORACLE/{tensors_pipeline,tensors_toplevel_weights,tensors_full_depth_weights,
// tensors_vae_weights}/*.npy and tensors_full_depth/block_rotary.npy. Returns 77 (CTest skip) when the
// oracle data is absent; returns 2 (runnable) when no CUDA device.

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
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/scheduler.h"
#include "models/qwen_image/rope.h"
#include "models/qwen_image/noise.h"
#include "models/qwen_image/dit_block.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qo = qioracle;
namespace qi = infernix::models::qwen_image;
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

using T = std::vector<float>;

// VAE decoder up-block config (mirrors the host oracle struct in test_image_vae_prod_cuda.cu).
struct UpCfg { std::size_t cin, cout; bool temporal, up; };

// ---- Host BF16 codec ----
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
void* upload(const T& v) {
    const auto bits = bits_of(v);
    void* d = nullptr;
    cudaMalloc(&d, bits.size() * sizeof(std::uint16_t));
    cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}
T download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    T out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}
// Run-to-run fingerprint of a device buffer: prints stable aggregates that change if ANY element
// differs between two runs, so a back-to-back double-run can localize the first non-deterministic
// op (the first fp_ line that differs is the culprit).
static void fp_print(const char* tag, void* d, std::size_t n) {
    const T v = download(d, n);
    double s = 0.0, ma = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double f = (double)v[i];
        s += f;
        const double a = f < 0.0 ? -f : f;
        if (a > ma) ma = a;
    }
    const double e0  = n > 0 ? (double)v[0] : 0.0;
    const double e10 = n > 10 ? (double)v[10] : 0.0;
    std::printf("STEP0 fp_%s sum=%.9g maxabs=%.9g e0=%.9g e10=%.9g\n", tag, s, ma, e0, e10);
    std::fflush(stdout);
}
Tensor view(void* p, int a, int b, int c = 1, int dd = 1) {
    return Tensor(p, DType::BF16, {a, b, c, dd});
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
        w             = Weight{};
        w.payload     = d;
        w.payload_bytes = bits.size() * sizeof(std::uint16_t);
        w.qtype       = infernix::QType::BF16;
        w.shape[0]    = n;   w.shape[1] = k;
        w.padded_shape[0] = n; w.padded_shape[1] = k;
        w.ndim        = 2;
        w.qdata       = d;
        w.n           = n;   w.k = k;
        w.layout      = infernix::QuantLayout::Contiguous;
    }
    ~BfWeight() { if (d) cudaFree(d); }
    BfWeight(const BfWeight&) = delete;
    BfWeight& operator=(const BfWeight&) = delete;
};

// DiT per-block scratch: the shared module owns the layout (one code path with the interlude/service).
using WS = qi::DiTWorkspace;

// One DiT block forward through the production ops (BF16 linears, BF16 everything else). din/dout
// are the running joint hidden (viewed [Di, Si] = Si tokens x Di features). Identical 18-op sequence
// to the qualified single/depth-block tests.
// One DiT block forward: shared module function (verbatim move from this test).
using qi::run_block;

std::string bwf(int l, const char* role) {
    std::string s = "blk"; s += std::to_string(l); s += "_"; s += role; s += ".npy";
    return s;
}

}  // namespace

int main() {
    try {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
        if (env == nullptr || std::string(env).empty()) {
            std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
            return 77;
        }
        const std::string oracle(env);
        // The pipeline reference dir (prompt/seed/resolution-dependent) defaults to
        // <oracle>/tensors_pipeline; INFERNIX_IMAGE_ORACLE_PIPELINE overrides just that dir so the
        // model-only weight dirs (toplevel/full_depth/vae/full_depth) stay on <oracle>.
        std::string TP = qo::join(oracle, "tensors_pipeline");
        if (const char* pe_pipeline = std::getenv("INFERNIX_IMAGE_ORACLE_PIPELINE"))
            if (*pe_pipeline) TP = pe_pipeline;
        const std::string TW  = qo::join(oracle, "tensors_toplevel_weights");
        const std::string WD  = qo::join(oracle, "tensors_full_depth_weights");
        const std::string WVAE = qo::join(oracle, "tensors_vae_weights");
        const std::string TD  = qo::join(oracle, "tensors_full_depth");
        const std::string TS  = qo::join(oracle, "tensors_step0");

        auto F  = [&](const std::string& dir, const std::string& name) {
            return qo::read_npy(qo::join(dir, name)).floats();
        };
        auto LW = [&](const std::string& dir, const std::string& name) {
            return qo::read_npy(qo::join(dir, name)).floats();
        };

        // ---- constants (model geometry) ----
        const std::size_t D = 4096, heads = 32, td = 256, mlp0 = 12288;
        const double time_factor = 1000.0;
        const float eps = 1e-6F;

        // ---- geometry: from the reference tensors, or a general-resolution override (no dump) ----
        qo::NpyArray pe_a = qo::read_npy(qo::join(TP, "prompt_embeds.npy"));   // text cond is resolution-independent
        qo::require(pe_a.shape.size() == 3 && pe_a.shape[0] == 1 && pe_a.shape[2] == (int)D,
                    "prompt_embeds not [1,?,4096]");
        const std::size_t tp = (std::size_t)pe_a.shape[1];         // text prefix length (variable)
        const T prompt_embeds = represent(pe_a.floats());          // [tp,4096] token-major
        std::uint64_t seed_val = 0;
        bool seeded = false;
        if (const char* se = std::getenv("INFERNIX_IMAGE_SEED"); se && *se) {
            seeded = true;
            seed_val = std::strtoull(se, nullptr, 10);
        }
        // A seeded run needs no reference final/image; INFERNIX_IMAGE_SIZE=HxW then sizes the job from the
        // env instead of image.npy, proving generation at a resolution that has NO oracle dump (#37 size+seed).
        int ovH = 0, ovW = 0;
        bool geom_override = false;
        if (seeded)
            if (const char* z = std::getenv("INFERNIX_IMAGE_SIZE"); z && *z) {
                int a = 0, b = 0;
                if (std::sscanf(z, "%dx%d", &a, &b) == 2 && a > 0 && b > 0 && a % 16 == 0 && b % 16 == 0) {
                    ovH = a; ovW = b; geom_override = true;
                }
            }
        const bool oracle_geom = !geom_override;   // the dumped 256x256 reference tensors are valid comparisons
        int H = 0, W = 0;
        T img_ref;
        if (geom_override) {
            H = ovH; W = ovW;
            std::printf("GEOMETRY_OVERRIDE HxW=%dx%d from INFERNIX_IMAGE_SIZE (no reference dump at this geometry)\n", H, W);
        } else {
            qo::NpyArray img_a = qo::read_npy(qo::join(TP, "image.npy"));
            qo::require(img_a.shape.size() == 4 && img_a.shape[0] == 1 && img_a.shape[1] == 4 &&
                        (int)img_a.shape[2] % 16 == 0 && (int)img_a.shape[3] % 16 == 0,
                        "image.npy not [1,4,H,W] (H,W multiples of 16)");
            H = (int)img_a.shape[2]; W = (int)img_a.shape[3];      // image spatial
            img_ref = img_a.floats();                              // [1,4,H,W]
        }
        const std::size_t Hl = (std::size_t)H / 16, Wl = (std::size_t)W / 16;  // latent spatial (16x)
        const std::size_t img_tokens = Hl * Wl;
        const std::size_t S = tp + img_tokens;

        // ---- init latent (seeded runs draw it from fill_normal; seed parsed above) ----
        T lat_init;
        if (seeded) {
            lat_init.assign(img_tokens * 64, 0.0F);
            qi::fill_normal(seed_val, std::span<float>(lat_init));
            lat_init = represent(lat_init);                        // bf16-represent, matches the stored f32->bf16 init
            std::printf("SEED_WIRING init latent from fill_normal(seed=%llu) count=%zu (NOT bit-comparable to latents_init.npy)\n",
                        (unsigned long long)seed_val, lat_init.size());
        } else {
            qo::NpyArray li_a = qo::read_npy(qo::join(TP, "latents_init.npy"));
            qo::require(li_a.shape.size() == 3 && li_a.shape[1] == (int)img_tokens && li_a.shape[2] == 64,
                        "latents_init not [1,img_tokens,64]");
            lat_init = represent(li_a.floats());                   // [img_tokens,64] token-major
        }
        // WIRED (variable-input #37): the denoise schedule is COMPUTED from the production scheduler
        // module (resolution-dependent shift mu from this job's geometry token count), not read from a
        // fixed .npy. The dumped sigmas.npy/timesteps.npy are kept as the parity reference to prove the
        // module is a drop-in; the loop then consumes the COMPUTED schedule.
        const qi::ImageGeometry geom = qi::ImageGeometry::derive((std::uint32_t)H, (std::uint32_t)W);
        const qi::SchedulerConfig scfg;
        const double mu = qi::calculate_shift(scfg, geom.tokens);
        std::size_t n_steps = 40;
        qi::Schedule sched;
        if (oracle_geom) {
            const T timesteps_ref = F(TP, "timesteps.npy");        // [40] reference
            const T sigmas_ref = F(TP, "sigmas.npy");             // [41] reference
            n_steps = timesteps_ref.size();
            qo::require(n_steps == 40 && sigmas_ref.size() == n_steps + 1, "timesteps/sigmas len mismatch");
            sched = qi::build_schedule(scfg, n_steps, mu);
            const double s_par = sched.sigmas.size() == sigmas_ref.size() ? qo::rel_fro(sched.sigmas, sigmas_ref) : 1e9;
            const double t_par = sched.timesteps.size() == timesteps_ref.size() ? qo::rel_fro(sched.timesteps, timesteps_ref) : 1e9;
            std::printf("SCHEDULER_WIRING HxW=%dx%d tokens=%u steps=%zu mu=%.8f sigmas_rel=%.3e timesteps_rel=%.3e %s\n",
                        H, W, geom.tokens, n_steps, mu, s_par, t_par, (s_par < 2e-5 && t_par < 2e-5) ? "OK" : "CHECK");
            qo::require(s_par < 2e-5 && t_par < 2e-5, "production scheduler module must reproduce the dumped schedule");
        } else {
            if (const char* st = std::getenv("INFERNIX_IMAGE_STEPS"); st && *st) { const int v = std::atoi(st); if (v > 0) n_steps = (std::size_t)v; }
            sched = qi::build_schedule(scfg, n_steps, mu);
            std::printf("SCHEDULER_COMPUTE HxW=%dx%d tokens=%u steps=%zu mu=%.8f (computed at an un-dumped geometry)\n",
                        H, W, geom.tokens, n_steps, mu);
        }
        const T timesteps = sched.timesteps;                       // the sampler consumes the COMPUTED values
        const T sigmas = sched.sigmas;
        const T lat_mean = F(TP, "latents_mean.npy");              // [64] resolution-independent, needed for decode
        const T lat_std = F(TP, "latents_std.npy");                // [64]
        qo::require(lat_mean.size() == 64 && lat_std.size() == 64, "latents_mean/std size");
        T lat_final_ref, noise_ref0, traj_full, traj0_ref;         // [img_tokens,64]-sized: only the dump geometry has them
        if (oracle_geom) {
            lat_final_ref = F(TP, "latents_final.npy");            // [img_tokens,64]
            qo::require(lat_final_ref.size() == img_tokens * 64, "latents_final size");
            qo::require(img_ref.size() == (std::size_t)4 * H * W, "image.npy size");
            noise_ref0 = F(TP, "noise_pred_step0.npy");            // [img_tokens,64] step-0 velocity, image tokens
            qo::require(noise_ref0.size() == img_tokens * 64, "noise_pred_step0 size");
            traj_full = F(TP, "latent_traj.npy");                  // [steps,img_tokens,64]
            traj0_ref.assign(traj_full.begin(), traj_full.begin() + img_tokens * 64);
        }
        // step-0 localization refs from the single-forward dumper (splits the step-0 DiT divergence
        // into front-end hidden [b0_in] vs 32-block stack [b31_out] vs post-block output [sample_full]).
        // Loaded only if present so the primary/secondary gates still evaluate when the dumper is absent.
        T b0_ref, sample_full_ref, hsimg_ref, hstxt_ref, rot_ref, temb_ref, mod_ref, blocks_ref;
        bool have_b0 = false, have_sample = false, have_hsimg = false, have_hstxt = false,
             have_rot = false, have_temb = false, have_mod = false, have_blocks = false;
        {
            std::ifstream fb0(qo::join(TS, "b0_in.npy"));
            if (fb0.good()) { have_b0 = true; b0_ref = F(TS, "b0_in.npy"); }
            std::ifstream fsf(qo::join(TS, "sample_full.npy"));
            if (fsf.good()) { have_sample = true; sample_full_ref = F(TS, "sample_full.npy"); }
            std::ifstream fhimg(qo::join(TS, "hs_img.npy"));
            if (fhimg.good()) { have_hsimg = true; hsimg_ref = F(TS, "hs_img.npy"); }
            std::ifstream fhstx(qo::join(TS, "hs_txt.npy"));
            if (fhstx.good()) { have_hstxt = true; hstxt_ref = F(TS, "hs_txt.npy"); }
            std::ifstream frot(qo::join(TS, "rot_inter.npy"));
            if (frot.good()) { have_rot = true; rot_ref = F(TS, "rot_inter.npy"); }
            std::ifstream ftemb(qo::join(TS, "temb.npy"));
            if (ftemb.good()) { have_temb = true; temb_ref = F(TS, "temb.npy"); }
            std::ifstream fmod(qo::join(TS, "mod_table.npy"));
            if (fmod.good()) { have_mod = true; mod_ref = F(TS, "mod_table.npy"); }
            std::ifstream fblk(qo::join(TS, "blocks_out.npy"));
            if (fblk.good()) { have_blocks = true; blocks_ref = F(TS, "blocks_out.npy"); }
        }

        // The tensors_step0 localization refs are S-dependent (dumped at one geometry): b0_in [S,D],
        // sample_full [S,64], hs_img [img_tokens,D], hs_txt [tp,D], blocks_out [32,S,D]. A 512 run sets
        // INFERNIX_IMAGE_ORACLE to the 256 base for model weights, so these are 256-sized; the per-block
        // window would read off the end of blocks_out (OOB host read in the std::vector range copy) and the
        // point refs would trip rel_fro's size-require. Gate each on its exact expected size so a foreign
        // geometry disables just those refs. temb (timestep embed) and mod ([4D,2] table) are S-independent
        // and stay. At the matching geometry (256) every size matches and nothing is disabled.
        {
            const std::size_t sD0 = S * D;
            if (have_b0 && b0_ref.size() != sD0) have_b0 = false;
            if (have_sample && sample_full_ref.size() != S * 64) have_sample = false;
            if (have_hsimg && hsimg_ref.size() != img_tokens * D) have_hsimg = false;
            if (have_hstxt && hstxt_ref.size() != tp * D) have_hstxt = false;
            if (have_blocks && blocks_ref.size() != (std::size_t)32 * sD0) have_blocks = false;
            if (!have_b0 && !have_blocks && !have_sample && !have_hsimg && !have_hstxt)
                std::printf("STEP0_LOCALIZATION off (tensors_step0 refs foreign/absent for S=%zu)\n", S);
        }

        // ---- RoPE self-check: build_rope(8,8,8) must match the dumped full-depth rotary ----
        {
            const std::vector<float> rot8 = qo::build_rope(8, 8, 8);
            const T rotref = F(TD, "block_rotary.npy");
            const double e = qo::rel_fro(rot8, rotref);
            std::printf("rope_selfcheck build_rope(8,8,8) vs block_rotary.npy rel_fro=%.3e %s\n",
                        e, (e < 1e-2 ? "OK" : "CHECK"));
        }

        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
            std::printf("NO_CUDA_DEVICE\n");
            std::fflush(stdout);
            return 2;
        }

        // ---- toplevel weights (BF16) ----
        const BfWeight Wimg(LW(TW, "toplevel_img_in_weight.npy"), (int)D, 64);
        const BfWeight Winl(LW(TW, "toplevel_txt_in_in_layer_weight.npy"), (int)D, (int)D);
        const BfWeight Woutl(LW(TW, "toplevel_txt_in_out_layer_weight.npy"), (int)D, (int)D);
        const BfWeight Wtl1(LW(TW, "toplevel_time_text_embed_timestep_embedder_linear_1_weight.npy"), (int)D, (int)td);
        const BfWeight Wtl2(LW(TW, "toplevel_time_text_embed_timestep_embedder_linear_2_weight.npy"), (int)D, (int)D);
        const BfWeight Wmod(LW(TW, "toplevel_modulation_1_weight.npy"), (int)(4 * D), (int)D);
        const BfWeight Wnolin(LW(TW, "toplevel_norm_out_linear_weight.npy"), (int)D, (int)D);
        const BfWeight Wproj(LW(TW, "toplevel_proj_out_weight.npy"), 64, (int)D);
        const T w_tnorm = represent(LW(TW, "toplevel_txt_in_text_norm_weight.npy"));  // [4096]

        // ---- all 32 block weights (BF16, resident for the whole 40-step loop) ----
        struct BW {
            std::unique_ptr<BfWeight> Wq, Wk, Wv, Wout, Wproj, Wgate, Wmlpout;
            void* d_nq = nullptr;
            void* d_nk = nullptr;
        };
        std::vector<BW> blocks(32);
        for (int l = 0; l < 32; ++l) {
            blocks[l].Wq     = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "attn_to_q_weight"))), (int)D, (int)D);
            blocks[l].Wk     = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "attn_to_k_weight"))), (int)D, (int)D);
            blocks[l].Wv     = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "attn_to_v_weight"))), (int)D, (int)D);
            blocks[l].Wout   = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "attn_to_out_0_weight"))), (int)D, (int)D);
            blocks[l].Wproj  = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "img_mlp_proj_weight"))), (int)mlp0, (int)D);
            blocks[l].Wgate  = std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "img_mlp_gate_layer_weight"))), (int)mlp0, (int)D);
            blocks[l].Wmlpout= std::make_unique<BfWeight>(represent(LW(WD, bwf(l, "img_mlp_out_weight"))), (int)D, (int)mlp0);
            blocks[l].d_nq   = upload(represent(LW(WD, bwf(l, "attn_norm_q_weight"))));
            blocks[l].d_nk   = upload(represent(LW(WD, bwf(l, "attn_norm_k_weight"))));
        }

        const int Di = (int)D, Si = (int)S, Hi = (int)heads, Mi = (int)mlp0, tp_i = (int)tp;
        const std::size_t sD = S * D;          // 267*4096
        const std::size_t sMlp = S * mlp0;     // 267*12288

        // ---- shared device constants ----
        // d_ones/d_zeros back both the single-token layer_norm views [D,1] (run_block, norm_out) and
        // the two-row time-embedding views [D,2] (temb / mod SiLU gate), so they must hold 2*D.
        const T ones(2 * D, 1.0F), zeros(2 * D, 0.0F);
        void* d_ones = upload(ones);
        void* d_zeros = upload(zeros);
        void* d_tnorm = upload(w_tnorm);
        // RoPE for this geometry: build_rope(11,16,16) -> [S,64] re/im interleaved -> [128,1,S] device.
        // WIRED (variable-input #37): the DiT rotary comes from the production build_joint_rope module
        // (identical 3-axis formula to the oracle-ref build_rope); verified equal so the assembled pipeline
        // consumes the module, and resolution enters through (Hl,Wl) rather than a per-geometry .npy.
        const std::vector<float> rot = qi::build_joint_rope((int)tp, (int)Hl, (int)Wl);
        {
            const std::vector<float> rot_oracle = qo::build_rope(tp, Hl, Wl);
            const double r_par = qo::rel_fro(rot, rot_oracle);
            std::printf("ROPE_WIRING build_joint_rope(%u,%u,%u) vs oracle-ref rel_fro=%.3e %s\n",
                        (unsigned)tp, (unsigned)Hl, (unsigned)Wl, r_par, (r_par < 1e-6) ? "OK" : "CHECK");
            qo::require(r_par < 1e-6, "production build_joint_rope must equal the oracle-ref rope");
        }
        void* drot = upload(rot);
        if (have_rot) {
            const double er = (rot_ref.size() == rot.size()) ? qo::rel_fro(rot, rot_ref) : 1e9;
            std::printf("STEP0 rotary    build_rope(11,16,16) vs rot_inter.npy rel_fro=%.4e (%s)\n",
                        er, (rot_ref.size() == rot.size() && er < 5e-2) ? "OK" : "CHECK");
        }

        // ---- DiT scratch ----
        WS ws{};
        void** wp[] = {&ws.dln1, &ws.dmod1, &ws.dq, &ws.dk, &ws.dv, &ws.dqr, &ws.dkr,
                       &ws.dattn, &ws.dattn_out, &ws.dx1, &ws.dln2, &ws.dmod2, &ws.dmlp_out};
        for (void** p : wp) { *p = nullptr; CUDA_CHECK(cudaMalloc(p, sD * 2)); }
        for (void** p : {&ws.dproj, &ws.dgate, &ws.dhidden}) { *p = nullptr; CUDA_CHECK(cudaMalloc(p, sMlp * 2)); }

        // per-step buffers
        void *d_latent, *d_hs_img, *d_txt, *d_txt2, *d_joint, *d_hb;
        void *d_tproj, *d_temb_a, *d_temb_a_silu, *d_temb, *d_temb_silu, *d_mod, *d_nlin;
        void *d_ln, *d_norm_out, *d_sample;
        CUDA_CHECK(cudaMalloc(&d_latent, img_tokens * 64 * 2));
        CUDA_CHECK(cudaMalloc(&d_hs_img, img_tokens * D * 2));
        CUDA_CHECK(cudaMalloc(&d_txt, tp * D * 2));
        CUDA_CHECK(cudaMalloc(&d_txt2, tp * D * 2));
        CUDA_CHECK(cudaMalloc(&d_joint, sD * 2));
        CUDA_CHECK(cudaMalloc(&d_hb, sD * 2));
        CUDA_CHECK(cudaMalloc(&d_tproj, 2 * td * 2));
        const std::size_t s2 = 2 * D;
        CUDA_CHECK(cudaMalloc(&d_temb_a, s2 * 2));
        CUDA_CHECK(cudaMalloc(&d_temb_a_silu, s2 * 2));
        CUDA_CHECK(cudaMalloc(&d_temb, s2 * 2));
        CUDA_CHECK(cudaMalloc(&d_temb_silu, s2 * 2));
        CUDA_CHECK(cudaMalloc(&d_mod, 4 * D * 2 * 2));
        CUDA_CHECK(cudaMalloc(&d_nlin, s2 * 2));
        CUDA_CHECK(cudaMalloc(&d_ln, sD * 2));
        CUDA_CHECK(cudaMalloc(&d_norm_out, sD * 2));
        CUDA_CHECK(cudaMalloc(&d_sample, S * 64 * 2));

        // seed the running latent (bf16, token-major [256,64] = view [64,256])
        {
            const auto bits = bits_of(lat_init);
            CUDA_CHECK(cudaMemcpy(d_latent, bits.data(), img_tokens * 64 * 2, cudaMemcpyHostToDevice));
        }
        const T pe_rep = prompt_embeds;
        void* d_pe = upload(pe_rep);  // [11,4096]
        // step-0 localization captures (filled only at i==0)
        T b0_mine, b31_mine, sample_full_mine, hsimg_mine, hstxt_mine, temb_mine, mod_mine, blk_mine;
        double blk_over[32] = {0}, blk_txt[32] = {0}, blk_img[32] = {0};
        // rel_fro of a vs b over the token range [t0,t1) of a token-major [S,D] buffer
        auto relf_slice = [&](const T& a, const T& b, std::size_t t0, std::size_t t1) {
            double nn = 0.0, dd = 0.0;
            for (std::size_t s = t0; s < t1; ++s)
                for (std::size_t d = 0; d < D; ++d) {
                    const double x = (double)a[s * D + d] - (double)b[s * D + d];
                    dd += x * x;
                    nn += (double)b[s * D + d] * (double)b[s * D + d];
                }
            return nn > 0.0 ? std::sqrt(dd / nn) : 0.0;
        };

        // ---- the 40-step flow-match denoise loop ----
        // INFERNIX_DBG_STEPS caps the denoise loop (default = full 40) so a single-step, full-32-block
        // chain can be probed (e.g. under compute-sanitizer memcheck) without running all 40 steps.
        const std::size_t n_steps_dbg = [&] { const char* e = std::getenv("INFERNIX_DBG_STEPS"); return e ? (std::size_t)std::atoi(e) : n_steps; }();
        for (std::size_t i = 0; i < n_steps_dbg; ++i) {
            const double sigma = sigmas[i], sigma_next = sigmas[i + 1];
            const float model_input_t = (float)(timesteps[i] / 1000.0);

            // timestep embedding -> temb -> modulation + nlin (all depend on t, recomputed each step)
            const std::vector<float> tvec = {model_input_t, 0.0F};
            const std::vector<float> tproj = qo::temporal_timesteps(tvec, td, time_factor);  // [2,256]
            {
                const auto bits = bits_of(tproj);
                CUDA_CHECK(cudaMemcpy(d_tproj, bits.data(), tproj.size() * 2, cudaMemcpyHostToDevice));
            }
            Tensor tproj_v = view(d_tproj, (int)td, 2), temb_a_v = view(d_temb_a, Di, 2);
            ops::linear(tproj_v, Wtl1.w, temb_a_v, 0);
            Tensor temb_a_silu_v = view(d_temb_a_silu, Di, 2);
            ops::silu_mul(temb_a_v, view(d_ones, Di, 2), temb_a_silu_v, 0);
            Tensor temb_v = view(d_temb, Di, 2);
            ops::linear(temb_a_silu_v, Wtl2.w, temb_v, 0);
            if (i == 0) temb_mine = download(d_temb, 2 * D);  // time_text_embed output [2,4096]
            Tensor temb_silu_v = view(d_temb_silu, Di, 2);
            ops::silu_mul(temb_v, view(d_ones, Di, 2), temb_silu_v, 0);
            Tensor mod_v = view(d_mod, 4 * Di, 2);
            ops::linear(temb_silu_v, Wmod.w, mod_v, 0);
            if (i == 0) mod_mine = download(d_mod, 4 * D * 2);  // modulation table [2,4D]
            Tensor nlin_v = view(d_nlin, Di, 2);
            ops::linear(temb_silu_v, Wnolin.w, nlin_v, 0);

            // img_in: latent [64,256] -> hs_img [4096,256]
            Tensor hs_img_v = view(d_hs_img, Di, (int)img_tokens);
            ops::linear(view(d_latent, 64, (int)img_tokens), Wimg.w, hs_img_v, 0);
            if (i == 0) hsimg_mine = download(d_hs_img, img_tokens * D);

            // txt_in: prompt_embeds [4096,11] -> ZeroCenterRMSNorm -> in_layer -> gelu(Tanh) -> out_layer.
            // The two linears use SEPARATE in/out buffers (tn_v, tn2_v): an in-place GEMM (input==output) at
            // M=K=4096,N=11 is run-to-run non-deterministic for larger-magnitude inputs (#33, isolated via
            // fingerprint double-run); the real model never aliases a linear's input and output.
            Tensor pe_v = view(d_pe, Di, (int)tp), tn_v = view(d_txt, Di, (int)tp), tn2_v = view(d_txt2, Di, (int)tp);
            if (i == 0) fp_print("pe_in", d_pe, tp * D);
            ops::rmsnorm(pe_v, view(d_tnorm, Di, 1), eps, /*unit_offset=*/true, tn_v, 0);
            if (i == 0) fp_print("rmsnorm", d_txt, tp * D);
            ops::linear(tn_v, Winl.w, tn2_v, 0);
            if (i == 0) fp_print("lin1", d_txt2, tp * D);
            ops::gelu(tn2_v, ops::GeluMode::Tanh, 0);
            if (i == 0) fp_print("gelu", d_txt2, tp * D);
            ops::linear(tn2_v, Woutl.w, tn_v, 0);
            if (i == 0) fp_print("lin2", d_txt, tp * D);
            if (i == 0) hstxt_mine = download(d_txt, tp * D);

            // assemble the joint [text(11) + image(256)] = 267-token token-major buffer (both parts
            // are token-major [tokens,4096], D=4096 innermost, so a contiguous memcpy composes them).
            {
                const std::size_t tpB = tp * D * 2, imgB = img_tokens * D * 2;
                CUDA_CHECK(cudaMemcpy(d_joint, d_txt, tpB, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy((char*)d_joint + tpB, d_hs_img, imgB, cudaMemcpyDeviceToDevice));
            }
            if (i == 0) b0_mine = download(d_joint, sD);  // block-0 input (front-end hidden)

            // 32-block chain: d_joint in, ping-pong with d_hb; after 32 (even) blocks the result is
            // back in d_joint. (INFERNIX_DBG_BOUNDS the chain for diagnostics; default 32.)
            const int nblocks_dbg = [] { const char* e = std::getenv("INFERNIX_DBG_BLOCKS"); return e ? std::atoi(e) : 32; }();
            void* hin = d_joint, *hout = d_hb;
            for (int l = 0; l < nblocks_dbg; ++l) {
                run_block(hin, hout, blocks[l].Wq->w, blocks[l].Wk->w, blocks[l].Wv->w, blocks[l].Wout->w,
                          blocks[l].Wproj->w, blocks[l].Wgate->w, blocks[l].Wmlpout->w,
                          blocks[l].d_nq, blocks[l].d_nk, d_mod, drot, d_ones, d_zeros, ws,
                          Si, Di, Hi, Mi, tp_i, eps);
                if (i == 0 && have_blocks) {
                    const std::size_t off = (std::size_t)l * sD;
                    blk_mine = download(hout, sD);
                    const T bref(blocks_ref.begin() + off, blocks_ref.begin() + off + sD);
                    blk_over[l] = qo::rel_fro(blk_mine, bref);
                    blk_txt[l] = relf_slice(blk_mine, bref, 0, tp);
                    blk_img[l] = relf_slice(blk_mine, bref, tp, S);
                }
                std::swap(hin, hout);
            }
            // after 32 swaps hin == d_joint holds the block-31 output
            void* h_final = hin;
            if (i == 0) {
                cudaError_t ec = cudaDeviceSynchronize();
                size_t freeB = 0, totalB = 0; cudaMemGetInfo(&freeB, &totalB);
                std::printf("  PRE.cpy sync=%s h_final=%p d_joint=%p d_hb=%p eq_joint=%d eq_hb=%d sD=%zu freeMiB=%.0f\n",
                            cudaGetErrorString(ec), h_final, d_joint, d_hb,
                            (h_final == d_joint) ? 1 : 0, (h_final == d_hb) ? 1 : 0, sD, freeB / (1024.0 * 1024.0));
                std::fflush(stdout);
                const T tiny = download(h_final, 64);
                std::printf("  PRE.tiny ok e0=%.4f e63=%.4f\n", (double)tiny[0], (double)tiny[63]);
                std::fflush(stdout);
            }
            if (i == 0) b31_mine = download(h_final, sD);  // block-31 output
            { const cudaError_t e0 = cudaDeviceSynchronize();
              if (i == 0) { std::printf("  POST.cpy sync=%s\n", cudaGetErrorString(e0)); std::fflush(stdout); } }
            auto psync = [&](const char* tag) {
                if (i == 0) {
                    const cudaError_t e = cudaDeviceSynchronize();
                    std::printf("  POST.%s sync=%s\n", tag, cudaGetErrorString(e));
                    std::fflush(stdout);
                }
            };

            // norm_out: layer_norm(affine=False) -> adaln_scale(scale-only, row-select) -> proj_out
            Tensor hfinal_v = view(h_final, Di, Si), ln_v = view(d_ln, Di, Si);
            ops::layer_norm(hfinal_v, view(d_ones, Di, 1), view(d_zeros, Di, 1), eps, ln_v, 0); psync("ln");
            Tensor norm_out_v = view(d_norm_out, Di, Si);
            ops::adaln_scale(ln_v, nlin_v, 0, tp_i, norm_out_v, 0); psync("sc");
            Tensor sample_v = view(d_sample, 64, Si);
            ops::linear(norm_out_v, Wproj.w, sample_v, 0); psync("proj");
            if (i == 0) sample_full_mine = download(d_sample, S * 64);  // full [267,64] velocity

            // noise_pred = the image tokens (last 256) of the [267,64] velocity, token-major.
            const std::size_t noise_off = tp * 64 * 2;  // skip the 11 text tokens
            const T lat_f = download(d_latent, img_tokens * 64);
            const T noise_f = download((char*)d_sample + noise_off, img_tokens * 64);
            T new_lat(img_tokens * 64);
            const double ds = (sigma_next - sigma);
            for (std::size_t p = 0; p < new_lat.size(); ++p)
                new_lat[p] = (float)((double)lat_f[p] + ds * (double)noise_f[p]);
            if (i == 0 && !seeded) {  // localize (oracle-init only): DiT forward vs scheduler accumulation
                if (have_hsimg)  std::printf("STEP0 hs_img     rel_fro=%.4e (vs hs_img.npy)\n", qo::rel_fro(hsimg_mine, hsimg_ref));
                if (have_hstxt)  std::printf("STEP0 hs_txt     rel_fro=%.4e (vs hs_txt.npy)\n", qo::rel_fro(hstxt_mine, hstxt_ref));
                if (have_temb)   std::printf("STEP0 temb       rel_fro=%.4e (vs temb.npy)\n", qo::rel_fro(temb_mine, temb_ref));
                if (have_mod)    std::printf("STEP0 mod_table  rel_fro=%.4e (vs mod_table.npy)\n", qo::rel_fro(mod_mine, mod_ref));
                if (have_b0)     std::printf("STEP0 b0_in      rel_fro=%.4e (vs b0_in.npy)\n", qo::rel_fro(b0_mine, b0_ref));
                if (have_blocks) {
                    for (int l = 0; l < 32; ++l)
                        std::printf("  blk %2d  all=%.3e  txt=%.3e  img=%.3e\n", l, blk_over[l], blk_txt[l], blk_img[l]);
                }
                if (have_sample) std::printf("STEP0 sample_full rel_fro=%.4e (vs sample_full.npy)\n", qo::rel_fro(sample_full_mine, sample_full_ref));
                std::printf("STEP0 noisepred rel_fro=%.4e (vs noise_pred_step0.npy)\n", qo::rel_fro(noise_f, noise_ref0));
                std::printf("STEP0 postlatent rel_fro=%.4e (vs latent_traj[0])\n", qo::rel_fro(new_lat, traj0_ref));
                // scale/direction: ||a||/||b||, best-fit scale c=(a.b)/(b.b) (c~1 => no scale bias),
                // and the residual norm relative to the reference after removing the fitted scale.
                double na2 = 0, nb2 = 0, dot = 0;
                for (std::size_t p = 0; p < noise_f.size(); ++p) {
                    na2 += double(noise_f[p]) * noise_f[p];
                    nb2 += double(noise_ref0[p]) * noise_ref0[p];
                    dot += double(noise_f[p]) * noise_ref0[p];
                }
                const double c = dot / nb2;
                double resid2 = 0;
                for (std::size_t p = 0; p < noise_f.size(); ++p) {
                    const double r = noise_f[p] - c * noise_ref0[p];
                    resid2 += r * r;
                }
                std::printf("STEP0 noisepred ||a||/||b||=%.4f  bestfit_c=%.4f  residual_rel_fro=%.4e\n",
                            std::sqrt(na2 / nb2), c, std::sqrt(resid2 / nb2));
                std::fflush(stdout);
            }
            {  // round to bf16 (reference upcasts to f32, steps, casts back to bf16)
                const auto bits = bits_of(new_lat);
                CUDA_CHECK(cudaMemcpy(d_latent, bits.data(), img_tokens * 64 * 2, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaGetLastError());
            if (!seeded) {
                // Free-running state error E_x(i) vs the reference trajectory: separates a smooth
                // chaotic accumulation (transformer fine; E_x climbs) from a step-specific spike
                // (systematic defect). latent_traj[i] = the reference state after step i.
                const std::size_t off = (std::size_t)i * img_tokens * 64;
                const T bref(traj_full.begin() + off, traj_full.begin() + off + img_tokens * 64);
                std::printf("  step %2zu/%zu  t=%8.4f  state E_x=%.4e\n", i, n_steps,
                            (double)timesteps[i], qo::rel_fro(new_lat, bref));
                std::fflush(stdout);
            }
        }

        if (n_steps_dbg < n_steps) {
            std::printf("DBG_STEPS_CAP_REACHED n_steps_dbg=%zu n_steps=%zu (probe: chain only; skipping gate/force/VAE)\n",
                        n_steps_dbg, n_steps);
            std::fflush(stdout);
            CUDA_CHECK(cudaDeviceSynchronize());
            return 0;
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- final latent (PRIMARY gate vs the reference; informational under a seeded init) ----
        const T lat_got = download(d_latent, img_tokens * 64);
        bool lat_got_finite = true;
        for (float v : lat_got) if (!std::isfinite(v)) lat_got_finite = false;
        const double e_lat = (oracle_geom && lat_final_ref.size() == lat_got.size())
                                 ? qo::rel_fro(lat_got, lat_final_ref) : 0.0;   // 0 = no dump at this geometry
        const double tol_lat = 0.5;
        if (seeded)
            std::printf("FINAL_LATENT(seeded) finite=%d  rel_to_torch_ref=%.3e (informational; seeded init differs by design)\n",
                        (int)lat_got_finite, e_lat);
        else
            std::printf("FINAL_LATENT rel_fro=%.3e (vs latents_final.npy) tol=%.1e %s\n",
                        e_lat, tol_lat, (e_lat < tol_lat ? "OK" : "FAIL"));

        // ---- TRAJECTORY-FORCE pass (decisive per-step velocity check) ----
        // Free-running drift (E_x above) conflates two things: (a) a fixable per-step SYSTEMATIC
        // velocity defect, and (b) the irreducible per-step bf16 kernel error that a sensitive
        // 40-step ODE amplifies (the velocity field is only ever measured at the drifted state).
        // This pass feeds the REFERENCE state latent_traj[i-1] (initial noise at i==0) into the DiT
        // and takes one Euler step; the result should equal the reference's own next state
        // latent_traj[i]. A large one-step error at some step = systematic defect (fix it); a small
        // one (~bf16 floor) at every sampled step = velocity field correct, free-running drift is
        // pure ODE-sensitivity accumulation. Sampled (not all 40) to bound runtime under the watchdog.
        const std::vector<int> force_steps = seeded ? std::vector<int>{}
                                                    : std::vector<int>{0, 5, 10, 15, 20, 25, 30, 39};
        if (!force_steps.empty())
            std::printf("=== TRAJECTORY-FORCE (one-step at reference states) ===\n");
        for (int si : force_steps) {
            const std::size_t i = (std::size_t)si;
            const T in = (i == 0) ? lat_init
                                  : T(traj_full.begin() + (std::size_t)(i - 1) * img_tokens * 64,
                                      traj_full.begin() + (std::size_t)i * img_tokens * 64);
            {
                const auto bits = bits_of(in);
                CUDA_CHECK(cudaMemcpy(d_latent, bits.data(), img_tokens * 64 * 2, cudaMemcpyHostToDevice));
            }
            const double sigma = sigmas[i], sigma_next = sigmas[i + 1];
            const std::vector<float> tvec = {(float)(timesteps[i] / 1000.0), 0.0F};
            const std::vector<float> tproj = qo::temporal_timesteps(tvec, td, time_factor);
            {
                const auto bits = bits_of(tproj);
                CUDA_CHECK(cudaMemcpy(d_tproj, bits.data(), tproj.size() * 2, cudaMemcpyHostToDevice));
            }
            Tensor tproj_v = view(d_tproj, (int)td, 2), temb_a_v = view(d_temb_a, Di, 2);
            ops::linear(tproj_v, Wtl1.w, temb_a_v, 0);
            Tensor temb_a_silu_v = view(d_temb_a_silu, Di, 2);
            ops::silu_mul(temb_a_v, view(d_ones, Di, 2), temb_a_silu_v, 0);
            Tensor temb_v = view(d_temb, Di, 2);
            ops::linear(temb_a_silu_v, Wtl2.w, temb_v, 0);
            Tensor temb_silu_v = view(d_temb_silu, Di, 2);
            ops::silu_mul(temb_v, view(d_ones, Di, 2), temb_silu_v, 0);
            Tensor mod_v = view(d_mod, 4 * Di, 2);
            ops::linear(temb_silu_v, Wmod.w, mod_v, 0);
            Tensor nlin_v = view(d_nlin, Di, 2);
            ops::linear(temb_silu_v, Wnolin.w, nlin_v, 0);
            Tensor hs_img_v = view(d_hs_img, Di, (int)img_tokens);
            ops::linear(view(d_latent, 64, (int)img_tokens), Wimg.w, hs_img_v, 0);
            Tensor pe_v = view(d_pe, Di, (int)tp), tn_v = view(d_txt, Di, (int)tp), tn2_v = view(d_txt2, Di, (int)tp);
            ops::rmsnorm(pe_v, view(d_tnorm, Di, 1), eps, /*unit_offset=*/true, tn_v, 0);
            ops::linear(tn_v, Winl.w, tn2_v, 0);
            ops::gelu(tn2_v, ops::GeluMode::Tanh, 0);
            ops::linear(tn2_v, Woutl.w, tn_v, 0);
            {
                const std::size_t tpB = tp * D * 2, imgB = img_tokens * D * 2;
                CUDA_CHECK(cudaMemcpy(d_joint, d_txt, tpB, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy((char*)d_joint + tpB, d_hs_img, imgB, cudaMemcpyDeviceToDevice));
            }
            void* hin = d_joint, *hout = d_hb;
            for (int l = 0; l < 32; ++l) {
                run_block(hin, hout, blocks[l].Wq->w, blocks[l].Wk->w, blocks[l].Wv->w, blocks[l].Wout->w,
                          blocks[l].Wproj->w, blocks[l].Wgate->w, blocks[l].Wmlpout->w,
                          blocks[l].d_nq, blocks[l].d_nk, d_mod, drot, d_ones, d_zeros, ws,
                          Si, Di, Hi, Mi, tp_i, eps);
                std::swap(hin, hout);
            }
            void* h_final = hin;
            Tensor hfinal_v = view(h_final, Di, Si), ln_v = view(d_ln, Di, Si);
            ops::layer_norm(hfinal_v, view(d_ones, Di, 1), view(d_zeros, Di, 1), eps, ln_v, 0);
            Tensor norm_out_v = view(d_norm_out, Di, Si);
            ops::adaln_scale(ln_v, nlin_v, 0, tp_i, norm_out_v, 0);
            Tensor sample_v = view(d_sample, 64, Si);
            ops::linear(norm_out_v, Wproj.w, sample_v, 0);
            const std::size_t noise_off = tp * 64 * 2;
            const T noise_f = download((char*)d_sample + noise_off, img_tokens * 64);
            const double ds = (sigma_next - sigma);
            T new_lat(img_tokens * 64);
            for (std::size_t p = 0; p < new_lat.size(); ++p)
                new_lat[p] = (float)((double)in[p] + ds * (double)noise_f[p]);
            const T bref(traj_full.begin() + (std::size_t)i * img_tokens * 64,
                         traj_full.begin() + (std::size_t)(i + 1) * img_tokens * 64);
            std::printf("  force step %2d  t=%8.4f  one-step rel_fro=%.4e\n", si,
                        (double)timesteps[i], qo::rel_fro(new_lat, bref));
            std::fflush(stdout);
        }

        // ---- VAE decode of the produced final latent -> image (secondary gate) ----
        // unpack [256,64] token-major (token h*Wl+w, feature c) -> [64,16,16] channel-major, denorm.
        T vae_in(64 * Hl * Wl);
        for (std::size_t c = 0; c < 64; ++c)
            for (std::size_t h = 0; h < Hl; ++h)
                for (std::size_t w = 0; w < Wl; ++w) {
                    const double u = lat_got[(h * Wl + w) * 64 + c];
                    vae_in[c * (Hl * Wl) + h * Wl + w] = (float)(u * lat_std[c] + lat_mean[c]);
                }
        void* d_vae = upload(vae_in);  // channel-major [64,16,16] bf16

        // ---- VAE weights (host-represented cache + device loaders), identical graph to #31 ----
        std::map<std::string, T> WM;
        const auto loadv = [&](const std::string& name) -> const T& {
            auto it = WM.find(name);
            if (it != WM.end()) return it->second;
            return WM.emplace(name, represent(LW(WVAE, name + ".npy"))).first->second;
        };
        const std::string mid0 = "decoder_mid_block_resnets_0_";
        const std::string mid1 = "decoder_mid_block_resnets_1_";
        const std::string attnp = "decoder_mid_block_attentions_0_";
        auto load_mid = [&](const std::string& pre) {
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
        const UpCfg cfg[5] = {
            {1152, 1152, true, true}, {1152, 1152, true, true}, {1152, 576, true, true},
            {576, 288, false, true},  {288, 144, false, false},
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

        // ---- VAE device decode (device-only; ops already qualified in #31) ----
        std::vector<void*> galloc;
        const auto alloc = [&](std::size_t bytes) -> void* {
            void* p = nullptr;
            if (cudaMalloc(&p, bytes) != cudaSuccess) {
                std::printf("VAE cudaMalloc failed (%zu bytes)\n", bytes);
                std::fflush(stdout);
                std::exit(3);
            }
            galloc.push_back(p);
            return p;
        };
        const auto xT = [&](void* in, void* out, int rows, int cols) {
            const int total = rows * cols;
            transpose_u16_kernel<<<(total + 255) / 256, 256, 0, 0>>>(
                reinterpret_cast<const std::uint16_t*>(in), reinterpret_cast<std::uint16_t*>(out), rows, cols);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { std::printf("transpose FAIL: %s\n", cudaGetErrorString(e)); std::abort(); }
        };
        std::map<std::string, void*> dt;
        std::map<std::string, Weight> dlin;
        const auto upl = [&](void* d, const std::string& name) {
            const auto bits = bits_of(WM.at(name));
            cudaError_t e = cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) { std::printf("weight upload FAIL %s: %s\n", name.c_str(), cudaGetErrorString(e)); std::abort(); }
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
        const auto dresnet = [&](void* x_dev, int Cin, int Cout, int H, int W, const std::string& pre,
                                 bool has_sc) -> void* {
            const int HW = H * W;
            Tensor xin = view(x_dev, Cin, H, W);
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
                res_buf = Tensor(short_dev, DType::BF16, {Cout, H, W});
                res = &res_buf;
            }
            void* n1 = drms(x_dev, Cin, HW, pre + "norm1_gamma");
            Tensor o1 = view(n1, Cin, HW);
            ops::silu(o1, 0);
            void* t1 = alloc((std::size_t)Cout * H * W * 2);
            Tensor c1 = view(t1, Cout, H, W);
            ops::conv2d(view(n1, Cin, H, W), d3(pre + "conv1_weight", Cout, Cin, 3, 3),
                        d3(pre + "conv1_bias", Cout, 1, 1, 1), nullptr, c1, 0);
            void* n2 = drms(t1, Cout, HW, pre + "norm2_gamma");
            Tensor o2 = view(n2, Cout, H, W);
            ops::silu(o2, 0);
            void* tfin = alloc((std::size_t)Cout * H * W * 2);
            Tensor cf = view(tfin, Cout, H, W);
            ops::conv2d(o2, d3(pre + "conv2_weight", Cout, Cout, 3, 3),
                        d3(pre + "conv2_bias", Cout, 1, 1, 1), res, cf, 0);
            return tfin;
        };
        const auto dattn = [&](void* x_dev, int H, int W) -> void* {
            const int C = 1152, HW = H * W;
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
        const auto dup_up = [&](void* x_dev, int Cin, int Cout, int H, int W, bool temporal, bool up,
                                const std::string& pre) -> void* {
            void* r0 = dresnet(x_dev, Cin, Cout, H, W, pre + "resnets_0_", Cin != Cout);
            void* r1 = dresnet(r0, Cout, Cout, H, W, pre + "resnets_1_", false);
            void* r2 = dresnet(r1, Cout, Cout, H, W, pre + "resnets_2_", false);
            if (up) {
                const int H0 = H, W0 = W, HH = 2 * H, WW = 2 * W;
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

        // run the decode at H0=W0=16
        double e_img = 0.0;
        bool img_finite = true, img_nondeg = false;   // outer scope: the seeded verdict reads these
        const double tol_img = 0.3;
        void* vx = d_vae;  // [64,16,16] channel-major
        {
            // post_quant (1x1 = linear + bias): chan->token, linear, bias, token->chan
            void* xt = alloc(64 * (Hl * Wl) * 2);
            xT(vx, xt, 64, Hl * Wl);
            void* x1t = alloc(64 * (Hl * Wl) * 2);
            Tensor o_xt = view(x1t, 64, Hl * Wl);
            ops::linear(view(xt, 64, Hl * Wl), dl("post_quant_conv_weight", 64, 64), o_xt, 0);
            ops::add_bias(d3("post_quant_conv_bias", 64, 1, 1, 1), o_xt, 0);
            void* x1 = alloc(64 * (Hl * Wl) * 2);
            xT(x1t, x1, Hl * Wl, 64);
            vx = x1;
            void* x2 = alloc((std::size_t)1152 * Hl * Wl * 2);
            Tensor o_x2 = view(x2, 1152, Hl, Wl);
            ops::conv2d(view(x1, 64, Hl, Wl), d3("decoder_conv_in_weight", 1152, 64, 3, 3),
                        d3("decoder_conv_in_bias", 1152, 1, 1, 1), nullptr, o_x2, 0);
            vx = x2;
            vx = dresnet(vx, 1152, 1152, Hl, Wl, mid0, false);
            vx = dattn(vx, Hl, Wl);
            vx = dresnet(vx, 1152, 1152, Hl, Wl, mid1, false);
            int Hh = Hl, Ww = Wl;
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
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            const std::size_t fout = 4 * (std::size_t)Hh * Ww;
            T dev_final = download(fin, fout);
            for (float v : dev_final) if (!std::isfinite(v)) img_finite = false;   // finiteness BEFORE clamp
            for (auto& v : dev_final) v = std::max(-1.0F, std::min(1.0F, v));
            if (oracle_geom && img_ref.size() == dev_final.size()) e_img = qo::rel_fro(dev_final, img_ref);
            if (seeded)
                std::printf("FINAL_IMAGE(seeded) finite=%d  rel_to_torch_ref=%.3e (informational)  (Hh=%d Ww=%d)\n",
                            (int)img_finite, e_img, Hh, Ww);
            else
                std::printf("FINAL_IMAGE  rel_fro=%.3e (vs image.npy) tol=%.1e %s  (Hh=%d Ww=%d)\n",
                            e_img, tol_img, (e_img < tol_img ? "OK" : "FAIL"), Hh, Ww);
            // per-channel stats for disclosure
            double mn = 1e30, mx = -1e30;
            for (auto v : dev_final) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
            img_nondeg = img_finite && (mx - mn) > 0.05;   // not a flat / collapsed image
            std::printf("image range [%.3f, %.3f]  %s  (ref range reported by dumper: [-1,1])\n",
                        mn, mx, img_nondeg ? "non-degenerate" : "DEGENERATE");
            // Dump the generated image (channels 0,1,2 -> RGB) as a P6 PPM so it is viewable.
            // (Matches the reference convention: ch3 is the constant ~1.0 alpha channel.)
            {
                const std::size_t HW = (std::size_t)Hh * Ww;
                const char* ppm_env = std::getenv("INFERNIX_IMAGE_OUT_PPM");
                std::string ppath_s;
                if (ppm_env && *ppm_env) ppath_s = ppm_env;
                else if (seeded) ppath_s = "C:/Users/Micke/Documents/Kodprojekt/Qwen Flash Next/image-spike/e2e_mine_seed" +
                                           std::to_string((unsigned long long)seed_val) + ".ppm";
                else ppath_s = "C:/Users/Micke/Documents/Kodprojekt/Qwen Flash Next/image-spike/e2e_mine.ppm";
                const char* ppath = ppath_s.c_str();
                FILE* pf = fopen(ppath, "wb");
                if (pf) {
                    std::fprintf(pf, "P6\n%d %d\n255\n", Ww, Hh);
                    std::vector<unsigned char> px((std::size_t)Hh * Ww * 3);
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
                    std::printf("IMAGE_WRITE_FAIL %s\n", ppath);
                }
            }
        }

        // ---- verdict ----
        // Oracle mode: the reference-comparison gates (unchanged).
        // Seeded mode: BEHAVIOURAL gate -- the pipeline ran to completion, the final latent and decoded
        // image are finite, and the image is non-degenerate (a real generated picture). Deliberately NOT
        // the reference tolerance: the seeded init is a different RNG draw than the torch-seeded reference.
        const bool ok = seeded ? (lat_got_finite && img_finite && img_nondeg && std::isfinite(e_img))
                               : ((e_lat < tol_lat) && std::isfinite(e_lat) && (e_img < tol_img) && std::isfinite(e_img));
        // free everything
        for (void* p : galloc) cudaFree(p);
        cudaFree(d_vae);
        for (void** p : wp) if (*p) cudaFree(*p);
        for (void** p : {&ws.dproj, &ws.dgate, &ws.dhidden}) if (*p) cudaFree(*p);
        cudaFree(d_latent); cudaFree(d_hs_img); cudaFree(d_txt); cudaFree(d_txt2); cudaFree(d_joint); cudaFree(d_hb);
        cudaFree(d_tproj); cudaFree(d_temb_a); cudaFree(d_temb_a_silu); cudaFree(d_temb); cudaFree(d_temb_silu);
        cudaFree(d_mod); cudaFree(d_nlin); cudaFree(d_ln); cudaFree(d_norm_out); cudaFree(d_sample);
        cudaFree(d_ones); cudaFree(d_zeros); cudaFree(d_tnorm); cudaFree(drot); cudaFree(d_pe);
        for (auto& b : blocks) { if (b.d_nq) cudaFree(b.d_nq); if (b.d_nk) cudaFree(b.d_nk); }

        std::printf("%s %s (latent=%.3e img=%.3e)\n",
                    seeded ? "SEEDED" : "ORACLE",
                    ok ? "IMAGE_PIPELINE_PROD_TEST: PASS" : "IMAGE_PIPELINE_PROD_TEST: FAIL",
                    e_lat, e_img);
        std::fflush(stdout);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("IMAGE_PIPELINE_PROD_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
