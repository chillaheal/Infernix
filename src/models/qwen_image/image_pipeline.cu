#include "models/qwen_image/image_pipeline.h"

#include "artifact/reader.h"
#include "core/tensor.h"
#include "core/vram_budget.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/dit_adaln.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/vae_decoder.h"
#include "models/qwen_image/dit_block.h"
#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/image_device_util.h"
#include "models/qwen_image/rope.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen_image {

namespace {

using infernix::DType;
using infernix::Tensor;
using infernix::Weight;
namespace artifact = infernix::artifact;
namespace ops      = infernix::ops;

constexpr double kTimeFactor = 1000.0; // the qualified timestep convention (t / 1000 -> embedder)

// Device transpose: in [rows,cols] row-major -> out [cols,rows] row-major. Bridges the channel-major
// [C,P] layout of the VAE conv/attn ops and the token-major [P,C] layout that ops::linear reads.
__global__ void transpose_u16_kernel(const std::uint16_t* __restrict__ in, std::uint16_t* __restrict__ out,
                                     int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    out[c * rows + r] = in[r * cols + c];
}


// The [text_prefix, HxW-grid] rope table is the host module's; this job's tables come from it.
std::vector<float> temporal_timesteps(const std::vector<float>& t, std::size_t dim, double time_factor) {
    // The qualified temporal-timestep formula: sinusoidal of (time_factor * t), cos in the first
    // half and sin in the second, one row per batch entry ([dim, batch] token-major after upload).
    const std::size_t half = dim / 2;
    std::vector<double> freqs(half);
    for (std::size_t j = 0; j < half; ++j) {
        freqs[j] = std::exp(-std::log(10000.0) * static_cast<double>(j) / static_cast<double>(half));
    }
    std::vector<float> out(t.size() * dim);
    for (std::size_t b = 0; b < t.size(); ++b) {
        const double tt = time_factor * static_cast<double>(t[b]);
        for (std::size_t j = 0; j < half; ++j) {
            const double a = tt * freqs[j];
            out[b * dim + j]         = static_cast<float>(std::cos(a));
            out[b * dim + half + j]  = static_cast<float>(std::sin(a));
        }
    }
    return out;
}

// One DiT block's nine roles as raw artifact object bytes, cached in PINNED host RAM for the whole
// job (the copy stream's H2D comes from here; the artifact is read once per object per job). Pinned
// so the async copy is genuinely asynchronous (pageable source forces a driver-side bounce that
// cannot overlap compute). The flags are chosen by host_cache_alloc_flags() (WC measured, default
// cached).
struct BlockBytes {
    PinnedBytes q, k, v, out, gate, up, down, query_norm, key_norm;
};

// The bundle-B fas 2 copy lane: a non-blocking copy stream, two device block slots, and the events
// that order the prefetch copies against the block compute enqueued on the job's (blocking) image
// stream. A non-blocking stream does NOT implicitly serialize with a blocking stream the way the
// legacy default stream does, which is what lets block i+1's H2D run while block i computes. The
// slots live only for the DiT loop (freed before the VAE decode) and are owned by Buf, so the image
// ledger still accounts them and the interlude's release gate sees them go to zero.
struct CopyLane {
    // The prefetch double-buffers exactly two slots (the [2] arrays and the i%2 parity below). This is
    // the resident-block count the need model reserves for; if it ever changes, update the parity AND
    // kImageResidentBlocks together -- the static_assert makes a silent drift between them a build error.
    static_assert(kImageResidentBlocks == 2,
                  "CopyLane double-buffers exactly two slots; keep kImageResidentBlocks at 2 or rework the parity.");
    cudaStream_t stream = nullptr;
    cudaEvent_t  copy_done[2] = {nullptr, nullptr}; // block l's nine objects landed in slot l%2
    cudaEvent_t  freed[2]     = {nullptr, nullptr}; // block l's compute finished reading slot l%2
    cudaEvent_t  slot_ready   = nullptr;            // the two slots exist (ordered vs the compute lane)
    cudaEvent_t  cp0s = nullptr, cp0e = nullptr;    // timing sample of one block's transfer
    Buf          slot[2];
    void*        slot_p[2] = {nullptr, nullptr};
    bool         active = false;

    CopyLane() = default;

    void start(cudaStream_t compute, std::size_t slot_bytes) {
        cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "image copy stream");
        for (int i = 0; i < 2; ++i) {
            cuda_ok(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming), "image copy event");
            cuda_ok(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming), "image free event");
        }
        cuda_ok(cudaEventCreateWithFlags(&slot_ready, cudaEventDisableTiming), "image slot event");
        cuda_ok(cudaEventCreate(&cp0s), "image timing event"); // default timing flag
        cuda_ok(cudaEventCreate(&cp0e), "image timing event");
        for (int i = 0; i < 2; ++i) {
            slot[i].alloc(slot_bytes); // stream-ordered on the compute lane's pool (g_image_stream)
            slot_p[i] = slot[i].d;
        }
        cuda_ok(cudaEventRecord(slot_ready, compute), "image slot ready record");
        cuda_ok(cudaStreamWaitEvent(stream, slot_ready, 0), "image slot ready wait");
        active = true;
    }
    ~CopyLane() {
        if (active) { (void)cudaStreamSynchronize(stream); }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
        for (int i = 0; i < 2; ++i) {
            if (copy_done[i] != nullptr) { cudaEventDestroy(copy_done[i]); }
            if (freed[i] != nullptr)     { cudaEventDestroy(freed[i]); }
        }
        if (slot_ready != nullptr) { cudaEventDestroy(slot_ready); }
        if (cp0s != nullptr) { cudaEventDestroy(cp0s); }
        if (cp0e != nullptr) { cudaEventDestroy(cp0e); }
        // Buf slot[0]/slot[1] free their device bytes via RAII after this body.
    }
    CopyLane(const CopyLane&) = delete;
    CopyLane& operator=(const CopyLane&) = delete;
};

// One VAE weight set's execution context, shared by the job's decode and the reference-image encode
// (#75): both chains are the same qualified primitives over different weight prefixes. Weights
// upload on first use and live in `va` until the graph ends; scratch is released BUFFER AT A TIME
// (retire) the moment its last consumer is enqueued, not at scope exit: holding every plane to the
// end summed ALL intermediates (~60+ GiB at 2048x2048) instead of the few planes a stage keeps live.
struct VaeGraph {
    artifact::Reader& reader;
    const VaeWeights& vae;
    cudaStream_t stream;
    std::map<std::string, Uploaded> va;
    std::vector<Buf> scratch;

    void* alloc(std::size_t bytes) {
        Buf b;
        b.alloc(bytes);
        void* p = b.d;
        scratch.push_back(std::move(b));
        return p;
    }
    // Every consumer of p is enqueued on this same stream before the retire, so the stream-ordered
    // free cannot precede a reader; retired Bufs own nothing and the scope-exit destructor skips them.
    void retire(void* p) {
        for (auto& b : scratch) {
            if (b.d == p) {
                b.retire(stream);
                return;
            }
        }
        throw std::runtime_error("VAE graph: retired a pointer the scratch does not own");
    }
    // in [rows,cols] row-major -> out [cols,rows] row-major: bridges the channel-major [C,P] conv/
    // attn layout and the token-major [P,C] layout ops::linear reads.
    void xT(void* in, void* dst, int rows, int cols) {
        const int total = rows * cols;
        transpose_u16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(in), reinterpret_cast<std::uint16_t*>(dst), rows, cols);
        cuda_ok(cudaGetLastError(), "VAE transpose");
    }
    Tensor d3(const std::string& name, int a, int b, int c, int dd) {
        auto it = va.find(name);
        if (it == va.end()) {
            it = va.emplace(name, upload_ref(reader, vae.by_name.at(name))).first;
        }
        return Tensor(it->second.buf.d, DType::BF16, {a, b, c, dd});
    }
    // A linear-route weight: the upload lives in the map (device memory outlives every kernel the
    // graph enqueues against it), so the Weight may be handed out by value.
    Weight dl(const std::string& name) {
        auto it = va.find(name);
        if (it == va.end()) {
            it = va.emplace(name, upload_ref(reader, vae.by_name.at(name))).first;
        }
        return it->second.w;
    }
    // The input is never retired here (dresnet/att keep it live for the residual/shortcut); pm and
    // po die as soon as rmsnorm and the channel-major transpose have consumed them.
    void* drms(void* x_dev, int C, int P, const std::string& gname) {
        const std::size_t tot = static_cast<std::size_t>(C) * P;
        void* pm = alloc(tot * 2);
        xT(x_dev, pm, C, P);
        void* po = alloc(tot * 2);
        Tensor xr = view(pm, C, P), or_t = view(po, C, P);
        ops::rmsnorm(xr, d3(gname, C, 1, 1, 1), 1e-12F, false, or_t, stream);
        retire(pm);
        void* chan = alloc(tot * 2);
        xT(po, chan, P, C);
        retire(po);
        return chan;
    }
    // owns_input: x_dev's last reader here is the shortcut/norm1 (has_sc) or conv2's residual add
    // (no shortcut). The caller keeps x_dev live when it re-reads it later (encoder avg_shortcut).
    void* dresnet(void* x_dev, int Cin_, int Cout, int Hs, int Ws, const std::string& pre, bool has_sc,
                  bool owns_input) {
        const int HW = Hs * Ws;
        Tensor xin = view(x_dev, Cin_, Hs, Ws);
        Tensor res_buf;
        Tensor* res = &xin;
        void* short_dev = nullptr;
        if (has_sc) {
            void* xin_tok = alloc(static_cast<std::size_t>(Cin_) * HW * 2);
            xT(x_dev, xin_tok, Cin_, HW);
            void* short_tok = alloc(static_cast<std::size_t>(Cout) * HW * 2);
            Tensor short_t = view(short_tok, Cout, HW);
            ops::linear(view(xin_tok, Cin_, HW), dl(pre + "conv_shortcut.weight"), short_t, stream);
            retire(xin_tok);
            ops::add_bias(d3(pre + "conv_shortcut.bias", Cout, 1, 1, 1), short_t, stream);
            short_dev = alloc(static_cast<std::size_t>(Cout) * HW * 2);
            xT(short_tok, short_dev, HW, Cout);
            retire(short_tok);
            res_buf = Tensor(short_dev, DType::BF16, {Cout, Hs, Ws});
            res     = &res_buf;
        }
        void* n1 = drms(x_dev, Cin_, HW, pre + "norm1.gamma");
        Tensor n1_flat = view(n1, Cin_, HW);
        ops::silu(n1_flat, stream);
        void* t1 = alloc(static_cast<std::size_t>(Cout) * Hs * Ws * 2);
        Tensor t1_v = view(t1, Cout, Hs, Ws);
        ops::conv2d(view(n1, Cin_, Hs, Ws), d3(pre + "conv1.weight", Cout, Cin_, 3, 3),
                    d3(pre + "conv1.bias", Cout, 1, 1, 1), nullptr, t1_v, stream);
        retire(n1);
        void* n2 = drms(t1, Cout, HW, pre + "norm2.gamma");
        retire(t1);
        Tensor n2_flat = view(n2, Cout, HW);
        ops::silu(n2_flat, stream);
        void* tfin = alloc(static_cast<std::size_t>(Cout) * Hs * Ws * 2);
        Tensor tfin_v = view(tfin, Cout, Hs, Ws);
        // conv2's x must be the 3-dim [C,H,W] channel-major view (conv2d derives H,W from x);
        // the 2-dim view above is only for the in-place silu.
        ops::conv2d(view(n2, Cout, Hs, Ws), d3(pre + "conv2.weight", Cout, Cout, 3, 3),
                    d3(pre + "conv2.bias", Cout, 1, 1, 1), res, tfin_v, stream);
        retire(n2);
        if (has_sc) retire(short_dev);
        if (owns_input) retire(x_dev);
        return tfin;
    }
    // The mid-block attention (encoder C=768 / decoder C=1152): RMS norm -> 1x1 to_qkv (a linear on
    // the token-major view) -> single-head full SDPA at 1/sqrt(C) -> 1x1 proj -> residual add.
    void* att(void* x_dev, int C, const std::string& attnp, int Hs, int Ws) {
        const int HW = Hs * Ws;
        Tensor xin = view(x_dev, C, HW);
        void* nrm = drms(x_dev, C, HW, attnp + "norm.gamma");
        void* on_tok = alloc(static_cast<std::size_t>(C) * HW * 2);
        xT(nrm, on_tok, C, HW);
        retire(nrm);
        void* qkv_tok = alloc(3 * static_cast<std::size_t>(C) * HW * 2);
        Tensor qkv_t = view(qkv_tok, 3 * C, HW);
        ops::linear(view(on_tok, C, HW), dl(attnp + "to_qkv.weight"), qkv_t, stream);
        retire(on_tok);
        ops::add_bias(d3(attnp + "to_qkv.bias", 3 * C, 1, 1, 1), qkv_t, stream);
        void* qkv = alloc(3 * static_cast<std::size_t>(C) * HW * 2);
        xT(qkv_tok, qkv, HW, 3 * C);
        retire(qkv_tok);
        char* qb = reinterpret_cast<char*>(qkv);
        const Tensor q = view(qb, C, HW);
        const Tensor k = view(qb + (C * HW) * 2, C, HW);
        const Tensor v = view(qb + (2 * C * HW) * 2, C, HW);
        void* att_buf = alloc(static_cast<std::size_t>(C) * HW * 2);
        const float scale = 1.0F / std::sqrt(static_cast<float>(C));
        Tensor att_t = view(att_buf, C, HW);
        ops::spatial_attn(q, k, v, nullptr, scale, att_t, stream);
        retire(qkv);
        void* ot_tok = alloc(static_cast<std::size_t>(C) * HW * 2);
        xT(att_buf, ot_tok, C, HW);
        retire(att_buf);
        void* prj_tok = alloc(static_cast<std::size_t>(C) * HW * 2);
        Tensor prj_t = view(prj_tok, C, HW);
        ops::linear(view(ot_tok, C, HW), dl(attnp + "proj.weight"), prj_t, stream);
        retire(ot_tok);
        ops::add_bias(d3(attnp + "proj.bias", C, 1, 1, 1), prj_t, stream);
        void* prj = alloc(static_cast<std::size_t>(C) * HW * 2);
        xT(prj_tok, prj, HW, C);
        retire(prj_tok);
        void* xn = alloc(static_cast<std::size_t>(C) * HW * 2);
        Tensor xn_t = view(xn, C, HW);
        ops::add(view(prj, C, HW), xin, xn_t, stream);
        retire(prj);
        retire(x_dev);
        return xn;
    }
    void* dup_up(void* x_dev, int Cin_, int Cout, int Hs, int Ws, bool temporal, bool up,
                 const std::string& pre) {
        // resnets.0 keeps x_dev alive (owned=false): the temporal shortcut below re-reads it. On the
        // up=false path dupup3d never runs, so x_dev dies after resnets.0.
        void* r0 = dresnet(x_dev, Cin_, Cout, Hs, Ws, pre + "resnets.0.", Cin_ != Cout, false);
        if (!up) retire(x_dev);
        void* r1 = dresnet(r0, Cout, Cout, Hs, Ws, pre + "resnets.1.", false, true);
        void* r2 = dresnet(r1, Cout, Cout, Hs, Ws, pre + "resnets.2.", false, true);
        if (!up) { return r2; }
        const int HH = 2 * Hs, WW = 2 * Ws;
        void* xup = alloc(static_cast<std::size_t>(Cout) * HH * WW * 2);
        Tensor xup_v = view(xup, Cout, HH, WW);
        ops::upsample_nearest_2x(view(r2, Cout, Hs, Ws), xup_v, stream);
        retire(r2);
        void* sc = alloc(static_cast<std::size_t>(Cout) * HH * WW * 2);
        Tensor sc_v = view(sc, Cout, HH, WW);
        ops::dupup3d(view(x_dev, Cin_, Hs, Ws), temporal ? 2 : 1, sc_v, stream);
        retire(x_dev);
        void* tfin = alloc(static_cast<std::size_t>(Cout) * HH * WW * 2);
        Tensor tfin_v = view(tfin, Cout, HH, WW);
        Tensor sc_buf = Tensor(sc, DType::BF16, {Cout, HH, WW});
        ops::conv2d(view(xup, Cout, HH, WW), d3(pre + "upsampler.resample.1.weight", Cout, Cout, 3, 3),
                    d3(pre + "upsampler.resample.1.bias", Cout, 1, 1, 1), &sc_buf, tfin_v, stream);
        retire(xup);
        retire(sc);
        return tfin;
    }
};

} // namespace

std::uint64_t image_device_bytes_live() noexcept { return g_image_device_live.load(); }

std::uint64_t image_device_bytes_peak() noexcept { return g_image_device_peak.load(); }

void reset_image_device_peak() noexcept { g_image_device_peak.store(g_image_device_live.load()); }

std::uint64_t host_block_cache_bytes(const artifact::Reader& reader, const DiTWeights& dit) {
    if (dit.blocks.empty()) { return 0; }
    const DiTBlockWeights& b = dit.blocks.front();
    // The nine roles of one block, in the order the cache fills them. Every block has the same
    // shapes, so one block's stored bytes times the blocks is exactly what the job page-locks.
    const artifact::ParameterReference refs[9] = {b.q,           b.k,
                                                  b.v,           b.attention_output,
                                                  b.mlp_gate,    b.mlp_up,
                                                  b.mlp_down,    b.query_norm,
                                                  b.key_norm};
    std::uint64_t one_block = 0;
    for (const auto& ref : refs) {
        one_block += object_geom(reader.geometry(ref.binding.parts.front().object)).bytes;
    }
    return one_block * dit.blocks.size();
}

GeneratedImage run_image_job(artifact::Reader& reader, const DiTWeights& dit, const VaeWeights& vae,
                             const ImageJobInputs& inputs, cudaStream_t stream,
                             ImageJobProgressView* progress,
                             std::function<void(const std::string&)> log_line) {
    // The stage telemetry goes through the caller's sink (the serving route's logger, serialized
    // against the console's stats panel); with no sink the certified route prints nothing.
    // The ledger is the module's own account of what it holds; the driver free and the stream-ordered
    // pool's reserved bytes are what the NEXT allocator (the LLM's restore, plain cudaMalloc) can
    // actually get. Reporting all three per stage is what separates a leak from a release that is still
    // queued on the image stream: live 0 with a large pool reserved means the latter.
    const auto log_mem = [&](const char* stage) {
        if (!log_line) { return; }
        std::size_t driver_free = 0, driver_total = 0;
        (void)cudaMemGetInfo(&driver_free, &driver_total);
        std::size_t pool_reserved = 0;
        cudaMemPool_t pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess && pool != nullptr) {
            (void)cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
        }
        log_line("image job: " + std::string(stage) + " | ledger live " +
                 std::to_string(g_image_device_live.load() / (1024 * 1024)) + " MiB peak " +
                 std::to_string(g_image_device_peak.load() / (1024 * 1024)) +
                 " MiB | driver free " + std::to_string(driver_free / (1024 * 1024)) +
                 " MiB pool reserved " + std::to_string(pool_reserved / (1024 * 1024)) + " MiB");
    };
    // Bundle B's attribution, measured only when a sink exists. In fas 2 the block copies are
    // asynchronous on a copy stream, so a host clock around the enqueue is meaningless; the transfer
    // time is measured on the device (one block's copy bracketed by events) and extrapolated over the
    // uniform block count. upload_blocks/upload_bytes stay as the inventory.
    std::uint64_t upload_blocks = 0, upload_bytes = 0;
    const std::size_t progress_units = inputs.timesteps.size() + 1;  // + the VAE decode unit
    const auto step    = [&](std::size_t units_done) {
        if (progress && !progress->step(units_done, progress_units)) {
            throw std::runtime_error("image job: stalled, flagged by the serving watchdog");
        }
    };
    // A stalled throw unwinds the same RAII buffers as any other failure: the job's device
    // memory is freed before the interlude restores the LLM's pools.
    ImageStreamScope stream_scope(stream);
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess && pool != nullptr) {
        // Release retired buffers to the driver once their stream passes them: the arena and the
        // LLM's restore allocate with plain cudaMalloc and cannot take bytes the pool keeps.
        std::uint64_t threshold = 0;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, static_cast<void*>(&threshold));
    }
    const std::size_t D     = dit.dims.hidden;
    const std::size_t heads = dit.dims.heads;
    const std::size_t mlp0  = dit.dims.intermediate;
    const std::size_t Cin   = dit.dims.image_in;
    const float eps         = dit.dims.eps;
    if (D == 0 || heads == 0 || mlp0 == 0 || Cin == 0 || dit.blocks.empty() || inputs.height_px % 16 ||
        inputs.width_px % 16 || inputs.height_px == 0 || inputs.width_px == 0) {
        throw std::runtime_error("image job: dims or job size are unusable");
    }
    const std::size_t Hl = inputs.height_px / 16, Wl = inputs.width_px / 16;
    const std::size_t img_tokens = Hl * Wl;
    const std::size_t tp = D == 0 ? 0 : inputs.prompt_embeds.size() / D;
    std::size_t ref_tokens = 0; // #75: the reference/edit-image block (0 = plain text-to-image)
    for (const auto& g : inputs.ref_grids) ref_tokens += static_cast<std::size_t>(g.first) * g.second;
    if (tp == 0 || inputs.prompt_embeds.size() != tp * D ||
        inputs.init_latent.size() != img_tokens * Cin || inputs.latents_mean.size() != Cin ||
        inputs.latents_std.size() != Cin || inputs.timesteps.empty() ||
        inputs.sigmas.size() != inputs.timesteps.size() + 1 ||
        inputs.ref_latents.size() != ref_tokens * Cin) {
        throw std::runtime_error("image job: job inputs do not match the model dims or geometry");
    }
    const std::size_t all_img = ref_tokens + img_tokens; // the img_in/latent extent: [ref, target]
    const std::size_t S = tp + all_img;
    const int Di = static_cast<int>(D), Hi = static_cast<int>(heads);
    const int Mi = static_cast<int>(mlp0), tp_i = static_cast<int>(tp);
    // True CFG adds a second pass with a DIFFERENT text length, so every sequence-sized buffer is sized
    // by the longer joint sequence and the extra pass carries its own rope table.
    const std::size_t tp_neg = inputs.negative_prompt_embeds.size() / D;
    const bool use_cfg = inputs.true_cfg_scale != 1.0F || tp_neg != 0;
    if (use_cfg && (tp_neg == 0 || inputs.negative_prompt_embeds.size() != tp_neg * D)) {
        throw std::runtime_error("image job: true CFG needs whole negative prompt embeds");
    }
    const std::size_t S_neg  = tp_neg + all_img;
    const std::size_t tp_max = use_cfg && tp_neg > tp ? tp_neg : tp;
    const std::size_t s_max  = use_cfg && S_neg > S ? S_neg : S;
    const std::size_t sD = s_max * D, sMlp = s_max * mlp0;

    // ---- the block host cache: nine roles x layers, read from the artifact once per job ----
    // Objects the process cannot page-lock fall back to plain host memory (see PinnedBytes); the
    // counter is per job and the timing line below reports it, so a degraded overlap is visible.
    // What the machine can still page-lock when the picture asks, beside the same quantity the RAM
    // ledger planned with at Engine startup (before the expert tier had pinned anything). A cache
    // that cannot pin then says what it found, instead of running its loop slower in silence.
    if (log_line) {
        const auto source       = open_vram_budget_source(0);
        const VramSnapshot vram = source->query();
        // The ledger's own margin (models/qwen4_exp/memory_plan.h's kPinRoomMarginBytes), restated
        // here rather than reached across a model boundary.
        constexpr std::uint64_t kMargin = 1ULL << 30;
        const std::uint64_t need = host_block_cache_bytes(reader, dit);
        const std::uint64_t used = vram.nonlocal_usage + kMargin;
        const std::uint64_t room = vram.shared_system > used ? vram.shared_system - used : 0;
        log_line("image job: host cache | need " + std::to_string(need >> 20) +
                 " MiB, pin room live " + std::to_string(room >> 20) + " MiB (shared " +
                 std::to_string(vram.shared_system >> 20) + " MiB, non-local used " +
                 std::to_string(vram.nonlocal_usage >> 20) + " MiB)");
    }
    unpinned_cache_objects().store(0);
    std::vector<BlockBytes> cache(dit.blocks.size());
    for (std::size_t l = 0; l < dit.blocks.size(); ++l) {
        const auto& b = dit.blocks[l];
        BlockBytes& c = cache[l];
        c.q          = read_object_pinned(reader, b.q);
        c.k          = read_object_pinned(reader, b.k);
        c.v          = read_object_pinned(reader, b.v);
        c.out        = read_object_pinned(reader, b.attention_output);
        c.gate       = read_object_pinned(reader, b.mlp_gate);
        c.up         = read_object_pinned(reader, b.mlp_up);
        c.down       = read_object_pinned(reader, b.mlp_down);
        c.query_norm = read_object_pinned(reader, b.query_norm);
        c.key_norm   = read_object_pinned(reader, b.key_norm);
    }

    // The copy-stream layout of one block, precomputed once from block 0 (every block has the same
    // nine shapes). The two device slots, the events and the copy stream live in `lane`; block_w[s]
    // holds the nine Weight operands for the slot of parity s as the copies land.
    const artifact::ParameterReference block_refs[9] = {
        dit.blocks[0].q, dit.blocks[0].k, dit.blocks[0].v, dit.blocks[0].attention_output,
        dit.blocks[0].mlp_gate, dit.blocks[0].mlp_up, dit.blocks[0].mlp_down, dit.blocks[0].query_norm,
        dit.blocks[0].key_norm};
    const BlockCopyPlan block_plan = make_block_copy_plan(reader, block_refs);
    Weight block_w[2][9];
    // INFERNIX_IMAGE_OVERLAP=0 forces the serialized route (copy -> wait -> compute, no lookahead) on
    // the SAME pinned host cache and copy stream, so a depth-0 run isolates the overlap itself from the
    // pinned/async change. Default = overlap.
    const bool overlap_prefetch = [] {
        const char* e = std::getenv("INFERNIX_IMAGE_OVERLAP");
        return !(e != nullptr && e[0] == '0');
    }();
    CopyLane lane;
    bool     cp0_sampled = false;

    GeneratedImage out;

    // ---- all image device memory lives in this scope; RAII frees it before the function returns ----

    // Shared device constants. d_ones/d_zeros back the [D,1] layer_norm views AND the [D,2]
    // time-embedding views, so they hold 2*D.
    Buf d_ones, d_zeros, d_tnorm, drot;
    const T ones(2 * D, 1.0F), zeros(2 * D, 0.0F);
    {
        const auto bits_o = bits_of(ones), bits_z = bits_of(zeros);
        d_ones.alloc(2 * D * 2);   h2d(d_ones.d, bits_o.data(), bits_o.size() * 2);
        d_zeros.alloc(2 * D * 2);  h2d(d_zeros.d, bits_z.data(), bits_z.size() * 2);
    }
    {
        Uploaded tnorm = upload_ref(reader, dit.txt_in_norm);
        d_tnorm = std::move(tnorm.buf);
    }
    std::vector<std::pair<int, int>> ref_grids_i;
    for (const auto& g : inputs.ref_grids) ref_grids_i.emplace_back(static_cast<int>(g.first), static_cast<int>(g.second));
    const std::vector<float> rot = build_joint_rope_blocks(tp_i, ref_grids_i, static_cast<int>(Hl), static_cast<int>(Wl));
    if (rot.size() != S * 64 * 2) { throw std::runtime_error("image job: rope table size differs from S"); }
    {
        const auto bits = bits_of(rot);
        drot.alloc(bits.size() * 2);
        h2d(drot.d, bits.data(), bits.size() * 2);
    }
    Buf drot_neg;
    if (use_cfg) {
        const std::vector<float> rot_neg =
            build_joint_rope_blocks(static_cast<int>(tp_neg), ref_grids_i, static_cast<int>(Hl),
                                    static_cast<int>(Wl));
        if (rot_neg.size() != S_neg * 64 * 2) {
            throw std::runtime_error("image job: negative rope table size differs from the sequence");
        }
        const auto bits = bits_of(rot_neg);
        drot_neg.alloc(bits.size() * 2);
        h2d(drot_neg.d, bits.data(), bits.size() * 2);
    }

    // Toplevel/front-end weights: resident for the whole loop.
    Uploaded Wimg  = upload_ref(reader, dit.img_in);
    Uploaded Winl  = upload_ref(reader, dit.txt_in_in);
    Uploaded Woutl = upload_ref(reader, dit.txt_in_out);
    Uploaded Wtl1  = upload_ref(reader, dit.time_linear_1);
    Uploaded Wtl2  = upload_ref(reader, dit.time_linear_2);
    Uploaded Wmod  = upload_ref(reader, dit.modulation);
    Uploaded Wnolin = upload_ref(reader, dit.norm_out);
    Uploaded Wproj = upload_ref(reader, dit.proj_out);
    const std::size_t td = Wtl1.w.k; // the time embedder's input width the artifact stores

    // DiT per-block scratch (allocated once, reused every block and every step).
    DiTWorkspace ws{};
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
    Buf d_lat, d_hs_img, d_txt, d_txt2, d_joint, d_hb;
    Buf d_tproj, d_temb_a, d_temb_a_silu, d_temb, d_temb_silu, d_mod, d_nlin;
    Buf d_ln, d_norm_out, d_sample, d_pe;
    d_lat.alloc(all_img * Cin * 2);
    // the evolving target latent lives AFTER the constant reference block: [ref, target] exactly like
    // the reference pipeline's cat([input_images_latents, latents]); d_latent points at its first byte.
    void* d_latent = static_cast<char*>(d_lat.d) + ref_tokens * Cin * 2;
    if (ref_tokens) { h2d(d_lat.d, inputs.ref_latents.data(), ref_tokens * Cin * 2); }
    d_hs_img.alloc(all_img * D * 2);
    d_txt.alloc(tp_max * D * 2);
    d_txt2.alloc(tp_max * D * 2);
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
    d_sample.alloc(s_max * Cin * 2);
    h2d(d_latent, inputs.init_latent.data(), img_tokens * Cin * 2);
    d_pe.alloc(tp_max * D * 2);
    h2d(d_pe.d, inputs.prompt_embeds.data(), tp * D * 2);
    Buf d_pe_neg, d_txt_neg;
    if (use_cfg) {
        d_pe_neg.alloc(tp_neg * D * 2);
        h2d(d_pe_neg.d, inputs.negative_prompt_embeds.data(), tp_neg * D * 2);
        d_txt_neg.alloc(tp_neg * D * 2);
    }
    log_mem("pre-loop resident");

    // Bring up the copy lane (non-blocking copy stream + two device block slots + events) for the
    // DiT loop; the slots are freed by its destructor before the VAE decode.
    lane.start(stream, block_plan.total);

    // txt_in front-end, computed once per prompt (it depends only on the embeds, not on t or the
    // latent): ZeroCenterRMSNorm -> in_layer -> gelu(Tanh) -> out_layer. Separate in/out buffers: an
    // in-place GEMM is run-to-run non-deterministic (#33).
    const auto txt_in = [&](void* d_pe_p, void* d_txt_p, int tp_p) {
        Tensor pe_v = view(d_pe_p, Di, tp_p), tn_v = view(d_txt_p, Di, tp_p);
        Tensor tn2_v = view(d_txt2.d, Di, tp_p);
        ops::rmsnorm(pe_v, view(d_tnorm.d, Di, 1), eps, /*unit_offset=*/true, tn_v, stream);
        ops::linear(tn_v, Winl.w, tn2_v, stream);
        ops::gelu(tn2_v, ops::GeluMode::Tanh, stream);
        ops::linear(tn2_v, Woutl.w, tn_v, stream);
    };
    txt_in(d_pe.d, d_txt.d, tp_i);
    if (use_cfg) { txt_in(d_pe_neg.d, d_txt_neg.d, static_cast<int>(tp_neg)); }

    // The AdaLN output modulation depends only on t, so both CFG passes share it.
    Tensor nlin_v = view(d_nlin.d, Di, 2);

    // ONE DiT forward: img_in over the [ref+target] latent, the joint [text + image] sequence, the
    // 32-block chain, norm_out + proj_out, then the target tokens' velocity to the host. The text
    // front-end, rope table and extents are parameters because true CFG runs this twice per step with
    // different text (the reference pipeline builds a separate negative rope the same way).
    const auto dit_velocity = [&](void* d_txt_p, void* d_rot_p, int tp_p, std::size_t S_p) -> T {
        Tensor hs_img_v = view(d_hs_img.d, Di, static_cast<int>(all_img));
        ops::linear(view(d_lat.d, static_cast<int>(Cin), static_cast<int>(all_img)), Wimg.w, hs_img_v,
                    stream);

        // assemble the joint [text + image(all_img)] token-major buffer (txt_in is stable in
        // d_txt_p; only hs_img changes per step).
        {
            const std::size_t tpB = static_cast<std::size_t>(tp_p) * D * 2, imgB = all_img * D * 2;
            cuda_ok(cudaMemcpy(d_joint.d, d_txt_p, tpB, cudaMemcpyDeviceToDevice), "joint memcpy txt");
            cuda_ok(cudaMemcpy(static_cast<char*>(d_joint.d) + tpB, d_hs_img.d, imgB,
                               cudaMemcpyDeviceToDevice),
                    "joint memcpy img");
        }

        // 32-block chain: d_joint in, ping-pong with d_hb; after an even block count the result is
        // back in d_joint. The nine weights of each block land on the copy stream into one of two
        // device slots; block l+1's copy is issued before block l's compute so the transfer runs
        // while the SM computes (fas 2). Events carry the ordering: copy_done[p] gates compute on
        // slot p, freed[p] gates the next copy into slot p. INFERNIX_IMAGE_OVERLAP=0 disables the
        // lookahead (copy -> wait -> compute, serialized) to isolate the overlap from the pinned/async
        // change. Two blocks are resident during the loop, so the peak rises by ~one block.
        const int Si_p = static_cast<int>(S_p);
        void* hin = d_joint.d;
        void* hout = d_hb.d;
        const std::size_t NB = dit.blocks.size();

        const auto issue_copy = [&](std::size_t l, int p) {
            const BlockBytes& cc = cache[l];
            const std::byte* host[9] = {cc.q.h, cc.k.h, cc.v.h, cc.out.h, cc.gate.h, cc.up.h,
                                        cc.down.h, cc.query_norm.h, cc.key_norm.h};
            const bool sample = !cp0_sampled && l == 0;
            if (sample) { cuda_ok(cudaEventRecord(lane.cp0s, lane.stream), "copy timing start"); }
            upload_block_async(lane.stream, lane.slot_p[p], block_plan, host, block_w[p]);
            if (sample) { cuda_ok(cudaEventRecord(lane.cp0e, lane.stream), "copy timing end");
                          cp0_sampled = true; }
            cuda_ok(cudaEventRecord(lane.copy_done[p], lane.stream), "copy done record");
            upload_bytes += cc.q.n + cc.k.n + cc.v.n + cc.out.n + cc.gate.n + cc.up.n + cc.down.n +
                            cc.query_norm.n + cc.key_norm.n;
        };

        if (overlap_prefetch) { // prologue copy of block 0
            cuda_ok(cudaStreamWaitEvent(lane.stream, lane.freed[0], 0), "slot 0 free wait");
            issue_copy(0, 0);
        }
        for (std::size_t l = 0; l < NB; ++l) {
            const int p = static_cast<int>(l & 1U);
            if (!overlap_prefetch) {
                cuda_ok(cudaStreamWaitEvent(lane.stream, lane.freed[p], 0), "slot free wait");
                issue_copy(l, p);
            } else if (l + 1 < NB) { // prefetch the next block into the other slot
                const int pn = static_cast<int>((l + 1) & 1U);
                cuda_ok(cudaStreamWaitEvent(lane.stream, lane.freed[pn], 0), "next slot free wait");
                issue_copy(l + 1, pn);
            }
            cuda_ok(cudaStreamWaitEvent(stream, lane.copy_done[p], 0), "copy done wait");
            Weight* W = block_w[p];
            // The head-norm operands are BF16 Contiguous objects; their device address is the slot
            // base plus that role's offset (Weight::qdata is const, and run_block takes void*).
            auto* slot_base = static_cast<char*>(lane.slot_p[p]);
            run_block(hin, hout, W[0], W[1], W[2], W[3], W[5], W[4], W[6],
                      slot_base + block_plan.offset[7], slot_base + block_plan.offset[8],
                      d_mod.d, d_rot_p, d_ones.d, d_zeros.d, ws, Si_p, Di, Hi, Mi, tp_p, eps, stream);
            cuda_ok(cudaEventRecord(lane.freed[p], stream), "slot freed record");
            ++upload_blocks;
            std::swap(hin, hout);
        }
        void* h_final = hin;

        // norm_out: layer_norm(affine=False) -> adaln_scale(scale-only) -> proj_out.
        Tensor hfinal_v = view(h_final, Di, Si_p), ln_v = view(d_ln.d, Di, Si_p);
        ops::layer_norm(hfinal_v, view(d_ones.d, Di, 1), view(d_zeros.d, Di, 1), eps, ln_v, stream);
        Tensor norm_out_v = view(d_norm_out.d, Di, Si_p);
        ops::adaln_scale(ln_v, nlin_v, 0, tp_p, norm_out_v, stream);
        Tensor sample_v = view(d_sample.d, static_cast<int>(Cin), Si_p);
        ops::linear(norm_out_v, Wproj.w, sample_v, stream);
        cuda_ok(cudaStreamSynchronize(stream), "DiT step did not complete");

        // noise_pred = the target image tokens (last img_tokens) of the [S,Cin] velocity, token-major:
        // the reference block's velocity exists in the [S,Cin] output but is discarded (it is constant).
        const std::size_t noise_off = (static_cast<std::size_t>(tp_p) + ref_tokens) * Cin * 2;
        std::vector<std::uint16_t> noise_bits(img_tokens * Cin);
        cuda_ok(cudaMemcpy(noise_bits.data(), static_cast<char*>(d_sample.d) + noise_off,
                           noise_bits.size() * 2, cudaMemcpyDeviceToHost), "velocity D2H");
        return from_bits(noise_bits);
    };

    // ---- the flow-match denoise loop ----
    const auto loop_t0 = std::chrono::steady_clock::now();
    const std::size_t n_steps = inputs.timesteps.size();
    for (std::size_t i = 0; i < n_steps; ++i) {
        const double sigma = inputs.sigmas[i], sigma_next = inputs.sigmas[i + 1];
        const float model_input_t = static_cast<float>(inputs.timesteps[i] / 1000.0);

        // timestep embedding -> temb -> modulation + nlin (all depend on t, recomputed each step)
        const std::vector<float> tproj = temporal_timesteps({model_input_t, 0.0F}, td, kTimeFactor);
        { const auto bits = bits_of(tproj); h2d(d_tproj.d, bits.data(), bits.size() * 2); }
        Tensor tproj_v = view(d_tproj.d, static_cast<int>(td), 2), temb_a_v = view(d_temb_a.d, Di, 2);
        ops::linear(tproj_v, Wtl1.w, temb_a_v, stream);
        Tensor temb_a_silu_v = view(d_temb_a_silu.d, Di, 2);
        ops::silu_mul(temb_a_v, view(d_ones.d, Di, 2), temb_a_silu_v, stream);
        Tensor temb_v = view(d_temb.d, Di, 2);
        ops::linear(temb_a_silu_v, Wtl2.w, temb_v, stream);
        Tensor temb_silu_v = view(d_temb_silu.d, Di, 2);
        ops::silu_mul(temb_v, view(d_ones.d, Di, 2), temb_silu_v, stream);
        Tensor mod_v = view(d_mod.d, 4 * Di, 2);
        ops::linear(temb_silu_v, Wmod.w, mod_v, stream);
        ops::linear(temb_silu_v, Wnolin.w, nlin_v, stream);

        T noise_f = dit_velocity(d_txt.d, drot.d, tp_i, S);
        if (use_cfg) {
            const T noise_neg =
                dit_velocity(d_txt_neg.d, drot_neg.d, static_cast<int>(tp_neg), S_neg);
            // neg + scale * (cond - neg), as the reference's true_cfg branch. Its model_output is
            // BF16, so the steered velocity is rounded to BF16 here instead of carried in float.
            T steered(noise_f.size());
            for (std::size_t p = 0; p < steered.size(); ++p) {
                steered[p] = static_cast<float>(
                    static_cast<double>(noise_neg[p]) +
                    static_cast<double>(inputs.true_cfg_scale) *
                        (static_cast<double>(noise_f[p]) - static_cast<double>(noise_neg[p])));
            }
            noise_f = from_bits(bits_of(steered));
        }

        // scheduler: new_lat = lat + (sigma_next - sigma) * noise, rounded to bf16, back to
        // d_latent (the reference upcasts to f32, steps, casts back to bf16).
        std::vector<std::uint16_t> lat_bits(img_tokens * Cin);
        cuda_ok(cudaMemcpy(lat_bits.data(), d_latent, lat_bits.size() * 2, cudaMemcpyDeviceToHost),
                "latent D2H");
        const T lat_f = from_bits(lat_bits);
        T new_lat(img_tokens * Cin);
        const double ds = sigma_next - sigma;
        for (std::size_t p = 0; p < new_lat.size(); ++p) {
            new_lat[p] = static_cast<float>(static_cast<double>(lat_f[p]) + ds * static_cast<double>(noise_f[p]));
        }
        { const auto bits = bits_of(new_lat); h2d(d_latent, bits.data(), bits.size() * 2); }
        cuda_ok(cudaStreamSynchronize(stream), "scheduler upload did not complete");
        step(i + 1);
    }

    log_mem("DiT loop done");
    if (log_line) {
        // The loop wall is the honest total; the copy estimate is one block's transfer time (a device
        // event pair around the first block's H2D on the copy stream) scaled over the uniform block
        // count. asyncEngineCount/deviceOverlap say whether copy+compute overlap is even possible on
        // this GPU -- if overlap is on yet the loop wall does not fall vs a serialized run, the copy
        // is not actually hiding behind compute and that is reported rather than dressed as a win.
        const double loop_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_t0).count();
        double copy_ms = 0.0;
        if (cp0_sampled) {
            float ms = 0.0F;
            // The transfer time of ONE block (the first H2D on the copy stream, so cold and
            // representative of the worst per-block copy). Reported as-is, NOT scaled: the loop-wall
            // reduction vs the §8 baseline is what shows the copy is hidden behind compute.
            if (cudaEventSynchronize(lane.cp0e) == cudaSuccess &&
                cudaEventElapsedTime(&ms, lane.cp0s, lane.cp0e) == cudaSuccess) {
                copy_ms = static_cast<double>(ms);
            }
        }
        // asyncEngineCount is the number of independent copy engines: >= 1 means an H2D on the copy
        // stream can run concurrently with the SM compute. (CUDA 13 dropped cudaDeviceProp's
        // deviceOverlap flag; asyncEngineCount is the surviving copy-overlap signal.)
        static int s_async_engine = -1;
        if (s_async_engine < 0) {
            cudaDeviceProp prop{};
            s_async_engine = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.asyncEngineCount : 0;
        }
        char loopb[32], copyb[32];
        std::snprintf(loopb, sizeof(loopb), "%.0f", loop_ms);
        std::snprintf(copyb, sizeof(copyb), "%.0f", copy_ms);
        log_line("image job: timing | block uploads " + std::to_string(upload_blocks) +
                 " x 9 objects, bytes " + std::to_string(upload_bytes / (1024 * 1024)) +
                 " MiB, loop wall " + loopb + " ms, block0 copy " + copyb + " ms (" +
                 std::string(overlap_prefetch ? "overlap" : "serialized") + ", asyncEngineCount " +
                 std::to_string(s_async_engine) + ", host cache unpinned " +
                 std::to_string(unpinned_cache_objects().load()) + " objects)");
    }
    // ---- the VAE decode of the final latent ----
    // unpack [img_tokens,Cin] token-major (token h*Wl+w, feature c) -> [Cin,Hl,Wl] channel-major,
    // denormalize with the VAE latents mean/std.
    std::vector<std::uint16_t> lat_bits(img_tokens * Cin);
    cuda_ok(cudaMemcpy(lat_bits.data(), d_latent, lat_bits.size() * 2, cudaMemcpyDeviceToHost), "final latent D2H");
    out.latent = lat_bits;
    const T lat_got = from_bits(lat_bits);
    T vae_in(Cin * Hl * Wl);
    for (std::size_t c = 0; c < Cin; ++c) {
        for (std::size_t h = 0; h < Hl; ++h) {
            for (std::size_t w = 0; w < Wl; ++w) {
                const double u = static_cast<double>(lat_got[(h * Wl + w) * Cin + c]);
                vae_in[c * (Hl * Wl) + h * Wl + w] =
                    static_cast<float>(u * static_cast<double>(inputs.latents_std[c]) +
                                       static_cast<double>(inputs.latents_mean[c]));
            }
        }
    }
    Buf d_vae;
    {
        const auto bits = bits_of(vae_in);
        d_vae.alloc(bits.size() * 2);
        h2d(d_vae.d, bits.data(), bits.size() * 2);
    }

    // VAE device decode. All primitives/scratch management live in the shared VaeGraph (also used
    // by encode_reference_image, #75).
    VaeGraph vg{reader, vae, stream};

    // run the decode: [Cin,Hl,Wl] -> post_quant -> conv_in -> mid -> 5 up -> norm_out -> conv_out.
    const struct UpCfg { int cin, cout; bool temporal, up; } cfg[5] = {
        {1152, 1152, true, true}, {1152, 1152, true, true}, {1152, 576, true, true},
        {576, 288, false, true}, {288, 144, false, false},
    };
    int Hh = static_cast<int>(Hl), Ww = static_cast<int>(Wl);
    void* vx = d_vae.d; // [Cin,Hl,Wl] channel-major
    {
        // post_quant_conv is a linear on the token-major view (the qualified bridge: transpose to
        // [P,C], linear, transpose back).
        void* xt = vg.alloc(Cin * (Hl * Wl) * 2);
        vg.xT(vx, xt, static_cast<int>(Cin), static_cast<int>(Hl * Wl));
        d_vae.retire(stream);
        void* x1t = vg.alloc(Cin * (Hl * Wl) * 2);
        Tensor x1t_v = view(x1t, static_cast<int>(Cin), static_cast<int>(Hl * Wl));
        ops::linear(view(xt, static_cast<int>(Cin), static_cast<int>(Hl * Wl)), vg.dl("vae/post_quant_conv.weight"),
                    x1t_v, stream);
        vg.retire(xt);
        ops::add_bias(vg.d3("vae/post_quant_conv.bias", static_cast<int>(Cin), 1, 1, 1), x1t_v, stream);
        void* x1 = vg.alloc(Cin * (Hl * Wl) * 2);
        vg.xT(x1t, x1, static_cast<int>(Hl * Wl), static_cast<int>(Cin));
        vg.retire(x1t);
        vx = x1;
        void* x2 = vg.alloc(static_cast<std::size_t>(1152) * Hl * Wl * 2);
        Tensor x2_v = view(x2, 1152, static_cast<int>(Hl), static_cast<int>(Wl));
        ops::conv2d(view(x1, static_cast<int>(Cin), static_cast<int>(Hl), static_cast<int>(Wl)),
                    vg.d3("vae/decoder.conv_in.weight", 1152, static_cast<int>(Cin), 3, 3),
                    vg.d3("vae/decoder.conv_in.bias", 1152, 1, 1, 1), nullptr, x2_v, stream);
        vg.retire(x1);
        vx = x2;
        vx = vg.dresnet(vx, 1152, 1152, static_cast<int>(Hl), static_cast<int>(Wl), "vae/decoder.mid_block.resnets.0.", false, true);
        vx = vg.att(vx, 1152, "vae/decoder.mid_block.attentions.0.", static_cast<int>(Hl), static_cast<int>(Wl));
        vx = vg.dresnet(vx, 1152, 1152, static_cast<int>(Hl), static_cast<int>(Wl), "vae/decoder.mid_block.resnets.1.", false, true);
        for (int b = 0; b < 5; ++b) {
            const std::string pre = "vae/decoder.up_blocks." + std::to_string(b) + ".";
            vx = vg.dup_up(vx, cfg[b].cin, cfg[b].cout, Hh, Ww, cfg[b].temporal, cfg[b].up, pre);
            if (cfg[b].up) { Hh *= 2; Ww *= 2; }
        }
        void* n = vg.drms(vx, 144, Hh * Ww, "vae/decoder.norm_out.gamma");
        vg.retire(vx);
        Tensor n_flat = view(n, 144, Hh * Ww);
        ops::silu(n_flat, stream);
        void* fin = vg.alloc(4 * static_cast<std::size_t>(Hh) * Ww * 2);
        Tensor fin_v = view(fin, 4, Hh, Ww);
        ops::conv2d(view(n, 144, Hh, Ww), vg.d3("vae/decoder.conv_out.weight", 4, 144, 3, 3),
                    vg.d3("vae/decoder.conv_out.bias", 4, 1, 1, 1), nullptr, fin_v, stream);
        vg.retire(n);
        cuda_ok(cudaStreamSynchronize(stream), "VAE decode did not complete");
        log_mem("VAE decode done");
        step(progress_units);
        const std::size_t fout = 4 * static_cast<std::size_t>(Hh) * Ww;
        std::vector<std::uint16_t> fin_bits(fout);
        cuda_ok(cudaMemcpy(fin_bits.data(), fin, fout * 2, cudaMemcpyDeviceToHost), "image D2H");
        vg.retire(fin);
        out.channels = from_bits(fin_bits);
        for (auto& v : out.channels) v = std::max(-1.0F, std::min(1.0F, v));
    }
    out.height = static_cast<std::uint32_t>(Hh);
    out.width  = static_cast<std::uint32_t>(Ww);
    return out; // every image allocation is freed with this scope
}

// #75 reference-image encode: the same VAE's ENCODER as the exact mirror of the qualified decode.
// Image path = pure 2D (a fresh feat_cache makes every time_conv take its first-call skip, so the
// time weights never run); CausalConv3d degenerates to the qualified symmetric pad-1 conv2d. Chain
// per the reference Encoder3d: conv_in 4->96; five ResidualDownBlocks dims [96,96,192,384,768,768]
// (each: 2 resnets on the main path, and for the first four a stride-2 resample conv on the resnet
// output PLUS AvgDown3D of the BLOCK INPUT, factor_t = 1 for b0 and 2 for b1-b3; b4 has no
// downsample and its factor-1 avg_shortcut is the identity, so it is a plain +input); mid block
// resnet -> full single-head attention C=768 -> resnet; norm_out + silu + conv_out 768->128;
// quant_conv as a token-major linear; the argmax sample is the first 64 channels; per-channel
// (x-mean)/std in the BF16 operation order; the token-major [P,128] slice IS the packed layout.
EncodedReference encode_reference_image(artifact::Reader& reader, const VaeWeights& vae,
                                        const ReferenceEncodeInputs& in, cudaStream_t stream) {
    ImageStreamScope stream_scope(stream);
    const std::uint32_t H = in.height_px, W = in.width_px;
    if (H == 0 || W == 0 || H % 16 || W % 16 || in.pixels.size() != 4ULL * H * W ||
        in.latents_mean.size() != 64 || in.latents_std.size() != 64) {
        throw std::runtime_error("reference encode: image size must be a positive multiple of 16 "
                                 "with [4,H,W] pixels and 64-wide latent stats");
    }
    int Hs = static_cast<int>(H), Ws = static_cast<int>(W);
    VaeGraph vg{reader, vae, stream};

    // Debug (#75): INFERNIX_VAE_STAGE_STATS prints each stage's magnitude so an encoder-stage
    // defect is localized from one device run. Inert unless the variable is set.
    const bool stage_stats = std::getenv("INFERNIX_VAE_STAGE_STATS") != nullptr;
    auto stats = [&](const char* tag, void* p, std::size_t n) {
        if (!stage_stats) { return; }
        cuda_ok(cudaStreamSynchronize(stream), "stage stats sync");
        std::vector<std::uint16_t> bits(n);
        cuda_ok(cudaMemcpy(bits.data(), p, n * 2, cudaMemcpyDeviceToHost), "stage stats D2H");
        double sum = 0.0, mx = 0.0;
        for (const std::uint16_t b : bits) {
            const double v = from_bf16(b);
            sum += v * v;
            mx = std::max(mx, std::fabs(v));
        }
        std::printf("VAESTAGE %s n=%zu rms=%.6g absmax=%.6g\n", tag, n,
                    n ? std::sqrt(sum / static_cast<double>(n)) : 0.0, mx);
        std::fflush(stdout);
    };

    Buf d_img;
    {
        const auto bits = bits_of(in.pixels);
        d_img.alloc(bits.size() * 2);
        h2d(d_img.d, bits.data(), bits.size() * 2);
    }
    void* x = d_img.d; // [4,H,W] channel-major BF16
    {
        void* xc = vg.alloc(96ULL * Hs * Ws * 2);
        Tensor xc_v = view(xc, 96, Hs, Ws);
        ops::conv2d(view(x, 4, Hs, Ws), vg.d3("vae/encoder.conv_in.weight", 96, 4, 3, 3),
                    vg.d3("vae/encoder.conv_in.bias", 96, 1, 1, 1), nullptr, xc_v, stream);
        stats("conv_in", xc, 96ULL * Hs * Ws);
        d_img.retire(stream);
        x = xc;
    }

    static constexpr int kCin[5] = {96, 96, 192, 384, 768}, kCout[5] = {96, 192, 384, 768, 768};
    for (int b = 0; b < 5; ++b) {
        const std::string pre = "vae/encoder.down_blocks." + std::to_string(b) + ".";
        const int ci = kCin[b], co = kCout[b];
        void* block_in = x; // the avg_shortcut re-reads it after the resnets -> owns_input=false
        void* r = vg.dresnet(block_in, ci, co, Hs, Ws, pre + "resnets.0.", ci != co, false);
        r = vg.dresnet(r, co, co, Hs, Ws, pre + "resnets.1.", false, true);
        if (b < 4) {
            const int H2 = Hs / 2, W2 = Ws / 2; // even extents: stride-2 floor formula == /2 here
            void* cs = vg.alloc(static_cast<std::size_t>(co) * H2 * W2 * 2);
            Tensor cs_v = view(cs, co, H2, W2);
            ops::conv2d_stride2(view(r, co, Hs, Ws),
                                vg.d3(pre + "downsampler.resample.1.weight", co, co, 3, 3),
                                vg.d3(pre + "downsampler.resample.1.bias", co, 1, 1, 1), cs_v, stream);
            stats((pre + "resample").c_str(), cs, static_cast<std::size_t>(co) * H2 * W2);
            vg.retire(r);
            void* av = vg.alloc(static_cast<std::size_t>(co) * H2 * W2 * 2);
            Tensor av_v = view(av, co, H2, W2);
            ops::avg_down(view(block_in, ci, Hs, Ws), b == 0 ? 1 : 2, av_v, stream);
            stats((pre + "avg_shortcut").c_str(), av, static_cast<std::size_t>(co) * H2 * W2);
            void* x2 = vg.alloc(static_cast<std::size_t>(co) * H2 * W2 * 2);
            Tensor x2_v = view(x2, co, H2, W2);
            ops::add(view(cs, co, H2, W2), view(av, co, H2, W2), x2_v, stream);
            stats((pre + "out").c_str(), x2, static_cast<std::size_t>(co) * H2 * W2);
            vg.retire(cs);
            vg.retire(av);
            vg.retire(block_in);
            x = x2;
            Hs = H2;
            Ws = W2;
        } else {
            // b4: downsampler absent, avg_shortcut = identity(factor 1) -> plain + block input.
            void* x2 = vg.alloc(static_cast<std::size_t>(co) * Hs * Ws * 2);
            Tensor x2_v = view(x2, co, Hs, Ws);
            ops::add(view(r, co, Hs, Ws), view(block_in, ci, Hs, Ws), x2_v, stream);
            stats("down_4.out", x2, static_cast<std::size_t>(co) * Hs * Ws);
            vg.retire(r);
            vg.retire(block_in);
            x = x2;
        }
    }

    x = vg.dresnet(x, 768, 768, Hs, Ws, "vae/encoder.mid_block.resnets.0.", false, true);
    stats("mid_resnets.0", x, 768ULL * Hs * Ws);
    x = vg.att(x, 768, "vae/encoder.mid_block.attentions.0.", Hs, Ws);
    stats("mid_attn", x, 768ULL * Hs * Ws);
    x = vg.dresnet(x, 768, 768, Hs, Ws, "vae/encoder.mid_block.resnets.1.", false, true);
    stats("mid", x, 768ULL * Hs * Ws);

    void* n = vg.drms(x, 768, Hs * Ws, "vae/encoder.norm_out.gamma");
    vg.retire(x);
    Tensor n_flat = view(n, 768, Hs * Ws);
    ops::silu(n_flat, stream);
    stats("norm_out_silu", n, 768ULL * Hs * Ws);
    void* z = vg.alloc(128ULL * Hs * Ws * 2);
    Tensor z_v = view(z, 128, Hs, Ws);
    ops::conv2d(view(n, 768, Hs, Ws), vg.d3("vae/encoder.conv_out.weight", 128, 768, 3, 3),
                vg.d3("vae/encoder.conv_out.bias", 128, 1, 1, 1), nullptr, z_v, stream);
    stats("conv_out", z, 128ULL * Hs * Ws);
    vg.retire(n);

    // quant_conv is a 1x1 conv = the token-major linear bridge; stay token-major [P,128] so the
    // first-64-channel mode slice is exactly the packed [tokens,64] the DiT consumes.
    const int P = Hs * Ws;
    void* qt = vg.alloc(static_cast<std::size_t>(P) * 128 * 2);
    vg.xT(z, qt, 128, P);
    vg.retire(z);
    void* qo = vg.alloc(static_cast<std::size_t>(P) * 128 * 2);
    Tensor qo_v = view(qo, 128, P);
    ops::linear(view(qt, 128, P), vg.dl("vae/quant_conv.weight"), qo_v, stream);
    vg.retire(qt);
    ops::add_bias(vg.d3("vae/quant_conv.bias", 128, 1, 1, 1), qo_v, stream);
    stats("quant_conv", qo, static_cast<std::size_t>(P) * 128);
    cuda_ok(cudaStreamSynchronize(stream), "reference encode did not complete");
    std::vector<std::uint16_t> qbits(static_cast<std::size_t>(P) * 128);
    cuda_ok(cudaMemcpy(qbits.data(), qo, qbits.size() * 2, cudaMemcpyDeviceToHost), "reference latent D2H");
    vg.retire(qo);

    EncodedReference out;
    out.height_lat = static_cast<std::uint32_t>(Hs);
    out.width_lat  = static_cast<std::uint32_t>(Ws);
    out.latents.resize(static_cast<std::size_t>(P) * 64);
    // mode sample = first 64 channels; normalize in the reference's BF16 operation order:
    // bf16(x - mean) then bf16(that / std), x already the represented BF16 value.
    for (int t = 0; t < P; ++t) {
        for (int c = 0; c < 64; ++c) {
            const std::uint16_t raw = qbits[static_cast<std::size_t>(t) * 128 + c];
            const float d = from_bf16(to_bf16(from_bf16(raw) - in.latents_mean[c]));
            out.latents[static_cast<std::size_t>(t) * 64 + c] = to_bf16(d / in.latents_std[c]);
        }
    }
    if (stage_stats) {
        double sum = 0.0, mx = 0.0;
        for (const std::uint16_t b : out.latents) {
            const double v = from_bf16(b);
            sum += v * v;
            mx = std::max(mx, std::fabs(v));
        }
        std::printf("VAESTAGE packed n=%zu rms=%.6g absmax=%.6g\n", out.latents.size(),
                    std::sqrt(sum / static_cast<double>(out.latents.size())), mx);
        std::fflush(stdout);
    }
    return out; // every reference-encode allocation is freed with this scope
}

} // namespace infernix::models::qwen_image
