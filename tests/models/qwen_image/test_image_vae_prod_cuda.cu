// GPU qualification of the FULL Qwen-Image-2.1 VAE decoder (single first-chunk frame) composed from
// the PRODUCTION ops (repo op layer), not a self-contained naive kernel. The whole decode path runs in
// BF16 (activations and weights), matching the production image route:
//
//   z -> post_quant_conv(1x1=linear+bias) -> conv_in(3x3) ->
//        mid[ resnet0(1152) -> spatial-attn -> resnet1(1152) ] ->
//        up0..up4 (3 resnets + optional nearest-2x + resample-conv3x3 + DupUp3D residual) ->
//        norm_out(RMS) -> silu -> conv_out(3x3) -> clamp[-1,1]
//
// Every device transformation is a repo op on the op-development contract:
//   1x1 conv   ->  ops::linear (general BF16 GEMM / registered Vision route) + ops::add_bias
//   3x3 conv   ->  ops::conv2d (zero-padded cross-correlation, bias + optional residual folded in)
//   RMS_norm   ->  ops::rmsnorm (normalize over channel dim0 per spatial position, unit_offset=false)
//   SiLU       ->  ops::silu (in place)
//   upsample   ->  ops::upsample_nearest_2x
//   DupUp3D    ->  ops::dupup3d
//   spatial-attn -> ops::spatial_attn (single-head full attention, head_dim=C, N=HW<=256) + ops::add
//
// Qualification oracle: the host FP32 (double-accumulation) re-derivation of the identical graph
// (ported from test_image_vae.cpp, the CPU-only oracle that matched vae_decode_frame0.npy at 3.3e-6),
// evaluated on the SAME BF16-represented operands the device receives. So the only difference between
// the host reference and the device path is the production kernels' BF16 compute/storage rounding.
// Every stage is compared per-stage so a single blackout run localizes a divergent op (the agent is
// offline during the run).
//
// Tolerance (a-priori, never narrowed to pass): one rel-Frobenius bound of 5e-2 applied to every
// per-stage row and to the final-vs-frame0 comparison. Rationale: the decoder is a ~125-op deep BF16
// chain, deeper and conv-heavier than the DiT 32-block chain qualified at 4e-2; the accumulated BF16
// storage floor for such a chain is expected ~1e-2..3e-2, so 5e-2 is a safe envelope one notch above the
// deep-DiT precedent. A real op defect is O(0.1..1), far above the bound, and localizes to the stage
// where the per-stage table jumps. The final is ALSO compared to vae_decode_frame0.npy (the diffusers
// FP32 reference) for the production-acceptance number.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors_vae/*.npy and .../tensors_vae_weights/*.npy. Returns 77 (CTest
// skip) when the oracle data is absent; returns 2 (runnable) when no CUDA device is present.

#include "infernix/ops/linear.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/vae_decoder.h"

#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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

using T = std::vector<float>;

// ---- Host BF16 codec (no device header needed for the reference path) ----
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
// The float value a BF16 operand actually represents.
T represent(const T& v) {
    T out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}
// Download a BF16 device buffer and decode to float.
T download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    T out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}
Tensor view(void* p, int a, int b, int c = 1, int dd = 1) {
    return Tensor(p, DType::BF16, {a, b, c, dd});
}
// Device transpose: `in` is [rows, cols] row-major, `out` is [cols, rows] row-major. Used to bridge
// the channel-major [C,P] layout of the conv/attn ops and the token-major [P,C]/[P,N] layout that
// ops::linear (the production GEMM) reads and writes.
__global__ void transpose_u16_kernel(const std::uint16_t* __restrict__ in, std::uint16_t* __restrict__ out,
                                     int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    out[c * rows + r] = in[r * cols + c];
}

// ---- Host FP32 (double-accumulation) oracle primitives; ported verbatim from test_image_vae.cpp ----
namespace ho {

T conv2d(const T& x, std::size_t Cin, std::size_t H, std::size_t W, const T& w, const T& b, std::size_t Cout,
         int k, int p) {
    T out(Cout * H * W);
    for (std::size_t oc = 0; oc < Cout; ++oc) {
        const double bb = b[oc];
        for (std::size_t ih = 0; ih < H; ++ih) {
            for (std::size_t iw = 0; iw < W; ++iw) {
                double acc = bb;
                for (std::size_t ic = 0; ic < Cin; ++ic) {
                    const float* xr = &x[ic * H * W];
                    const float* wr = &w[((oc * Cin + ic) * std::size_t(k)) * std::size_t(k)];
                    for (int u = 0; u < k; ++u) {
                        const int hy = (int)ih - p + u;
                        if (hy < 0 || hy >= (int)H) continue;
                        for (int v = 0; v < k; ++v) {
                            const int hx = (int)iw - p + v;
                            if (hx < 0 || hx >= (int)W) continue;
                            acc += (double)xr[hy * W + hx] * (double)wr[u * k + v];
                        }
                    }
                }
                out[(oc * H + ih) * W + iw] = (float)acc;
            }
        }
    }
    return out;
}

// QwenImage21RMS_norm (channel_first, bias=False): y[c,p] = x[c,p]/sqrt(sum_c x[c,p]^2+1e-12)*sqrt(C)*gamma[c].
T rmsnorm(const T& x, std::size_t C, std::size_t H, std::size_t W, const T& gamma) {
    T out(C * H * W);
    const double scale = std::sqrt((double)C);
    const std::size_t HW = H * W;
    for (std::size_t p = 0; p < HW; ++p) {
        double ss = 0.0;
        for (std::size_t c = 0; c < C; ++c) { const double v = x[c * HW + p]; ss += v * v; }
        const double norm = std::sqrt(ss + 1e-12);
        const double inv = scale / norm;
        for (std::size_t c = 0; c < C; ++c) out[c * HW + p] = (float)((double)x[c * HW + p] * inv * (double)gamma[c]);
    }
    return out;
}

T silu(const T& x) {
    T o(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) o[i] = x[i] / (1.0F + std::exp(-x[i]));
    return o;
}

// Single-head full self-attention over H*W positions (head_dim=C, scale 1/sqrt(C)):
// norm(RMS) -> to_qkv(1x1) -> split q/k/v -> softmax(qk^T/sqrt(C)) v -> proj(1x1) -> +identity.
T attention(const T& xn, std::size_t C, std::size_t H, std::size_t W, const T& wqkv, const T& bqkv, const T& wproj,
            const T& bproj, const T& gamma) {
    const std::size_t HW = H * W;
    const T normed = rmsnorm(xn, C, H, W, gamma);
    T qkv(3 * C * HW);
    for (std::size_t oc = 0; oc < 3 * C; ++oc) {
        const float* wr = &wqkv[oc * C];
        const double bb = bqkv[oc];
        for (std::size_t p = 0; p < HW; ++p) {
            double acc = bb;
            for (std::size_t c = 0; c < C; ++c) acc += (double)normed[c * HW + p] * (double)wr[c];
            qkv[oc * HW + p] = (float)acc;
        }
    }
    const double scale = 1.0 / std::sqrt((double)C);
    T attn(C * HW);
    for (std::size_t n = 0; n < HW; ++n) {
        std::vector<double> s(HW);
        double mx = -1e30;
        for (std::size_t m = 0; m < HW; ++m) {
            double dot = 0.0;
            for (std::size_t c = 0; c < C; ++c) dot += (double)qkv[c * HW + n] * (double)qkv[(C + c) * HW + m];
            const double sc = dot * scale;
            s[m] = sc;
            if (sc > mx) mx = sc;
        }
        double sum = 0.0;
        for (std::size_t m = 0; m < HW; ++m) { s[m] = std::exp(s[m] - mx); sum += s[m]; }
        for (std::size_t m = 0; m < HW; ++m) s[m] /= sum;
        for (std::size_t c = 0; c < C; ++c) {
            double acc = 0.0;
            for (std::size_t m = 0; m < HW; ++m) acc += s[m] * (double)qkv[(2 * C + c) * HW + m];
            attn[c * HW + n] = (float)acc;
        }
    }
    T proj(C * HW);
    for (std::size_t oc = 0; oc < C; ++oc) {
        const float* wr = &wproj[oc * C];
        const double bb = bproj[oc];
        for (std::size_t p = 0; p < HW; ++p) {
            double acc = bb;
            for (std::size_t c = 0; c < C; ++c) acc += (double)attn[c * HW + p] * (double)wr[c];
            proj[oc * HW + p] = (float)acc;
        }
    }
    for (std::size_t i = 0; i < proj.size(); ++i) proj[i] += xn[i];
    return proj;
}

T resblock(const T& x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t W,
           const std::map<std::string, T>& WM, const std::string& pre, bool has_sc) {
    T h;
    if (has_sc) {
        h = conv2d(x, Cin, H, W, WM.at(pre + "conv_shortcut_weight"), WM.at(pre + "conv_shortcut_bias"), Cout, 1, 0);
    } else {
        h = x;
    }
    T t = rmsnorm(x, Cin, H, W, WM.at(pre + "norm1_gamma"));
    t = silu(t);
    t = conv2d(t, Cin, H, W, WM.at(pre + "conv1_weight"), WM.at(pre + "conv1_bias"), Cout, 3, 1);
    t = rmsnorm(t, Cout, H, W, WM.at(pre + "norm2_gamma"));
    t = silu(t);
    t = conv2d(t, Cout, H, W, WM.at(pre + "conv2_weight"), WM.at(pre + "conv2_bias"), Cout, 3, 1);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] += h[i];
    return t;
}

T upsample2x(const T& x, std::size_t C, std::size_t H, std::size_t W) {
    T out(C * (2 * H) * (2 * W));
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t h = 0; h < H; ++h)
            for (std::size_t w = 0; w < W; ++w) {
                const float v = x[c * H * W + h * W + w];
                const std::size_t base = c * (2 * H) * (2 * W) + (2 * h) * (2 * W) + 2 * w;
                out[base] = v; out[base + 1] = v; out[base + 2 * W] = v; out[base + 2 * W + 1] = v;
            }
    return out;
}

T dupup3d(const T& x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t W, int factor_t) {
    const std::size_t factor = (std::size_t)factor_t * 4;
    const std::size_t repeats = Cout * factor / Cin;
    const std::size_t tt = (std::size_t)factor_t - 1;
    T out(Cout * (2 * H) * (2 * W));
    for (std::size_t oc = 0; oc < Cout; ++oc)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                const std::size_t cin = (oc * factor + tt * 4 + (std::size_t)a * 2 + (std::size_t)b) / repeats;
                for (std::size_t h = 0; h < H; ++h)
                    for (std::size_t w = 0; w < W; ++w)
                        out[oc * (2 * H) * (2 * W) + (2 * h + a) * (2 * W) + (2 * w + b)] = x[cin * H * W + h * W + w];
            }
    return out;
}

T upblock(T x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t W, bool temporal, bool up,
          const std::map<std::string, T>& WM, const std::string& pre) {
    T xcopy = x;
    x = resblock(x, Cin, Cout, H, W, WM, pre + "resnets_0_", Cin != Cout);
    x = resblock(x, Cout, Cout, H, W, WM, pre + "resnets_1_", false);
    x = resblock(x, Cout, Cout, H, W, WM, pre + "resnets_2_", false);
    if (up) {
        const std::size_t H0 = H, W0 = W;
        x = upsample2x(x, Cout, H0, W0);
        H = 2 * H0; W = 2 * W0;
        x = conv2d(x, Cout, H, W, WM.at(pre + "upsampler_resample_1_weight"),
                   WM.at(pre + "upsampler_resample_1_bias"), Cout, 3, 1);
        const T sc = dupup3d(xcopy, Cin, Cout, H0, W0, temporal ? 2 : 1);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += sc[i];
    }
    return x;
}

}  // namespace ho

// Host reference stage set (every checkpoint compared against the device path).
struct HostStages {
    T after_post_quant;
    T after_conv_in, after_mid_r0, after_attn, after_mid_r1;
    T after_up[5];
    T after_norm_silu, final;
};

struct UpCfg { std::size_t cin, cout; bool temporal, up; };

int run() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // flush every stage line even if a kernel aborts
    const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (env == nullptr || std::string(env).empty()) {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string oracle(env);
    const std::string TDIR = qo::join(oracle, "tensors_vae");
    const std::string WDIR = qo::join(oracle, "tensors_vae_weights");

    qo::NpyArray in_arr = qo::read_npy(qo::join(TDIR, "vae_latent_denormed.npy"));
    qo::require(in_arr.shape.size() == 5 && in_arr.shape[0] == 1 && in_arr.shape[2] == 1,
                "vae_latent_denormed not [1,64,1,H,W]");
    qo::require(in_arr.shape[1] == 64 && in_arr.shape[3] == 8 && in_arr.shape[4] == 8,
                "expected [1,64,1,8,8] latent");
    const T latent = represent(in_arr.floats());  // [64,8,8]

    qo::NpyArray ref_arr = qo::read_npy(qo::join(TDIR, "vae_decode_frame0.npy"));
    qo::require(ref_arr.shape.size() == 4 && ref_arr.shape[0] == 1 && ref_arr.shape[2] == 128 &&
                    ref_arr.shape[3] == 128,
                "vae_decode_frame0 not [1,4,128,128]");
    const T ref = ref_arr.floats();  // diffusers FP32 reference (already clamped)

    // Represented (BF16) weights, keyed by sanitized .npy base name, cached.
    std::map<std::string, T> WM;
    const auto load = [&](const std::string& name) -> const T& {
        auto it = WM.find(name);
        if (it != WM.end()) return it->second;
        qo::NpyArray a = qo::read_npy(qo::join(WDIR, name + ".npy"));
        return WM.emplace(name, represent(a.floats())).first->second;
    };
    // Preload every weight the graph uses (so .at never throws and both sides see the same data).
    const std::string mid0 = "decoder_mid_block_resnets_0_";
    const std::string mid1 = "decoder_mid_block_resnets_1_";
    const std::string attn = "decoder_mid_block_attentions_0_";
    auto load_mid = [&](const std::string& pre) {
        load(pre + "conv1_weight"); load(pre + "conv1_bias");
        load(pre + "conv2_weight"); load(pre + "conv2_bias");
        load(pre + "norm1_gamma");  load(pre + "norm2_gamma");
    };
    load("post_quant_conv_weight"); load("post_quant_conv_bias");
    load("decoder_conv_in_weight"); load("decoder_conv_in_bias");
    load_mid(mid0); load_mid(mid1);
    load(attn + "to_qkv_weight"); load(attn + "to_qkv_bias");
    load(attn + "proj_weight");   load(attn + "proj_bias");
    load(attn + "norm_gamma");
    const UpCfg cfg[5] = {
        {1152, 1152, true, true},
        {1152, 1152, true, true},
        {1152, 576, true, true},
        {576, 288, false, true},
        {288, 144, false, false},
    };
    for (int b = 0; b < 5; ++b) {
        std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
        for (int r = 0; r < 3; ++r) {
            std::string rp = pre + "resnets_" + std::to_string(r) + "_";
            load(rp + "conv1_weight"); load(rp + "conv1_bias");
            load(rp + "conv2_weight"); load(rp + "conv2_bias");
            load(rp + "norm1_gamma");  load(rp + "norm2_gamma");
            if (r == 0 && cfg[b].cin != cfg[b].cout) {
                load(rp + "conv_shortcut_weight"); load(rp + "conv_shortcut_bias");
            }
        }
        if (cfg[b].up) { load(pre + "upsampler_resample_1_weight"); load(pre + "upsampler_resample_1_bias"); }
    }
    load("decoder_norm_out_gamma");
    load("decoder_conv_out_weight"); load("decoder_conv_out_bias");

    // ---- Host FP32 (double-accumulation) reference on the BF16-represented operands ----
    HostStages H;
    {
        T x = latent;
        x = ho::conv2d(x, 64, 8, 8, WM.at("post_quant_conv_weight"), WM.at("post_quant_conv_bias"), 64, 1, 0);
        H.after_post_quant = x;
        x = ho::conv2d(x, 64, 8, 8, WM.at("decoder_conv_in_weight"), WM.at("decoder_conv_in_bias"), 1152, 3, 1);
        H.after_conv_in = x;
        x = ho::resblock(x, 1152, 1152, 8, 8, WM, mid0, false);
        H.after_mid_r0 = x;
        x = ho::attention(x, 1152, 8, 8, WM.at(attn + "to_qkv_weight"), WM.at(attn + "to_qkv_bias"),
                          WM.at(attn + "proj_weight"), WM.at(attn + "proj_bias"), WM.at(attn + "norm_gamma"));
        H.after_attn = x;
        x = ho::resblock(x, 1152, 1152, 8, 8, WM, mid1, false);
        H.after_mid_r1 = x;
        std::size_t Hh = 8, Ww = 8;
        for (int b = 0; b < 5; ++b) {
            std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
            x = ho::upblock(std::move(x), cfg[b].cin, cfg[b].cout, Hh, Ww, cfg[b].temporal, cfg[b].up, WM, pre);
            if (cfg[b].up) { Hh *= 2; Ww *= 2; }
            H.after_up[b] = x;
        }
        x = ho::rmsnorm(x, 144, Hh, Ww, WM.at("decoder_norm_out_gamma"));
        x = ho::silu(x);
        H.after_norm_silu = x;
        x = ho::conv2d(x, 144, Hh, Ww, WM.at("decoder_conv_out_weight"), WM.at("decoder_conv_out_bias"), 4, 3, 1);
        for (auto& v : x) v = std::max(-1.0F, std::min(1.0F, v));
        H.final = x;
    }
    std::printf("host_vs_frame0 rel_fro=%.3e (oracle sanity, CPU-only reference)\n", qo::rel_fro(H.final, ref));

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    // ---- Device forward through the production ops ----
    std::vector<void*> galloc;
    const auto alloc = [&](std::size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            std::printf("VAE PROD: cudaMalloc failed (%zu bytes)\n", bytes);
            std::fflush(stdout);
            std::exit(3);
        }
        galloc.push_back(p);
        return p;
    };
    // Transpose a device buffer in-place-of: in [rows,cols] row-major -> out [cols,rows] row-major.
    // chan->token: xT(chan, tok, C, P); token->chan: xT(tok, chan, P, N). Valid in any lambda body.
    const auto xT = [&](void* in, void* out, int rows, int cols) {
        const int total = rows * cols;
        transpose_u16_kernel<<<(total + 255) / 256, 256, 0, 0>>>(
            reinterpret_cast<const std::uint16_t*>(in), reinterpret_cast<std::uint16_t*>(out), rows,
            cols);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::printf("transpose launch FAIL: %s\n", cudaGetErrorString(e));
            std::fflush(stdout);
            std::abort();
        }
    };
    // Device weight accessors: 3x3 convs / biases / gammas as raw BF16 Tensors; 1x1 linears as Weight.
    std::map<std::string, void*> dt;
    std::map<std::string, Weight> dlin;
    // Copy a weight's BF16 bits (host WM) to an already-allocated device buffer.
    const auto upl = [&](void* d, const std::string& name) {
        const auto bits = bits_of(WM.at(name));
        const cudaError_t e = cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t),
                                         cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            std::printf("weight upload FAIL %s: %s\n", name.c_str(), cudaGetErrorString(e));
            std::fflush(stdout);
            std::abort();
        }
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
            w.payload = d;
            w.payload_bytes = WM.at(name).size() * 2;
            w.qtype = infernix::QType::BF16;
            w.shape[0] = n;      w.shape[1] = k;
            w.padded_shape[0] = n; w.padded_shape[1] = k;
            w.ndim = 2;
            w.qdata = d;
            w.n = n;             w.k = k;
            w.layout = infernix::QuantLayout::Contiguous;
            it = dlin.emplace(name, w).first;
        }
        return it->second;
    };

    // Per-position RMSNorm over C channels on a channel-major [C, P] buffer. ops::rmsnorm consumes a
    // position-major [P, C] operand (ne[0] = C, C innermost), so transpose in, run the production
    // RMSNorm, and transpose the result back to channel-major. Returns a freshly allocated
    // channel-major [C, P] buffer holding the RMSNorm output (caller owns it).
    const auto drms = [&](void* x_dev, int C, int P, const std::string& gname) -> void* {
        const std::size_t tot = (std::size_t)C * P;
        void* pm = alloc(tot * 2);
        xT(x_dev, pm, C, P);  // chan [C,P] -> token [P,C]
        void* po = alloc(tot * 2);
        Tensor o_pm = view(pm, C, P);
        Tensor o_po = view(po, C, P);
        ops::rmsnorm(o_pm, d3(gname, C, 1, 1, 1), 1e-12F, false, o_po, 0);
        void* chan = alloc(tot * 2);
        xT(po, chan, P, C);  // token [P,C] -> chan [C,P]
        cudaFree(pm);
        cudaFree(po);
        return chan;
    };

    // One QwenImage21ResidualBlock: h = shortcut(x); t = conv1(silu(rms1(x)));
    // t = conv2(silu(rms2(t))) + h.
    const auto dresnet = [&](void* x_dev, int Cin, int Cout, int H, int W, const std::string& pre,
                             bool has_sc) -> void* {
        const int HW = H * W;
        // Identity residual for has_sc=false (Cin==Cout here); overridden by the 1x1 shortcut below.
        // 3-D view matching the conv output so ops::conv2d's residual-shape check accepts it.
        Tensor xin = view(x_dev, Cin, H, W);
        Tensor res_buf;
        Tensor* res = &xin;
        void* short_dev = nullptr;
        if (has_sc) {
            // ops::linear is token-major [T,K]->[T,N]; the residual stream is channel-major [C,P], so
            // transpose in, run the GEMM, transpose out. xin is [Cin,HW] channel-major.
            void* xin_tok = alloc(Cin * HW * 2);
            xT(x_dev, xin_tok, Cin, HW);  // chan [Cin,HW] -> token [HW,Cin]
            void* short_tok = alloc(Cout * HW * 2);
            Tensor o_short = view(short_tok, Cout, HW);
            ops::linear(view(xin_tok, Cin, HW), dl(pre + "conv_shortcut_weight", Cout, Cin),
                        o_short, 0);
            ops::add_bias(d3(pre + "conv_shortcut_bias", Cout, 1, 1, 1), o_short, 0);  // Cout innermost
            short_dev = alloc(Cout * HW * 2);
            xT(short_tok, short_dev, HW, Cout);  // token [HW,Cout] -> chan [Cout,HW]
            cudaFree(xin_tok);
            cudaFree(short_tok);
            res_buf = Tensor(short_dev, DType::BF16, {Cout, H, W});
            res = &res_buf;
        }
        void* n1 = drms(x_dev, Cin, HW, pre + "norm1_gamma");
        Tensor o1 = view(n1, Cin, HW);
        ops::silu(o1, 0);
        void* t1 = alloc(Cout * H * W * 2);
        Tensor c1 = view(t1, Cout, H, W);
        ops::conv2d(view(n1, Cin, H, W), d3(pre + "conv1_weight", Cout, Cin, 3, 3),
                    d3(pre + "conv1_bias", Cout, 1, 1, 1), nullptr, c1, 0);
        void* n2 = drms(t1, Cout, HW, pre + "norm2_gamma");
        Tensor o2 = view(n2, Cout, H, W);
        ops::silu(o2, 0);
        void* tfin = alloc(Cout * H * W * 2);
        Tensor cf = view(tfin, Cout, H, W);
        ops::conv2d(o2, d3(pre + "conv2_weight", Cout, Cout, 3, 3),
                    d3(pre + "conv2_bias", Cout, 1, 1, 1), res, cf, 0);
        cudaFree(n1); cudaFree(t1); cudaFree(n2);
        if (has_sc) cudaFree(short_dev);
        return tfin;
    };

    // The mid spatial-attention block: norm -> to_qkv(+bias) -> [q|k|v] -> attn -> proj(+bias) -> +identity.
    const auto dattn = [&](void* x_dev) -> void* {
        const int C = 1152, HW = 8 * 8;
        Tensor xin = view(x_dev, C, HW);
        void* nrm = drms(x_dev, C, HW, attn + "norm_gamma");
        void* on_tok = alloc(C * HW * 2);
        xT(nrm, on_tok, C, HW);  // chan [C,HW] -> token [HW,C]
        void* qkv_tok = alloc(3 * C * HW * 2);
        Tensor o_qkv = view(qkv_tok, 3 * C, HW);
        ops::linear(view(on_tok, C, HW), dl(attn + "to_qkv_weight", 3 * C, C), o_qkv, 0);
        ops::add_bias(d3(attn + "to_qkv_bias", 3 * C, 1, 1, 1), o_qkv, 0);  // 3C innermost
        void* qkv = alloc(3 * C * HW * 2);
        xT(qkv_tok, qkv, HW, 3 * C);  // token [HW,3C] -> chan [3C,HW]
        cudaFree(on_tok);
        cudaFree(qkv_tok);
        char* qb = reinterpret_cast<char*>(qkv);
        const Tensor q = view(qb, C, HW);
        const Tensor k = view(qb + (C * HW) * 2, C, HW);
        const Tensor v = view(qb + (2 * C * HW) * 2, C, HW);
        void* att = alloc(C * HW * 2);
        Tensor ot = view(att, C, HW);
        const float scale = 1.0F / std::sqrt(1152.0F);
        ops::spatial_attn(q, k, v, nullptr, scale, ot, 0);
        void* ot_tok = alloc(C * HW * 2);
        xT(att, ot_tok, C, HW);  // chan [C,HW] -> token [HW,C]
        void* prj_tok = alloc(C * HW * 2);
        Tensor o_prj = view(prj_tok, C, HW);
        ops::linear(view(ot_tok, C, HW), dl(attn + "proj_weight", C, C), o_prj, 0);
        ops::add_bias(d3(attn + "proj_bias", C, 1, 1, 1), o_prj, 0);  // C innermost
        void* prj = alloc(C * HW * 2);
        xT(prj_tok, prj, HW, C);  // token [HW,C] -> chan [C,HW]
        cudaFree(ot_tok);
        cudaFree(prj_tok);
        Tensor op = view(prj, C, HW);
        void* xn = alloc(C * HW * 2);
        Tensor oxn = view(xn, C, HW);
        ops::add(op, xin, oxn, 0);
        cudaFree(nrm); cudaFree(qkv); cudaFree(att); cudaFree(prj);
        return xn;
    };

    // One QwenImage21ResidualUpBlock: 3 resnets; optional nearest-2x + resample-conv3x3 + DupUp3D residual.
    const auto dup_up = [&](void* x_dev, int Cin, int Cout, int H, int W, bool temporal, bool up,
                            const std::string& pre) -> void* {
        // x_dev is the DupUp3D source; kept alive until the upsample, then freed here.
        void* r0 = dresnet(x_dev, Cin, Cout, H, W, pre + "resnets_0_", Cin != Cout);
        void* r1 = dresnet(r0, Cout, Cout, H, W, pre + "resnets_1_", false);
        cudaFree(r0);
        void* r2 = dresnet(r1, Cout, Cout, H, W, pre + "resnets_2_", false);
        cudaFree(r1);
        if (up) {
            const int H0 = H, W0 = W, HH = 2 * H, WW = 2 * W;
            void* xup = alloc(Cout * HH * WW * 2);
            Tensor o_up = view(xup, Cout, HH, WW);
            ops::upsample_nearest_2x(view(r2, Cout, H0, W0), o_up, 0);
            void* sc = alloc(Cout * HH * WW * 2);
            Tensor o_sc = view(sc, Cout, HH, WW);
            ops::dupup3d(view(x_dev, Cin, H0, W0), temporal ? 2 : 1, o_sc, 0);
            void* tfin = alloc(Cout * HH * WW * 2);
            Tensor sc_buf = Tensor(sc, DType::BF16, {Cout, HH, WW});
            Tensor o_tf = view(tfin, Cout, HH, WW);
            ops::conv2d(o_up, d3(pre + "upsampler_resample_1_weight", Cout, Cout, 3, 3),
                        d3(pre + "upsampler_resample_1_bias", Cout, 1, 1, 1), &sc_buf, o_tf, 0);
            cudaFree(xup); cudaFree(sc); cudaFree(r2); cudaFree(x_dev);
            return tfin;
        }
        cudaFree(x_dev);
        return r2;
    };

    const auto check = [&](void* d, std::size_t n, const T& refv, const char* name, double tol, bool& ok) {
        const T got = download(d, n);
        const double e = qo::rel_fro(got, refv);
        const bool s_ok = (e < tol) && std::isfinite(e);
        ok = ok && s_ok;
        std::printf("stage %-12s rel_fro=%.3e %s\n", name, e, s_ok ? "OK" : "FAIL");
    };

    const double tol = 5e-2;  // a-priori deep-chain BF16 floor bound (see file header).
    bool ok = true;

    void* x = alloc(64 * 64 * 2);  // latent [64,8,8]
    {
        const auto bits = bits_of(latent);
        cudaMemcpy(x, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    }
    void* x_tok = alloc(64 * 64 * 2);
    xT(x, x_tok, 64, 64);  // chan [64,64] -> token [64,64]
    void* x1_tok = alloc(64 * 64 * 2);
    Tensor o_xtok = view(x1_tok, 64, 64);
    ops::linear(view(x_tok, 64, 64), dl("post_quant_conv_weight", 64, 64), o_xtok, 0);
    ops::add_bias(d3("post_quant_conv_bias", 64, 1, 1, 1), o_xtok, 0);  // Cout innermost (token-major)
    void* x1 = alloc(64 * 64 * 2);
    xT(x1_tok, x1, 64, 64);  // token [64,64] -> chan [64,64]
    cudaFree(x_tok);
    cudaFree(x1_tok);
    check(x1, 64 * 64, H.after_post_quant, "post_quant", tol, ok);
    cudaFree(x);
    void* x2 = alloc(1152 * 8 * 8 * 2);
    Tensor o_x2 = view(x2, 1152, 8, 8);
    ops::conv2d(view(x1, 64, 8, 8), d3("decoder_conv_in_weight", 1152, 64, 3, 3),
                d3("decoder_conv_in_bias", 1152, 1, 1, 1), nullptr, o_x2, 0);
    cudaFree(x1);
    check(x2, 1152 * 8 * 8, H.after_conv_in, "conv_in", tol, ok);

    {
        void* xr = dresnet(x2, 1152, 1152, 8, 8, mid0, false);
        cudaFree(x2);
        x = xr;
        check(x, 1152 * 8 * 8, H.after_mid_r0, "mid_r0", tol, ok);
        void* xa = dattn(x);
        cudaFree(x);
        x = xa;
        check(x, 1152 * 8 * 8, H.after_attn, "attn", tol, ok);
        void* xr1 = dresnet(x, 1152, 1152, 8, 8, mid1, false);
        cudaFree(x);
        x = xr1;
        check(x, 1152 * 8 * 8, H.after_mid_r1, "mid_r1", tol, ok);
    }

    int Hh = 8, Ww = 8;
    for (int b = 0; b < 5; ++b) {
        const std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
        x = dup_up(x, (int)cfg[b].cin, (int)cfg[b].cout, Hh, Ww, cfg[b].temporal, cfg[b].up, pre);
        if (cfg[b].up) { Hh *= 2; Ww *= 2; }
        char buf[16];
        std::snprintf(buf, sizeof(buf), "up_block_%d", b);
        check(x, (std::size_t)cfg[b].cout * (std::size_t)Hh * (std::size_t)Ww, H.after_up[b], buf, tol, ok);
    }

    void* n = drms(x, 144, Hh * Ww, "decoder_norm_out_gamma");
    Tensor o_n = view(n, 144, Hh * Ww);
    ops::silu(o_n, 0);
    cudaFree(x);
    check(n, 144 * (std::size_t)Hh * Ww, H.after_norm_silu, "norm_silu", tol, ok);

    void* fin = alloc(4 * (std::size_t)Hh * Ww * 2);
    Tensor o_fin = view(fin, 4, Hh, Ww);
    ops::conv2d(view(n, 144, Hh, Ww), d3("decoder_conv_out_weight", 4, 144, 3, 3),
                d3("decoder_conv_out_bias", 4, 1, 1, 1), nullptr, o_fin, 0);
    cudaFree(n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Final: device decoded image (clamped to [-1,1]) vs host reference and vs the diffusers FP32 frame0.
    const std::size_t fout = 4 * (std::size_t)Hh * Ww;
    T dev_final = download(fin, fout);
    for (auto& v : dev_final) v = std::max(-1.0F, std::min(1.0F, v));
    const double e_host = qo::rel_fro(dev_final, H.final);
    const double e_ref = qo::rel_fro(dev_final, ref);
    const bool fin_ok = (e_host < tol) && (e_ref < tol) && std::isfinite(e_host) && std::isfinite(e_ref);
    ok = ok && fin_ok;
    std::printf("final vs host-refs  rel_fro=%.3e %s\n", e_host, (e_host < tol) ? "OK" : "FAIL");
    std::printf("final vs frame0.npy rel_fro=%.3e %s\n", e_ref, (e_ref < tol) ? "OK" : "FAIL");
    std::fflush(stdout);

    for (void* p : galloc) cudaFree(p);

    std::printf(ok ? "IMAGE_VAE_PROD_TEST: PASS\n" : "IMAGE_VAE_PROD_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_VAE_PROD_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
