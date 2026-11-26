// CPU-only numeric oracle for the Qwen-Image-2.1 VAE DECODER (AutoencoderKLQwenImage21), single frame.
//
// Naive float32 (double-accumulation) re-derivation of the full decode path against the authoritative
// diffusers model, which ran the identical forward in float32 on CPU (image-spike/oracle/dump_vae.py):
//   z -> post_quant_conv(1x1) -> conv_in(3x3) -> mid_block(resnet->attn->resnet) ->
//   5 up_blocks -> norm_out(RMS) -> silu -> conv_out(3x3) -> clamp[-1,1].
//
// Single-frame reduction (T=1, first_chunk, no feat_cache): every CausalConv3d runs as a plain 2D conv
// (squeeze T -> F.pad -> Conv2d -> unsqueeze T); every Resample (upsample2d/3d) skips time_conv and the
// temporal reshape, so the spatial upsample == nn.Upsample(2x nearest-exact) + Conv2d(3x3 pad1). The
// ResidualUpBlock residual path is DupUp3D(avg_shortcut) on a spatial-2x + channel-mixing repeat.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors_vae/*.npy (input vae_latent_denormed, expected vae_decode_frame0)
// and $INFERNIX_IMAGE_ORACLE/tensors_vae_weights/*.npy (decoder + post_quant_conv, FP32). Returns 77
// (CTest skip) when the data is absent, so CI without the dump is unaffected.
//
// Gate (a-priori, not narrowed to pass): rel-Frobenius < 1e-3. Both sides are float32; the C++ side uses
// double accumulators (at least as precise as torch), so a residual above 1e-3 indicates a graph/wiring
// defect (an order-of-magnitude gap would be the signature of a real bug).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using T = std::vector<float>;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

// Minimal .npy reader: v1/v2 header, C-contiguous, little-endian '<f4'.
struct NpyArray {
    std::string descr;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
    std::size_t numel = 0;

    T floats() const {
        require(descr == "<f4", "floats() on non-f32 array");
        require(bytes.size() == numel * 4, "float array byte count mismatch");
        T out(numel);
        std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
        return out;
    }
};

NpyArray read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    require(f.is_open(), ("npy open failed: " + path).c_str());
    std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    require(raw.size() > 12, "npy too small");
    const std::uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    require(std::memcmp(raw.data(), magic, 6) == 0, "npy bad magic");
    const int major = raw[6];
    const int minor = raw[7];
    std::size_t header_len = 0;
    std::size_t data_off = 0;
    if (major == 1 && minor == 0) {
        header_len = std::uint16_t(raw[8]) | (std::uint16_t(raw[9]) << 8);
        data_off = 10 + header_len;
    } else if (major == 2 && minor == 0) {
        header_len = std::uint32_t(raw[8]) | (std::uint32_t(raw[9]) << 8) | (std::uint32_t(raw[10]) << 16) |
                     (std::uint32_t(raw[11]) << 24);
        data_off = 12 + header_len;
    } else {
        require(false, "unsupported npy version");
    }
    require(raw.size() >= data_off, "npy truncated header");
    const std::string header(reinterpret_cast<const char*>(raw.data() + (data_off - header_len)), header_len);

    NpyArray a;
    const std::size_t dpos = header.find("'descr'");
    require(dpos != std::string::npos, "npy header missing descr");
    const std::size_t q1 = header.find('\'', header.find(':', dpos) + 1);
    const std::size_t q2 = header.find('\'', q1 + 1);
    require(q1 != std::string::npos && q2 != std::string::npos, "npy descr quote parse");
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    require(header.find("'fortran_order': False") != std::string::npos, "npy not C-contiguous");

    const std::size_t spos = header.find("'shape'");
    require(spos != std::string::npos, "npy header missing shape");
    const std::size_t popen = header.find('(', spos);
    const std::size_t pclose = header.find(')', popen);
    require(popen != std::string::npos && pclose != std::string::npos, "npy shape paren parse");
    const std::string dims = header.substr(popen + 1, pclose - popen - 1);
    std::size_t i = 0;
    while (i < dims.size()) {
        while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) ++i;
        std::size_t j = i;
        while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') ++j;
        if (j > i) { a.shape.push_back(std::stoull(dims.substr(i, j - i))); i = j; } else { ++i; }
    }
    a.numel = 1;
    for (std::size_t s : a.shape) a.numel *= s;
    require(raw.size() == data_off + a.numel * 4, "npy payload size mismatch");
    a.bytes.assign(raw.begin() + data_off, raw.end());
    return a;
}

// conv2d cross-correlation (torch Conv2d). x [Cin,H,W], w [Cout,Cin,k,k] (torch layout), b [Cout].
// out [Cout,H,W] with pad p on each spatial side. Double accumulation.
T conv2d(const T& x, std::size_t Cin, std::size_t H, std::size_t W, const T& w, const T& b, std::size_t Cout,
         int k, int p) {
    require(w.size() == Cout * Cin * std::size_t(k) * std::size_t(k), "conv2d weight size");
    require(b.size() == Cout, "conv2d bias size");
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

// QwenImage21RMS_norm (channel_first, bias=False): normalize over the channel dim per spatial position,
// scale = sqrt(C), gamma [C]. y[c,p] = x[c,p]/sqrt(sum_c x[c,p]^2 + eps)*sqrt(C)*gamma[c].
T rmsnorm(const T& x, std::size_t C, std::size_t H, std::size_t W, const T& gamma) {
    require(gamma.size() == C, "rmsnorm gamma size");
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

// QwenImage21AttentionBlock: single-head full self-attention over H*W positions (head_dim=C, scale 1/sqrt(C)),
// norm(RMS) -> to_qkv(1x1) -> split q/k/v -> softmax(qk^T/sqrt(C)) v -> proj(1x1) -> +identity.
T attention(const T& xn, std::size_t C, std::size_t H, std::size_t W, const T& wqkv, const T& bqkv, const T& wproj,
            const T& bproj, const T& gamma) {
    const std::size_t HW = H * W;
    const T normed = rmsnorm(xn, C, H, W, gamma);
    // qkv = conv1x1(normed)+bias, [3C, HW]
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
    // proj = conv1x1(attn)+bias
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

// QwenImage21ResidualBlock (single frame): h = shortcut(x); x = silu(rms1(x)); x = conv3(x);
// x = silu(rms2(x)); x = conv3(x); return x + h. shortcut = conv1x1 if Cin!=Cout else identity.
T resblock(const T& x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t Wd,
           const std::map<std::string, T>& WM, const std::string& pre, bool has_sc) {
    T h;
    if (has_sc) {
        h = conv2d(x, Cin, H, Wd, WM.at(pre + "conv_shortcut_weight"), WM.at(pre + "conv_shortcut_bias"), Cout, 1, 0);
    } else {
        h = x;
    }
    T t = rmsnorm(x, Cin, H, Wd, WM.at(pre + "norm1_gamma"));
    t = silu(t);
    t = conv2d(t, Cin, H, Wd, WM.at(pre + "conv1_weight"), WM.at(pre + "conv1_bias"), Cout, 3, 1);
    t = rmsnorm(t, Cout, H, Wd, WM.at(pre + "norm2_gamma"));
    t = silu(t);
    t = conv2d(t, Cout, H, Wd, WM.at(pre + "conv2_weight"), WM.at(pre + "conv2_bias"), Cout, 3, 1);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] += h[i];
    return t;
}

// nn.Upsample(2x, nearest-exact) on an integer 2x scale: replicate each pixel to a 2x2 block.
T upsample2x(const T& x, std::size_t C, std::size_t H, std::size_t W) {
    T out(C * (2 * H) * (2 * W));
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
                const float v = x[c * H * W + h * W + w];
                const std::size_t base = c * (2 * H) * (2 * W) + (2 * h) * (2 * W) + 2 * w;
                out[base] = v;
                out[base + 1] = v;
                out[base + 2 * W] = v;
                out[base + 2 * W + 1] = v;
            }
        }
    }
    return out;
}

// QwenImage21DupUp3D for a single first-chunk frame. Spatial 2x + channel mixing via repeat_interleave.
// factor = factor_t*4, repeats = Cout*factor/Cin, tt = factor_t-1 (the kept temporal slice).
// out[oc, 2h+a, 2w+b] = x[cin, h, w], cin = (oc*factor + tt*4 + a*2 + b) / repeats.
T dupup3d(const T& x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t W, int factor_t) {
    const std::size_t fs = 2;
    const std::size_t factor = (std::size_t)factor_t * fs * fs;
    const std::size_t repeats = Cout * factor / Cin;
    const std::size_t tt = (std::size_t)factor_t - 1;
    T out(Cout * (2 * H) * (2 * W));
    for (std::size_t oc = 0; oc < Cout; ++oc) {
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                const std::size_t cin = (oc * factor + tt * fs * fs + (std::size_t)a * fs + (std::size_t)b) / repeats;
                for (std::size_t h = 0; h < H; ++h) {
                    for (std::size_t w = 0; w < W; ++w) {
                        out[oc * (2 * H) * (2 * W) + (2 * h + a) * (2 * W) + (2 * w + b)] = x[cin * H * W + h * W + w];
                    }
                }
            }
        }
    }
    return out;
}

// QwenImage21ResidualUpBlock (single frame): 3 resnets (Cin->Cout then Cout->Cout), optional spatial
// upsampler (nearest-2x + conv3x3), optional residual DupUp3D on the input copy.
T upblock(T x, std::size_t Cin, std::size_t Cout, std::size_t H, std::size_t W, bool temporal, bool up,
          const std::map<std::string, T>& WM, const std::string& pre) {
    T xcopy = x;
    std::string r0 = pre + "resnets_0_";
    x = resblock(x, Cin, Cout, H, W, WM, r0, Cin != Cout);
    std::string r1 = pre + "resnets_1_";
    x = resblock(x, Cout, Cout, H, W, WM, r1, false);
    std::string r2 = pre + "resnets_2_";
    x = resblock(x, Cout, Cout, H, W, WM, r2, false);
    if (up) {
        const std::size_t H0 = H, W0 = W;
        x = upsample2x(x, Cout, H0, W0);
        H = 2 * H0;
        W = 2 * W0;
        x = conv2d(x, Cout, H, W, WM.at(pre + "upsampler_resample_1_weight"), WM.at(pre + "upsampler_resample_1_bias"),
                   Cout, 3, 1);
        const T sc = dupup3d(xcopy, Cin, Cout, H0, W0, temporal ? 2 : 1);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += sc[i];
    }
    return x;
}

double rel_fro(const T& a, const T& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return std::sqrt(num) / (den > 0.0 ? std::sqrt(den) : 1.0);
}

double max_abs(const T& a, const T& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

}  // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
        if (env == nullptr || std::string(env).empty()) {
            std::cout << "INFERNIX_IMAGE_ORACLE unset; skipping Qwen-Image VAE oracle\n";
            return 77;
        }
        const std::string oracle(env);
        const std::string tdir = join(oracle, "tensors_vae");
        const std::string wdir = join(oracle, "tensors_vae_weights");

        NpyArray in_arr = read_npy(join(tdir, "vae_latent_denormed.npy"));
        require(in_arr.shape.size() == 5 && in_arr.shape[0] == 1 && in_arr.shape[2] == 1,
                "vae_latent_denormed not [1,64,1,H,W]");
        const std::size_t Z = in_arr.shape[1];
        const std::size_t H0 = in_arr.shape[3];
        const std::size_t W0 = in_arr.shape[4];
        require(Z == 64 && H0 == 8 && W0 == 8, "expected [1,64,1,8,8] latent");
        T x = in_arr.floats();  // [1,64,1,8,8] collapses to [64,8,8]

        NpyArray ref_arr = read_npy(join(tdir, "vae_decode_frame0.npy"));
        require(ref_arr.shape.size() == 4 && ref_arr.shape[0] == 1 && ref_arr.shape[2] == 128 &&
                    ref_arr.shape[3] == 128,
                "vae_decode_frame0 not [1,4,128,128]");
        const T ref = ref_arr.floats();  // [4,128,128]

        // Weight loader: reads tensors_vae_weights/<name>.npy into a map keyed by sanitized name (no .npy).
        std::map<std::string, T> WM;
        const auto load = [&](const std::string& name) -> const T& {
            auto it = WM.find(name);
            if (it != WM.end()) return it->second;
            NpyArray a = read_npy(join(wdir, name + ".npy"));
            auto res = WM.emplace(name, a.floats());
            return res.first->second;
        };
        auto load_mid = [&](const std::string& pre) {
            load(pre + "conv1_weight"); load(pre + "conv1_bias");
            load(pre + "conv2_weight"); load(pre + "conv2_bias");
            load(pre + "norm1_gamma");  load(pre + "norm2_gamma");
        };

        // ---- post_quant_conv + conv_in ----
        x = conv2d(x, 64, 8, 8, load("post_quant_conv_weight"), load("post_quant_conv_bias"), 64, 1, 0);
        x = conv2d(x, 64, 8, 8, load("decoder_conv_in_weight"), load("decoder_conv_in_bias"), 1152, 3, 1);

        // ---- mid_block: resnet0 -> attn -> resnet1 (all 1152) ----
        const std::string mid0 = "decoder_mid_block_resnets_0_";
        const std::string mid1 = "decoder_mid_block_resnets_1_";
        load_mid(mid0);
        load_mid(mid1);
        x = resblock(x, 1152, 1152, 8, 8, WM, mid0, false);
        x = attention(x, 1152, 8, 8, load("decoder_mid_block_attentions_0_to_qkv_weight"),
                      load("decoder_mid_block_attentions_0_to_qkv_bias"),
                      load("decoder_mid_block_attentions_0_proj_weight"),
                      load("decoder_mid_block_attentions_0_proj_bias"),
                      load("decoder_mid_block_attentions_0_norm_gamma"));
        x = resblock(x, 1152, 1152, 8, 8, WM, mid1, false);

        // ---- up_blocks (dims, temporal, up_flag) ----
        struct UpCfg { std::size_t cin, cout; bool temporal, up; };
        const UpCfg cfg[5] = {
            {1152, 1152, true, true},
            {1152, 1152, true, true},
            {1152, 576, true, true},
            {576, 288, false, true},
            {288, 144, false, false},
        };
        std::size_t H = 8, Wp = 8;
        for (int b = 0; b < 5; ++b) {
            std::string pre = "decoder_up_blocks_" + std::to_string(b) + "_";
            // Preload every weight this block uses so WM.at() never throws.
            for (int r = 0; r < 3; ++r) {
                std::string rp = pre + "resnets_" + std::to_string(r) + "_";
                load(rp + "conv1_weight"); load(rp + "conv1_bias");
                load(rp + "conv2_weight"); load(rp + "conv2_bias");
                load(rp + "norm1_gamma");  load(rp + "norm2_gamma");
                if (r == 0 && cfg[b].cin != cfg[b].cout) {
                    load(rp + "conv_shortcut_weight"); load(rp + "conv_shortcut_bias");
                }
            }
            if (cfg[b].up) {
                load(pre + "upsampler_resample_1_weight"); load(pre + "upsampler_resample_1_bias");
            }
            const std::size_t cin = cfg[b].cin, cout = cfg[b].cout;
            x = upblock(std::move(x), cin, cout, H, Wp, cfg[b].temporal, cfg[b].up, WM, pre);
            if (cfg[b].up) { H *= 2; Wp *= 2; }
        }

        // ---- norm_out -> silu -> conv_out -> clamp ----
        x = rmsnorm(x, 144, H, Wp, load("decoder_norm_out_gamma"));
        x = silu(x);
        x = conv2d(x, 144, H, Wp, load("decoder_conv_out_weight"), load("decoder_conv_out_bias"), 4, 3, 1);
        for (auto& v : x) v = std::max(-1.0F, std::min(1.0F, v));

        require(x.size() == ref.size(), "decoded size mismatch");
        const double rel = rel_fro(x, ref);
        const double mad = max_abs(x, ref);
        std::cout << "VAE decode (single frame, FP32): rel_fro=" << rel << " max_abs=" << mad
                  << " out=" << x.size() << "\n";
        constexpr double kGate = 1e-3;
        if (!(rel < kGate)) {
            std::cerr << "VAE decode rel_fro " << rel << " >= gate " << kGate << "\n";
            return 1;
        }
        std::cout << "VAE_DECODE_OK rel_fro=" << rel << " < " << kGate << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Qwen-Image VAE oracle failed: " << e.what() << "\n";
        return 1;
    }
}
