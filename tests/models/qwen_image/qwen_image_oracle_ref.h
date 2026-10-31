// Shared naive float32 (double-accumulation) reference for the Qwen-Image-2.1 DiT oracle tests.
// Everything here mirrors diffusers/models/transformers/transformer_qwenimage21.py exactly, on the
// CPU, in float32 with double accumulators. Two consumers: the single-block oracle (block_in ->
// block_out) and the full-forward oracle (packed inputs -> sample). Kept in a header so the two
// tests validate the identical primitives. Header-only, all inline.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qioracle {

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

// Minimal .npy reader: v1/v2 header, C-contiguous, little-endian '<f4' / '<c8' / '|b1'.
struct NpyArray {
    std::string descr;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
    std::size_t numel = 0;

    std::size_t float_count() const { return descr == "<c8" ? numel * 2 : numel; }

    std::vector<float> floats() const {
        require(descr == "<f4" || descr == "<c8", "floats() on non-float array");
        require(bytes.size() == float_count() * 4, "float array byte count mismatch");
        std::vector<float> out(float_count());
        std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
        return out;
    }
    std::vector<std::uint8_t> bools() const {
        require(descr == "|b1" || descr == "<b1", "bools() on non-bool array");
        require(bytes.size() == numel, "bool array byte count mismatch");
        return bytes;
    }
    // A torch token-id dump is int64 ('<i8'); the towers take int, so the values are narrowed here.
    std::vector<int> ints() const {
        require(descr == "<i8", "ints() on non-int64 array");
        require(bytes.size() == numel * 8, "int array byte count mismatch");
        std::vector<std::int64_t> wide(numel);
        std::memcpy(wide.data(), bytes.data(), bytes.size());
        std::vector<int> out(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            require(wide[i] >= 0 && wide[i] <= 0x7fffffffLL, "token id outside the int range");
            out[i] = static_cast<int>(wide[i]);
        }
        return out;
    }
    std::vector<double> doubles() const {
        require(descr == "<f8" || descr == "<f4", "doubles() on non-float array");
        if (descr == "<f8") {
            require(bytes.size() == numel * 8, "double array byte count mismatch");
            std::vector<double> out(numel);
            std::memcpy(out.data(), bytes.data(), out.size() * sizeof(double));
            return out;
        }
        std::vector<float> f = floats();
        return std::vector<double>(f.begin(), f.end());
    }
};

inline NpyArray read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(n));
    if (n > 0) f.read(reinterpret_cast<char*>(raw.data()), n);

    require(raw.size() > 12, "npy file too small");
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    require(std::memcmp(raw.data(), magic, 6) == 0, "not a npy file (magic)");
    const std::uint8_t major = raw[6];
    std::size_t header_len = 0, data_off = 0;
    if (major == 1) {
        header_len = static_cast<std::size_t>(raw[8]) | (static_cast<std::size_t>(raw[9]) << 8);
        data_off = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = static_cast<std::size_t>(raw[8]) | (static_cast<std::size_t>(raw[9]) << 8) |
                     (static_cast<std::size_t>(raw[10]) << 16) | (static_cast<std::size_t>(raw[11]) << 24);
        data_off = 12 + header_len;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    require(raw.size() >= data_off, "npy truncated header");
    const std::string header(reinterpret_cast<const char*>(raw.data() + (major == 1 ? 10 : 12)), header_len);

    NpyArray a;
    const std::size_t dpos = header.find("'descr'");
    require(dpos != std::string::npos, "npy header missing descr");
    const std::size_t q1 = header.find('\'', header.find(':', dpos) + 1);
    const std::size_t q2 = header.find('\'', q1 + 1);
    require(q1 != std::string::npos && q2 != std::string::npos, "npy descr quote parse");
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    // Four of the M6 latents fixtures (target_latents_init_*, latents_after_step_*) were dumped from
    // non-contiguous torch tensors and carry 'fortran_order': True. Their payload is the same array
    // with the FIRST axis varying fastest, so it is gathered into row-major order below instead of
    // refused.
    const bool fortran_order = header.find("'fortran_order': True") != std::string::npos;

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
    const std::size_t elem =
        (a.descr == "<f4") ? 4u : ((a.descr == "<c8" || a.descr == "<f8" || a.descr == "<i8") ? 8u : 1u);
    a.numel = 1;
    for (std::size_t s : a.shape) a.numel *= s;
    require(raw.size() == data_off + a.numel * elem, "npy payload size mismatch");
    a.bytes.assign(raw.begin() + data_off, raw.end());
    if (fortran_order && a.shape.size() > 1) {
        const std::size_t n = a.shape.size();
        std::vector<std::size_t> stride_c(n), stride_f(n);   // element strides, C- and Fortran-order
        stride_c[n - 1] = 1;
        for (std::size_t k = n - 1; k-- > 0;) {
            stride_c[k] = stride_c[k + 1] * a.shape[k + 1];
        }
        // Fortran order varies the FIRST axis fastest, so stride_f[0] = 1 and every later stride
        // multiplies in the axis BEFORE it. Deriving stride_f from shape[k+1] (as an earlier revision
        // did) makes it identical to stride_c and turns the gather below into a silent no-op: the four
        // F-ordered M6 latents fixtures then reached the job transposed -- same numbers, wrong layout,
        // Frobenius norm preserved, and every gate that compared them against the same misread file
        // called that bit-exact. Blackouts #21-#25 chased it as a DiT bug.
        stride_f[0] = 1;
        for (std::size_t k = 1; k < n; ++k) {
            stride_f[k] = stride_f[k - 1] * a.shape[k - 1];
        }
        std::vector<std::uint8_t> row_major(a.bytes.size());
        for (std::size_t flat = 0; flat < a.numel; ++flat) {
            std::size_t src = 0, rem = flat;
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t coord = rem / stride_c[k];
                rem -= coord * stride_c[k];
                src += coord * stride_f[k] * elem;
            }
            for (std::size_t b = 0; b < elem; ++b) row_major[flat * elem + b] = a.bytes[src + b];
        }
        a.bytes.swap(row_major);
    }
    return a;
}

// C[m][n] = sum_k A[m][k] * W[n][k]   (x @ W^T for a torch Linear weight [out,in]).
inline void matmul_xwt(const std::vector<float>& A, const std::vector<float>& W, std::vector<float>& C,
                       std::size_t M, std::size_t N, std::size_t K) {
    C.assign(M * N, 0.0F);
    for (std::size_t m = 0; m < M; ++m) {
        const float* arow = A.data() + m * K;
        float* crow = C.data() + m * N;
        for (std::size_t n = 0; n < N; ++n) {
            const float* wrow = W.data() + n * K;
            double acc = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<double>(arow[k]) * static_cast<double>(wrow[k]);
            }
            crow[n] = static_cast<float>(acc);
        }
    }
}

inline float gelu_tanh(float x) {
    const double c = 0.7978845608028654;  // sqrt(2/pi)
    const double t = std::tanh(c * (static_cast<double>(x) + 0.044715 * static_cast<double>(x) * x * x));
    return static_cast<float>(0.5 * static_cast<double>(x) * (1.0 + t));
}

inline float silu(float x) { return x / (1.0F + std::exp(-x)); }

// torch.nn.LayerNorm(elementwise_affine=False): (x-mean)/sqrt(var+eps), biased variance.
inline void layernorm(const std::vector<float>& x, std::vector<float>& y, std::size_t rows, std::size_t d, float eps) {
    y.assign(x.size(), 0.0F);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + r * d;
        float* yr = y.data() + r * d;
        double mean = 0.0;
        for (std::size_t i = 0; i < d; ++i) mean += xr[i];
        mean /= static_cast<double>(d);
        double var = 0.0;
        for (std::size_t i = 0; i < d; ++i) { const double t = static_cast<double>(xr[i]) - mean; var += t * t; }
        var /= static_cast<double>(d);
        const float inv = 1.0F / std::sqrt(static_cast<float>(var) + eps);
        for (std::size_t i = 0; i < d; ++i) yr[i] = static_cast<float>((static_cast<double>(xr[i]) - mean) * static_cast<double>(inv));
    }
}

// diffusers RMSNorm over the last axis: x*rsqrt(mean(x^2)+eps)*weight.
inline void rmsnorm_head(std::vector<float>& qh, std::size_t rows, std::size_t d,
                         const std::vector<float>& w, float eps) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* xr = qh.data() + r * d;
        double ss = 0.0;
        for (std::size_t i = 0; i < d; ++i) ss += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        const float rr = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + eps);
        for (std::size_t i = 0; i < d; ++i) xr[i] = xr[i] * rr * w[i];
    }
}

// diffusers ZeroCenterRMSNorm: x*rsqrt(mean(x^2)+eps)*(weight+1).
inline void zerocenter_rmsnorm(const std::vector<float>& x, std::vector<float>& y,
                               const std::vector<float>& w, float eps) {
    y.assign(x.size(), 0.0F);
    const std::size_t rows = x.size() / w.size();
    const std::size_t d = w.size();
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + r * d;
        float* yr = y.data() + r * d;
        double ss = 0.0;
        for (std::size_t i = 0; i < d; ++i) ss += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        const float rr = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + eps);
        for (std::size_t i = 0; i < d; ++i) yr[i] = xr[i] * rr * (w[i] + 1.0F);
    }
}

// QwenImage21TemporalTimesteps: sinusoidal of (time_factor * t), cos in first half, sin second.
inline std::vector<float> temporal_timesteps(const std::vector<float>& t, std::size_t dim, double time_factor) {
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
            out[b * dim + j] = static_cast<float>(std::cos(a));
            out[b * dim + half + j] = static_cast<float>(std::sin(a));
        }
    }
    return out;
}

// 3-axis complex RoPE (use_real=False): pairs (2j,2j+1) as complex, multiply by cis[token, j].
inline void apply_rope_c64(std::vector<float>& qh, std::size_t seq, std::size_t heads,
                           std::size_t head_dim, const std::vector<float>& rot) {
    const std::size_t rope_dim = head_dim / 2;
    for (std::size_t s = 0; s < seq; ++s) {
        for (std::size_t h = 0; h < heads; ++h) {
            float* base = qh.data() + (s * heads + h) * head_dim;
            const float* c = rot.data() + s * rope_dim * 2;
            for (std::size_t j = 0; j < rope_dim; ++j) {
                const std::complex<float> z(base[2 * j], base[2 * j + 1]);
                const std::complex<float> cis(c[2 * j], c[2 * j + 1]);
                const std::complex<float> r = z * cis;
                base[2 * j] = r.real();
                base[2 * j + 1] = r.imag();
            }
        }
    }
}

// Rebuild QwenImage21Rope positional cis for one image block [1,H,W] on a text-prefix image-suffix
// sequence. text_prefix = leading non-image tokens. Returns [S, 64] interleaved (re,im). Matches the
// dumped block_rotary.npy (axes_dim [16,56,56], theta 10000).
inline std::vector<float> build_rope(std::size_t text_prefix, std::size_t H, std::size_t W) {
    const std::size_t S = text_prefix + H * W;
    const std::size_t axes[3] = {16, 56, 56};
    // rope_params(index, dim) = polar(1, outer(index, 1/theta^(arange(0,dim,2)/dim))). Each freqs row
    // stores angle = index_value * theta^(-2*col/dim); the array index remap (pos rows 0..8191, neg
    // rows 9216+v) is irrelevant to the angle, which depends only on the integer position value v.
    auto lookup = [&](std::size_t axis, long long v, std::size_t col, float& re, float& im) {
        const std::size_t dim = axes[axis];
        const double exponent = static_cast<double>(2 * col) / static_cast<double>(dim);
        const double angle = static_cast<double>(v) * std::pow(10000.0, -exponent);
        re = static_cast<float>(std::cos(angle));
        im = static_cast<float>(std::sin(angle));
    };
    std::vector<float> rot(S * 64 * 2, 0.0F);

    // Build frame / height / width indices exactly as QwenImage21Rope.forward.
    std::vector<long long> frame(S), height(S), width(S);
    {
        long long cursor = 0, position = 0;
        // single image block
        const long long block_start = static_cast<long long>(text_prefix);
        const long long text_len = block_start - cursor;
        for (long long i = 0; i < text_len; ++i) frame[cursor + i] = position + i;
        position += text_len;
        cursor = block_start + static_cast<long long>(H * W);
        for (long long i = 0; i < static_cast<long long>(H * W); ++i) frame[block_start + i] = position;
        position += static_cast<long long>(std::max(H, W));
        // height/width grids for image tokens, row-major over (H, W)
        for (long long hh = 0; hh < static_cast<long long>(H); ++hh) {
            for (long long ww = 0; ww < static_cast<long long>(W); ++ww) {
                const long long idx = block_start + hh * static_cast<long long>(W) + ww;
                height[idx] = -(static_cast<long long>(H) - static_cast<long long>(H) / 2) + hh;
                width[idx] = -(static_cast<long long>(W) - static_cast<long long>(W) / 2) + ww;
            }
        }
    }
    for (std::size_t s = 0; s < text_prefix; ++s) {  // text: all three axes = frame index
        height[s] = frame[s];
        width[s] = frame[s];
    }
    for (std::size_t s = 0; s < S; ++s) {
        float* o = rot.data() + s * 64 * 2;
        float re, im;
        std::size_t col = 0;
        for (std::size_t j = 0; j < axes[0] / 2; ++j, ++col) { lookup(0, frame[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
        for (std::size_t j = 0; j < axes[1] / 2; ++j, ++col) { lookup(1, height[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
        for (std::size_t j = 0; j < axes[2] / 2; ++j, ++col) { lookup(2, width[s], j, re, im); o[col * 2] = re; o[col * 2 + 1] = im; }
    }
    return rot;
}

// One full QwenImage21TransformerBlock (single layer) in float32.
struct BlockWeights {
    const std::vector<float>* to_q;
    const std::vector<float>* to_k;
    const std::vector<float>* to_v;
    const std::vector<float>* to_out;
    const std::vector<float>* norm_q;
    const std::vector<float>* norm_k;
    const std::vector<float>* mlp_proj;   // [mlp, D]
    const std::vector<float>* mlp_gate;   // [mlp, D]
    const std::vector<float>* mlp_out;    // [D, mlp]
};

// hidden_states [S,D], modulation [2,4D] (causal_condition rows), rot [S,64] re/im interleaved,
// target_mask [S] (target suffix -> row 0 else row 1), text_prefix = number of strictly-causal prefix
// tokens. Writes out [S,D].
inline void block_forward(const std::vector<float>& xin, const std::vector<float>& mod,
                          const std::vector<float>& rot, const std::vector<std::uint8_t>& target_mask,
                          const BlockWeights& w, std::size_t S, std::size_t D, std::size_t heads,
                          std::size_t text_prefix, float eps, std::vector<float>& out) {
    const std::size_t head_dim = D / heads;
    const std::size_t mlp = w.mlp_proj->size() / D;

    auto select_scale = [&](std::size_t col_off, const std::vector<float>& normed) {
        std::vector<float> o(normed.size());
        for (std::size_t s = 0; s < S; ++s) {
            const std::size_t row = target_mask[s] ? 0 : 1;
            const float* m = mod.data() + row * (4 * D) + col_off;
            const float* x = normed.data() + s * D;
            float* p = o.data() + s * D;
            for (std::size_t d = 0; d < D; ++d) p[d] = x[d] * (1.0F + m[d]);
        }
        return o;
    };
    auto gate_of = [&](std::size_t col_off, std::size_t s) {
        const std::size_t row = target_mask[s] ? 0 : 1;
        return mod.data() + row * (4 * D) + col_off;
    };

    std::vector<float> ln1, mod1;
    layernorm(xin, ln1, S, D, eps);
    mod1 = select_scale(0, ln1);

    std::vector<float> q, k, v;
    matmul_xwt(mod1, *w.to_q, q, S, D, D);
    matmul_xwt(mod1, *w.to_k, k, S, D, D);
    matmul_xwt(mod1, *w.to_v, v, S, D, D);
    rmsnorm_head(q, S * heads, head_dim, *w.norm_q, eps);
    rmsnorm_head(k, S * heads, head_dim, *w.norm_k, eps);
    apply_rope_c64(q, S, heads, head_dim, rot);
    apply_rope_c64(k, S, heads, head_dim, rot);

    const float qscale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> attn(S * D, 0.0F);
    std::vector<float> scores(S), probs(S), acc(head_dim);
    for (std::size_t s = 0; s < S; ++s) {
        const bool is_text = (s < text_prefix);
        const std::size_t kv_hi = is_text ? text_prefix : S;
        for (std::size_t h = 0; h < heads; ++h) {
            const float* qs = q.data() + (s * heads + h) * head_dim;
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                const float* ks = k.data() + (kk * heads + h) * head_dim;
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d) dot += static_cast<double>(qs[d]) * static_cast<double>(ks[d]);
                scores[kk] = static_cast<float>(dot) * qscale;
            }
            probs.assign(S, 0.0F);
            float maxv = -std::numeric_limits<float>::infinity();
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (!(is_text && kk > s) && scores[kk] > maxv) maxv = scores[kk];
            }
            double sum = 0.0;
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (is_text && kk > s) continue;
                const double e = std::exp(static_cast<double>(scores[kk]) - static_cast<double>(maxv));
                probs[kk] = static_cast<float>(e);
                sum += e;
            }
            require(sum > 0.0, "fully masked attention row");
            const float inv = static_cast<float>(1.0 / sum);
            for (std::size_t kk = 0; kk < kv_hi; ++kk) probs[kk] *= inv;
            std::fill(acc.begin(), acc.end(), 0.0F);
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (probs[kk] == 0.0F) continue;
                const float* vs = v.data() + (kk * heads + h) * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d) acc[d] += probs[kk] * vs[d];
            }
            float* od = attn.data() + (s * heads + h) * head_dim;
            for (std::size_t d = 0; d < head_dim; ++d) od[d] = acc[d];
        }
    }

    std::vector<float> attn_out;
    matmul_xwt(attn, *w.to_out, attn_out, S, D, D);
    std::vector<float> x1(xin.size());
    for (std::size_t s = 0; s < S; ++s) {
        const float* g = gate_of(D, s);  // gate_attn chunk
        for (std::size_t d = 0; d < D; ++d) x1[s * D + d] = xin[s * D + d] + std::tanh(g[d]) * attn_out[s * D + d];
    }

    std::vector<float> ln2, mod2;
    layernorm(x1, ln2, S, D, eps);
    mod2 = select_scale(2 * D, ln2);

    std::vector<float> proj, gate, hidden, mlp_out;
    matmul_xwt(mod2, *w.mlp_proj, proj, S, mlp, D);
    matmul_xwt(mod2, *w.mlp_gate, gate, S, mlp, D);
    hidden.resize(proj.size());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = silu(gate[i]) * proj[i];
    matmul_xwt(hidden, *w.mlp_out, mlp_out, S, D, mlp);

    out.assign(xin.size(), 0.0F);
    for (std::size_t s = 0; s < S; ++s) {
        const float* g = gate_of(3 * D, s);  // gate_mlp chunk
        for (std::size_t d = 0; d < D; ++d) out[s * D + d] = x1[s * D + d] + std::tanh(g[d]) * mlp_out[s * D + d];
    }
}

// Relative Frobenius error of got vs ref.
inline double rel_fro(const std::vector<float>& got, const std::vector<float>& ref) {
    double d = 0.0, r = 0.0;
    require(got.size() == ref.size(), "rel_fro size mismatch");
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double dv = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        d += dv * dv;
        r += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return std::sqrt(d) / (std::sqrt(r) + 1e-30);
}

// Relative Frobenius error of got vs ref (double operands).
inline double rel_fro_d(const std::vector<double>& got, const std::vector<double>& ref) {
    double d = 0.0, r = 0.0;
    require(got.size() == ref.size(), "rel_fro_d size mismatch");
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double dv = got[i] - ref[i];
        d += dv * dv;
        r += ref[i] * ref[i];
    }
    return std::sqrt(d) / (std::sqrt(r) + 1e-30);
}

}  // namespace qioracle
