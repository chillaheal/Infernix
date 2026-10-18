// CPU-only numeric oracle for one Qwen-Image-2.1 DiT block. It loads the authoritative block-0
// forward reference (block_in -> block_out) and its FP32 weights, both produced by the upstream
// diffusers QwenImage21Transformer2DModel in an isolated oracle venv, then re-derives the block in
// naive float32 (double accumulation) and requires it to match block_out. It qualifies the whole
// novel-block composition in one falsifiable unit: LayerNorm(affine=False) -> scale-only AdaLN with
// the [scale_attn, gate_attn, scale_mlp, gate_mlp] chunk order and causal_condition row select ->
// shared QKV -> per-head RMSNorm BEFORE RoPE -> 3-axis complex RoPE -> block-causal attention ->
// gate.tanh() residual -> LayerNorm -> SwiGLU -> gate.tanh() residual.
//
// The reference is FP32 (the original checkpoint weights), so the FP8 route's quantization error is
// deliberately kept out of this unit; FP8 RowScale decode is qualified separately by the load oracle.
//
// Reads $INFERNIX_IMAGE_ORACLE/tensors/*.npy and $INFERNIX_IMAGE_ORACLE/tensors_block_weights/*.npy.
// Returns 77 (CTest skip) when the oracle data is absent, so CI without the dump is unaffected.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A minimal .npy reader: v1/v2 header, C-contiguous, little-endian '<f4' / '<c8' / '|b1'.
struct NpyArray {
    std::string descr;                 // "<f4", "<c8", "|b1"
    std::vector<std::size_t> shape;    // element count per axis (not counting the c8 pair)
    std::vector<std::uint8_t> bytes;   // raw little-endian payload
    std::size_t numel = 0;             // product(shape)

    std::size_t float_count() const {
        return descr == "<c8" ? numel * 2 : numel;
    }
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
};

std::string read_all_bytes(const std::string& path, std::vector<std::uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(n));
    if (n > 0) {
        f.read(reinterpret_cast<char*>(out.data()), n);
    }
    return path;
}

std::size_t first_true_index(const std::vector<std::uint8_t>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i]) return i;
    }
    return v.size();
}

NpyArray read_npy(const std::string& path) {
    std::vector<std::uint8_t> raw;
    read_all_bytes(path, raw);
    require(raw.size() > 12, "npy file too small");
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    require(std::memcmp(raw.data(), magic, 6) == 0, "not a npy file (magic)");
    const std::uint8_t major = raw[6];
    std::size_t header_len = 0;
    std::size_t data_off = 0;
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

    require(header.find("'fortran_order': False") != std::string::npos, "npy not C-contiguous");

    const std::size_t spos = header.find("'shape'");
    require(spos != std::string::npos, "npy header missing shape");
    const std::size_t popen = header.find('(', spos);
    const std::size_t pclose = header.find(')', popen);
    require(popen != std::string::npos && pclose != std::string::npos, "npy shape paren parse");
    std::string dims = header.substr(popen + 1, pclose - popen - 1);
    std::size_t i = 0;
    while (i < dims.size()) {
        while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) {
            ++i;
        }
        std::size_t j = i;
        while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') {
            ++j;
        }
        if (j > i) {
            a.shape.push_back(std::stoull(dims.substr(i, j - i)));
            i = j;
        } else {
            ++i;
        }
    }

    const std::size_t item = (a.descr == "<c8") ? 8 : 4;  // f4/b1 = 4/1 handled below
    const std::size_t elem = (a.descr == "<f4" || a.descr == "|b1" || a.descr == "<b1") ?
                                 ((a.descr == "<f4") ? 4u : 1u) : item;
    a.numel = 1;
    for (std::size_t s : a.shape) {
        a.numel *= s;
    }
    require(raw.size() == data_off + a.numel * elem, "npy payload size mismatch");
    a.bytes.assign(raw.begin() + data_off, raw.end());
    return a;
}

// C[m][n] = sum_k A[m][k] * W[n][k]  (W is a torch Linear weight [out,in]; this is x @ W^T).
void matmul_xwt(const std::vector<float>& A, const std::vector<float>& W, std::vector<float>& C,
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

// torch.nn.LayerNorm(elementwise_affine=False): (x-mean)/sqrt(var+eps), biased variance (divide by N).
void layernorm(const std::vector<float>& x, std::vector<float>& y, std::size_t rows, std::size_t d, float eps) {
    y.assign(x.size(), 0.0F);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + r * d;
        float* yr = y.data() + r * d;
        double mean = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            mean += xr[i];
        }
        mean /= static_cast<double>(d);
        double var = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double t = static_cast<double>(xr[i]) - mean;
            var += t * t;
        }
        var /= static_cast<double>(d);
        const float inv = 1.0F / std::sqrt(static_cast<float>(var) + eps);
        for (std::size_t i = 0; i < d; ++i) {
            yr[i] = static_cast<float>((static_cast<double>(xr[i]) - mean) * static_cast<double>(inv));
        }
    }
}

// diffusers RMSNorm over the last axis (head_dim): x*rsqrt(mean(x^2)+eps)*weight.
void rmsnorm_head(std::vector<float>& qh, std::size_t rows, std::size_t d,
                  const std::vector<float>& w, float eps) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* xr = qh.data() + r * d;
        double ss = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            ss += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        const float rr = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + eps);
        for (std::size_t i = 0; i < d; ++i) {
            xr[i] = xr[i] * rr * w[i];
        }
    }
}

float silu(float x) {
    return x / (1.0F + std::exp(-x));
}

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    const char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_IMAGE_ORACLE");
        if (env == nullptr || std::string(env).empty()) {
            std::cout << "INFERNIX_IMAGE_ORACLE unset; skipping Qwen-Image block oracle\n";
            return 77;
        }
        const std::string oracle(env);
        const std::string tdir = join(oracle, "tensors");
        const std::string wdir = join(oracle, "tensors_block_weights");

        // Block inputs + reference output.
        NpyArray block_in = read_npy(join(tdir, "block_in.npy"));
        NpyArray block_out = read_npy(join(tdir, "block_out.npy"));
        NpyArray modulation = read_npy(join(tdir, "block_modulation.npy"));
        NpyArray rotary = read_npy(join(tdir, "block_rotary.npy"));
        NpyArray target_mask = read_npy(join(tdir, "block_target_mask.npy"));

        require(block_in.shape.size() == 3 && block_in.shape[0] == 1, "block_in not [1,S,D]");
        require(block_out.shape == block_in.shape, "block_out shape mismatch");
        const std::size_t S = block_in.shape[1];
        const std::size_t D = block_in.shape[2];
        const std::size_t heads = 32;
        const std::size_t head_dim = D / heads;
        require(head_dim == 128, "expected head_dim 128");
        const std::size_t rope_dim = head_dim / 2;  // 64 complex pairs, shared across heads

        require(modulation.shape.size() == 2 && modulation.shape[1] == 4 * D, "modulation not [B+1,4D]");
        const std::size_t mod_rows = modulation.shape[0];
        require(mod_rows == 2, "expected causal_condition modulation rows = 2 (t, t=0)");

        require(rotary.descr == "<c8" && rotary.shape.size() == 2 && rotary.shape[0] == S &&
                    rotary.shape[1] == rope_dim, "block_rotary not [S,64] complex");
        require(target_mask.shape.size() == 1 && target_mask.shape[0] == S, "block_target_mask not [S]");

        // causal_condition row select: target tokens read row 0 (real t), everything else row 1 (t=0).
        // The dumped layout is a text prefix then one contiguous target-image block.
        const std::vector<std::uint8_t> tmask = target_mask.bools();
        const std::size_t prefix_len = first_true_index(tmask);
        require(prefix_len < S, "target_mask has no target tokens");
        for (std::size_t s = prefix_len; s < S; ++s) {
            require(tmask[s] != 0, "target tokens must be a contiguous suffix");
        }

        // Weights (FP32).
        auto W = [&](const std::string& name) {
            NpyArray a = read_npy(join(wdir, name));
            return a.floats();
        };
        const std::vector<float> w_q = W("blk0_attn_to_q_weight.npy");
        const std::vector<float> w_k = W("blk0_attn_to_k_weight.npy");
        const std::vector<float> w_v = W("blk0_attn_to_v_weight.npy");
        const std::vector<float> w_out = W("blk0_attn_to_out_0_weight.npy");
        const std::vector<float> w_nq = W("blk0_attn_norm_q_weight.npy");
        const std::vector<float> w_nk = W("blk0_attn_norm_k_weight.npy");
        const std::vector<float> w_mlp_proj = W("blk0_img_mlp_proj_weight.npy");
        const std::vector<float> w_mlp_gate = W("blk0_img_mlp_gate_layer_weight.npy");
        const std::vector<float> w_mlp_out = W("blk0_img_mlp_out_weight.npy");
        require(w_q.size() == D * D && w_out.size() == D * D, "attn weight size");
        require(w_nq.size() == head_dim && w_nk.size() == head_dim, "qk norm weight size");
        const std::size_t mlp = w_mlp_proj.size() / D;
        require(w_mlp_gate.size() == D * mlp && w_mlp_out.size() == mlp * D, "mlp weight size");

        const std::vector<float> x_in = block_in.floats();
        const std::vector<float> mod = modulation.floats();
        const std::vector<float> rot = rotary.floats();  // interleaved re,im; [S][rope_dim][2]
        const std::vector<float> ref = block_out.floats();

        const float eps = 1e-6F;

        // Select per-token scale/gate rows from the shared [2,4D] modulation, chunk order
        // [scale_attn, gate_attn, scale_mlp, gate_mlp]. row = target ? 0 : 1.
        auto scale_gate = [&](const std::vector<float>& xin, std::size_t col_off) {
            std::vector<float> out(xin.size());
            for (std::size_t s = 0; s < S; ++s) {
                const std::size_t row = tmask[s] ? 0 : 1;
                const float* m = mod.data() + row * (4 * D) + col_off;
                const float* x = xin.data() + s * D;
                float* o = out.data() + s * D;
                for (std::size_t d = 0; d < D; ++d) {
                    o[d] = x[d] * (1.0F + m[d]);
                }
            }
            return out;
        };

        // norm1 + scale-only AdaLN.
        std::vector<float> ln1, mod1;
        layernorm(x_in, ln1, S, D, eps);
        mod1 = scale_gate(ln1, 0);  // scale_attn chunk at column offset 0

        // QKV projections.
        std::vector<float> q_lin, k_lin, v_lin;
        matmul_xwt(mod1, w_q, q_lin, S, D, D);
        matmul_xwt(mod1, w_k, k_lin, S, D, D);
        matmul_xwt(mod1, w_v, v_lin, S, D, D);

        // Per-head QK RMSNorm, then 3-axis complex RoPE (use_real=False), pairs (2j, 2j+1).
        const auto apply_rope = [&](std::vector<float>& qh) {
            for (std::size_t s = 0; s < S; ++s) {
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
        };
        // rmsnorm over (S*heads) head-rows of head_dim each (before RoPE):
        rmsnorm_head(q_lin, S * heads, head_dim, w_nq, eps);
        rmsnorm_head(k_lin, S * heads, head_dim, w_nk, eps);
        apply_rope(q_lin);
        apply_rope(k_lin);
        // q_lin/k_lin/v_lin are now [S, heads, head_dim] viewed contiguously = [S, D].

        // Block-causal attention. Dumped structure: one strictly-causal text prefix [0,prefix_len)
        // then the target image [prefix_len,S) attending the full sequence. No padding key mask.
        const float qscale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> attn(S * D, 0.0F);
        {
            std::vector<float> scores(S);
            std::vector<char> allowed(S);
            std::vector<float> probs(S);
            std::vector<float> acc(head_dim);
            for (std::size_t s = 0; s < S; ++s) {
                const bool is_text = (s < prefix_len);
                const std::size_t kv_hi = is_text ? prefix_len : S;
                for (std::size_t h = 0; h < heads; ++h) {
                    const float* qs = q_lin.data() + (s * heads + h) * head_dim;
                    // scores over keys [0, kv_hi)
                    std::fill(scores.begin(), scores.begin() + kv_hi, 0.0F);
                    for (std::size_t k = 0; k < kv_hi; ++k) {
                        const float* ks = k_lin.data() + (k * heads + h) * head_dim;
                        double dot = 0.0;
                        for (std::size_t d = 0; d < head_dim; ++d) {
                            dot += static_cast<double>(qs[d]) * static_cast<double>(ks[d]);
                        }
                        scores[k] = static_cast<float>(dot) * qscale;
                    }
                    for (std::size_t k = 0; k < kv_hi; ++k) {
                        allowed[k] = (is_text && k > s) ? 0 : 1;  // text: causal triangle; target: full
                    }
                    probs.assign(S, 0.0F);
                    {
                        // softmax over only the first kv_hi slots
                        float maxv = -std::numeric_limits<float>::infinity();
                        for (std::size_t k = 0; k < kv_hi; ++k) {
                            if (allowed[k] && scores[k] > maxv) maxv = scores[k];
                        }
                        double sum = 0.0;
                        for (std::size_t k = 0; k < kv_hi; ++k) {
                            if (allowed[k]) {
                                const double e = std::exp(static_cast<double>(scores[k]) - static_cast<double>(maxv));
                                probs[k] = static_cast<float>(e);
                                sum += e;
                            }
                        }
                        require(sum > 0.0, "fully masked attention row");
                        const float inv = static_cast<float>(1.0 / sum);
                        for (std::size_t k = 0; k < kv_hi; ++k) probs[k] *= inv;
                    }
                    std::fill(acc.begin(), acc.end(), 0.0F);
                    for (std::size_t k = 0; k < kv_hi; ++k) {
                        if (probs[k] == 0.0F) continue;
                        const float* vs = v_lin.data() + (k * heads + h) * head_dim;
                        for (std::size_t d = 0; d < head_dim; ++d) {
                            acc[d] += probs[k] * vs[d];
                        }
                    }
                    float* od = attn.data() + (s * heads + h) * head_dim;
                    for (std::size_t d = 0; d < head_dim; ++d) od[d] = acc[d];
                }
            }
        }

        // Output projection + gated residual 1.
        std::vector<float> attn_out;
        matmul_xwt(attn, w_out, attn_out, S, D, D);
        std::vector<float> x1(x_in.size());
        for (std::size_t s = 0; s < S; ++s) {
            const std::size_t row = tmask[s] ? 0 : 1;
            const float* g = mod.data() + row * (4 * D) + D;  // gate_attn chunk at offset D
            for (std::size_t d = 0; d < D; ++d) {
                x1[s * D + d] = x_in[s * D + d] + std::tanh(g[d]) * attn_out[s * D + d];
            }
        }

        // norm2 + scale-only AdaLN + SwiGLU + gated residual 2.
        std::vector<float> ln2, mod2;
        layernorm(x1, ln2, S, D, eps);
        mod2 = scale_gate(ln2, 2 * D);  // scale_mlp chunk at column offset 2D

        std::vector<float> gproj, ggate, gh, mlp_out;
        matmul_xwt(mod2, w_mlp_proj, gproj, S, mlp, D);
        matmul_xwt(mod2, w_mlp_gate, ggate, S, mlp, D);
        gh.resize(gproj.size());
        for (std::size_t i = 0; i < gh.size(); ++i) {
            gh[i] = silu(ggate[i]) * gproj[i];
        }
        matmul_xwt(gh, w_mlp_out, mlp_out, S, D, mlp);

        std::vector<float> x_out(x_in.size());
        for (std::size_t s = 0; s < S; ++s) {
            const std::size_t row = tmask[s] ? 0 : 1;
            const float* g = mod.data() + row * (4 * D) + 3 * D;  // gate_mlp chunk at offset 3D
            for (std::size_t d = 0; d < D; ++d) {
                x_out[s * D + d] = x1[s * D + d] + std::tanh(g[d]) * mlp_out[s * D + d];
            }
        }

        // Compare against the authoritative block_out.
        double diff_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
        for (std::size_t i = 0; i < x_out.size(); ++i) {
            const double dv = static_cast<double>(x_out[i]) - static_cast<double>(ref[i]);
            diff_sq += dv * dv;
            ref_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            max_abs = std::max(max_abs, std::fabs(dv));
        }
        const double rel_fro = std::sqrt(diff_sq) / (std::sqrt(ref_sq) + 1e-30);
        std::cout << "Qwen-Image block oracle S=" << S << " D=" << D << " heads=" << heads
                  << " mlp=" << mlp << " prefix_len=" << prefix_len
                  << " rel_fro_err=" << rel_fro << " max_abs_err=" << max_abs << "\n";

        require(rel_fro < 1e-3, "block forward diverges from oracle (rel Frobenius >= 1e-3)");
        require(std::isfinite(rel_fro), "non-finite block output");
        std::cout << "Qwen-Image block oracle PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen-Image block oracle FAILED: " << error.what() << "\n";
        return 1;
    }
}
