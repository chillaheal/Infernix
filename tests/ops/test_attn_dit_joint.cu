// GPU numeric test for the Qwen-Image DiT joint block-causal attention Op (ops::attn_dit_joint).
//
// Independent oracle: the masked softmax evaluated naively in FP64 from the BF16-represented q/k/v
// over the same channel-first index. The block-causal mask allow(s, kk) = (s >= text_prefix) ||
// (kk <= s) is applied exactly as the Op's contract. Skips (77) when no CUDA device is available.

#include "infernix/ops/attn_dit_joint.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

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
std::vector<std::uint16_t> pack(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}

struct Lcg {
    std::uint32_t s = 1u;
    float next(float lo, float hi) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
        return lo + u * (hi - lo);
    }
};

const std::size_t kDim = 128;

// Represented BF16 operands and the FP64 masked-softmax oracle written to `ref` (same index).
void make_case(std::size_t heads, std::size_t S, std::size_t text_prefix,
               std::vector<float>& q_repr, std::vector<float>& k_repr, std::vector<float>& v_repr,
               std::vector<double>& ref, unsigned seed) {
    const std::size_t n = S * heads * kDim;
    std::vector<float> q_f(n), k_f(n), v_f(n);
    Lcg rng;
    rng.s = seed;
    for (auto& x : q_f) x = rng.next(-3.f, 3.f);
    for (auto& x : k_f) x = rng.next(-3.f, 3.f);
    for (auto& x : v_f) x = rng.next(-3.f, 3.f);
    q_repr.resize(n); k_repr.resize(n); v_repr.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        q_repr[i] = from_bf16(to_bf16(q_f[i]));
        k_repr[i] = from_bf16(to_bf16(k_f[i]));
        v_repr[i] = from_bf16(to_bf16(v_f[i]));
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
    ref.assign(n, 0.0);
    std::vector<double> scores(S), probs(S);
    for (std::size_t s = 0; s < S; ++s) {
        const bool is_text = (s < text_prefix);
        const std::size_t kv_hi = is_text ? text_prefix : S;
        for (std::size_t h = 0; h < heads; ++h) {
            const auto qbase = [&](std::size_t tok, std::size_t d) { return d + kDim * (h + heads * tok); };
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (is_text && kk > s) { probs[kk] = 0.0; continue; }
                double dot = 0.0;
                for (std::size_t d = 0; d < kDim; ++d)
                    dot += static_cast<double>(q_repr[qbase(s, d)]) * static_cast<double>(k_repr[qbase(kk, d)]);
                scores[kk] = dot * scale;
            }
            double maxv = -std::numeric_limits<double>::infinity();
            for (std::size_t kk = 0; kk < kv_hi; ++kk)
                if (!(is_text && kk > s) && scores[kk] > maxv) maxv = scores[kk];
            double sum = 0.0;
            for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                if (is_text && kk > s) { probs[kk] = 0.0; continue; }
                const double e = std::exp(scores[kk] - maxv);
                probs[kk] = e;
                sum += e;
            }
            const double inv = 1.0 / sum;
            for (std::size_t d = 0; d < kDim; ++d) {
                double acc = 0.0;
                for (std::size_t kk = 0; kk < kv_hi; ++kk) {
                    if (probs[kk] == 0.0) continue;
                    acc += probs[kk] * static_cast<double>(v_repr[qbase(kk, d)]);
                }
                ref[qbase(s, d)] = acc * inv;
            }
        }
    }
}

bool run(const char* label, std::size_t heads, std::size_t S, std::size_t text_prefix,
         double tol, unsigned seed) {
    std::vector<float> q_repr, k_repr, v_repr;
    std::vector<double> ref;
    make_case(heads, S, text_prefix, q_repr, k_repr, v_repr, ref, seed);

    const auto q_dev = pack(q_repr), k_dev = pack(k_repr), v_dev = pack(v_repr);
    const std::size_t bytes = q_dev.size() * 2;

    void *dq, *dk, *dv, *dout;
    if (cudaMalloc(&dq, bytes) != cudaSuccess) return false;
    cudaMalloc(&dk, bytes); cudaMalloc(&dv, bytes); cudaMalloc(&dout, bytes);
    cudaMemcpy(dq, q_dev.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k_dev.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_dev.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, bytes);

    infernix::Tensor q_t(dq, infernix::DType::BF16, {(std::int32_t)kDim, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::Tensor k_t(dk, infernix::DType::BF16, {(std::int32_t)kDim, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::Tensor v_t(dv, infernix::DType::BF16, {(std::int32_t)kDim, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::Tensor o_t(dout, infernix::DType::BF16, {(std::int32_t)kDim, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::ops::attn_dit_joint(q_t, k_t, v_t, (std::int32_t)text_prefix, o_t, 0);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> got(q_dev.size());
    cudaMemcpy(got.data(), dout, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);

    double d = 0.0, e = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = from_bf16(got[i]), x = ref[i];
        d += (g - x) * (g - x);
        e += x * x;
    }
    const double rel = std::sqrt(d) / (std::sqrt(e) + 1e-30);
    const bool ok = (rel < tol) && std::isfinite(rel);
    std::printf("%-30s heads=%zu S=%zu tp=%zu rel=%.3e %s\n", label, heads, S, text_prefix, rel,
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const double tol = 6e-3;  // BF16 storage of an FP32 online softmax over BF16 operands
    bool ok = true;
    ok &= run("joint tp8 16x16", 32, 264, 8, tol, 11);
    ok &= run("joint tp8 8x8",   32, 72,  8, tol, 12);
    ok &= run("joint tp4 4x4",   32, 20,  4, tol, 13);
    ok &= run("image-only tp0",  32, 64,  0, tol, 14);
    ok &= run("causal tp=S",     32, 48, 48, tol, 15);
    ok &= run("wide tp100 S700", 32, 700, 100, tol, 16);
    ok &= run("wide tp680 S700", 32, 700, 680, tol, 17);
    ok &= run("single tp1 S1",   32, 1,   1, tol, 18);
    std::printf(ok ? "ATTN_DIT_JOINT_TEST: PASS\n" : "ATTN_DIT_JOINT_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
