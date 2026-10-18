// GPU numeric test for the DiT AdaLN-Zero modulation Ops (ops::adaln_scale, ops::adaln_gate_add).
//
// Independent oracle: the modulation formula evaluated naively in FP64 from the BF16-represented
// operands, using the same channel-first [D,S,1,1] index and the same [M,2] modulation table row
// selection (row 1 for s < text_prefix, row 0 for the target suffix) as the kernels. Random inputs
// exercise the grid; real DiT D=4096 and the S=1 / all-text / all-target / wide-grid edges cover the
// contract domain. Skips (77) when no CUDA device is available.

#include "infernix/ops/dit_adaln.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
std::vector<float> represent(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}

struct Lcg {
    std::uint32_t s = 12345u;
    float next(float lo, float hi) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
        return lo + u * (hi - lo);
    }
};

double rel(const std::vector<std::uint16_t>& got, const std::vector<double>& exp) {
    double d = 0.0, e = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = from_bf16(got[i]), x = exp[i];
        d += (g - x) * (g - x);
        e += x * x;
    }
    return std::sqrt(d) / (std::sqrt(e) + 1e-30);
}

// One adaln_scale case over [D,S,1,1] with a [4D,2] table; col_off selects a chunk, text_prefix the row split.
bool run_scale(std::size_t D, std::size_t S, std::size_t text_prefix, std::size_t col_off,
               double max_rel) {
    const std::size_t M = 4 * D, stride = M;
    Lcg rng;
    rng.s = static_cast<std::uint32_t>(D * 3 + S * 17 + col_off * 5 + 1);
    std::vector<float> x_f(D * S), mod_f(M * 2);
    for (auto& v : x_f) v = rng.next(-4.f, 4.f);
    for (auto& v : mod_f) v = rng.next(-1.f, 1.f);
    const auto x_repr = represent(x_f), mod_repr = represent(mod_f);

    std::vector<double> exp(D * S);
    for (std::size_t i = 0; i < D * S; ++i) {
        const std::size_t d = i % D, s = i / D;
        const std::size_t row = (s < text_prefix) ? 1 : 0;
        exp[i] = static_cast<double>(x_repr[i]) * (1.0 + mod_repr[col_off + d + row * stride]);
    }

    const auto x_dev = pack(x_f), mod_dev = pack(mod_f);
    void *dx, *dmod, *dout;
    if (cudaMalloc(&dx, x_dev.size() * 2) != cudaSuccess) return false;
    cudaMalloc(&dmod, mod_dev.size() * 2);
    cudaMalloc(&dout, (D * S) * 2);
    cudaMemcpy(dx, x_dev.data(), x_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dmod, mod_dev.data(), mod_dev.size() * 2, cudaMemcpyHostToDevice);

    infernix::Tensor x_t(dx, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)S, 1, 1});
    infernix::Tensor m_t(dmod, infernix::DType::BF16, {(std::int32_t)M, 2, 1, 1});
    infernix::Tensor o_t(dout, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)S, 1, 1});
    infernix::ops::adaln_scale(x_t, m_t, (std::int32_t)col_off, (std::int32_t)text_prefix, o_t, 0);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> got(D * S);
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dmod); cudaFree(dout);

    const double r = rel(got, exp);
    const bool ok = (r < max_rel) && std::isfinite(r);
    std::printf("scale   D=%-5zu S=%-5zu tp=%-5zu off=%-6zu rel=%.3e %s\n", D, S, text_prefix,
                col_off, r, ok ? "OK" : "FAIL");
    return ok;
}

// One adaln_gate_add case over [D,S,1,1].
bool run_gate(std::size_t D, std::size_t S, std::size_t text_prefix, std::size_t col_off,
              double max_rel) {
    const std::size_t M = 4 * D, stride = M;
    Lcg rng;
    rng.s = static_cast<std::uint32_t>(D * 7 + S * 19 + col_off * 11 + 3);
    std::vector<float> res_f(D * S), br_f(D * S), mod_f(M * 2);
    for (auto& v : res_f) v = rng.next(-4.f, 4.f);
    for (auto& v : br_f) v = rng.next(-4.f, 4.f);
    for (auto& v : mod_f) v = rng.next(-1.f, 1.f);
    const auto res_repr = represent(res_f), br_repr = represent(br_f), mod_repr = represent(mod_f);

    std::vector<double> exp(D * S);
    for (std::size_t i = 0; i < D * S; ++i) {
        const std::size_t d = i % D, s = i / D;
        const std::size_t row = (s < text_prefix) ? 1 : 0;
        const double mv = mod_repr[col_off + d + row * stride];
        exp[i] = res_repr[i] + std::tanh(mv) * br_repr[i];
    }

    const auto res_dev = pack(res_f), br_dev = pack(br_f), mod_dev = pack(mod_f);
    void *dres, *dbr, *dmod, *dout;
    if (cudaMalloc(&dres, res_dev.size() * 2) != cudaSuccess) return false;
    cudaMalloc(&dbr, br_dev.size() * 2);
    cudaMalloc(&dmod, mod_dev.size() * 2);
    cudaMalloc(&dout, (D * S) * 2);
    cudaMemcpy(dres, res_dev.data(), res_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dbr, br_dev.data(), br_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dmod, mod_dev.data(), mod_dev.size() * 2, cudaMemcpyHostToDevice);

    infernix::Tensor r_t(dres, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)S, 1, 1});
    infernix::Tensor b_t(dbr, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)S, 1, 1});
    infernix::Tensor m_t(dmod, infernix::DType::BF16, {(std::int32_t)M, 2, 1, 1});
    infernix::Tensor o_t(dout, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)S, 1, 1});
    infernix::ops::adaln_gate_add(r_t, b_t, m_t, (std::int32_t)col_off, (std::int32_t)text_prefix,
                                  o_t, 0);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> got(D * S);
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dres); cudaFree(dbr); cudaFree(dmod); cudaFree(dout);

    const double r = rel(got, exp);
    const bool ok = (r < max_rel) && std::isfinite(r);
    std::printf("gate    D=%-5zu S=%-5zu tp=%-5zu off=%-6zu rel=%.3e %s\n", D, S, text_prefix,
                col_off, r, ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const double tol = 6e-3;
    const std::size_t D = 4096;  // DiT inner_dim
    bool ok = true;
    // scale: real DiT D, the two scale chunks, all-text, all-target, single token, wide grid edge.
    ok &= run_scale(D, 20, 8, 0, tol);          // scale_attn1 chunk
    ok &= run_scale(D, 20, 8, 2 * D, tol);       // scale_attn2 chunk
    ok &= run_scale(D, 16, 16, D, tol);          // all tokens in the text row (row 1)
    ok &= run_scale(D, 1, 0, 3 * D, tol);         // single token, target row (row 0)
    ok &= run_scale(64, 16384, 8192, 0, tol);     // wide launch grid, small D
    // gate_add: real DiT D, the two gate chunks, all-target, single token.
    ok &= run_gate(D, 20, 8, D, tol);             // gate_attn chunk
    ok &= run_gate(D, 20, 8, 3 * D, tol);         // gate_mlp chunk
    ok &= run_gate(D, 1, 0, D, tol);              // single token, target row
    std::printf(ok ? "DIT_ADALN_TEST: PASS\n" : "DIT_ADALN_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
