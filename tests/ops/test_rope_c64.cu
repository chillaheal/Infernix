// GPU numeric test for the Qwen-Image DiT complex RoPE Op (ops::rope_c64).
//
// Independent oracle: the complex rotation evaluated naively in FP64 from the BF16-represented
// inputs. The coefficient table is the authoritative 3-axis positional cis builder from the shared
// oracle header (build_rope, axes [16,56,56], theta 10000) for real DiT geometries, plus a random
// unit-complex table to exercise grid edges. Skips (77) when no CUDA device is available.

#include "infernix/ops/rope_c64.h"

#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace qo = qioracle;

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
    std::uint32_t s = 12345u;
    float next(float lo, float hi) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
        return lo + u * (hi - lo);
    }
};

bool run_pair(const char* label, std::size_t text_prefix, std::size_t H, std::size_t W,
              double max_rel) {
    const std::size_t heads = 32;
    const std::size_t S = text_prefix + H * W;
    const std::size_t D = 128;

    const std::vector<float> rot_f = qo::build_rope(text_prefix, H, W);  // [S,128] re/im interleaved
    std::vector<float> rot_repr(rot_f.size());
    for (std::size_t i = 0; i < rot_f.size(); ++i) rot_repr[i] = from_bf16(to_bf16(rot_f[i]));

    Lcg rng;
    rng.s = static_cast<std::uint32_t>(S * 7 + 13);
    std::vector<float> q_f(S * heads * D), k_f(S * heads * D);
    for (auto& v : q_f) v = rng.next(-4.f, 4.f);
    for (auto& v : k_f) v = rng.next(-4.f, 4.f);
    std::vector<float> q_repr(q_f.size()), k_repr(k_f.size());
    for (std::size_t i = 0; i < q_f.size(); ++i) q_repr[i] = from_bf16(to_bf16(q_f[i]));
    for (std::size_t i = 0; i < k_f.size(); ++i) k_repr[i] = from_bf16(to_bf16(k_f[i]));

    // FP64 reference over the BF16-represented operands, same channel-first index as the kernel.
    std::vector<double> q_ref(q_repr.size()), k_ref(k_repr.size());
    for (std::size_t s = 0; s < S; ++s) {
        for (std::size_t h = 0; h < heads; ++h) {
            for (int j = 0; j < 64; ++j) {
                const std::size_t base = 2 * j + D * (h + heads * s);
                const std::size_t rb   = 2 * j + D * s;
                const double x0 = q_repr[base], x1 = q_repr[base + 1];
                const double c0 = rot_repr[rb], c1 = rot_repr[rb + 1];
                q_ref[base]     = x0 * c0 - x1 * c1;
                q_ref[base + 1] = x0 * c1 + x1 * c0;
                const double y0 = k_repr[base], y1 = k_repr[base + 1];
                k_ref[base]     = y0 * c0 - y1 * c1;
                k_ref[base + 1] = y0 * c1 + y1 * c0;
            }
        }
    }

    const auto q_dev = pack(q_f);
    const auto k_dev = pack(k_f);
    const auto rot_dev = pack(rot_repr);

    void *dq, *dk, *drot;
    if (cudaMalloc(&dq, q_dev.size() * 2) != cudaSuccess) return false;
    cudaMalloc(&dk, k_dev.size() * 2);
    cudaMalloc(&drot, rot_dev.size() * 2);
    cudaMemcpy(dq, q_dev.data(), q_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k_dev.data(), k_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(drot, rot_dev.data(), rot_dev.size() * 2, cudaMemcpyHostToDevice);

    infernix::Tensor rot_t(drot, infernix::DType::BF16, {(std::int32_t)D, 1, (std::int32_t)S, 1});
    infernix::Tensor q_t(dq, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::Tensor k_t(dk, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::ops::rope_c64(rot_t, q_t, k_t, 0);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> q_got(q_dev.size()), k_got(k_dev.size());
    cudaMemcpy(q_got.data(), dq, q_got.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(k_got.data(), dk, k_got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(drot);

    auto rel = [&](const std::vector<std::uint16_t>& got, const std::vector<double>& exp) {
        double d = 0.0, e = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double g = from_bf16(got[i]), x = exp[i];
            d += (g - x) * (g - x);
            e += x * x;
        }
        return std::sqrt(d) / (std::sqrt(e) + 1e-30);
    };
    const double eq = rel(q_got, q_ref), ek = rel(k_got, k_ref);
    const bool ok = (eq < max_rel) && (ek < max_rel) && std::isfinite(eq) && std::isfinite(ek);
    std::printf("%-34s S=%zu rel_q=%.3e rel_k=%.3e %s\n", label, S, eq, ek, ok ? "OK" : "FAIL");
    return ok;
}

// Single-form with a random unit-complex table, exercising the launch grid and the single-x entry.
bool run_single_random(const char* label, std::size_t S, double max_rel) {
    const std::size_t heads = 32, D = 128;
    Lcg rng;
    rng.s = static_cast<std::uint32_t>(S * 31 + 5);
    std::vector<float> x_f(S * heads * D);
    for (auto& v : x_f) v = rng.next(-4.f, 4.f);
    std::vector<float> x_repr(x_f.size());
    for (std::size_t i = 0; i < x_f.size(); ++i) x_repr[i] = from_bf16(to_bf16(x_f[i]));

    std::vector<float> rot_f(S * D);
    for (std::size_t i = 0; i < rot_f.size(); i += 2) {
        const double a = rng.next(-3.14159265f, 3.14159265f);
        rot_f[i] = (float)std::cos(a);
        rot_f[i + 1] = (float)std::sin(a);
    }
    std::vector<float> rot_repr(rot_f.size());
    for (std::size_t i = 0; i < rot_f.size(); ++i) rot_repr[i] = from_bf16(to_bf16(rot_f[i]));

    std::vector<double> ref(x_repr.size());
    for (std::size_t s = 0; s < S; ++s)
        for (std::size_t h = 0; h < heads; ++h)
            for (int j = 0; j < 64; ++j) {
                const std::size_t base = 2 * j + D * (h + heads * s);
                const std::size_t rb   = 2 * j + D * s;
                const double x0 = x_repr[base], x1 = x_repr[base + 1];
                const double c0 = rot_repr[rb], c1 = rot_repr[rb + 1];
                ref[base]     = x0 * c0 - x1 * c1;
                ref[base + 1] = x0 * c1 + x1 * c0;
            }

    const auto x_dev = pack(x_f), rot_dev = pack(rot_repr);
    void *dx, *drot;
    if (cudaMalloc(&dx, x_dev.size() * 2) != cudaSuccess) return false;
    cudaMalloc(&drot, rot_dev.size() * 2);
    cudaMemcpy(dx, x_dev.data(), x_dev.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(drot, rot_dev.data(), rot_dev.size() * 2, cudaMemcpyHostToDevice);

    infernix::Tensor rot_t(drot, infernix::DType::BF16, {(std::int32_t)D, 1, (std::int32_t)S, 1});
    infernix::Tensor x_t(dx, infernix::DType::BF16, {(std::int32_t)D, (std::int32_t)heads, (std::int32_t)S, 1});
    infernix::ops::rope_c64(rot_t, x_t, 0);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> got(x_dev.size());
    cudaMemcpy(got.data(), dx, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(drot);

    double d = 0.0, e = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = from_bf16(got[i]), x = ref[i];
        d += (g - x) * (g - x);
        e += x * x;
    }
    const double rel = std::sqrt(d) / (std::sqrt(e) + 1e-30);
    const bool ok = (rel < max_rel) && std::isfinite(rel);
    std::printf("%-34s S=%zu rel=%.3e %s\n", label, S, rel, ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const double tol = 5e-3;  // BF16 storage of a complex product of two BF16 operands
    bool ok = true;
    ok &= run_pair("dit text8 16x16", 8, 16, 16, tol);
    ok &= run_pair("dit text8 8x8", 8, 8, 8, tol);
    ok &= run_pair("dit text4 4x4", 4, 4, 4, tol);
    ok &= run_pair("dit single-token", 0, 1, 1, tol);
    ok &= run_single_random("random rot S=2048", 2048, tol);
    ok &= run_single_random("random rot S=1", 1, tol);
    std::printf(ok ? "ROPE_C64_TEST: PASS\n" : "ROPE_C64_TEST: FAIL\n");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
