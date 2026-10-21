// GPU qualification of ops::spatial_attn against a naive FP64 oracle over the channel planes [C,N]
// (single-head full attention). This qualifies the ONLINE-SOFTMAX rewrite: K and V stream through
// shared in 256-column tiles and the running max/sum/accumulator rescale when a later tile raises
// the max, so block memory is O(C + 256) floats instead of O(N) and N reaches a full 2048x2048
// latent grid (65536). The pre-rewrite kernel needed an N-float softmax row in dynamic shared memory
// and could not launch beyond N = 12288.
//
// Cases: small exhaustive full-plane rows (tile tails tn < 256, strided C, C < blockDim, residual and
// no-residual), the production mid-attention C = 1152, a beyond-old-limit case (N = 16640), the 2K
// production smoke (N = 65536) and the 47-KiB shared-memory boundary (C = 5888, the largest C the
// guard accepts; an exact-48-KiB launch is refused by the driver even with the opt-in). Exhaustive oracles
// run the whole [C,N] plane; the two large-N cases compare 64 uniformly sampled rows exhaustively in
// FP64 (an exhaustive oracle is O(N^2 * C)) and check the whole output plane is finite. The oracle
// consumes the BF16-REPRESENTED operands the device receives; only kernel arithmetic differs.
//
// Guards: N = 65537 and C = 5889 must throw (new caps 65536 / 5888); N = 65536 and C = 5888 pass.
//
// Tolerance (a-priori, never narrowed to pass): 2e-2 on every rel-Frobenius and rel-row bound. The
// BF16 output storage floor against the FP64 oracle is ~4e-3; a real online-softmax defect (missing
// rescale, tile misalignment, barrier race) is O(0.1..1), orders above the bound.
//
// No oracle data required. Returns 2 (runnable) with no CUDA device.

#include "infernix/ops/vae_decoder.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using infernix::DType;
using infernix::Tensor;
namespace ops = infernix::ops;

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        const cudaError_t err_ = (call);                                 \
        if (err_ != cudaSuccess) {                                       \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__, \
                        __LINE__, cudaGetErrorString(err_));             \
            std::fflush(stdout);                                         \
            return 1;                                                    \
        }                                                                \
    } while (0)

namespace {

using T = std::vector<float>;

// ---- Host BF16 codec (same convention as the VAE prod test) ----
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

Tensor view(void* p, int a, int b) { return Tensor(p, DType::BF16, {a, b, 1, 1}); }

void upload(void* dst, const T& host) {
    std::vector<std::uint16_t> bits(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) bits[i] = to_bf16(host[i]);
    if (cudaMemcpy(dst, bits.data(), bits.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("UPLOAD_FAIL\n");
        std::fflush(stdout);
        std::exit(3);
    }
}

T download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    if (cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::printf("DOWNLOAD_FAIL\n");
        std::fflush(stdout);
        std::exit(3);
    }
    T out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}

// BF16-represented uniform inputs in [-1,1], channel plane [C,N] (c*N + n).
T random_plane(std::mt19937& rng, int C, int N) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    T v(static_cast<std::size_t>(C) * N);
    for (auto& x : v) x = from_bf16(to_bf16(dist(rng)));
    return v;
}

// Naive FP64 ideal row n (all channels) for the represented operands; matches the header contract
//   S[n,m] = scale * sum_c q[c,n]*k[c,m]; P = softmax_m(S); out[c,n] = sum_m P[n,m]*v[c,m] + res.
void oracle_row(int n, const T& q, const T& k, const T& v, const T* res, int C, int N, float scale,
                T& ideal) {
    std::vector<double> score(static_cast<std::size_t>(N));
    double mx = -std::numeric_limits<double>::infinity();
    for (int m = 0; m < N; ++m) {
        double s = 0.0;
        for (int c = 0; c < C; ++c)
            s += static_cast<double>(q[static_cast<std::size_t>(c) * N + n]) *
                 static_cast<double>(k[static_cast<std::size_t>(c) * N + m]);
        s *= static_cast<double>(scale);
        score[m] = s;
        if (s > mx) mx = s;
    }
    double sum = 0.0;
    for (int m = 0; m < N; ++m) {
        score[m] = std::exp(score[m] - mx);
        sum += score[m];
    }
    const double inv = 1.0 / sum;
    ideal.assign(static_cast<std::size_t>(C), 0.0F);
    for (int c = 0; c < C; ++c) {
        double a = 0.0;
        for (int m = 0; m < N; ++m)
            a += score[m] * inv * static_cast<double>(v[static_cast<std::size_t>(c) * N + m]);
        if (res != nullptr) a += static_cast<double>((*res)[static_cast<std::size_t>(c) * N + n]);
        ideal[c] = static_cast<float>(a);
    }
}

double rel_fro(const T& a, const T& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-300);
}

bool g_all_pass = true;
constexpr double kTol = 2e-2;

void check_bound(const char* label, double got) {
    const bool ok = got < kTol;
    g_all_pass = g_all_pass && ok;
    std::printf("%s err=%.3e tol=%.1e %s\n", label, got, kTol, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
}

// One device case: random represented operands, device op, oracle over `rows` (all rows when empty)
// as worst-row rel error; with rows empty also the full-plane rel_fro. Times the kernel.
bool run_case(int C, int N, bool with_residual, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    const std::size_t cn = static_cast<std::size_t>(C) * N;
    const T q = random_plane(rng, C, N), k = random_plane(rng, C, N), v = random_plane(rng, C, N);
    const T res = with_residual ? random_plane(rng, C, N) : T{};
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dr = nullptr, *dout = nullptr;
    if (cudaMalloc(&dq, cn * 2) != cudaSuccess || cudaMalloc(&dk, cn * 2) != cudaSuccess ||
        cudaMalloc(&dv, cn * 2) != cudaSuccess ||
        cudaMalloc(&dout, cn * 2) != cudaSuccess ||
        (with_residual && cudaMalloc(&dr, cn * 2) != cudaSuccess)) {
        std::printf("%s ALLOC_FAIL\n", label);
        return false;
    }
    upload(dq, q);
    upload(dk, k);
    upload(dv, v);
    if (with_residual) upload(dr, res);
    const float scale = 1.0F / std::sqrt(static_cast<float>(C));
    const Tensor tq = view(dq, C, N), tk = view(dk, C, N), tv = view(dv, C, N);
    Tensor tr = view(dr, C, N);
    Tensor to = view(dout, C, N);
    const cudaEvent_t ev0 = [] { cudaEvent_t e; cudaEventCreate(&e); return e; }();
    const cudaEvent_t ev1 = [] { cudaEvent_t e; cudaEventCreate(&e); return e; }();
    cudaEventRecord(ev0);
    ops::spatial_attn(tq, tk, tv, with_residual ? &tr : nullptr, scale, to, 0);
    cudaEventRecord(ev1);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("%s KERNEL_FAIL (see CUDA_CHECK-less sync above)\n", label);
        return false;
    }
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, ev0, ev1);
    const T dev = download(dout, cn);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dr);
    cudaFree(dout);

    int fails = 0;
    for (float x : dev) fails += std::isfinite(x) ? 0 : 1;

    T ideal;
    if (cn <= static_cast<std::size_t>(3) * 1200 * 1200) {  // exhaustive plane (small cases)
        T full(cn);
        for (int n = 0; n < N; ++n) {
            oracle_row(n, q, k, v, with_residual ? &res : nullptr, C, N, scale, ideal);
            for (int c = 0; c < C; ++c) full[static_cast<std::size_t>(c) * N + n] = ideal[c];
        }
        check_bound(label, rel_fro(dev, full));
    } else {  // 64 uniformly sampled rows, worst-row rel error (large-N cases)
        double worst = 0.0;
        for (int j = 0; j < 64; ++j) {
            const int n = static_cast<int>((static_cast<long long>(j) * N) / 64) +
                          (N > 64 ? N / 128 : 0);
            oracle_row(n, q, k, v, with_residual ? &res : nullptr, C, N, scale, ideal);
            T row_dev(static_cast<std::size_t>(C)), row_ideal(ideal);
            for (int c = 0; c < C; ++c) row_dev[c] = dev[static_cast<std::size_t>(c) * N + n];
            worst = std::max(worst, rel_fro(row_dev, row_ideal));
        }
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s rows64", label);
        check_bound(buf, worst);
    }
    if (fails != 0) {
        std::printf("%s NONFINITE=%d FAIL\n", label, fails);
        g_all_pass = false;
    }
    std::printf("%s C=%d N=%d res=%d kernel_ms=%.1f nonfinite=%d\n", label, C, N,
                with_residual ? 1 : 0, ms, fails);
    std::fflush(stdout);
    return fails == 0;
}

bool expect_reject(int C, int N, const char* label) {
    void* p = nullptr;
    if (cudaMalloc(&p, static_cast<std::size_t>(C) * N * 2) != cudaSuccess) return false;
    const Tensor q = view(p, C, N), k = view(p, C, N), v = view(p, C, N), o = view(p, C, N);
    bool threw = false;
    try {
        ops::spatial_attn(q, k, v, nullptr, 0.05F, const_cast<Tensor&>(o), 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (...) {
    }
    cudaFree(p);
    std::printf("%s rejected=%s %s\n", label, threw ? "true" : "false",
                threw ? "PASS" : "FAIL");
    std::fflush(stdout);
    return threw;
}

} // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }
    // Small exhaustive cases: tile tail (N<256 and N>256), C<blockDim, strided C, residual.
    g_all_pass = run_case(32, 64, true, 1, "attn_c32n64") && g_all_pass;
    g_all_pass = run_case(48, 300, false, 2, "attn_c48n300") && g_all_pass;
    g_all_pass = run_case(1152, 1200, true, 3, "attn_c1152n1200") && g_all_pass;
    g_all_pass = run_case(5888, 8, false, 6, "attn_c5888n8") && g_all_pass;
    // Beyond the old shared-memory limit (N > 12288) and the full 2K latent grid.
    g_all_pass = run_case(1152, 16640, true, 4, "attn_c1152n16640") && g_all_pass;
    g_all_pass = run_case(1152, 65536, true, 5, "attn_c1152n65536") && g_all_pass;
    // New caps: N <= 65536 and C <= 5888 must reject the first value above.
    g_all_pass = expect_reject(1, 65537, "guard_N65537") && g_all_pass;
    g_all_pass = expect_reject(5889, 4, "guard_C5889") && g_all_pass;
    std::printf(g_all_pass ? "SPATIAL_ATTN ALL PASS\n" : "SPATIAL_ATTN FAIL\n");
    std::fflush(stdout);
    return g_all_pass ? 0 : 1;
}
