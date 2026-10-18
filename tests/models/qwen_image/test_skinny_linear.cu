// CPU-oracle qualification of the BF16 linear's small-token (skinny GEMV) path (tokens <= 8), the
// route the Qwen-Image time-embedding linears use (temb_a: [4096,256] t=2; temb: [4096,4096] t=2).
// Each op is checked against a naive FP32-accumulate / BF16-store CPU reference at the exact
// (N,K) shapes the assembly uses, sweeping the token count across the skinny->general boundary so
// the dispatch switch is exercised directly. Independent of any model or oracle data.
//
// Returns 0 on success, 1 on any failed config, 2 with no CUDA device.

#include "infernix/ops/linear.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using infernix::Tensor;
using infernix::DType;
using infernix::Weight;
namespace ops = infernix::ops;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::printf("CUDA_ERROR %s at %s:%d: %s\n", #call, __FILE__,          \
                        __LINE__, cudaGetErrorString(err_));                      \
            std::fflush(stdout);                                                  \
            return 1;                                                             \
        }                                                                         \
    } while (0)

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
std::vector<std::uint16_t> bits_of(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}
void* upload(const std::vector<float>& v) {
    const auto bits = bits_of(v);
    void* d = nullptr;
    cudaMalloc(&d, bits.size() * sizeof(std::uint16_t));
    cudaMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}
Tensor view(void* p, int a, int b) { return Tensor(p, DType::BF16, {a, b, 1, 1}); }

// Deterministic varied fill in [-0.5, 0.5), depends on (row, col).
float fill_val(int r, int c) {
    unsigned h = static_cast<unsigned>(static_cast<unsigned int>(r) * 2654435761u ^
                                       static_cast<unsigned int>(c) * 40503u);
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return (float)((h & 0xFFFFFFu) / (float)0x1000000 - 0.5f);
}

struct BfWeight {
    int n, k;
    void* d;
    Weight w;
    BfWeight(int rows, int cols, const std::vector<float>& vals) : n(rows), k(cols) {
        d = upload(vals);
        w.payload = d;
        w.payload_bytes = (long long)rows * cols * sizeof(std::uint16_t);
        w.qtype = infernix::QType::BF16;
        w.shape[0] = n;
        w.shape[1] = k;
        w.padded_shape[0] = n;
        w.padded_shape[1] = k;
        w.ndim = 2;
        w.qdata = d;
        w.n = n;
        w.k = k;
        w.layout = infernix::QuantLayout::Contiguous;
    }
    ~BfWeight() {
        if (d) cudaFree(d);
    }
};

// Run ops::linear at shape (N,K,tokens) vs the CPU reference; print metrics + (when show) the
// per-element pattern. Returns true if rel_fro <= tol.
bool run_case(int N, int K, int tokens, double tol, bool show, const std::vector<std::uint16_t>& wbits,
              const std::vector<std::uint16_t>& xbits) {
    // CPU reference: out[t*N+n] = bf16( sum_k fp32(W[n,k]) * fp32(x[t,k]) )
    std::vector<float> ref((std::size_t)N * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += (double)from_bf16(wbits[(std::size_t)n * K + k]) *
                                             (double)from_bf16(xbits[(std::size_t)t * K + k]);
            ref[(std::size_t)t * N + n] = to_bf16((float)s);
        }

    void* dw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, wbits.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpy(dw, wbits.data(), wbits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    Weight W;
    W.payload = dw;
    W.payload_bytes = (long long)N * K * sizeof(std::uint16_t);
    W.qtype = infernix::QType::BF16;
    W.shape[0] = N;
    W.shape[1] = K;
    W.padded_shape[0] = N;
    W.padded_shape[1] = K;
    W.ndim = 2;
    W.qdata = dw;
    W.n = N;
    W.k = K;
    W.layout = infernix::QuantLayout::Contiguous;

    void* dx = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, xbits.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpy(dx, xbits.data(), xbits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    void* dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dout, (std::size_t)N * tokens * sizeof(std::uint16_t)));

    Tensor x_v = view(dx, K, tokens);
    Tensor out_v = view(dout, N, tokens);
    ops::linear(x_v, W, out_v, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got = download(dout, (std::size_t)N * tokens);

    double dd = 0.0, nn = 0.0, maxe = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = ref[i];
        dd += (a - b) * (a - b);
        nn += b * b;
        const double e = std::fabs(a - b);
        if (e > maxe) maxe = e;
    }
    const double relf = nn > 0.0 ? std::sqrt(dd / nn) : 0.0;

    std::printf("N=%-5d K=%-5d t=%-4d rel_fro=%.4e max_abs=%.4e  %s\n", N, K, tokens, relf, maxe,
                relf <= tol ? "PASS" : "FAIL");
    if (show) {
        for (int t = 0; t < tokens && t < 2; ++t) {
            std::printf("  tok%d out[0..7]:", t);
            for (int n = 0; n < 8; ++n) std::printf(" %9.4f", got[(std::size_t)t * N + n]);
            std::printf("\n  tok%d ref[0..7]:", t);
            for (int n = 0; n < 8; ++n) std::printf(" %9.4f", ref[(std::size_t)t * N + n]);
            std::printf("\n");
        }
    }

    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dout);
    return relf <= tol;
}

bool run_config(int N, int K, int tokens, double tol, bool show) {
    std::vector<float> wf((std::size_t)N * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) wf[(std::size_t)n * K + k] = fill_val(n, k);
    std::vector<float> xf((std::size_t)tokens * K);
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < K; ++k) xf[(std::size_t)t * K + k] = fill_val(t, k);
    const auto wbits = bits_of(wf);
    const auto xbits = bits_of(xf);
    return run_case(N, K, tokens, tol, show, wbits, xbits);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("NO CUDA DEVICE\n");
        return 2;
    }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("skinny linear oracle on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // A-priori tolerance: FP32-accumulate vs FP32-accumulate differ only by FMA association order;
    // the single real risk is a dispatch/shape bug (O(1) rel_fro), so a loose 1e-2 separates a
    // genuine bug from FMA rounding noise without hiding a real defect.
    const double tol = 1e-2;
    bool ok = true;

    // temb_a shape [4096,256] across the skinny->general boundary.
    for (int t : {1, 2, 4, 8, 9, 16, 33}) {
        ok = run_config(4096, 256, t, tol, t == 2) && ok;
    }
    // temb shape [4096,4096] t=2 (skinny) + general control.
    ok = run_config(4096, 4096, 2, tol, true) && ok;
    ok = run_config(4096, 4096, 33, tol, false) && ok;
    // hs_img shape [4096,64] t=256 (general, known-good control) + a skinny t=2 for the same K.
    ok = run_config(4096, 64, 256, tol, false) && ok;
    ok = run_config(4096, 64, 2, tol, true) && ok;
    // Small-N skinny instance path (like the Qwen4Exp GDN [96,2560]).
    ok = run_config(96, 2560, 1, tol, true) && ok;
    ok = run_config(96, 2560, 2, tol, true) && ok;

    std::printf(ok ? "SKINNY_LINEAR_OK\n" : "SKINNY_LINEAR_FAIL\n");
    return ok ? 0 : 1;
}
