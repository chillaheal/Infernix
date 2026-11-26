// GPU qualification of the two new VAE-encoder ops (ops::conv2d_stride2, ops::avg_down) against
// naive FP64 oracles from the header contracts. These are the Resample stride-2 downsample (torch
// ZeroPad2d((0,1,0,1)) + Conv2d 3x3 s2: right/bottom-only pad, so border windows differ from the
// qualified symmetric conv2d) and the AvgDown3D image shortcut (2x2 spatial + channel-group mean,
// divisor = group_size alone; the FRONT temporal zero pad keeps only the last temporal slot, so at
// group=4 every even output channel is exactly zero and the odd ones hold the plain 2x2 mean).
//
// Cases: stride-2 exhaustive small planes (odd/even borders, residual-free), a production-width
// 384->384 case checked on uniformly sampled output pixels (exhaustive oracle is Cin*9 per pixel),
// avg_down exhaustive group=1, group=4 factor_t=1 (pure 2x2 mean) and factor_t=2 (alternating
// zero/mean channels -- the case a self-mirroring oracle cannot distinguish), and a production
// 384->768 factor_t=2 plane.
//
// Guards: stride-2 H<2 and wrong out extent must reject; avg_down factor_t=3, odd H and an
// indivisible Cin/Cout pair must reject.
//
// Tolerance (a-priori, never narrowed to pass): 2e-2 on rel-Frobenius. The BF16 storage floor is
// ~4e-3; a border/padding defect shifts entire rows (O(1)).
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
#include <random>
#include <vector>

using infernix::DType;
using infernix::Tensor;
namespace ops = infernix::ops;

namespace {

using T = std::vector<float>;

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

Tensor view(void* p, int a, int b, int c) { return Tensor(p, DType::BF16, {a, b, c, 1}); }
Tensor view4(void* p, int a, int b, int c, int d) { return Tensor(p, DType::BF16, {a, b, c, d}); }

bool upload(void* dst, const T& host) {
    std::vector<std::uint16_t> bits(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) bits[i] = to_bf16(host[i]);
    return cudaMemcpy(dst, bits.data(), bits.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

T download(void* d, std::size_t n) {
    std::vector<std::uint16_t> bits(n);
    if (cudaMemcpy(bits.data(), d, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("DOWNLOAD_FAIL\n");
        std::exit(3);
    }
    T out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = from_bf16(bits[i]);
    return out;
}

T random_plane(std::mt19937& rng, std::size_t n) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    T v(n);
    for (auto& x : v) x = from_bf16(to_bf16(dist(rng)));
    return v;
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

// FP64 ideal for one conv2d_stride2 output element: right/bottom-only zero pad, stride 2.
double oracle_conv2d_s2(int oc, int oh, int ow, const T& x, const T& w, const T& b, int Cin, int H,
                        int W) {
    double acc = static_cast<double>(b[oc]);
    for (int ci = 0; ci < Cin; ++ci) {
        for (int u = 0; u < 3; ++u) {
            const int hy = 2 * oh + u;
            if (hy >= H) continue;
            for (int v = 0; v < 3; ++v) {
                const int hx = 2 * ow + v;
                if (hx >= W) continue;
                acc += static_cast<double>(x[static_cast<std::size_t>(ci) * H * W + hy * W + hx]) *
                       static_cast<double>(w[((static_cast<std::size_t>(oc) * Cin + ci) * 3 + u) * 3 + v]);
            }
        }
    }
    return acc;
}

bool run_conv2d_s2(int Cin, int Cout, int H, int W, unsigned seed, const char* label, int sampled) {
    const int H2 = (H - 2) / 2 + 1, W2 = (W - 2) / 2 + 1;
    std::mt19937 rng(seed);
    const T x = random_plane(rng, static_cast<std::size_t>(Cin) * H * W);
    const T w = random_plane(rng, static_cast<std::size_t>(Cout) * Cin * 9);
    const T b = random_plane(rng, Cout);
    void *dx = nullptr, *dw = nullptr, *db = nullptr, *do_ = nullptr;
    if (cudaMalloc(&dx, x.size() * 2) != cudaSuccess || cudaMalloc(&dw, w.size() * 2) != cudaSuccess ||
        cudaMalloc(&db, b.size() * 2) != cudaSuccess ||
        cudaMalloc(&do_, static_cast<std::size_t>(Cout) * H2 * W2 * 2) != cudaSuccess) {
        std::printf("%s ALLOC_FAIL\n", label);
        return false;
    }
    if (!upload(dx, x) || !upload(dw, w) || !upload(db, b)) {
        std::printf("%s UPLOAD_FAIL\n", label);
        return false;
    }
    Tensor out_v = view(do_, Cout, H2, W2);
    ops::conv2d_stride2(view(dx, Cin, H, W), view4(dw, Cout, Cin, 3, 3), view(db, Cout, 1, 1), out_v, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("%s KERNEL_FAIL\n", label);
        return false;
    }
    const std::size_t on = static_cast<std::size_t>(Cout) * H2 * W2;
    const T dev = download(do_, on);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(do_);

    if (sampled == 0) { // exhaustive plane
        T ideal(on);
        for (int oc = 0; oc < Cout; ++oc)
            for (int oh = 0; oh < H2; ++oh)
                for (int ow = 0; ow < W2; ++ow)
                    ideal[(static_cast<std::size_t>(oc) * H2 + oh) * W2 + ow] =
                        static_cast<float>(oracle_conv2d_s2(oc, oh, ow, x, w, b, Cin, H, W));
        check_bound(label, rel_fro(dev, ideal));
    } else { // uniformly sampled output pixels: max abs deviation / sampled-ideal RMS
        double den = 0.0, worst = 0.0;
        std::uniform_int_distribution<int> pick(0, static_cast<int>(on) - 1);
        for (int j = 0; j < sampled; ++j) {
            const std::size_t i = static_cast<std::size_t>(pick(rng)) % on;
            const int ow = static_cast<int>(i % W2);
            const int rem = static_cast<int>(i / W2);
            const int oh = rem % H2, oc = rem / H2;
            const double id = oracle_conv2d_s2(oc, oh, ow, x, w, b, Cin, H, W);
            den += id * id;
            worst = std::max(worst, std::fabs(static_cast<double>(dev[i]) - id));
        }
        den = std::sqrt(den / sampled);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s sampled", label);
        check_bound(buf, worst / (den + 1e-300));
    }
    for (float v : dev)
        if (!std::isfinite(v)) {
            std::printf("%s NONFINITE FAIL\n", label);
            return false;
        }
    std::printf("%s C=%d/%d HW=%d done\n", label, Cin, Cout, H);
    std::fflush(stdout);
    return true;
}

// FP64 reference rebuilt from the fork's QwenImage21AvgDown3D.forward: F.pad the single frame to
// factor_t frames AT THE FRONT, then the fork's view/permute makes each output group the slots
// (channel slowest, then ft, a, b) of that PADDED tensor, meaned over group_size. The zeros come from
// the padded tensor itself, not from a skip rule, so this cannot mirror the kernel's index algebra --
// which is exactly how the inverted temporal slot stayed invisible here once already.
T oracle_avg_down(const T& x, int Cin, int Cout, int H, int W, int factor_t) {
    const int pad_t = factor_t - 1;  // F.pad(x, (0,0,0,0,pad_t,0)): front frames are zeros
    const int Tp = pad_t + 1;
    const int factor = factor_t * 4;
    const int group = Cin * factor / Cout;
    const std::size_t plane = static_cast<std::size_t>(Tp) * H * W;
    T padded(static_cast<std::size_t>(Cin) * plane, 0.0);
    for (int c = 0; c < Cin; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                padded[c * plane + static_cast<std::size_t>(pad_t) * H * W + h * W + w] =
                    x[static_cast<std::size_t>(c) * H * W + h * W + w];
    const int H2 = H / 2, W2 = W / 2;
    T ideal(static_cast<std::size_t>(Cout) * H2 * W2, 0.0);
    for (int oc = 0; oc < Cout; ++oc)
        for (int oh = 0; oh < H2; ++oh)
            for (int ow = 0; ow < W2; ++ow) {
                double acc = 0.0;
                for (int g = 0; g < group; ++g) {
                    const int k = oc * group + g;
                    const int c = k / factor, r = k % factor;
                    const int ft = r / 4, a = (r / 2) % 2, b = r % 2;
                    acc += padded[c * plane + static_cast<std::size_t>(ft) * H * W +
                                  (2 * oh + a) * W + 2 * ow + b];
                }
                ideal[(static_cast<std::size_t>(oc) * H2 + oh) * W2 + ow] =
                    static_cast<float>(acc / group);
            }
    return ideal;
}

bool run_avg_down(int Cin, int Cout, int factor_t, int H, int W, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    const T x = random_plane(rng, static_cast<std::size_t>(Cin) * H * W);
    const int H2 = H / 2, W2 = W / 2;
    void *dx = nullptr, *do_ = nullptr;
    if (cudaMalloc(&dx, x.size() * 2) != cudaSuccess ||
        cudaMalloc(&do_, static_cast<std::size_t>(Cout) * H2 * W2 * 2) != cudaSuccess) {
        std::printf("%s ALLOC_FAIL\n", label);
        return false;
    }
    if (!upload(dx, x)) {
        std::printf("%s UPLOAD_FAIL\n", label);
        return false;
    }
    Tensor out_v = view(do_, Cout, H2, W2);
    ops::avg_down(view(dx, Cin, H, W), factor_t, out_v, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("%s KERNEL_FAIL\n", label);
        return false;
    }
    const std::size_t on = static_cast<std::size_t>(Cout) * H2 * W2;
    const T dev = download(do_, on);
    cudaFree(dx);
    cudaFree(do_);
    const T ideal = oracle_avg_down(x, Cin, Cout, H, W, factor_t);
    check_bound(label, rel_fro(dev, ideal));
    for (float v : dev)
        if (!std::isfinite(v)) {
            std::printf("%s NONFINITE FAIL\n", label);
            return false;
        }
    std::printf("%s C=%d/%d ft=%d HW=%d done\n", label, Cin, Cout, factor_t, H);
    std::fflush(stdout);
    return true;
}

template <typename F>
bool expect_reject(const char* label, F&& f) {
    bool threw = false;
    try {
        f();
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (...) {
    }
    std::printf("%s rejected=%s %s\n", label, threw ? "true" : "false", threw ? "PASS" : "FAIL");
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
    g_all_pass = run_conv2d_s2(4, 8, 16, 16, 1, "s2_c4n8", 0) && g_all_pass;
    g_all_pass = run_conv2d_s2(96, 96, 64, 64, 2, "s2_c96", 0) && g_all_pass;
    g_all_pass = run_conv2d_s2(384, 384, 128, 128, 3, "s2_c384prod", 4096) && g_all_pass;

    g_all_pass = run_avg_down(768, 768, 1, 8, 8, 4, "avg_identity") && g_all_pass;
    g_all_pass = run_avg_down(96, 96, 1, 32, 32, 5, "avg_g4_ft1") && g_all_pass;
    g_all_pass = run_avg_down(96, 192, 2, 32, 32, 6, "avg_g4_ft2") && g_all_pass;
    g_all_pass = run_avg_down(384, 768, 2, 64, 64, 7, "avg_prod384_768_ft2") && g_all_pass;

    // Guards.
    std::mt19937 rng(99);
    void* p = nullptr;
    if (cudaMalloc(&p, static_cast<std::size_t>(64) * 64 * 2) != cudaSuccess) return 1;
    Tensor o_s2a = view(p, 8, 31, 31), o_s2b = view(p, 8, 64, 64);
    Tensor o_ft3 = view(p, 96, 32, 32), o_odd = view(p, 96, 31, 32), o_ind = view(p, 7, 32, 32);
    g_all_pass = expect_reject("guard_s2_H1", [&] {
                   ops::conv2d_stride2(view(p, 4, 1, 64), view4(p, 8, 4, 3, 3), view(p, 8, 1, 1), o_s2a, 0);
               }) && g_all_pass;
    g_all_pass = expect_reject("guard_s2_badout", [&] {
                   ops::conv2d_stride2(view(p, 4, 64, 64), view4(p, 8, 4, 3, 3), view(p, 8, 1, 1), o_s2b, 0);
               }) && g_all_pass;
    g_all_pass = expect_reject("guard_avg_ft3", [&] {
                   ops::avg_down(view(p, 96, 64, 64), 3, o_ft3, 0);
               }) && g_all_pass;
    g_all_pass = expect_reject("guard_avg_oddH", [&] {
                   ops::avg_down(view(p, 96, 63, 64), 1, o_odd, 0);
               }) && g_all_pass;
    g_all_pass = expect_reject("guard_avg_indivisible", [&] {
                   ops::avg_down(view(p, 10, 64, 64), 1, o_ind, 0);
               }) && g_all_pass;
    cudaFree(p);
    (void)rng;

    std::printf(g_all_pass ? "VAE_ENCODE_OPS ALL PASS\n" : "VAE_ENCODE_OPS FAIL\n");
    std::fflush(stdout);
    return g_all_pass ? 0 : 1;
}
