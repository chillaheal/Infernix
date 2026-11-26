// Seeded standard-normal init-latent noise (infernix_qwen_image::fill_normal). Host-only, no oracle
// file: the independent reference is the N(0,1) distribution itself, so the numeric gate is the
// sample moments (mean ~ 0, std ~ 1) over a large draw accumulated in double (not the production
// float path), plus the seed contract (same seed -> bit-identical, different seed -> differs) and
// the odd-count tail path. Tolerances are a-priori from the moments' own standard errors (sample-mean
// SE = 1/sqrt(N), sample-std SE ~ 1/sqrt(2N)); never narrowed to pass.

#include "models/qwen_image/noise.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace qi = infernix::models::qwen_image;

int main() {
    int failures = 0;
    auto check = [&](const char* name, bool ok, const char* detail) {
        std::printf("  %-16s %s  %s\n", name, ok ? "OK" : "FAIL", detail);
        if (!ok) ++failures;
    };
    char d[128];

    const std::size_t n = 1u << 20; // 1,048,576 samples
    std::vector<float> a(n), b(n);
    qi::fill_normal(42, std::span<float>(a.data(), a.size()));
    qi::fill_normal(42, std::span<float>(b.data(), b.size()));

    // Independent double-accumulation moments over the N(0,1) draw (production path is float).
    bool finite = true;
    double sum = 0.0, sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i])) { finite = false; break; }
        const double x = a[i];
        sum += x;
        sumsq += x * x;
    }
    const double mean = sum / static_cast<double>(n);
    const double sd   = std::sqrt(sumsq / static_cast<double>(n) - mean * mean);
    std::snprintf(d, sizeof d, "mean=%.5f std=%.5f (n=%zu)", mean, sd, n);
    check("finite", finite, d);
    check("mean~0", std::abs(mean) < 0.005, d);
    std::snprintf(d, sizeof d, "std=%.5f", sd);
    check("std~1", std::abs(sd - 1.0) < 0.01, d);

    // Seed contract: same seed -> bit-identical.
    bool identical = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) { identical = false; break; }
    }
    std::snprintf(d, sizeof d, "seed 42 twice", (std::size_t)0);
    check("determinism", identical, d);

    // Different seed -> a different latent.
    std::vector<float> c(n);
    qi::fill_normal(43, std::span<float>(c.data(), c.size()));
    check("seed-sensitive", a[0] != c[0] || a[1] != c[1], "seed 43 vs 42");

    // Odd-count tail path: 1001 samples, finite + sane moments (looser a-priori SE at N=1001).
    std::vector<float> oddv; oddv.resize(1001);
    qi::fill_normal(7, std::span<float>(oddv.data(), oddv.size()));
    double os = 0.0, o2 = 0.0;
    bool ofin = true;
    for (std::size_t i = 0; i < oddv.size(); ++i) {
        if (!std::isfinite(oddv[i])) { ofin = false; break; }
        const double x = oddv[i];
        os += x;
        o2 += x * x;
    }
    const double omean = os / 1001.0;
    const double ostd  = std::sqrt(o2 / 1001.0 - omean * omean);
    std::snprintf(d, sizeof d, "mean=%.4f std=%.4f (n=1001)", omean, ostd);
    check("odd-tail", ofin && std::abs(omean) < 0.15 && std::abs(ostd - 1.0) < 0.15, d);

    if (failures) {
        std::printf("IMAGE_NOISE_TEST: FAIL (%d check(s))\n", failures);
        return 1;
    }
    std::printf("IMAGE_NOISE_TEST: PASS (n=%zu: moments / determinism / seed / odd-tail)\n", n);
    return 0;
}
