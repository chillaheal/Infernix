#include "models/qwen_image/noise.h"

#include <cmath>

namespace infernix::models::qwen_image {
namespace {

// The canonical SplitMix64 step, using the same 64-bit constants as qwen4_exp/frontend/ngram_hash.cpp
// (the codebase's shared 64-bit mix), so the image RNG and the n-gram hash share one well-tested mixer.
constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ULL;

struct SplitMix64 {
    std::uint64_t state;
    explicit SplitMix64(std::uint64_t s) noexcept : state(s) {}
    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t z = state += kGamma;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // 53-bit double in [0,1).
    [[nodiscard]] double uniform01() noexcept { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

} // namespace

void fill_normal(std::uint64_t seed, std::span<float> out) noexcept {
    SplitMix64 rng(seed);
    const double two_pi = 2.0 * std::acos(-1.0);
    std::size_t i = 0;
    while (i + 1 < out.size()) {
        const double u1 = 1.0 - rng.uniform01(); // (0,1]: keep u1 > 0 so -2*ln(u1) is finite
        const double u2 = rng.uniform01();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = two_pi * u2;
        out[i]     = static_cast<float>(r * std::cos(th));
        out[i + 1] = static_cast<float>(r * std::sin(th));
        i += 2;
    }
    if (i < out.size()) { // odd tail: one sample, discard the polar partner
        const double u1 = 1.0 - rng.uniform01();
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = static_cast<float>(r * std::cos(two_pi * rng.uniform01()));
    }
}

} // namespace infernix::models::qwen_image
