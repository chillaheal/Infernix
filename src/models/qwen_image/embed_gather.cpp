#include "models/qwen_image/embed_gather.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen_image {

namespace {
// BF16 -> FP32 is exact: BF16 is the top 16 bits of the FP32 representation, so the value is the raw
// bits shifted into the high half. memcpy (not a reinterpret_cast through a pointer) for strict aliasing.
inline float bf16_to_f32(std::uint16_t u) {
    const std::uint32_t bits = static_cast<std::uint32_t>(u) << 16;
    float f = 0.0F;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
} // namespace

std::vector<float> build_inputs_embeds(const std::uint16_t* embed_bf16, long long vocab, int hidden,
                                        const long long* ids, int n) {
    if (embed_bf16 == nullptr || ids == nullptr)
        throw std::invalid_argument("build_inputs_embeds: null table or ids pointer");
    if (vocab <= 0 || hidden <= 0 || n < 0)
        throw std::invalid_argument("build_inputs_embeds: degenerate table/shape");
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab)
            throw std::invalid_argument("build_inputs_embeds: id " + std::to_string(ids[i]) +
                                        " out of range [0," + std::to_string(vocab) + ")");
    }
    std::vector<float> out(static_cast<std::size_t>(n) * hidden);
    for (int i = 0; i < n; ++i) {
        const std::uint16_t* row = embed_bf16 + static_cast<std::size_t>(ids[i]) * hidden;
        float* dst = out.data() + static_cast<std::size_t>(i) * hidden;
        for (int d = 0; d < hidden; ++d) dst[d] = bf16_to_f32(row[d]);
    }
    return out;
}

} // namespace infernix::models::qwen_image
