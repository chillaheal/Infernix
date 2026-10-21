#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::media::encode {

// Encode 8-bit RGBA pixels as a 24+alpha PNG (color type 6, filter 0, non-interlaced). The IDAT
// stream uses stored DEFLATE blocks: no compression library is needed and the output is a valid
// PNG on every platform (the OpenAI images response contract only requires PNG bytes).
// Throws std::invalid_argument when rgba.size() != 4 * width * height or a dimension is zero.
[[nodiscard]] std::vector<std::uint8_t> encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                                                         std::span<const std::uint8_t> rgba);

} // namespace infernix::media::encode
