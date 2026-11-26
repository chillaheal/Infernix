#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::media::encode {

// Encode 8-bit RGBA pixels as a 24+alpha PNG (color type 6, non-interlaced). Each scanline picks
// its own filter from the five PNG filters by the cheapest filtered bytes, and the IDAT stream is
// DEFLATE-compressed when zlib is on the build (INFERNIX_MEDIA_ZLIB_ENCODER, set by
// src/media/CMakeLists.txt on Windows); without it the stream falls back to stored DEFLATE
// blocks, which is still a valid PNG on every platform (the OpenAI images response contract only
// requires PNG bytes). Pixels are lossless either way: the filter and DEFLATE choices only change
// how the same bytes are written.
// Throws std::invalid_argument when rgba.size() != 4 * width * height or a dimension is zero.
[[nodiscard]] std::vector<std::uint8_t> encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                                                         std::span<const std::uint8_t> rgba);

} // namespace infernix::media::encode
