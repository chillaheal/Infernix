#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "media/decode/decode.h"

namespace infernix::media::decode {

// True when the first 8 bytes are the PNG file signature.
bool is_png(std::span<const std::uint8_t> bytes);

// Display dimensions of a PNG from its IHDR chunk. Enforces the decoded-pixel
// budget and throws std::invalid_argument for a non-PNG or truncated header.
ImageInfo png_image_info(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels);

// Decode a non-interlaced PNG into 24-bit RGB. Supports color types 0,2,3,4,6
// (gray, RGB, palette, gray+alpha, RGBA) at bit depths 1,2,4,8,16. Alpha is
// composited over white, matching the FFmpeg image path. Throws
// std::invalid_argument for malformed input and Error(ErrorKind::BudgetExceeded)
// when the pixel budget is exceeded.
Image decode_png(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels);

// RGBA8 raster; alpha is a real fourth channel (never composited).
struct RgbaImage {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba; // 4 bytes per pixel, row-major
};

// Decode a non-interlaced PNG into raw RGBA8. Same container support as decode_png; unlike it,
// alpha is preserved as the fourth channel (types 0/2 and non-tRNS palette entries get alpha 255)
// because the image-VAE reference path consumes alpha directly. Same error contract as decode_png.
RgbaImage decode_png_rgba(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels);

} // namespace infernix::media::decode