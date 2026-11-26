// The image lane's RGBA PNG reader. Ours-only file (docs/maintainer/image-integration-points.md,
// "Ours-only lane files"), re-added on this port on top of upstream's own PNG decoder instead of the
// fork's copy of it: decode_png with keep_alpha returns the stored colour samples plus the source's
// own alpha plane, which is exactly what the lane's contract asks for.
//
// The lane consumes alpha directly (the image-VAE reference path reads a fourth channel), so unlike
// decode_png nothing is composited here: a source with no transparency gets an opaque alpha, and a
// source with one -- an alpha channel of color types 4/6, or PNG tRNS for a palette or a keyed gray
// or RGB sample -- keeps the alpha it declares. Built on the native PNG path, so it is compiled where
// that path is (WIN32), exactly like decode/png_decode.cpp.

#include "media/decode/png_decode.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::media::decode {

RgbaImage decode_png_rgba(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels) {
    const Image rgb = decode_png(bytes, max_decoded_pixels, /*keep_alpha=*/true);

    RgbaImage out;
    out.width  = rgb.width;
    out.height = rgb.height;
    const std::size_t pixels = static_cast<std::size_t>(rgb.width) * static_cast<std::size_t>(rgb.height);
    out.rgba.resize(pixels * 4);
    // decode_png checked the pixel budget and rejects a truncated header, so rgb.rgb holds 3 bytes per
    // pixel and rgb.alpha, when it is there, exactly one.
    for (std::size_t i = 0; i < pixels; ++i) {
        out.rgba[4 * i + 0] = rgb.rgb[3 * i + 0];
        out.rgba[4 * i + 1] = rgb.rgb[3 * i + 1];
        out.rgba[4 * i + 2] = rgb.rgb[3 * i + 2];
        out.rgba[4 * i + 3] = rgb.alpha.empty() ? static_cast<std::uint8_t>(255) : rgb.alpha[i];
    }
    return out;
}

} // namespace infernix::media::decode
