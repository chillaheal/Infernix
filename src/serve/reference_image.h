#pragma once

// #75 reference/edit-image host preparation: the OpenAI `image` payload becomes the exact
// pixel plane the reference pipeline hands the VAE encoder. Owns base64, media decode and the
// geometry/resize/normalize math -- all pure host work that must finish BEFORE the interlude
// takes VRAM. The device encode itself is the model layer's encode_reference_image.

#include <cstdint>
#include <string>
#include <vector>

namespace infernix::serve {

// The VAE encoder's input plane: channel-major [4, height_px, width_px] float in [-1,1] with
// the source alpha carried RAW as the fourth channel (never composited); height/width are the
// reference geometry, multiples of 32.
struct ReferencePixels {
    std::vector<float> pixels;
    std::uint32_t height_px = 0;
    std::uint32_t width_px  = 0;
};

// Decode the base64 payload, decode the image (PNG keeps its alpha as the fourth channel; other
// formats get an opaque alpha), snap to the reference geometry -- the pipeline's
// calculate_dimensions(1024^2, aspect ratio), rounded to multiples of 32 -- and resize with the
// reference pipeline's PIL bilinear mapping (no antialias, half-pixel centers).
// Throws std::invalid_argument for invalid base64 or an undecodable/degenerate image and
// media::decode::Error for a decoded-size budget violation.
[[nodiscard]] ReferencePixels prepare_reference_image(const std::string& payload);

} // namespace infernix::serve
