#pragma once

// #75 reference/edit-image host preparation: the OpenAI `image` payload becomes the exact
// pixel plane the reference pipeline hands the VAE encoder. Owns base64, media decode and the
// geometry/resize/normalize math -- all pure host work that must finish BEFORE the interlude
// takes VRAM. The device encode itself is the model layer's encode_reference_image.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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

// The vision encoder's copy of the SAME resized picture: row-major RGB8 with the alpha composited
// over white (PIL paste semantics) -- the checkpoint was trained on this flattened copy for the
// Qwen3-VL tower only; the VAE plane above keeps the raw alpha and is never composited.
struct ReferenceVisionPlane {
    std::vector<std::uint8_t> rgb;
    std::uint32_t height_px = 0;
    std::uint32_t width_px  = 0;
};

// Both copies from one decode + one resize. `vae` is byte-identical to what
// prepare_reference_image returns. source_width_px/source_height_px are the DECODED size, before the
// geometry snap: the canvas of an edit follows the reference's aspect, and it must follow the picture
// the client sent rather than the plane the VAE was fed (see openai_images.h resolve_canvas_geometry).
struct ReferencePlanes {
    ReferencePixels      vae;
    ReferenceVisionPlane vision;
    std::uint32_t source_width_px  = 0;
    std::uint32_t source_height_px = 0;
};

// Strict base64 (the OpenAI `image` payload's data URI and every reference plane input). Exposed
// because the M1 tokenization gate grades it against an oracle vector whose length is not a multiple
// of 3 -- the case that a fixed 16/8/0 byte shift silently decodes as a zero last byte. Throws
// std::invalid_argument on a bad length, character, or misplaced/short padding.
[[nodiscard]] std::vector<std::uint8_t> base64_decode(std::string_view text);

// Flatten an RGBA8 run over white with PIL's paste(mask=alpha) rounding. Exposed for the
// tokenization gate's oracle vector; production uses prepare_reference_planes.
[[nodiscard]] std::vector<std::uint8_t> composite_rgba_over_white(const std::uint8_t* rgba,
                                                                   std::size_t pixels);

// Vision-tower grid over one snapped plane: patch 16 x spatial_merge 2 = 32 px per vision token.
// Each vision token expands to 4 latent slots in the joint sequence, so for a plane on the
// 32-grid 4 * vision_token_count == the VAE token count (H/16 * W/16). build_token_metadata
// raises if that equality ever fails.
[[nodiscard]] constexpr std::uint64_t vision_token_count(std::uint32_t height_px,
                                                         std::uint32_t width_px) noexcept {
    return static_cast<std::uint64_t>(height_px / 32) * (width_px / 32);
}

// The pipeline's condition-image resize, byte-exact to
// PIL.Image.resize((dst_w, dst_h), resample=LANCZOS, reducing_gap=None) on RGBA input: the
// premultiplied-"RGBa" round trip Image.resize performs around an RGBA resize, then the separable
// windowed-sinc resample (horizontal pass first) with Pillow's 22-bit fixed-point coefficients,
// per-window normalization and clip8. `rgba` is row-major RGBA8 of src_w x src_h; `dst` comes back
// row-major RGBA8 of dst_w x dst_h. Exposed because the P1 gate grades it against Pillow's own
// output; production uses prepare_reference_planes. Throws std::invalid_argument on a null plane or a
// non-positive dimension.
void resize_rgba_pil_lanczos(const std::uint8_t* rgba, int src_w, int src_h, int dst_w, int dst_h,
                             std::vector<std::uint8_t>& dst);

// Decode the base64 payload, decode the image (PNG keeps its alpha as the fourth channel; other
// formats get an opaque alpha), snap to the reference geometry -- the pipeline's
// calculate_dimensions(1024^2, aspect ratio), rounded to multiples of 32 -- and resize it with
// resize_rgba_pil_lanczos above, which is what VaeImageProcessor does (its processor is constructed
// with no resample override, so the "lanczos" default governs BOTH the VAE copy and the vision copy).
// Throws std::invalid_argument for invalid base64 or an undecodable/degenerate image and
// media::decode::Error for a decoded-size budget violation.
[[nodiscard]] ReferencePlanes prepare_reference_planes(const std::string& payload);

// The VAE plane alone (the historical entry point; == prepare_reference_planes(payload).vae).
[[nodiscard]] ReferencePixels prepare_reference_image(const std::string& payload);

} // namespace infernix::serve
