#include "serve/reference_image.h"

#include "media/decode/decode.h"
#include "media/decode/png_decode.h"
#include "serve/openai_images.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace infernix::serve {
namespace {

namespace md = media::decode;

// Reference-geometry area: pipeline_qwenimage21 always resizes a condition image to an
// output_resolution^2 area (1024 default) regardless of the requested output size.
constexpr double kReferenceTargetArea = 1024.0 * 1024.0;

[[nodiscard]] int snap32(double v) {
    // Python round(): banker's rounding of v/32, back up by 32. Exact .5 fractions are the only
    // divergence from round-half-away; matching them keeps boundary geometries oracle-identical.
    const double q = v / 32.0;
    const double floor_q = std::floor(q);
    const double rest = q - floor_q;
    double snapped = 0.0;
    if (rest > 0.5) { snapped = floor_q + 1.0; }
    else if (rest < 0.5) { snapped = floor_q; }
    else { snapped = (std::fmod(floor_q, 2.0) == 0.0) ? floor_q : floor_q + 1.0; }
    if (snapped < 1.0) { return 0; }
    return static_cast<int>(snapped) * 32;
}

} // namespace (the geometry helpers above stay internal)

// Strict base64. The significant characters of a group carry six bits each, so the emitted bytes are
// read at 6*chars - 8*(b+1) from the top of the group. Reading fixed 16/8/0 shifts assumes four
// significant characters, which zeroes the LAST BYTE of every payload whose length is not a multiple
// of 3 -- two of every three images -- and hands the codec a PNG with a corrupted tail.
[[nodiscard]] std::vector<std::uint8_t> base64_decode(std::string_view text) {
    if (text.size() % 4 != 0) { throw std::invalid_argument("image base64 length is invalid"); }
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') { return c - 'A'; }
        if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
        if (c >= '0' && c <= '9') { return c - '0' + 52; }
        if (c == '+') { return 62; }
        if (c == '/') { return 63; }
        return -1;
    };
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 4 * 3);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        std::uint32_t group = 0;
        std::size_t   chars = 0;
        for (std::size_t k = i; k < i + 4 && text[k] != '='; ++k) {
            const int v = value_of(text[k]);
            if (v < 0) { throw std::invalid_argument("image base64 has an invalid character"); }
            group = (group << 6) | static_cast<std::uint32_t>(v);
            ++chars;
        }
        if (chars != 4) {  // pad only ever closes the final group, and nothing may follow it
            if (i + 4 != text.size()) { throw std::invalid_argument("image base64 has padding in the middle"); }
            for (std::size_t k = i + chars; k < i + 4; ++k) {
                if (text[k] != '=') { throw std::invalid_argument("image base64 has a character after padding"); }
            }
            if (chars < 2) { throw std::invalid_argument("image base64 has an incomplete group"); }
        }
        for (std::size_t b = 0; b + 1 < chars; ++b) {
            out.push_back(static_cast<std::uint8_t>((group >> (6 * chars - 8 * (b + 1))) & 0xFFu));
        }
    }
    return out;
}

namespace {

// PIL Image.resize(resample=BILINEAR) mapping: source centers at (x+0.5)/scale - 0.5, clamped,
// 2x2 bilinear weights, no antialiasing prefilter, result rounded to uint8.
void resize_rgba_pil_bilinear(const md::RgbaImage& src, int dst_w, int dst_h,
                              std::vector<std::uint8_t>& dst) {
    dst.resize(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h) * 4);
    const double sx = static_cast<double>(src.width) / dst_w;
    const double sy = static_cast<double>(src.height) / dst_h;
    for (int dy = 0; dy < dst_h; ++dy) {
        double fy = (dy + 0.5) * sy - 0.5;
        fy        = std::min(std::max(fy, 0.0), static_cast<double>(src.height - 1));
        const int y0 = static_cast<int>(std::floor(fy));
        const int y1 = std::min(y0 + 1, src.height - 1);
        const double wy = fy - y0;
        for (int dx = 0; dx < dst_w; ++dx) {
            double fx = (dx + 0.5) * sx - 0.5;
            fx        = std::min(std::max(fx, 0.0), static_cast<double>(src.width - 1));
            const int x0 = static_cast<int>(std::floor(fx));
            const int x1 = std::min(x0 + 1, src.width - 1);
            const double wx = fx - x0;
            const std::uint8_t* row0 = src.rgba.data() + static_cast<std::size_t>(y0) * src.width * 4;
            const std::uint8_t* row1 = src.rgba.data() + static_cast<std::size_t>(y1) * src.width * 4;
            std::uint8_t* out =
                dst.data() + (static_cast<std::size_t>(dy) * dst_w + dx) * 4;
            for (int c = 0; c < 4; ++c) {
                const double top = row0[x0 * 4 + c] * (1.0 - wx) + row0[x1 * 4 + c] * wx;
                const double bot = row1[x0 * 4 + c] * (1.0 - wx) + row1[x1 * 4 + c] * wx;
                const double v   = top * (1.0 - wy) + bot * wy;
                out[c] = static_cast<std::uint8_t>(
                    std::min(std::max(std::lround(v), 0L), 255L));
            }
        }
    }
}

} // namespace

std::vector<std::uint8_t> composite_rgba_over_white(const std::uint8_t* rgba, std::size_t pixels) {
    std::vector<std::uint8_t> out(pixels * 3);
    for (std::size_t i = 0; i < pixels; ++i) {
        const std::uint32_t a = rgba[4 * i + 3];
        for (int c = 0; c < 3; ++c) {
            const std::uint32_t src = rgba[4 * i + c];
            // PIL ImagingPaste blend: (src*a + 255*(255-a) + 127) / 255.
            out[3 * i + c] = static_cast<std::uint8_t>((src * a + 255u * (255u - a) + 127u) / 255u);
        }
    }
    return out;
}

ReferencePlanes prepare_reference_planes(const std::string& payload) {
    std::vector<std::uint8_t> bytes = base64_decode(payload);
    if (bytes.size() > kMaximumImageInputBytes) {
        throw std::invalid_argument("the decoded reference image exceeds the size limit");
    }
    if (bytes.empty()) { throw std::invalid_argument("the reference image is empty"); }

    md::RgbaImage rgba;
    if (md::is_png(bytes)) {
        rgba = md::decode_png_rgba(bytes, md::Policy{}.max_decoded_pixels);
    } else {
        const md::Image rgb = md::decode_image(bytes, md::Policy{});
        rgba.width  = rgb.width;
        rgba.height = rgb.height;
        rgba.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
        for (std::size_t i = 0; i < static_cast<std::size_t>(rgb.width) * rgb.height; ++i) {
            rgba.rgba[4 * i + 0] = rgb.rgb[3 * i + 0];
            rgba.rgba[4 * i + 1] = rgb.rgb[3 * i + 1];
            rgba.rgba[4 * i + 2] = rgb.rgb[3 * i + 2];
            rgba.rgba[4 * i + 3] = 255;
        }
    }
    if (rgba.width <= 0 || rgba.height <= 0) {
        throw std::invalid_argument("the reference image has no pixels");
    }

    const int target_w = snap32(std::sqrt(kReferenceTargetArea *
                                          static_cast<double>(rgba.width) / rgba.height));
    const int target_h = snap32(std::sqrt(kReferenceTargetArea *
                                          static_cast<double>(rgba.height) / rgba.width));
    if (target_w < 32 || target_h < 32) {
        throw std::invalid_argument("the reference image aspect ratio is too extreme");
    }

    std::vector<std::uint8_t> resized;
    resize_rgba_pil_bilinear(rgba, target_w, target_h, resized);

    // VaeImageProcessor.preprocess: float32 x / 255 * 2 - 1, channel-major.
    const std::size_t pixels = static_cast<std::size_t>(target_w) * target_h;
    ReferencePlanes out;
    out.vae.height_px = static_cast<std::uint32_t>(target_h);
    out.vae.width_px  = static_cast<std::uint32_t>(target_w);
    out.vae.pixels.resize(4 * pixels);
    for (std::size_t c = 0; c < 4; ++c) {
        float* plane = out.vae.pixels.data() + c * pixels;
        for (std::size_t i = 0; i < pixels; ++i) {
            plane[i] = static_cast<float>(resized[4 * i + c]) / 255.0F * 2.0F - 1.0F;
        }
    }
    // The vision tower's flattened copy (pipeline: white.paste(img, mask=img.getchannel("A"))).
    out.vision.height_px = out.vae.height_px;
    out.vision.width_px  = out.vae.width_px;
    out.vision.rgb       = composite_rgba_over_white(resized.data(), pixels);
    return out;
}

ReferencePixels prepare_reference_image(const std::string& payload) {
    return prepare_reference_planes(payload).vae;
}

} // namespace infernix::serve
