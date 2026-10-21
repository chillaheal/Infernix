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

std::vector<std::uint8_t> base64_decode(std::string_view text) {
    if (text.size() % 4 != 0) { throw std::invalid_argument("image base64 length is invalid"); }
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') { return c - 'A'; }
        if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
        if (c >= '0' && c <= '9') { return c - '0' + 52; }
        if (c == '+') { return 62; }
        if (c == '/') { return 63; }
        return -1;
    };
    std::size_t pad = 0;
    if (!text.empty() && text.back() == '=') {
        pad = (text.size() >= 2 && text[text.size() - 2] == '=') ? 2 : 1;
    }
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 4 * 3);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        std::uint32_t group = 0;
        const std::size_t payload = i + (4 - ((i + 4 == text.size()) ? pad : 0));
        for (std::size_t k = i; k < i + 4; ++k) {
            if (k >= payload) { break; } // only the final group's trailing pad is skipped
            if (text[k] == '=') { throw std::invalid_argument("image base64 has padding in the middle"); }
            const int v = value_of(text[k]);
            if (v < 0) { throw std::invalid_argument("image base64 has an invalid character"); }
            group = (group << 6) | static_cast<std::uint32_t>(v);
        }
        const std::size_t bytes = (i + 4 == text.size()) ? 3 - pad : 3;
        for (std::size_t b = 0; b < bytes; ++b) {
            out.push_back(static_cast<std::uint8_t>(group >> (16 - 8 * b)));
        }
    }
    return out;
}

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

ReferencePixels prepare_reference_image(const std::string& payload) {
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
    ReferencePixels out;
    out.height_px = static_cast<std::uint32_t>(target_h);
    out.width_px  = static_cast<std::uint32_t>(target_w);
    out.pixels.resize(4 * pixels);
    for (std::size_t c = 0; c < 4; ++c) {
        float* plane = out.pixels.data() + c * pixels;
        for (std::size_t i = 0; i < pixels; ++i) {
            plane[i] = static_cast<float>(resized[4 * i + c]) / 255.0F * 2.0F - 1.0F;
        }
    }
    return out;
}

} // namespace infernix::serve
