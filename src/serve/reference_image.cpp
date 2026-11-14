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

// The 32-snap itself is serve::snap_to_32 (openai_images.h): the reference plane and the output
// canvas are the same rounding of the same calculate_dimensions, so they share one implementation.

} // namespace (the geometry constants above stay internal)

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

// ---------------------------------------------------------------------------
// The condition-image resize the reference pipeline performs, byte-exact to
//   PIL.Image.resize((w, h), resample=LANCZOS, reducing_gap=None)
// on an RGBA image. VaeImageProcessor is built with no resample override, so its "lanczos" default
// applies to both the vision copy and the VAE copy
// (docs/superpowers/specs/2026-10-31-image-edit-lanczos-resize-design.md). Everything below is a
// transcription of Pillow 12.3.0 -- src/libImaging/Resample.c, src/PIL/Image.py, and the RGBa
// conversions in src/libImaging/Convert.c -- and tests/models/qwen_image/test_image_resize.cpp
// grades it against Pillow's own output, byte for byte. Do not "simplify" any line here: each one
// is the reason a golden pixel matches.
// ---------------------------------------------------------------------------

// M_PI as Pillow sees it: the same correctly-rounded double. (The tree uses no math constants.)
constexpr double kPi = 3.14159265358979323846;

// Resample.c:93. Two bits above the 8-bit result: one for the filter's negative lobe, one for
// coefficient sums above 1.0.
constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr int kPrecisionHalf = 1 << (kPrecisionBits - 1);

constexpr double kLanczosSupport = 3.0;  // Resample.c:87

[[nodiscard]] double sinc_filter(double x) {
    if (x == 0.0) { return 1.0; }
    const double p = x * kPi;
    return std::sin(p) / p;
}

[[nodiscard]] double lanczos_filter(double x) {
    // Resample.c:75-81. The window is asymmetric: -3 is inside it, 3 is not.
    if (-3.0 <= x && x < 3.0) { return sinc_filter(x) * sinc_filter(x / 3.0); }
    return 0.0;
}

// Resample.c:96-180: clip8_lookups[v >> 22]. The +0.5 seed in the accumulator is the rounding; this
// floor-then-clamp is the whole conversion to uint8. Pillow reads a 1280-entry table indexed from
// -640 and is out of bounds beyond it; 0 and 255 are what the table's own shape says to return.
[[nodiscard]] std::uint8_t clip8(int v) {
    const int i = v >> kPrecisionBits;  // arithmetic shift: floor(v / 2^22)
    if (i < 0) { return 0; }
    if (i > 255) { return 255; }
    return static_cast<std::uint8_t>(i);
}

// ImagingUtils.h MULDIV255/SHIFTFORDIV255/CLIP8, the arithmetic the mode conversions use.
[[nodiscard]] inline unsigned shift_for_div255(unsigned a) { return ((a >> 8) + a) >> 8; }
[[nodiscard]] inline unsigned muldiv255(unsigned a, unsigned b) { return shift_for_div255(a * b + 128); }
[[nodiscard]] inline unsigned clip8u(unsigned v) { return v <= 0u ? 0u : (v < 256u ? v : 255u); }

// Convert.c:420-431 rgbA2rgba: premultiply the color channels, carry alpha unchanged.
void premultiply_rgba(std::vector<std::uint8_t>& px) {
    for (std::size_t i = 0; i < px.size(); i += 4) {
        const unsigned a = px[i + 3];
        px[i + 0] = static_cast<std::uint8_t>(muldiv255(px[i + 0], a));
        px[i + 1] = static_cast<std::uint8_t>(muldiv255(px[i + 1], a));
        px[i + 2] = static_cast<std::uint8_t>(muldiv255(px[i + 2], a));
    }
}

// Convert.c:435-452 rgba2rgbA: undo it. alpha 255 and 0 are exact; the rest divides, so a
// semi-transparent pixel is the only thing that can round.
void unpremultiply_rgba(std::vector<std::uint8_t>& px) {
    for (std::size_t i = 0; i < px.size(); i += 4) {
        const unsigned a = px[i + 3];
        if (a == 255u || a == 0u) { continue; }
        px[i + 0] = static_cast<std::uint8_t>(clip8u(255u * px[i + 0] / a));
        px[i + 1] = static_cast<std::uint8_t>(clip8u(255u * px[i + 1] / a));
        px[i + 2] = static_cast<std::uint8_t>(clip8u(255u * px[i + 2] / a));
    }
}

struct AxisCoeffs {
    std::vector<int> bounds;        // 2 per output index: {first source index, count}
    std::vector<std::int32_t> kk;   // ksize per output index, in 1/2^22
    int ksize = 0;
};

// Resample.c:182-267 plus the in-place quantization of :270-284.
[[nodiscard]] AxisCoeffs precompute_coeffs(int in_size, double in0, double in1, int out_size) {
    AxisCoeffs a;
    const double scale = (in1 - in0) / static_cast<double>(out_size);
    double filterscale = scale;
    if (filterscale < 1.0) { filterscale = 1.0; }  // upscale never widens the kernel
    const double support = kLanczosSupport * filterscale;
    a.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    a.bounds.assign(static_cast<std::size_t>(out_size) * 2, 0);
    std::vector<double> kk(static_cast<std::size_t>(out_size) * a.ksize, 0.0);
    const double inv_filterscale = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;  // not (x+0.5)/scale - 0.5
        int xmin = static_cast<int>(center - support + 0.5);  // C truncation, both sides of zero
        if (xmin < 0) { xmin = 0; }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) { xmax = in_size; }
        xmax -= xmin;
        double* k = kk.data() + static_cast<std::size_t>(xx) * a.ksize;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double w = lanczos_filter((x + xmin - center + 0.5) * inv_filterscale);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0) {  // normalize the WINDOW, per output pixel
            for (int x = 0; x < xmax; ++x) { k[x] /= ww; }
        }
        a.bounds[2 * xx]     = xmin;
        a.bounds[2 * xx + 1] = xmax;
    }
    const double scale_bits = static_cast<double>(1 << kPrecisionBits);
    a.kk.resize(kk.size());
    for (std::size_t i = 0; i < kk.size(); ++i) {
        const double v = kk[i] * scale_bits;
        a.kk[i] = static_cast<std::int32_t>(kk[i] < 0.0 ? v - 0.5 : v + 0.5);
    }
    return a;
}

// Resample.c:364-385, the bands == 4 branch. `rows` source rows starting at `row_offset` produce the
// same number of output rows of width out_w.
void resample_horizontal(const std::vector<std::uint8_t>& src, int src_w, int row_offset, int rows,
                         int out_w, const AxisCoeffs& a, std::vector<std::uint8_t>& dst) {
    dst.assign(static_cast<std::size_t>(out_w) * static_cast<std::size_t>(rows) * 4, 0);
    for (int yy = 0; yy < rows; ++yy) {
        const std::uint8_t* line_in =
            src.data() + (static_cast<std::size_t>(yy) + row_offset) * src_w * 4;
        std::uint8_t* line_out = dst.data() + static_cast<std::size_t>(yy) * out_w * 4;
        for (int xx = 0; xx < out_w; ++xx) {
            const int xmin = a.bounds[2 * xx];
            const int xmax = a.bounds[2 * xx + 1];
            const std::int32_t* k = a.kk.data() + static_cast<std::size_t>(xx) * a.ksize;
            int ss0 = kPrecisionHalf, ss1 = kPrecisionHalf, ss2 = kPrecisionHalf, ss3 = kPrecisionHalf;
            for (int x = 0; x < xmax; ++x) {
                const std::uint8_t* p = line_in + static_cast<std::size_t>(x + xmin) * 4;
                ss0 += p[0] * k[x];
                ss1 += p[1] * k[x];
                ss2 += p[2] * k[x];
                ss3 += p[3] * k[x];
            }
            std::uint8_t* o = line_out + static_cast<std::size_t>(xx) * 4;
            o[0] = clip8(ss0);
            o[1] = clip8(ss1);
            o[2] = clip8(ss2);
            o[3] = clip8(ss3);
        }
    }
}

// Resample.c:465-485, the bands == 4 branch: the vertical pass runs over the ALREADY horizontally
// resampled buffer, so it is the one that sees `w` columns, not the source width.
void resample_vertical(const std::vector<std::uint8_t>& src, int w, int out_h, const AxisCoeffs& a,
                       std::vector<std::uint8_t>& dst) {
    dst.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(out_h) * 4, 0);
    for (int yy = 0; yy < out_h; ++yy) {
        const int ymin = a.bounds[2 * yy];
        const int ymax = a.bounds[2 * yy + 1];
        const std::int32_t* k = a.kk.data() + static_cast<std::size_t>(yy) * a.ksize;
        std::uint8_t* line_out = dst.data() + static_cast<std::size_t>(yy) * w * 4;
        for (int xx = 0; xx < w; ++xx) {
            int ss0 = kPrecisionHalf, ss1 = kPrecisionHalf, ss2 = kPrecisionHalf, ss3 = kPrecisionHalf;
            for (int y = 0; y < ymax; ++y) {
                const std::uint8_t* p = src.data() + (static_cast<std::size_t>(y + ymin) * w + xx) * 4;
                ss0 += p[0] * k[y];
                ss1 += p[1] * k[y];
                ss2 += p[2] * k[y];
                ss3 += p[3] * k[y];
            }
            std::uint8_t* o = line_out + static_cast<std::size_t>(xx) * 4;
            o[0] = clip8(ss0);
            o[1] = clip8(ss1);
            o[2] = clip8(ss2);
            o[3] = clip8(ss3);
        }
    }
}

// ImagingResampleInner (Resample.c:772-865): horizontal first, then vertical, both coefficient sets
// computed from the ORIGINAL box, and only the rows the vertical pass can reach are resampled
// horizontally.
void resample_once(const std::vector<std::uint8_t>& src, int src_w, int src_h, int dst_w, int dst_h,
                   double bx0, double by0, double bx1, double by1, std::vector<std::uint8_t>& dst) {
    const bool need_h = dst_w != src_w || bx0 != 0.0 || bx1 != dst_w;
    const bool need_v = dst_h != src_h || by0 != 0.0 || by1 != dst_h;
    AxisCoeffs vert = precompute_coeffs(src_h, by0, by1, dst_h);
    const int ybox_first = vert.bounds[0];
    const int ybox_last  = vert.bounds[2 * dst_h - 2] + vert.bounds[2 * dst_h - 1];

    const std::vector<std::uint8_t>* cur    = &src;
    int cur_w = src_w;
    std::vector<std::uint8_t> temp;
    if (need_h) {
        const AxisCoeffs horiz = precompute_coeffs(src_w, bx0, bx1, dst_w);
        for (int i = 0; i < dst_h; ++i) { vert.bounds[2 * i] -= ybox_first; }  // shift for the pass below
        resample_horizontal(src, src_w, ybox_first, std::max(ybox_last - ybox_first, 0), dst_w, horiz,
                            temp);
        cur    = &temp;
        cur_w  = dst_w;
    }
    if (need_v) {
        resample_vertical(*cur, cur_w, dst_h, vert, dst);
        return;
    }
    dst = *cur;
}

} // namespace

void resize_rgba_pil_lanczos(const std::uint8_t* rgba, int src_w, int src_h, int dst_w, int dst_h,
                             std::vector<std::uint8_t>& dst) {
    if (!rgba) { throw std::invalid_argument("the reference image has no pixels"); }
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::invalid_argument("the reference resize geometry is invalid");
    }
    const std::size_t total = static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h) * 4;
    // Image.py:2401-2402: an identity resize is a copy, not a resample.
    if (src_w == dst_w && src_h == dst_h) {
        dst.assign(rgba, rgba + total);
        return;
    }
    // Image.py:2407-2410: RGBA is resampled in the premultiplied "RGBa" mode and un-premultiplied
    // afterwards. Resampling alpha linearly is what makes dark fringes appear around transparency.
    std::vector<std::uint8_t> px(rgba, rgba + total);
    premultiply_rgba(px);
    // Image.py:2432-2436: an image far taller than it is wide is shrunk vertically first. The route
    // rejects this aspect at the geometry snap, so it stays correct rather than unreachable-by-luck.
    if (src_h > src_w * 100 && dst_h < src_h) {
        std::vector<std::uint8_t> mid;
        resample_once(px, src_w, src_h, src_w, dst_h, 0.0, 0.0, static_cast<double>(src_w),
                      static_cast<double>(src_h), mid);
        resample_once(mid, src_w, dst_h, dst_w, dst_h, 0.0, 0.0, static_cast<double>(src_w),
                      static_cast<double>(dst_h), px);
    } else {
        resample_once(px, src_w, src_h, dst_w, dst_h, 0.0, 0.0, static_cast<double>(src_w),
                      static_cast<double>(src_h), px);
    }
    unpremultiply_rgba(px);
    dst = std::move(px);
}

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

    const int target_w = snap_to_32(std::sqrt(kReferenceTargetArea *
                                              static_cast<double>(rgba.width) / rgba.height));
    const int target_h = snap_to_32(std::sqrt(kReferenceTargetArea *
                                              static_cast<double>(rgba.height) / rgba.width));
    if (target_w < 32 || target_h < 32) {
        throw std::invalid_argument("the reference image aspect ratio is too extreme");
    }

    std::vector<std::uint8_t> resized;
    resize_rgba_pil_lanczos(rgba.rgba.data(), rgba.width, rgba.height, target_w, target_h, resized);

    // VaeImageProcessor.preprocess: float32 x / 255 * 2 - 1, channel-major.
    const std::size_t pixels = static_cast<std::size_t>(target_w) * target_h;
    ReferencePlanes out;
    out.vae.height_px      = static_cast<std::uint32_t>(target_h);
    out.vae.width_px       = static_cast<std::uint32_t>(target_w);
    out.source_width_px    = static_cast<std::uint32_t>(rgba.width);
    out.source_height_px   = static_cast<std::uint32_t>(rgba.height);
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
