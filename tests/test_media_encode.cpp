// PNG encoder round trip: encode PNG from RGBA with our encoder and decode it back with the
// engine's own PNG decoder. Opaque pixels survive the decoder's alpha-over-white composite
// exactly, so the decoded RGB must equal the source byte-for-byte. The large image spans several
// stored DEFLATE blocks (the 64 KiB block loop and its final block) and a row filter byte.
#include "media/encode/png_encode.h"

#ifdef INFERNIX_MEDIA_NATIVE_PNG
#include "media/decode/png_decode.h"
#endif

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n';
    if (!ok) { ++failures; }
}

std::vector<std::uint8_t> gradient(std::uint32_t w, std::uint32_t h) {
    std::vector<std::uint8_t> rgba(4ULL * w * h);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            std::uint8_t* p    = rgba.data() + 4ULL * (static_cast<std::size_t>(y) * w + x);
            p[0]               = static_cast<std::uint8_t>(x % 256U);
            p[1]               = static_cast<std::uint8_t>(y % 256U);
            p[2]               = static_cast<std::uint8_t>((x * 3U + y * 5U) % 256U);
            p[3]               = 255U; // fully opaque: the decoder's white composite is identity
        }
    }
    return rgba;
}

// The size a stored-block PNG would occupy: every byte of the filtered scanlines plus one byte
// of DEFLATE header per 64 KiB block, plus the fixed chunk overhead. The assertion below only
// means something for a build that compresses.
[[nodiscard]] std::size_t stored_floor(std::uint32_t w, std::uint32_t h) {
    const std::size_t raw     = static_cast<std::size_t>(h) * (1U + 4ULL * w);
    const std::size_t blocks  = (raw + 65534U) / 65535U; // at least one, even for an empty raw
    return 8U + 4U + 4U + 13U + 4U /* sig + IHDR */ + 4U + 4U + 4U /* IDAT framing */ +
           2U + 5U * blocks + raw + 4U + 4U + 12U /* IEND */;
}

std::vector<std::uint8_t> noise(std::uint32_t w, std::uint32_t h) {
    // Deterministic pseudo-random pixels: no filter can predict them, so this is the case where
    // compression must not grow the file much and must still decode back exactly.
    std::vector<std::uint8_t> rgba(4ULL * w * h);
    std::uint32_t             s = 0x12345678U;
    for (std::uint8_t& byte : rgba) {
        s ^= s << 13U; s ^= s >> 17U; s ^= s << 5U;
        byte = static_cast<std::uint8_t>(s >> 24U);
    }
    for (std::size_t i = 3U; i < rgba.size(); i += 4U) { rgba[i] = 255U; } // opaque, as in gradient
    return rgba;
}

// Encode and, where the engine's own decoder is built, decode back and compare. Returns the
// encoded size so the caller can assert on it.
std::size_t round_trip_pixels(std::uint32_t w, std::uint32_t h,
                              const std::vector<std::uint8_t>& rgba, const std::string& tag) {
    const auto png    = infernix::media::encode::encode_png_rgba8(w, h, rgba);
    const auto bytes  = std::span<const std::uint8_t>(png);
#ifdef INFERNIX_MEDIA_NATIVE_PNG
    check(infernix::media::decode::is_png(bytes), tag + ": the output carries the PNG signature");
    const auto info = infernix::media::decode::png_image_info(bytes, 64ULL << 20);
    check(info.width == static_cast<int>(w) && info.height == static_cast<int>(h),
          tag + ": IHDR reports the requested dimensions");
    const auto image = infernix::media::decode::decode_png(bytes, 64ULL << 20);
    check(image.width == static_cast<int>(w) && image.height == static_cast<int>(h),
          tag + ": the decoder reports the requested dimensions");
    bool same = image.rgb.size() == 3ULL * w * h;
    for (std::uint32_t y = 0; same && y < h; ++y) {
        for (std::uint32_t x = 0; same && x < w; ++x) {
            const std::uint8_t* src = rgba.data() + 4ULL * (static_cast<std::size_t>(y) * w + x);
            const std::uint8_t* got = image.rgb.data() + 3ULL * (static_cast<std::size_t>(y) * w + x);
            same                    = src[0] == got[0] && src[1] == got[1] && src[2] == got[2];
        }
    }
    check(same, tag + ": decoded RGB equals the source byte-for-byte");
#else
    (void)bytes;
    check(png.size() > 8ULL, tag + ": encoded without a decoder to round-trip against");
#endif
    return png.size();
}

void round_trip(std::uint32_t w, std::uint32_t h, const std::string& tag) {
    const auto    rgba = gradient(w, h);
    const std::size_t size = round_trip_pixels(w, h, rgba, tag);
#ifdef INFERNIX_MEDIA_ZLIB_ENCODER
    // The whole point of the change: an image must come out well under what stored blocks cost.
    // Half that size is a large margin over the measured 5x. Only for an image big enough for
    // the ratio to mean anything: a 3x2 strip is 26 bytes of scanline, where DEFLATE's fixed
    // overhead alone eats the comparison and the stored path is the right answer anyway.
    const std::size_t raw = static_cast<std::size_t>(h) * (1U + 4ULL * w);
    if (raw >= 4096U) {
        check(size * 2U < stored_floor(w, h),
              tag + ": compressed to " + std::to_string(size) + " bytes, under half the " +
                  std::to_string(stored_floor(w, h)) + " a stored-block PNG would take");
    } else {
        check(size <= stored_floor(w, h), tag + ": the tiny image is no larger than stored");
    }
#else
    (void)size;
#endif
}

void round_trip_noise(std::uint32_t w, std::uint32_t h, const std::string& tag) {
    const auto    rgba = noise(w, h);
    const std::size_t size = round_trip_pixels(w, h, rgba, tag);
    // Incompressible pixels are the case where DEFLATE loses to its own header, so the encoder
    // must fall back to stored rather than ship the bigger stream: the size then equals what
    // stored costs, and never grows past it. The pixel check in round_trip_pixels is the real
    // test here -- a buffer left half-filled by a discarded compression attempt passes nothing
    // else, and this is the image that caught it.
    check(size <= stored_floor(w, h),
          tag + ": incompressible pixels take the stored path, not a larger deflated one (" +
              std::to_string(size) + " vs at most " + std::to_string(stored_floor(w, h)) + ")");
}

} // namespace

int main() {
    try {
        round_trip(3U, 2U, "small");
        // 128 * (1 + 4 * 256) = 131200 raw bytes > two stored 65535-byte blocks.
        round_trip(256U, 128U, "multiblock");
        // A row of identical pixels: the Up filter alone makes it nearly free, which is what the
        // answer looks like for a flat background.
        round_trip(512U, 200U, "gradient");
        round_trip_noise(128U, 96U, "noise");

        bool threw = false;
        try {
            (void)infernix::media::encode::encode_png_rgba8(2U, 2U, std::vector<std::uint8_t>(15U));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "a wrong-size buffer is rejected");

        threw = false;
        try {
            (void)infernix::media::encode::encode_png_rgba8(0U, 1U, std::vector<std::uint8_t>(0U));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "a zero dimension is rejected");
    } catch (const std::exception& error) {
        std::cout << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "MEDIA_ENCODE_TEST: PASS\n" : "MEDIA_ENCODE_TEST: FAIL\n");
    return failures == 0 ? 0 : 1;
}
