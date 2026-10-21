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

void round_trip(std::uint32_t w, std::uint32_t h, const std::string& tag) {
    const auto rgba   = gradient(w, h);
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
}

} // namespace

int main() {
    try {
        round_trip(3U, 2U, "small");
        // 128 * (1 + 4 * 256) = 131200 raw bytes > two stored 65535-byte blocks.
        round_trip(256U, 128U, "multiblock");

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
