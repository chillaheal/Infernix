#include "media/encode/png_encode.h"

#include <array>
#include <cstdlib>
#include <stdexcept>

// The prebuilt Windows vcpkg tree carries zlib next to the FFmpeg import libs, and the decode
// path already links it (src/media/CMakeLists.txt). When the same library is available the IDAT
// stream is real DEFLATE; without it the encoder still emits valid PNG via stored blocks.
#ifdef INFERNIX_MEDIA_ZLIB_ENCODER
extern "C" {
#include <zlib.h>
}
#endif

namespace infernix::media::encode {
namespace {

// PNG chunk CRC (CRC-32/ISO-HDLC) and the zlib stream checksum (Adler-32), same polynomials as
// the decoder path; kept local because the encoder must not depend on a compression library.
class Crc32 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            crc_ ^= data[i];
            for (int k = 0; k < 8; ++k) {
                crc_ = (crc_ >> 1) ^ (0xEDB88320U & (~(crc_ & 1U) + 1U));
            }
        }
    }
    [[nodiscard]] std::uint32_t value() const { return ~crc_; }

private:
    std::uint32_t crc_ = 0xFFFFFFFFU;
};

void append_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 24));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}

void append_chunk(std::vector<std::uint8_t>& out, const char* type, const std::uint8_t* data,
                  std::size_t size) {
    append_be32(out, static_cast<std::uint32_t>(size));
    const std::array<std::uint8_t, 4> tag{static_cast<std::uint8_t>(type[0]), static_cast<std::uint8_t>(type[1]),
                                          static_cast<std::uint8_t>(type[2]), static_cast<std::uint8_t>(type[3])};
    out.insert(out.end(), tag.begin(), tag.end());
    out.insert(out.end(), data, data + size);
    Crc32 crc;
    crc.update(tag.data(), tag.size());
    crc.update(data, size);
    append_be32(out, crc.value());
}

// A zlib stream of stored DEFLATE blocks: 2-byte header, 64 KiB blocks (BFINAL on the last),
// Adler-32 of the raw bytes. RFC 1951 section 3.2.4: LEN + its one's complement NLEN. This is
// the fallback for a build without zlib; with zlib, append_idat compresses instead.
void append_stored_zlib(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& raw) {
    out.push_back(0x78U); // CM=8 (deflate), CINFO=7 (32 KiB window)
    out.push_back(0x01U); // FCHECK making CM/FLG divisible by 31; FLEVEL fastest
    std::uint32_t adler = 1U;
    std::uint32_t sum_a = 1U, sum_b = 0U;
    for (std::uint8_t byte : raw) {
        sum_a = (sum_a + byte) % 65521U;
        sum_b = (sum_b + sum_a) % 65521U;
    }
    adler = (sum_b << 16U) | sum_a;

    std::size_t offset = 0;
    do {
        const std::size_t chunk    = raw.size() - offset > 65535U ? 65535U : raw.size() - offset;
        const bool      is_final   = offset + chunk == raw.size();
        const auto      len        = static_cast<std::uint16_t>(chunk);
        const auto      nlen       = static_cast<std::uint16_t>(~len);
        out.push_back(is_final ? 0x01U : 0x00U); // BFINAL, BTYPE=00 (stored)
        out.push_back(static_cast<std::uint8_t>(len));
        out.push_back(static_cast<std::uint8_t>(len >> 8));
        out.push_back(static_cast<std::uint8_t>(nlen));
        out.push_back(static_cast<std::uint8_t>(nlen >> 8));
        out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                   raw.begin() + static_cast<std::ptrdiff_t>(offset + chunk));
        offset += chunk;
    } while (offset < raw.size());
    append_be32(out, adler);
}

// The predictor the decoder adds back for each filter type, copied from
// png_decode.cpp:173-192. It must agree with it byte for byte: the encoder stores
// raw - predictor and the decoder reconstructs raw as filtered + predictor, so any difference
// in these expressions (including the Paeth tie-breaks) yields a different picture.
// left/up/upleft are the ORIGINAL neighbouring bytes, pixel-aligned at bpp bytes, and read as
// 0 outside the image.
[[nodiscard]] std::uint8_t predictor(int filter, std::uint8_t left, std::uint8_t up,
                                     std::uint8_t upleft) {
    switch (filter) {
        case 1: return left;
        case 2: return up;
        case 3: return static_cast<std::uint8_t>((static_cast<int>(left) + static_cast<int>(up)) >> 1);
        case 4: {
            const int pa = std::abs(static_cast<int>(up) - static_cast<int>(upleft));
            const int pb = std::abs(static_cast<int>(left) - static_cast<int>(upleft));
            const int pc = std::abs(static_cast<int>(left) + static_cast<int>(up) -
                                    2 * static_cast<int>(upleft));
            return static_cast<std::uint8_t>((pa <= pb && pa <= pc) ? left : (pb <= pc ? up : upleft));
        }
        default: return 0U; // filter 0 (None)
    }
}

// Append one scanline: a filter-type byte then raw - predictor per byte. Each of the five
// filters is tried and the cheapest kept, by the sum of the absolute values of the filtered
// bytes read as signed - the heuristic the PNG spec recommends, and what makes a photographic
// row shrink before DEFLATE ever sees it. Ties keep the lower filter number, so the choice is
// reproducible from the pixels alone.
void append_filtered_row(std::vector<std::uint8_t>& out, const std::uint8_t* cur,
                         const std::uint8_t* up, std::size_t row_bytes, std::size_t bpp) {
    std::vector<std::uint8_t> scratch(row_bytes);
    std::vector<std::uint8_t> best(row_bytes);
    std::int64_t              best_cost   = -1; // -1 until a filter has been kept
    std::uint8_t              best_filter = 0U;
    for (std::uint8_t filter = 0U; filter <= 4U; ++filter) {
        std::int64_t cost = 0;
        for (std::size_t x = 0; x < row_bytes; ++x) {
            const std::uint8_t left   = x >= bpp ? cur[x - bpp] : 0U;
            const std::uint8_t upb    = up != nullptr ? up[x] : 0U;
            const std::uint8_t upleft = (up != nullptr && x >= bpp) ? up[x - bpp] : 0U;
            const std::uint8_t f =
                static_cast<std::uint8_t>(cur[x] - predictor(filter, left, upb, upleft));
            scratch[x] = f;
            cost += static_cast<std::int64_t>(static_cast<std::uint8_t>(f) > 127U
                                                  ? 256U - static_cast<std::int64_t>(f)
                                                  : static_cast<std::int64_t>(f));
        }
        if (best_cost < 0 || cost < best_cost) { // a tie keeps the earlier, lower filter
            best_cost   = cost;
            best_filter = filter;
            best        = scratch;
        }
    }
    out.push_back(best_filter);
    out.insert(out.end(), best.begin(), best.end());
}

// One IDAT chunk holding the filtered scanlines as a zlib stream, compressed when zlib is on the
// build and stored otherwise. Both paths build the stream first and then frame it, because the
// chunk's length field and CRC cover exactly those bytes.
void append_idat(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& raw) {
#ifdef INFERNIX_MEDIA_ZLIB_ENCODER
    // Level 9: the answer is encoded once per image job and costs milliseconds against a
    // diffusion of minutes, so the bytes on the wire decide the level. compress2 emits the
    // zlib header and the trailing Adler-32 that PNG's IDAT requires.
    std::vector<std::uint8_t> deflated(compressBound(static_cast<uLong>(raw.size())));
    uLongf                    size = static_cast<uLongf>(deflated.size());
    const int rc = compress2(deflated.data(), &size, raw.data(), static_cast<uLong>(raw.size()), 9);
    // Smaller-than-raw is the test, not just Z_OK: DEFLATE costs about 13 bytes of block
    // overhead, so a tiny or incompressible image comes back LARGER than its input and is
    // better served stored. The measured 3x2 case and a noise image both land here.
    if (rc == Z_OK && size < raw.size()) {
        deflated.resize(static_cast<std::size_t>(size));
        append_chunk(out, "IDAT", deflated.data(), deflated.size());
        return;
    }
#endif
    std::vector<std::uint8_t> idat; // fresh, so nothing of a failed compression attempt is kept
    append_stored_zlib(idat, raw);
    append_chunk(out, "IDAT", idat.data(), idat.size());
}

} // namespace

std::vector<std::uint8_t> encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                                           std::span<const std::uint8_t> rgba) {
    if (width == 0U || height == 0U || rgba.size() != 4ULL * width * height) {
        throw std::invalid_argument("png_encode: rgba must hold exactly 4 bytes per pixel");
    }
    std::vector<std::uint8_t> out;
    out.insert(out.end(), {0x89U, 'P',  'N',  'G',  '\r', '\n', 0x1AU, '\n'});

    std::array<std::uint8_t, 13> ihdr{};
    const auto put = [&ihdr](std::size_t at, std::uint32_t v) {
        ihdr[at]     = static_cast<std::uint8_t>(v >> 24);
        ihdr[at + 1] = static_cast<std::uint8_t>(v >> 16);
        ihdr[at + 2] = static_cast<std::uint8_t>(v >> 8);
        ihdr[at + 3] = static_cast<std::uint8_t>(v);
    };
    put(0, width);
    put(4, height);
    ihdr[8] = 8U;  // bit depth
    ihdr[9] = 6U;  // color type RGBA
    ihdr[10] = 0U; // compression
    ihdr[11] = 0U; // filter method 0: the five adaptive filters, chosen per row below
    ihdr[12] = 0U; // interlace: none
    append_chunk(out, "IHDR", ihdr.data(), ihdr.size());

    const std::size_t row_bytes = 4ULL * width;
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (1U + row_bytes));
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* cur = rgba.data() + 4ULL * width * y;
        const std::uint8_t* up  = y > 0U ? cur - row_bytes : nullptr;
        append_filtered_row(raw, cur, up, row_bytes, 4ULL);
    }
    append_idat(out, raw);
    append_chunk(out, "IEND", nullptr, 0U);
    return out;
}

} // namespace infernix::media::encode
