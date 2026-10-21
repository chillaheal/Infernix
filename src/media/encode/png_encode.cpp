#include "media/encode/png_encode.h"

#include <array>
#include <stdexcept>

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
// Adler-32 of the raw bytes. RFC 1951 section 3.2.4: LEN + its one's complement NLEN.
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
    ihdr[11] = 0U; // filter
    ihdr[12] = 0U; // interlace: none
    append_chunk(out, "IHDR", ihdr.data(), ihdr.size());

    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (1U + 4ULL * width));
    for (std::uint32_t y = 0; y < height; ++y) {
        raw.push_back(0U); // filter type 0 (None)
        const std::size_t begin = 4ULL * width * y;
        raw.insert(raw.end(), rgba.begin() + static_cast<std::ptrdiff_t>(begin),
                   rgba.begin() + static_cast<std::ptrdiff_t>(begin + 4ULL * width));
    }
    std::vector<std::uint8_t> idat;
    append_stored_zlib(idat, raw);
    append_chunk(out, "IDAT", idat.data(), idat.size());
    append_chunk(out, "IEND", nullptr, 0U);
    return out;
}

} // namespace infernix::media::encode
