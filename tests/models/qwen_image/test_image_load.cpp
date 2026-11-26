// CPU-only load oracle for a Qwen-Image-2.1 v3 artifact: qualifies the Python-writes / C++-reads
// FP8 RowScale framing seam on a real .infernix without touching the GPU or the running server.
// It (1) validates every object, (2) recomputes RowScale geometry independently and requires it to
// equal the geometry the loader will use and the stored object byte count, (3) confirms the engine's
// weight_row_planes agree with those independent offsets, and (4) naively decodes a sample of FP8
// codes and BF16 scales, requiring finite values and non-negative scales.

#include "artifact/reader.h"
#include "core/weight_view.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace infernix;
using namespace infernix::artifact;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A stored FP8_E4M3FN byte. Exponent 4 (bias 7), mantissa 3, no infinities, 0x7F is NaN.
float e4m3_to_f32(std::uint8_t byte) {
    const int sign       = (byte >> 7) & 1;
    const int exponent   = (byte >> 3) & 0xF;
    const int mantissa   = byte & 0x7;
    if (exponent == 15 && mantissa == 7) {
        return std::nanf("");
    }
    float value;
    if (exponent == 0) {
        value = static_cast<float>(mantissa) / 8.0F * std::ldexp(1.0F, -6);
    } else {
        value = (1.0F + static_cast<float>(mantissa) / 8.0F) * std::ldexp(1.0F, exponent - 7);
    }
    return sign ? -value : value;
}

float bf16_to_f32(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}

std::uint16_t read_bf16_word(const std::byte* at) {
    const auto low  = std::to_integer<std::uint16_t>(at[0]);
    const auto high = std::to_integer<std::uint16_t>(at[1]);
    return static_cast<std::uint16_t>(low | (high << 8));
}

void check_fp8_row_scale(const Reader& reader, const TensorObject& tensor, ObjectHandle handle,
                        bool sample_decode) {
    const WeightGeometry& geometry = reader.geometry(handle);
    require(geometry.format == QType::FP8_E4M3FN_ROW_BF16, "FP8 format enum mismatch");
    require(geometry.layout == QuantLayout::RowScale, "FP8 layout enum mismatch");
    require(tensor.shape.size() == 2, "FP8 RowScale tensor is not 2-D");

    const std::uint64_t n = tensor.shape[0];
    const std::uint64_t k = tensor.shape[1];

    // Independently recompute the RowScale layout the converter wrote.
    const std::uint64_t code_bytes = checked_mul(n, k, "code bytes");
    const std::uint64_t scale_off  = align_up(code_bytes, 256, "scale offset");
    const std::uint64_t scale_bytes = checked_mul(n, 2, "scale bytes");
    const std::uint64_t total_bytes = checked_add(scale_off, scale_bytes, "parent bytes");

    require(geometry.elements == code_bytes, "geometry elements mismatch");
    require(geometry.code_bytes == code_bytes, "geometry code_bytes mismatch");
    require(geometry.scale_offset == scale_off, "geometry scale_offset mismatch");
    require(geometry.scale_bytes == scale_bytes, "geometry scale_bytes mismatch");
    require(geometry.bytes == total_bytes, "geometry bytes mismatch");
    require(object_bytes(tensor) == total_bytes, "stored object bytes != recomputed geometry");

    const std::vector<std::byte> buffer = reader.read_object(handle);
    require(buffer.size() == total_bytes, "read_object size != recomputed geometry");

    const WeightParent parent{geometry, buffer.data()};
    const WeightRegion region{&parent, 0, code_bytes};
    const WeightRowPlanes planes = weight_row_planes(region);
    require(planes.codes == buffer.data(), "engine codes plane origin differs");
    require(planes.scales == buffer.data() + scale_off, "engine scales plane origin differs");
    require(planes.row_begin == 0 && planes.row_count == n, "engine row span differs");
    require(planes.scale_row_bytes == 2, "engine scale row stride differs");
    require(planes.high_row_bytes == 0, "FP8 RowScale must have no high plane");

    if (sample_decode) {
        const std::byte* codes  = planes.codes;
        const std::byte* scales = planes.scales;
        const std::uint64_t rows = std::min<std::uint64_t>(n, 16);
        const std::uint64_t cols = std::min<std::uint64_t>(k, 16);
        for (std::uint64_t row = 0; row < rows; ++row) {
            const float scale = bf16_to_f32(read_bf16_word(scales + row * 2));
            require(std::isfinite(scale), "scale not finite");
            require(scale >= 0.0F, "scale negative");
            for (std::uint64_t col = 0; col < cols; ++col) {
                const auto code_byte = std::to_integer<std::uint8_t>(codes[row * k + col]);
                const float code     = e4m3_to_f32(code_byte);
                require(!std::isnan(code), "FP8 code decoded to NaN");
                const double value =
                    static_cast<double>(scale) * static_cast<double>(code);
                require(std::isfinite(value), "dequantized value not finite");
            }
        }
    }
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping Qwen-Image load oracle\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "Qwen-Image artifact absent; skipping: " << env << "\n";
            return 77;
        }

        const Reader reader(path);
        const Directory& directory = reader.directory();

        std::uint64_t total          = 0;
        std::size_t fp8_rowscale     = 0;
        bool sampled                 = false;
        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const Object& object = directory.objects[i];
            reader.validate_object({i});
            total = checked_add(total, object_bytes(object), "artifact bytes");

            const auto* tensor = std::get_if<TensorObject>(&object);
            if (tensor == nullptr || tensor->format != "fp8_e4m3fn_row_bf16") {
                continue;
            }
            check_fp8_row_scale(reader, *tensor, {i}, !sampled);
            ++fp8_rowscale;
            sampled = true;
        }

        require(fp8_rowscale > 0, "no FP8 RowScale tensors found in the artifact");
        std::cout << "Qwen-Image load oracle passed objects=" << directory.objects.size()
                  << " fp8_rowscale=" << fp8_rowscale << " bytes=" << total << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen-Image load oracle FAILED: " << error.what() << "\n";
        return 1;
    }
}
