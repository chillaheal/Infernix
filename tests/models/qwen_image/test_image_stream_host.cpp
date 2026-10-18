// CPU-only host staging bound for the Qwen-Image-2.1 artifact: the memory claim behind the bounded
// streaming image loader (task #13 coexistence). It streams the ENTIRE real .infernix, one object at
// a time in directory order, through a single grow-only reused host buffer via Reader::read_into, so
// the peak host staging memory equals the largest SINGLE weight object, never the sum of all weights.
// That is what lets an image job run next to the live LLM server without a second full weight set
// resident in host RAM (the host-RAM wall). For every FP8 RowScale tensor it re-derives the RowScale
// planes from geometry and requires them to match the bytes currently held in the shared buffer, so
// reuse is proven to materialize each object correctly -- not merely the last one. It also requires
// sum(object bytes) to be many times the high-water mark: if all weights were resident the bound
// would collapse to the sum. No GPU, no server. Returns 77 (CTest skip) when INFERNIX_TEST_ARTIFACT
// is unset or the artifact is absent.
#include "artifact/reader.h"
#include "core/weight_view.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Independently recompute the RowScale framing the converter wrote and require the planes derived
// from the geometry to match the object currently staged at `base`. Complements (does not mirror) the
// load oracle: it is run over a REUSED buffer, on every FP8 object, not once on a freshly allocated one.
void check_row_scale_planes(const Reader& reader, const TensorObject& tensor, ObjectHandle handle,
                            const std::byte* base) {
    const WeightGeometry& g = reader.geometry(handle);
    require(g.format == QType::FP8_E4M3FN_ROW_BF16, "FP8 format enum mismatch");
    require(g.layout == QuantLayout::RowScale, "FP8 layout enum mismatch");
    require(tensor.shape.size() == 2, "FP8 RowScale tensor is not 2-D");

    const std::uint64_t n = tensor.shape[0];
    const std::uint64_t k = tensor.shape[1];
    const std::uint64_t code_bytes  = checked_mul(n, k, "code bytes");
    const std::uint64_t scale_off   = align_up(code_bytes, 256, "scale offset");
    const std::uint64_t scale_bytes = checked_mul(n, 2, "scale bytes");
    const std::uint64_t total_bytes = checked_add(scale_off, scale_bytes, "total bytes");

    require(g.code_bytes == code_bytes, "geometry code_bytes mismatch");
    require(g.scale_offset == scale_off, "geometry scale_offset mismatch");
    require(g.scale_bytes == scale_bytes, "geometry scale_bytes mismatch");
    require(g.bytes == total_bytes, "geometry bytes mismatch");
    require(tensor.bytes == total_bytes, "stored object bytes != recomputed geometry");

    const WeightParent parent{g, base};
    const WeightRegion region{&parent, 0, code_bytes};
    const WeightRowPlanes planes = weight_row_planes(region);
    require(planes.codes == base, "reused-buffer codes plane origin differs");
    require(planes.scales == base + scale_off, "reused-buffer scales plane origin differs");
    require(planes.row_begin == 0 && planes.row_count == n, "engine row span differs");
    require(planes.scale_row_bytes == 2, "engine scale row stride differs");
    require(planes.high_row_bytes == 0, "FP8 RowScale must have no high plane");
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping bounded-stream bound\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "artifact absent; skipping bounded-stream bound: " << env << "\n";
            return 77;
        }

        const Reader reader(path);
        const Directory& directory = reader.directory();

        std::vector<std::byte> pool; // single grow-only reused staging buffer
        std::uint64_t max_bytes = 0;
        std::uint64_t sum_bytes = 0;
        std::size_t fp8_rowscale = 0;
        std::size_t visited = 0;
        std::string_view max_id;

        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const Object& object = directory.objects[i];
            reader.validate_object({i});

            const std::uint64_t offset = object_offset(object);
            const std::uint64_t length = object_bytes(object);

            // Grow only: the buffer never shrinks, so its high-water mark is the largest single object.
            if (length > pool.size()) { pool.resize(length); }
            require(pool.size() >= length, "staging buffer not grown to object size");
            reader.read_into(offset, std::span<std::byte>(pool.data(), length));

            sum_bytes = checked_add(sum_bytes, length, "sum bytes");
            if (length > max_bytes) {
                max_bytes = length;
                max_id    = object_id(object);
            }

            const auto* tensor = std::get_if<TensorObject>(&object);
            if (tensor != nullptr && tensor->format == "fp8_e4m3fn_row_bf16") {
                check_row_scale_planes(reader, *tensor, {i}, pool.data());
                ++fp8_rowscale;
            }
            ++visited;
        }

        require(visited == directory.objects.size(), "did not stream every object once");
        require(pool.size() == max_bytes, "high-water mark is not the largest single object");
        require(max_bytes > 0, "no objects in artifact");
        // The weights are never all resident: the whole artifact is many times the staging peak.
        require(sum_bytes >= 8 * max_bytes, "artifact is not many times the staging bound");
        require(fp8_rowscale > 0, "no FP8 RowScale tensors found in the artifact");

        std::cout << "bounded-stream host bound passed objects=" << visited
                  << " peak_host=" << pool.size() << " max_obj=" << max_bytes << " (" << max_id
                  << ") sum=" << sum_bytes << " fp8_rowscale=" << fp8_rowscale << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bounded-stream host bound FAILED: " << error.what() << "\n";
        return 1;
    }
}
