// Pinned-host staging bound for the Qwen-Image-2.1 artifact (coexistence task #13, C4 foundation).
// The in-process image lane stages every streamed weight object through the HostContextArena -- a
// fixed-capacity, NON-growing pinned pool -- instead of a grow-only pageable buffer. Sizing the pool
// to the largest single object (never the sum) and refusing growth (allocate -> nullopt) is what keeps
// host-RAM pressure flat at runtime: exhaustion is reported as unsatisfiable, not grown. This is the
// C4 staging primitive: it proves the HostContextArena serves the real artifact's streaming contract
// (peak staging == the largest single object, and every FP8 RowScale object's planes re-derive
// correctly on the arena's pinned span) and that the pool is bounded (a second concurrent largest
// object is refused). Host + CUDA-context only (cudaMallocHost): NO device VRAM is allocated, so it
// runs beside the live server (no blackout). Returns 77 (CTest skip) when the artifact is unset,
// absent, or no usable CUDA device is present.
#include "artifact/reader.h"
#include "core/host_context_arena.h"
#include "core/weight_view.h"

#include <cuda_runtime_api.h>

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

// Re-derive the RowScale framing from the geometry and require the planes point into the arena's
// pinned span (same check as test_image_stream_host, but over the fixed pool's span, not a grow-only
// vector). Complements the load oracle: it proves the arena materializes each object correctly.
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
    require(planes.codes == base, "arena codes plane origin differs");
    require(planes.scales == base + scale_off, "arena scales plane origin differs");
    require(planes.row_begin == 0 && planes.row_count == n, "engine row span differs");
    require(planes.scale_row_bytes == 2, "engine scale row stride differs");
    require(planes.high_row_bytes == 0, "FP8 RowScale must have no high plane");
}

} // namespace

int main() {
    int count = 0;
    if (const cudaError_t error = cudaGetDeviceCount(&count);
        error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device (HostContextArena needs a context for cudaMallocHost)\n";
        return 77;
    }
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping pinned-arena staging bound\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "artifact absent; skipping pinned-arena staging bound: " << env << "\n";
            return 77;
        }

        const Reader reader(path);
        const Directory& directory = reader.directory();

        // The pool is sized to the LARGEST single object (from the directory, no read): that is the
        // peak a streaming loader ever needs, so the pool is neither over- nor under-reserved, and it
        // never has to grow (allocate -> optional).
        std::uint64_t max_bytes = 0;
        std::uint64_t sum_bytes = 0;
        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const std::uint64_t length = object_bytes(directory.objects[i]);
            sum_bytes = checked_add(sum_bytes, length, "sum bytes");
            if (length > max_bytes) { max_bytes = length; }
        }
        require(max_bytes > 0, "no objects in artifact");
        const std::size_t capacity = static_cast<std::size_t>(align_up(max_bytes, 256, "capacity"));

        HostContextArena arena(capacity, 256);
        require(arena.capacity_bytes() >= max_bytes, "arena capacity covers the largest object");

        std::size_t fp8_rowscale = 0;
        std::size_t visited = 0;
        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const Object& object = directory.objects[i];
            reader.validate_object({i});
            const std::uint64_t offset = object_offset(object);
            const std::uint64_t length = object_bytes(object);
            auto allocation = arena.allocate(length);
            require(allocation.has_value(), "arena refused a single object it should hold (pool too small)");
            reader.read_into(offset, std::span<std::byte>(allocation->data(), length));
            const auto* tensor = std::get_if<TensorObject>(&object);
            if (tensor != nullptr && tensor->format == "fp8_e4m3fn_row_bf16") {
                check_row_scale_planes(reader, *tensor, {i}, allocation->data());
                ++fp8_rowscale;
            }
            (void)allocation->release();
            ++visited;
        }
        require(visited == directory.objects.size(), "did not stream every object once");
        require(fp8_rowscale > 0, "no FP8 RowScale tensors found in the artifact");

        // The pool is bounded and non-growing: the largest object fills it, so a second concurrent
        // largest object must be refused (reported as unsatisfiable, not grown).
        auto first = arena.allocate(max_bytes);
        require(first.has_value(), "first largest-object allocation failed");
        require(!arena.allocate(max_bytes).has_value(),
                "a second concurrent largest object must fail (the pool is bounded, not grown)");
        (void)first->release();

        // Peak staging == the largest single object (plus alignment), and the whole artifact is many
        // times that: the weights are never all resident -- the bound that makes the coexistence lane fit.
        const std::uint64_t peak = arena.peak_occupied_bytes();
        require(peak >= max_bytes && peak <= static_cast<std::size_t>(align_up(max_bytes, 256, "peak")),
                "peak staging is not the largest single object");
        require(sum_bytes >= 8 * max_bytes, "artifact is not many times the staging bound");
        require(arena.occupied_bytes() == 0, "every allocation released");

        std::cout << "pinned-arena staging bound passed objects=" << visited
                  << " peak_host=" << peak << " max_obj=" << max_bytes << " sum=" << sum_bytes
                  << " fp8_rowscale=" << fp8_rowscale << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pinned-arena staging bound FAILED: " << error.what() << "\n";
        return 1;
    }
}
