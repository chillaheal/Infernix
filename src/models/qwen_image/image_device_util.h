#pragma once

// Shared device utilities for the runtime-internal image modules (the DiT/VAE pipeline, the text
// tower): the BF16 codec, Tensor views, CUDA error wrapping, the #17 ledger's RAII Buf, and the
// streamed upload of one artifact object into the qualified Contiguous-BF16 / RowScale-FP8 Weight
// ABI. Every image-lane device allocation goes through Buf so ONE ledger accounts it. Header-only
// (everything inline); __global__ kernels stay in the consuming .cu translation units. Consumed
// only by infernix_qwen_image_runtime sources.

#include "artifact/reader.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen_image {

using T = std::vector<float>;

// ---- host BF16 codec (identical to the qualified route) ----
inline std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((x >> 16) | 0x0040U);
    const std::uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}
inline float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}
inline std::vector<std::uint16_t> bits_of(const T& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}
inline T from_bits(const std::vector<std::uint16_t>& bits) {
    T out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = from_bf16(bits[i]);
    return out;
}
inline Tensor view(void* p, int a, int b, int c = 1, int dd = 1) { return Tensor(p, DType::BF16, {a, b, c, dd}); }

inline void cuda_ok(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

// #17 release-gate ledger: the device bytes the image modules' Bufs currently own. Buf is their
// ONLY device allocator, so every image-lane alloc/free passes through here; the live count must be
// 0 between jobs on the success AND the throw path (cudaMemGetInfo deltas are the diagnostic, this
// ledger is the app-owned account). image_pipeline.h exports the read/reset API.
inline std::atomic<std::uint64_t> g_image_device_live{0};
inline std::atomic<std::uint64_t> g_image_device_peak{0};

// The stream of the image region executing now (its Bufs' consumers enqueue on it), or null for a
// legacy region (the text tower on the default stream). Alloc inside a region goes to the
// stream-ordered pool on this stream so a retired buffer's bytes are REUSABLE by the lane's next
// alloc: a plain cudaMalloc lane with cudaFreeAsync retires strands the retired bytes in the pool
// (the driver counts them used, legacy allocs cannot take them) -- that OOMed the 2048x2048 decode
// at 6.5 GiB of ledger live against a full card. RAII so a nested call cannot leak a foreign stream.
inline std::atomic<cudaStream_t> g_image_stream{nullptr};
struct ImageStreamScope {
    explicit ImageStreamScope(cudaStream_t s) : prev_(g_image_stream.exchange(s)) {}
    ~ImageStreamScope() { g_image_stream.store(prev_); }
    ImageStreamScope(const ImageStreamScope&)            = delete;
    ImageStreamScope& operator=(const ImageStreamScope&) = delete;

private:
    cudaStream_t prev_;
};

// A device buffer the image job owns: freed (RAII) at scope exit -- always before the interlude
// restores the LLM's pools. Movable so a job can collect its scratch buffers in a vector.
struct Buf {
    void* d = nullptr;
    std::size_t bytes  = 0;
    cudaStream_t stream = nullptr; // the pool it was allocated from; null = legacy cudaMalloc
    void alloc(std::size_t b) {
        bytes  = b;
        stream = g_image_stream.load();
        const cudaError_t err =
            (stream != nullptr) ? cudaMallocAsync(&d, b, stream) : cudaMalloc(&d, b);
        if (err != cudaSuccess) {
            // The bare OOM told nothing; the ledger + driver view say which size died against what.
            cudaGetLastError();
            std::size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); }
            const auto MiB = [](std::size_t x) { return x / (1024ULL * 1024ULL); };
            throw std::runtime_error(std::string(stream != nullptr ? "cudaMallocAsync" : "cudaMalloc") +
                                     " (image buffer): out of memory (request " + std::to_string(MiB(b)) +
                                     " MiB, image ledger live " +
                                     std::to_string(MiB(g_image_device_live.load())) + " MiB, peak " +
                                     std::to_string(MiB(g_image_device_peak.load())) +
                                     " MiB, driver free " + std::to_string(MiB(free_b)) + " of " +
                                     std::to_string(MiB(total_b)) + " MiB)");
        }
        const std::uint64_t live = g_image_device_live.fetch_add(b) + b;
        std::uint64_t peak       = g_image_device_peak.load();
        while (live > peak && !g_image_device_peak.compare_exchange_weak(peak, live)) {}
    }
    // Stream-ordered release: the memory returns to the device once the stream passes this point (no
    // device-wide cudaFree), and the ledger slot comes back now. The Buf owns nothing afterwards.
    void retire(cudaStream_t s) {
        if (d == nullptr) return;
        if (stream != nullptr) cuda_ok(cudaFreeAsync(d, s), "cudaFreeAsync (image buffer)");
        else                   cuda_ok(cudaFree(d), "cudaFree (image buffer)");
        d      = nullptr;
        stream = nullptr;
        g_image_device_live.fetch_sub(bytes);
        bytes = 0;
    }
    ~Buf() { release(); }
    Buf() = default;
    Buf(Buf&& o) noexcept : d(o.d), bytes(o.bytes), stream(o.stream) {
        o.d = nullptr; o.bytes = 0; o.stream = nullptr;
    }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            release();
            d = o.d; bytes = o.bytes; stream = o.stream;
            o.d = nullptr; o.bytes = 0; o.stream = nullptr;
        }
        return *this;
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;

private:
    void release() {
        if (d == nullptr) return;
        if (stream != nullptr) {
            if (cudaFreeAsync(d, stream) != cudaSuccess) cudaGetLastError();
        } else {
            cudaFree(d);
        }
        d      = nullptr;
        stream = nullptr;
        g_image_device_live.fetch_sub(bytes);
        bytes = 0;
    }
};

inline void h2d(void* dst, const void* src, std::size_t bytes) {
    cuda_ok(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "H2D (image weight)");
}

// One streamed parameter uploaded to the device, wrapped the way the qualified linear routes read
// it: a contiguous BF16 object as a Contiguous Weight, a RowScale FP8 object as the single
// [codes][scales] plane with the FP8 RowScale Weight ABI the device route verified.
struct Uploaded {
    Buf buf;
    Weight w{};
};

inline std::vector<std::byte> read_object(const artifact::Reader& reader, const artifact::ParameterReference& ref) {
    if (ref.binding.parts.size() != 1) {
        throw std::runtime_error(ref.name + ": an image weight must bind exactly one whole object");
    }
    return reader.read_object(ref.binding.parts.front().object);
}

// Upload one streamed parameter from its already-read object bytes (a host cache) or straight from
// the artifact. The Weight it returns addresses the device copy.
inline Uploaded upload_streamed(const artifact::Reader& reader, const artifact::ParameterReference& ref,
                                const std::vector<std::byte>* object_bytes) {
    if (ref.binding.parts.size() != 1) {
        throw std::runtime_error(ref.name + ": an image weight must bind exactly one whole object");
    }
    const WeightGeometry& g = reader.geometry(ref.binding.parts.front().object);
    std::vector<std::byte> owned;
    const std::vector<std::byte>* obj = object_bytes;
    if (obj == nullptr) {
        owned   = reader.read_object(ref.binding.parts.front().object);
        obj     = &owned;
    }
    if (obj->size() != g.bytes) {
        throw std::runtime_error(ref.name + ": object size differs from its geometry");
    }
    const std::uint64_t n = g.shape.empty() ? 0 : g.shape[0];
    if (n == 0 || g.elements % n != 0) {
        throw std::runtime_error(ref.name + ": geometry does not frame as [rows, cols]");
    }
    const std::uint64_t k = g.elements / n;

    Uploaded u;
    if (g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) {
        u.buf.alloc(g.bytes);
        h2d(u.buf.d, obj->data(), g.bytes);
        u.w                = Weight{};
        u.w.payload        = u.buf.d;
        u.w.payload_bytes  = g.bytes;
        u.w.qtype          = QType::BF16;
        u.w.shape[0]       = n;   u.w.shape[1]       = k;
        u.w.shape[2]       = 1;   u.w.shape[3]       = 1;
        u.w.padded_shape[0] = n;  u.w.padded_shape[1] = k;
        u.w.padded_shape[2] = 1;  u.w.padded_shape[3] = 1;
        u.w.ndim           = 2;
        u.w.qdata          = u.buf.d;
        u.w.n              = n;   u.w.k              = k;
        u.w.layout         = QuantLayout::Contiguous;
        return u;
    }
    if (g.format == QType::FP8_E4M3FN_ROW_BF16 && g.layout == QuantLayout::RowScale) {
        if (g.high_bytes != 0 || g.code_bytes != n * k || g.scale_bytes != n * 2 ||
            g.scale_offset < g.code_bytes) {
            throw std::runtime_error(ref.name + ": FP8 object is not a plain [codes][scales] plane");
        }
        const std::size_t total = static_cast<std::size_t>(g.code_bytes + g.scale_bytes);
        u.buf.alloc(total);
        h2d(u.buf.d, obj->data(), static_cast<std::size_t>(g.code_bytes));
        h2d(static_cast<std::uint8_t*>(u.buf.d) + g.code_bytes, obj->data() + g.scale_offset,
            static_cast<std::size_t>(g.scale_bytes));
        u.w                = Weight{};
        u.w.payload        = u.buf.d;
        u.w.payload_bytes  = total;
        u.w.qtype          = QType::FP8_E4M3FN_ROW_BF16;
        u.w.shape[0]       = n;   u.w.shape[1]       = k;
        u.w.shape[2]       = 1;   u.w.shape[3]       = 1;
        u.w.padded_shape[0] = n;  u.w.padded_shape[1] = k;
        u.w.padded_shape[2] = 1;  u.w.padded_shape[3] = 1;
        u.w.ndim           = 2;
        u.w.qdata          = u.buf.d;
        u.w.qhigh          = nullptr;
        u.w.high_plane_bytes = 0;
        u.w.scales         = static_cast<std::uint8_t*>(u.buf.d) + g.code_bytes;
        u.w.n              = n;   u.w.k              = k;
        u.w.group          = k;   u.w.group_size     = static_cast<std::uint32_t>(k);
        u.w.layout         = QuantLayout::RowScale;
        u.w.scale_dtype    = DType::BF16;
        u.w.scale_ne[0]    = n;   u.w.scale_ne[1]    = 1;
        u.w.scale_ne[2]    = 1;   u.w.scale_ne[3]    = 1;
        const std::int64_t ss = static_cast<std::int64_t>(n) * 2;
        u.w.scale_nb[0]    = 2;   u.w.scale_nb[1]    = ss;
        u.w.scale_nb[2]    = ss;  u.w.scale_nb[3]    = ss;
        return u;
    }
    throw std::runtime_error(ref.name + ": unsupported image weight representation");
}

inline Uploaded upload_ref(const artifact::Reader& reader, const artifact::ParameterReference& ref) {
    return upload_streamed(reader, ref, nullptr);
}

} // namespace infernix::models::qwen_image
