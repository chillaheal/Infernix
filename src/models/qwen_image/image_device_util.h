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

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
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

// The representation of one image weight object, flattened from the artifact geometry so the ONE
// place that fills the device Weight ABI (fill_uploaded_weight) and the one that sizes a device
// region (object_device_bytes) never drift apart. The synchronous upload and the bundle-B fas 2
// copy-stream upload both build from here, so the bytes on the device -- and therefore every kernel
// that reads them -- are identical across the two routes.
struct ObjectGeom {
    QType      format;
    QuantLayout layout;
    std::uint64_t bytes = 0;        // the stored object size
    std::uint64_t n     = 0;        // rows
    std::uint64_t k     = 0;        // cols
    std::uint64_t high_bytes  = 0;
    std::uint64_t code_bytes  = 0;
    std::uint64_t scale_bytes = 0;
    std::uint64_t scale_offset = 0;
};

inline ObjectGeom object_geom(const WeightGeometry& g) {
    ObjectGeom o;
    o.format        = g.format;
    o.layout        = g.layout;
    o.bytes         = g.bytes;
    o.high_bytes    = g.high_bytes;
    o.code_bytes    = g.code_bytes;
    o.scale_bytes   = g.scale_bytes;
    o.scale_offset  = g.scale_offset;
    o.n             = g.shape.empty() ? 0 : g.shape[0];
    o.k             = (o.n == 0 || g.elements % o.n != 0) ? 0 : g.elements / o.n;
    return o;
}

// Device bytes this object occupies after upload: a BF16 object is stored as-is; a RowScale FP8
// object is repacked to a compact [codes][scales] plane (the artifact may carry a gap the device
// drops). The async copy lays the nine block roles out with this stride.
inline std::size_t object_device_bytes(const ObjectGeom& g) {
    if (g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) return g.bytes;
    if (g.format == QType::FP8_E4M3FN_ROW_BF16 && g.layout == QuantLayout::RowScale)
        return static_cast<std::size_t>(g.code_bytes + g.scale_bytes);
    throw std::runtime_error("image weight: unsupported object representation");
}

// Fill the Weight the qualified linear routes read: a contiguous BF16 object as a Contiguous
// Weight, a RowScale FP8 object as the single [codes][scales] plane. `dptr` addresses the object's
// device payload (already uploaded). Both upload routes call this, so the ABI has one definition.
inline void fill_uploaded_weight(Weight& w, const ObjectGeom& g, void* dptr) {
    if (g.n == 0 || g.k == 0) {
        throw std::runtime_error("image weight: geometry does not frame as [rows, cols]");
    }
    const std::uint64_t n = g.n, k = g.k;
    if (g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) {
        w                = Weight{};
        w.payload        = dptr;
        w.payload_bytes  = g.bytes;
        w.qtype          = QType::BF16;
        w.shape[0]       = n;   w.shape[1]       = k;
        w.shape[2]       = 1;   w.shape[3]       = 1;
        w.padded_shape[0] = n;  w.padded_shape[1] = k;
        w.padded_shape[2] = 1;  w.padded_shape[3] = 1;
        w.ndim           = 2;
        w.qdata          = dptr;
        w.n              = n;   w.k              = k;
        w.layout         = QuantLayout::Contiguous;
        return;
    }
    if (g.format == QType::FP8_E4M3FN_ROW_BF16 && g.layout == QuantLayout::RowScale) {
        if (g.high_bytes != 0 || g.code_bytes != n * k || g.scale_bytes != n * 2 ||
            g.scale_offset < g.code_bytes) {
            throw std::runtime_error("image weight: FP8 object is not a plain [codes][scales] plane");
        }
        w                = Weight{};
        w.payload        = dptr;
        w.payload_bytes  = g.code_bytes + g.scale_bytes;
        w.qtype          = QType::FP8_E4M3FN_ROW_BF16;
        w.shape[0]       = n;   w.shape[1]       = k;
        w.shape[2]       = 1;   w.shape[3]       = 1;
        w.padded_shape[0] = n;  w.padded_shape[1] = k;
        w.padded_shape[2] = 1;  w.padded_shape[3] = 1;
        w.ndim           = 2;
        w.qdata          = dptr;
        w.qhigh          = nullptr;
        w.high_plane_bytes = 0;
        w.scales         = static_cast<std::uint8_t*>(dptr) + g.code_bytes;
        w.n              = n;   w.k              = k;
        w.group          = k;   w.group_size     = static_cast<std::uint32_t>(k);
        w.layout         = QuantLayout::RowScale;
        w.scale_dtype    = DType::BF16;
        w.scale_ne[0]    = n;   w.scale_ne[1]    = 1;
        w.scale_ne[2]    = 1;   w.scale_ne[3]    = 1;
        const std::int64_t ss = static_cast<std::int64_t>(n) * 2;
        w.scale_nb[0]    = 2;   w.scale_nb[1]    = ss;
        w.scale_nb[2]    = ss;  w.scale_nb[3]    = ss;
        return;
    }
    throw std::runtime_error("image weight: unsupported image weight representation");
}

// One streamed parameter uploaded to the device, wrapped the way the qualified linear routes read
// it. The bundle-B fas 2 streamed block weights do NOT use this (they land on the copy stream into a
// shared slot); it is the path for the resident top-level/VAE weights and any artifact-direct upload.
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
    const ObjectGeom g = object_geom(reader.geometry(ref.binding.parts.front().object));
    std::vector<std::byte> owned;
    const std::vector<std::byte>* obj = object_bytes;
    if (obj == nullptr) {
        owned   = reader.read_object(ref.binding.parts.front().object);
        obj     = &owned;
    }
    if (obj->size() != g.bytes) {
        throw std::runtime_error(ref.name + ": object size differs from its geometry");
    }

    Uploaded u;
    if ((g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) ||
        (g.format == QType::FP8_E4M3FN_ROW_BF16 && g.layout == QuantLayout::RowScale)) {
        const std::size_t total = object_device_bytes(g);
        u.buf.alloc(total);
        if (g.format == QType::BF16) {
            h2d(u.buf.d, obj->data(), static_cast<std::size_t>(g.bytes));
        } else {
            h2d(u.buf.d, obj->data(), static_cast<std::size_t>(g.code_bytes));
            h2d(static_cast<std::uint8_t*>(u.buf.d) + g.code_bytes, obj->data() + g.scale_offset,
                static_cast<std::size_t>(g.scale_bytes));
        }
        fill_uploaded_weight(u.w, g, u.buf.d);
        return u;
    }
    throw std::runtime_error(ref.name + ": unsupported image weight representation");
}

inline Uploaded upload_ref(const artifact::Reader& reader, const artifact::ParameterReference& ref) {
    return upload_streamed(reader, ref, nullptr);
}

// ---- bundle B fas 2: pinned host cache + copy-stream block upload ----

// The DiT block host cache is pinned so the copy stream's H2D is genuinely asynchronous (a pageable
// source forces a driver-side bounce that cannot overlap compute). cudaHostAllocWriteCombined is the
// measured variant: good for pure H2D, slow for CPU reads, so it is opt-in (INFERNIX_IMAGE_HOST_CACHE_WC)
// and kept only if the measurement says it wins.
inline std::uint32_t host_cache_alloc_flags() {
    const char* wc = std::getenv("INFERNIX_IMAGE_HOST_CACHE_WC");
    return (wc != nullptr && wc[0] != '\0' && wc[0] != '0') ? cudaHostAllocWriteCombined
                                                            : cudaHostAllocDefault;
}

// How many block-cache objects this job could NOT page-lock. A serving process that already
// page-locked most of RAM for its expert tier leaves too little for the cache, and that is the normal
// configuration rather than a shortage: the copy stream then stages through the driver (correct, just
// not overlapped), so the job says so in its timing line instead of failing the picture.
[[nodiscard]] inline std::atomic<std::size_t>& unpinned_cache_objects() {
    static std::atomic<std::size_t> count{0};
    return count;
}

struct PinnedBytes {
    std::byte*  h      = nullptr;
    std::size_t n      = 0;
    bool        pinned = false;

    void alloc(std::size_t b) {
        release();
        n = b;
        if (b == 0) return;
        if (cudaHostAlloc(reinterpret_cast<void**>(&h), b, host_cache_alloc_flags()) == cudaSuccess) {
            pinned = true;
            return;
        }
        // The refusal is handled here, but a failed runtime call also SETS the thread's last-error
        // flag, and the next cudaGetLastError() anywhere (an unrelated op's check) would report this
        // out-of-memory and abort the process for a condition that was already recovered. Consume it.
        (void)cudaGetLastError();
        h = static_cast<std::byte*>(::operator new[](b, std::nothrow));
        if (h == nullptr) {
            throw std::runtime_error("image block host cache: neither pinned nor plain memory is available");
        }
        ++unpinned_cache_objects();
    }
    void release() {
        if (h == nullptr) return;
        if (pinned) { cudaFreeHost(h); } else { ::operator delete[](h, std::nothrow); }
        h      = nullptr;
        pinned = false;
    }
    ~PinnedBytes() { release(); }
    PinnedBytes() = default;
    PinnedBytes(const PinnedBytes&) = delete;
    PinnedBytes& operator=(const PinnedBytes&) = delete;
    PinnedBytes(PinnedBytes&& o) noexcept : h(o.h), n(o.n), pinned(o.pinned) {
        o.h = nullptr; o.n = 0; o.pinned = false;
    }
    PinnedBytes& operator=(PinnedBytes&& o) noexcept {
        if (this != &o) {
            release();
            h = o.h; n = o.n; pinned = o.pinned;
            o.h = nullptr; o.n = 0; o.pinned = false;
        }
        return *this;
    }
};

// Read one object straight into a pinned host buffer (the block cache read, once per job).
inline PinnedBytes read_object_pinned(const artifact::Reader& reader,
                                      const artifact::ParameterReference& ref) {
    std::vector<std::byte> v = read_object(reader, ref);
    PinnedBytes p;
    p.alloc(v.size());
    if (!v.empty()) { std::memcpy(p.h, v.data(), v.size()); }
    return p;
}

// The nine DiT block roles in upload order, with their (uniform) geometry and slot offsets precomputed
// once per job from block 0 -- every block has the same shapes, so the same layout applies. The async
// copy uses these to lay the block out in a device slot and hand back the nine Weight operands.
struct BlockCopyPlan {
    std::array<artifact::ParameterReference, 9> refs;
    std::array<ObjectGeom, 9>                   geom;
    std::array<std::size_t, 9>                  offset;
    std::array<std::size_t, 9>                  dev_bytes;
    std::size_t                                 total = 0;
};

// Build the plan from one block's nine references (geometry is the same for every block).
inline BlockCopyPlan make_block_copy_plan(const artifact::Reader& reader,
                                          const artifact::ParameterReference refs[9]) {
    BlockCopyPlan plan;
    std::size_t off = 0;
    for (int j = 0; j < 9; ++j) {
        plan.refs[j] = refs[j];
        plan.geom[j] = object_geom(reader.geometry(refs[j].binding.parts.front().object));
        plan.offset[j]     = off;
        plan.dev_bytes[j]  = object_device_bytes(plan.geom[j]);
        off += plan.dev_bytes[j];
    }
    plan.total = off;
    return plan;
}

// Issue one block's nine objects into [slot_base, slot_base + plan.total) as asynchronous H2D on the
// copy stream, from the caller's pinned host cache rows, and fill w[9] to address the device copies.
// `host[j]` points at the stored object bytes for role j (its size is checked against the geometry).
inline void upload_block_async(cudaStream_t s, void* slot_base, const BlockCopyPlan& plan,
                               const std::byte* const host[9], Weight w[9]) {
    auto* base = static_cast<std::uint8_t*>(slot_base);
    for (int j = 0; j < 9; ++j) {
        const ObjectGeom&  g   = plan.geom[j];
        const std::byte*   src = host[j];
        std::uint8_t*      dst = base + plan.offset[j];
        if (g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) {
            cuda_ok(cudaMemcpyAsync(dst, src, g.bytes, cudaMemcpyHostToDevice, s),
                    "H2D async (image block BF16)");
        } else if (g.format == QType::FP8_E4M3FN_ROW_BF16 && g.layout == QuantLayout::RowScale) {
            cuda_ok(cudaMemcpyAsync(dst, src, g.code_bytes, cudaMemcpyHostToDevice, s),
                    "H2D async (image block FP8 codes)");
            cuda_ok(cudaMemcpyAsync(dst + g.code_bytes, src + g.scale_offset, g.scale_bytes,
                                    cudaMemcpyHostToDevice, s),
                    "H2D async (image block FP8 scales)");
        } else {
            throw std::runtime_error("image block: unsupported weight representation");
        }
        fill_uploaded_weight(w[j], g, dst);
    }
}

} // namespace infernix::models::qwen_image
