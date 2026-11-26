#include "ops/rope_c64/launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// Channel-first layout: a head row is the 128 channels of one (head, token), the 64 complex pairs
// interleaved. One thread handles one pair of one head of one token; the pair partner is the next
// channel. q and k share the table; the pointer is chosen by blockIdx.y (0 = q, 1 = k).
__global__ __launch_bounds__(256) void rope_c64_kernel(
    const __nv_bfloat16* __restrict__ rot, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k, int heads, int pairs) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const int j      = pair & 63;                 // complex pair index within a head
    const int head   = (pair >> 6) % heads;       // 64 pairs per head
    const int token  = pair / (heads * 64);
    const bool query = (blockIdx.y == 0);
    __nv_bfloat16* __restrict__ data = query ? q : k;

    const int channel = 2 * j;
    const int base    = channel + 128 * (head + heads * token);
    const int rbase   = channel + 128 * token;

    const float x0 = __bfloat162float(data[base]);
    const float x1 = __bfloat162float(data[base + 1]);
    const float c0 = __bfloat162float(rot[rbase]);
    const float c1 = __bfloat162float(rot[rbase + 1]);

    data[base]     = __float2bfloat16_rn(x0 * c0 - x1 * c1);
    data[base + 1] = __float2bfloat16_rn(x0 * c1 + x1 * c0);
}

} // namespace

void rope_c64_launch(const Tensor& rot, Tensor* q, Tensor* k, std::int32_t heads,
                     std::int32_t tokens, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(tokens) * heads * 64;  // complex pairs
    const int threads = 256;
    const unsigned grid = static_cast<unsigned>((total + threads - 1) / threads);
    const auto* table = static_cast<const __nv_bfloat16*>(rot.data);
    if (q != nullptr && k != nullptr) {
        rope_c64_kernel<<<dim3(grid, 2), threads, 0, stream>>>(
            table, static_cast<__nv_bfloat16*>(q->data), static_cast<__nv_bfloat16*>(k->data),
            heads, static_cast<int>(total));
    } else {
        __nv_bfloat16* x = static_cast<__nv_bfloat16*>((q != nullptr ? q : k)->data);
        rope_c64_kernel<<<dim3(grid, 1), threads, 0, stream>>>(table, x, nullptr, heads,
                                                               static_cast<int>(total));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
