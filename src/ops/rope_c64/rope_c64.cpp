#include "infernix/ops/rope_c64.h"

#include "ops/rope_c64/launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kHeadDim   = 128;  // 64 complex pairs
constexpr std::int32_t kHeads     = 32;   // DiT full (non-GQA) attention
constexpr std::int32_t kMaxHeads  = 40;
constexpr std::int32_t kMaxTokens = 20480;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_head_tensor(const Tensor& tensor, const std::int32_t heads, const std::int32_t tokens,
                         const char* label) {
    const std::array<std::int32_t, 4> shape = {kHeadDim, heads, tokens, 1};
    bool shape_matches = true;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        shape_matches = shape_matches && tensor.ne[dim] == shape[dim];
    }
    if (tensor.dtype != DType::BF16 || !shape_matches || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 4)) {
        throw std::invalid_argument(std::string("rope_c64: invalid ") + label);
    }
}

void require_rot(const Tensor& rot, const std::int32_t tokens) {
    const std::array<std::int32_t, 4> shape = {kHeadDim, 1, tokens, 1};
    bool shape_matches = true;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        shape_matches = shape_matches && rot.ne[dim] == shape[dim];
    }
    if (rot.dtype != DType::BF16 || !shape_matches || !rot.is_contiguous() ||
        !aligned_to(rot.data, 4)) {
        throw std::invalid_argument("rope_c64: invalid rot");
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_begin  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data);
    return first_begin < second_begin + second.bytes() &&
           second_begin < first_begin + first.bytes();
}

void require_geometry(const Tensor& rot, const Tensor& q, const Tensor* k) {
    const std::int32_t heads = q.ne[1];
    const std::int32_t tokens = q.ne[2];
    if (heads < 1 || heads > kMaxHeads) {
        throw std::invalid_argument("rope_c64: heads must be 1..40");
    }
    if (tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("rope_c64: S must be 1..20480");
    }
    require_rot(rot, tokens);
    require_head_tensor(q, heads, tokens, "q");
    if (overlaps(rot, q)) {
        throw std::invalid_argument("rope_c64: q overlaps rot");
    }
    if (k != nullptr) {
        require_head_tensor(*k, heads, tokens, "k");
        if (overlaps(rot, *k) || overlaps(q, *k)) {
            throw std::invalid_argument("rope_c64: k overlaps another operand");
        }
    }
}

} // namespace

void rope_c64(const Tensor& rot, Tensor& q, Tensor& k, cudaStream_t stream) {
    require_geometry(rot, q, &k);
    if (k.ne[1] != q.ne[1]) {
        throw std::invalid_argument("rope_c64: pair q and k must share head count");
    }
    detail::rope_c64_launch(rot, &q, &k, q.ne[1], q.ne[2], stream);
}

void rope_c64(const Tensor& rot, Tensor& x, cudaStream_t stream) {
    require_geometry(rot, x, nullptr);
    detail::rope_c64_launch(rot, &x, nullptr, x.ne[1], x.ne[2], stream);
}

} // namespace infernix::ops
