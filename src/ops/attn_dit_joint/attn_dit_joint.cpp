#include "infernix/ops/attn_dit_joint.h"

#include "ops/attn_dit_joint/launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kHeadDim   = 128;
constexpr std::int32_t kMaxHeads  = 40;
constexpr std::int32_t kMaxTokens = 20480;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool head_shape(const Tensor& tensor, const std::int32_t heads, const std::int32_t tokens) {
    const std::array<std::int32_t, 4> shape = {kHeadDim, heads, tokens, 1};
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        if (tensor.ne[dim] != shape[dim]) return false;
    }
    return tensor.dtype == DType::BF16 && tensor.is_contiguous() && aligned_to(tensor.data, 4);
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_begin  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data);
    return first_begin < second_begin + second.bytes() &&
           second_begin < first_begin + first.bytes();
}

} // namespace

void attn_dit_joint(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t text_prefix,
                    Tensor& out, cudaStream_t stream) {
    const std::int32_t heads  = q.ne[1];
    const std::int32_t tokens = q.ne[2];
    if (heads < 1 || heads > kMaxHeads) {
        throw std::invalid_argument("attn_dit_joint: heads must be 1..40");
    }
    if (tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("attn_dit_joint: S must be 1..20480");
    }
    if (text_prefix < 0 || text_prefix > tokens) {
        throw std::invalid_argument("attn_dit_joint: text_prefix must be 0..S");
    }
    if (!head_shape(q, heads, tokens)) {
        throw std::invalid_argument("attn_dit_joint: invalid q");
    }
    if (!head_shape(k, heads, tokens) || !head_shape(v, heads, tokens) ||
        !head_shape(out, heads, tokens)) {
        throw std::invalid_argument("attn_dit_joint: k, v and out must match q's shape and dtype");
    }
    if (overlaps(out, q) || overlaps(out, k) || overlaps(out, v)) {
        throw std::invalid_argument("attn_dit_joint: out overlaps an input");
    }
    detail::attn_dit_joint_launch(q, k, v, text_prefix, &out, heads, tokens, stream);
}

} // namespace infernix::ops
