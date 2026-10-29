#include "infernix/ops/dit_adaln.h"

#include "ops/dit_adaln/launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kMaxSeq = 20480;
constexpr std::int32_t kMaxDim = 65536;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool shape_is(const Tensor& tensor, std::int32_t a, std::int32_t b) {
    return tensor.ne[0] == a && tensor.ne[1] == b && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

void require_activation(const Tensor& tensor, std::int32_t dim, std::int32_t tokens,
                        const char* label) {
    if (tensor.dtype != DType::BF16 || !shape_is(tensor, dim, tokens) || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 4)) {
        throw std::invalid_argument(std::string("dit_adaln: invalid activation ") + label);
    }
}

std::int32_t require_mod(const Tensor& mod, std::int32_t dim, std::int32_t col_off) {
    const std::int32_t stride = mod.ne[0];
    if (mod.dtype != DType::BF16 || mod.ne[1] != 2 || mod.ne[2] != 1 || mod.ne[3] != 1 ||
        stride < 1 || !mod.is_contiguous() || !aligned_to(mod.data, 4)) {
        throw std::invalid_argument("dit_adaln: invalid mod [M,2,1,1]");
    }
    if (col_off < 0 || col_off + dim > stride) {
        throw std::invalid_argument("dit_adaln: col_off slice exceeds mod rows");
    }
    return stride;
}

void require_mask(const Tensor& mask, std::int32_t tokens, const char* label) {
    // One byte per token, on the device, and nothing but: the kernel indexes it with the token
    // index alone, so a shape that is not exactly [S,1,1,1] would silently read past its end.
    if (mask.dtype != DType::U8 || mask.ne[0] != tokens || mask.ne[1] != 1 || mask.ne[2] != 1 ||
        mask.ne[3] != 1 || !mask.is_contiguous() || mask.data == nullptr) {
        throw std::invalid_argument(std::string("dit_adaln: invalid target mask ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_begin  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data);
    return first_begin < second_begin + second.bytes() &&
           second_begin < first_begin + first.bytes();
}

void require_geometry(std::int32_t dim, std::int32_t tokens, std::int32_t text_prefix) {
    if (dim < 1 || dim > kMaxDim) {
        throw std::invalid_argument("dit_adaln: D must be 1..65536");
    }
    if (tokens < 1 || tokens > kMaxSeq) {
        throw std::invalid_argument("dit_adaln: S must be 1..20480");
    }
    if (text_prefix < 0 || text_prefix > tokens) {
        throw std::invalid_argument("dit_adaln: text_prefix must be 0..S");
    }
}

}  // namespace

void adaln_scale(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                 std::int32_t text_prefix, Tensor& out, cudaStream_t stream) {
    const std::int32_t dim = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    require_geometry(dim, tokens, text_prefix);
    require_activation(x, dim, tokens, "x");
    require_activation(out, dim, tokens, "out");
    require_mod(mod, dim, col_off);
    if (overlaps(out, x) || overlaps(out, mod)) {
        throw std::invalid_argument("dit_adaln: adaln_scale out overlaps an input");
    }
    detail::adaln_scale_launch(x, mod, col_off, text_prefix, nullptr, &out, tokens, stream);
}

void adaln_scale_masked(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                        const Tensor& target_mask, Tensor& out, cudaStream_t stream) {
    const std::int32_t dim = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    require_geometry(dim, tokens, 0);
    require_activation(x, dim, tokens, "x");
    require_activation(out, dim, tokens, "out");
    require_mod(mod, dim, col_off);
    require_mask(target_mask, tokens, "target_mask");
    if (overlaps(out, x) || overlaps(out, mod) || overlaps(out, target_mask)) {
        throw std::invalid_argument("dit_adaln: adaln_scale_masked out overlaps an input");
    }
    detail::adaln_scale_launch(x, mod, col_off, 0, &target_mask, &out, tokens, stream);
}

void adaln_gate_add(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                    std::int32_t col_off, std::int32_t text_prefix, Tensor& out,
                    cudaStream_t stream) {
    const std::int32_t dim = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    require_geometry(dim, tokens, text_prefix);
    require_activation(residual, dim, tokens, "residual");
    require_activation(branch, dim, tokens, "branch");
    require_activation(out, dim, tokens, "out");
    require_mod(mod, dim, col_off);
    if (overlaps(residual, branch) || overlaps(out, residual) || overlaps(out, branch) ||
        overlaps(out, mod)) {
        throw std::invalid_argument("dit_adaln: adaln_gate_add operands overlap");
    }
    detail::adaln_gate_add_launch(residual, branch, mod, col_off, text_prefix, nullptr, &out, tokens,
                                  stream);
}

void adaln_gate_add_masked(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                           std::int32_t col_off, const Tensor& target_mask, Tensor& out,
                           cudaStream_t stream) {
    const std::int32_t dim = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    require_geometry(dim, tokens, 0);
    require_activation(residual, dim, tokens, "residual");
    require_activation(branch, dim, tokens, "branch");
    require_activation(out, dim, tokens, "out");
    require_mod(mod, dim, col_off);
    require_mask(target_mask, tokens, "target_mask");
    if (overlaps(residual, branch) || overlaps(out, residual) || overlaps(out, branch) ||
        overlaps(out, mod) || overlaps(out, target_mask)) {
        throw std::invalid_argument("dit_adaln: adaln_gate_add_masked operands overlap");
    }
    detail::adaln_gate_add_launch(residual, branch, mod, col_off, 0, &target_mask, &out, tokens,
                                  stream);
}

}  // namespace infernix::ops
