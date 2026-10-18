#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void adaln_scale_launch(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                        std::int32_t text_prefix, Tensor* out, std::int32_t tokens,
                        cudaStream_t stream);

void adaln_gate_add_launch(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                           std::int32_t col_off, std::int32_t text_prefix, Tensor* out,
                           std::int32_t tokens, cudaStream_t stream);

}  // namespace infernix::ops::detail
