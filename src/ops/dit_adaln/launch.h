#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// `target_mask` is null for the prefix form (row selected by text_prefix) or a device uint8
// [S,1,1,1] tensor for the edit route's per-token selection; see launch.cu for the semantics.
void adaln_scale_launch(const Tensor& x, const Tensor& mod, std::int32_t col_off,
                        std::int32_t text_prefix, const Tensor* target_mask, Tensor* out,
                        std::int32_t tokens, cudaStream_t stream);

void adaln_gate_add_launch(const Tensor& residual, const Tensor& branch, const Tensor& mod,
                           std::int32_t col_off, std::int32_t text_prefix,
                           const Tensor* target_mask, Tensor* out, std::int32_t tokens,
                           cudaStream_t stream);

}  // namespace infernix::ops::detail
