#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void attn_dit_joint_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                           std::int32_t text_prefix, Tensor* out, std::int32_t heads,
                           std::int32_t tokens, cudaStream_t stream);

} // namespace infernix::ops::detail
