#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void rope_c64_launch(const Tensor& rot, Tensor* q, Tensor* k, std::int32_t heads,
                     std::int32_t tokens, cudaStream_t stream);

} // namespace infernix::ops::detail
