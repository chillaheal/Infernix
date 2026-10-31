#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// `image_ids` is null for the prefix rule, or a device I32 [S,1,1,1] of image block ids (-1 on text
// rows) for the block-causal edit form; see launch.cu.
void attn_dit_joint_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                           std::int32_t text_prefix, const Tensor* image_ids, Tensor* out,
                           std::int32_t heads, std::int32_t tokens, cudaStream_t stream);

} // namespace infernix::ops::detail
