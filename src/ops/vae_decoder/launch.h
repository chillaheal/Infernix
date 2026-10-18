#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void vae_conv2d_launch(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor* residual,
                       const Tensor& out, cudaStream_t stream);

void vae_silu_launch(Tensor& x, cudaStream_t stream);

void vae_add_launch(const Tensor& a, const Tensor& b, const Tensor& out, cudaStream_t stream);

void vae_upsample2x_launch(const Tensor& x, const Tensor& out, cudaStream_t stream);

void vae_dupup3d_launch(const Tensor& x, std::int32_t factor_t, const Tensor& out, cudaStream_t stream);

void vae_spatial_attn_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* residual,
                             const Tensor& out, float scale, cudaStream_t stream);

} // namespace infernix::ops::detail
