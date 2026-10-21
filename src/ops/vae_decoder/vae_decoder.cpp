#include "infernix/ops/vae_decoder.h"

#include "ops/vae_decoder/launch.h"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

bool is_bf16_contiguous(const Tensor& t) {
    return t.dtype == DType::BF16 && t.is_contiguous() && t.data != nullptr;
}

bool shape3(const Tensor& t, int c, int h, int w) {
    return is_bf16_contiguous(t) && t.ne[0] == c && t.ne[1] == h && t.ne[2] == w;
}

bool shape2(const Tensor& t, int a, int b) {
    return is_bf16_contiguous(t) && t.ne[0] == a && t.ne[1] == b;
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto fb = reinterpret_cast<std::uintptr_t>(first.data);
    const auto sb = reinterpret_cast<std::uintptr_t>(second.data);
    return fb < sb + second.bytes() && sb < fb + first.bytes();
}

} // namespace

void conv2d(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor* residual, Tensor& out,
            cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2];
    const int Cout = w.ne[0];
    if (!shape3(x, Cin, H, W)) throw std::invalid_argument("conv2d: invalid x");
    if (!is_bf16_contiguous(w) || w.ne[1] != Cin || w.ne[2] != 3 || w.ne[3] != 3)
        throw std::invalid_argument("conv2d: invalid w");
    if (!is_bf16_contiguous(b) || b.ne[0] != Cout) throw std::invalid_argument("conv2d: invalid b");
    if (!shape3(out, Cout, H, W)) throw std::invalid_argument("conv2d: invalid out");
    if (residual != nullptr && !shape3(*residual, Cout, H, W))
        throw std::invalid_argument("conv2d: invalid residual");
    if (overlaps(out, x) || overlaps(out, w) || overlaps(out, b))
        throw std::invalid_argument("conv2d: out overlaps an input");
    if (residual != nullptr && overlaps(out, *residual))
        throw std::invalid_argument("conv2d: out overlaps residual");
    detail::vae_conv2d_launch(x, w, b, residual, out, stream);
}

void conv2d_stride2(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& out,
                    cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2];
    const int Cout = w.ne[0];
    if (!shape3(x, Cin, H, W) || H < 2 || W < 2) throw std::invalid_argument("conv2d_stride2: invalid x");
    if (!is_bf16_contiguous(w) || w.ne[1] != Cin || w.ne[2] != 3 || w.ne[3] != 3)
        throw std::invalid_argument("conv2d_stride2: invalid w");
    if (!is_bf16_contiguous(b) || b.ne[0] != Cout) throw std::invalid_argument("conv2d_stride2: invalid b");
    const int H2 = (H - 2) / 2 + 1, W2 = (W - 2) / 2 + 1;
    if (!shape3(out, Cout, H2, W2)) throw std::invalid_argument("conv2d_stride2: invalid out");
    if (overlaps(out, x) || overlaps(out, w) || overlaps(out, b))
        throw std::invalid_argument("conv2d_stride2: out overlaps an input");
    detail::vae_conv2d_stride2_launch(x, w, b, out, stream);
}

void avg_down(const Tensor& x, std::int32_t factor_t, Tensor& out, cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2];
    const int Cout = out.ne[0];
    if (!shape3(x, Cin, H, W) || H % 2 != 0 || W % 2 != 0)
        throw std::invalid_argument("avg_down: invalid x (H,W must be even)");
    if (factor_t != 1 && factor_t != 2) throw std::invalid_argument("avg_down: factor_t must be 1 or 2");
    if (static_cast<std::int64_t>(Cin) * factor_t * 4 % Cout != 0)
        throw std::invalid_argument("avg_down: Cin*factor not divisible by Cout");
    if (!shape3(out, Cout, H / 2, W / 2)) throw std::invalid_argument("avg_down: invalid out");
    if (overlaps(out, x)) throw std::invalid_argument("avg_down: out overlaps x");
    detail::vae_avg_down_launch(x, factor_t, out, stream);
}

void silu(Tensor& x, cudaStream_t stream) {
    if (!is_bf16_contiguous(x)) throw std::invalid_argument("silu: invalid x");
    detail::vae_silu_launch(x, stream);
}

void add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    const std::int64_t na = a.numel();
    if (!is_bf16_contiguous(a) || !is_bf16_contiguous(b) || !is_bf16_contiguous(out))
        throw std::invalid_argument("add: invalid tensors");
    if (b.numel() != na || out.numel() != na) throw std::invalid_argument("add: shape mismatch");
    if (overlaps(out, a) || overlaps(out, b)) throw std::invalid_argument("add: out overlaps an input");
    detail::vae_add_launch(a, b, out, stream);
}

void upsample_nearest_2x(const Tensor& x, Tensor& out, cudaStream_t stream) {
    const int C = x.ne[0], H = x.ne[1], W = x.ne[2];
    if (!shape3(x, C, H, W)) throw std::invalid_argument("upsample_nearest_2x: invalid x");
    if (!shape3(out, C, 2 * H, 2 * W)) throw std::invalid_argument("upsample_nearest_2x: invalid out");
    if (overlaps(out, x)) throw std::invalid_argument("upsample_nearest_2x: out overlaps x");
    detail::vae_upsample2x_launch(x, out, stream);
}

void dupup3d(const Tensor& x, std::int32_t factor_t, Tensor& out, cudaStream_t stream) {
    const int Cin = x.ne[0], H = x.ne[1], W = x.ne[2];
    const int Cout = out.ne[0];
    if (!shape3(x, Cin, H, W)) throw std::invalid_argument("dupup3d: invalid x");
    if (!shape3(out, Cout, 2 * H, 2 * W)) throw std::invalid_argument("dupup3d: invalid out");
    if (factor_t != 1 && factor_t != 2) throw std::invalid_argument("dupup3d: factor_t must be 1 or 2");
    if ((static_cast<std::int64_t>(Cout) * static_cast<std::int64_t>(factor_t) * 4) % Cin != 0)
        throw std::invalid_argument("dupup3d: Cout*factor not divisible by Cin");
    if (overlaps(out, x)) throw std::invalid_argument("dupup3d: out overlaps x");
    detail::vae_dupup3d_launch(x, factor_t, out, stream);
}

void spatial_attn(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* residual, float scale,
                  Tensor& out, cudaStream_t stream) {
    const int C = q.ne[0], N = q.ne[1];
    if (!shape2(q, C, N) || !shape2(k, C, N) || !shape2(v, C, N) || !shape2(out, C, N))
        throw std::invalid_argument("spatial_attn: invalid q/k/v/out");
    if (N < 1 || N > 65536) throw std::invalid_argument("spatial_attn: N must be 1..65536");
    if (C < 1 || 2 * C + 256 > 12032)
        throw std::invalid_argument("spatial_attn: C must be 1..5888");
    if (residual != nullptr && !shape2(*residual, C, N))
        throw std::invalid_argument("spatial_attn: invalid residual");
    if (!(std::isfinite(scale) && scale > 0.0F)) throw std::invalid_argument("spatial_attn: invalid scale");
    if (overlaps(out, q) || overlaps(out, k) || overlaps(out, v))
        throw std::invalid_argument("spatial_attn: out overlaps an input");
    if (residual != nullptr && overlaps(out, *residual))
        throw std::invalid_argument("spatial_attn: out overlaps residual");
    detail::vae_spatial_attn_launch(q, k, v, residual, out, scale, stream);
}

} // namespace infernix::ops
