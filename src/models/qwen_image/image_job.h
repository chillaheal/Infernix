#pragma once

// The caller-facing numeric inputs of one Qwen-Image-2.1 image job and the resolution-derived VAE
// latent geometry (docs/maintainer/qwen-image-2-1-variables.md). The geometry's token count drives
// the resolution-dependent scheduler shift (calculate_shift). Header-only, all inline.

#include <cstdint>

#include "models/qwen_image/scheduler.h"

namespace infernix::models::qwen_image {

// VAE latent geometry, fixed for Qwen-Image-2.1: a 16x pixel-to-latent downscale with 2x2 latent
// packing, so a pixel side must be a multiple of 32. The pixel dims are snapped down to it.
struct ImageGeometry {
    std::uint32_t height = 0; // snapped pixel height
    std::uint32_t width  = 0; // snapped pixel width
    std::uint32_t hl     = 0; // latent height = height / 16
    std::uint32_t wl     = 0; // latent width  = width  / 16
    std::uint32_t tokens = 0; // target latent tokens = hl * wl (drives the scheduler shift)

    [[nodiscard]] static constexpr std::uint32_t multiple_of() noexcept { return 32u; }
    [[nodiscard]] static constexpr std::uint32_t downscale() noexcept { return 16u; }

    [[nodiscard]] static ImageGeometry derive(std::uint32_t height_px, std::uint32_t width_px) noexcept {
        ImageGeometry g;
        g.height = height_px / multiple_of() * multiple_of();
        g.width  = width_px / multiple_of() * multiple_of();
        g.hl     = g.height / downscale();
        g.wl     = g.width / downscale();
        g.tokens = g.hl * g.wl;
        return g;
    }
};

// The caller-facing numeric inputs of one image job, served as `size` and `steps`. The defaults are the
// fixed 512x512 oracle geometry the in-process interlude reproduces. Everything else a job takes
// (prompt and negative prompt embeds, seed, reference image, true CFG) is a non-numeric or
// device-produced input and arrives through ImageJobInputs.
struct ImageJob {
    std::uint32_t height = 512;
    std::uint32_t width  = 512;
    std::uint32_t steps  = 40; // num_inference_steps == the number of DiT forwards

    [[nodiscard]] ImageGeometry geometry() const noexcept { return ImageGeometry::derive(height, width); }

    // The resolution-dependent denoise schedule for this job (mu from the latent token count).
    [[nodiscard]] Schedule schedule(const SchedulerConfig& cfg) const {
        const ImageGeometry g = geometry();
        return build_schedule(cfg, steps, calculate_shift(cfg, g.tokens));
    }
};

} // namespace infernix::models::qwen_image
