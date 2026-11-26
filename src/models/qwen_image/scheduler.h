#pragma once

// Qwen-Image-2.1 denoise scheduler (FlowMatchEulerDiscrete), the host-side math only. The
// resolution-dependent shift (mu) and the sigma/timestep schedule are recomputed per job from the
// latent token count; the per-step Euler update is applied by the CUDA execution. Nothing here
// carries a model constant: the six values are read from the artifact by read_scheduler_config
// (models/qwen_image/artifact_constants.h) and arrive as a SchedulerConstants, so this module cannot
// draw a job with a schedule it never read. docs/maintainer/qwen-image-2-1-variables.md.

#include "models/qwen_image/artifact_constants.h"

#include <cstddef>
#include <vector>

namespace infernix::models::qwen_image {

// The denoise schedule for one job. sigmas holds steps + 1 entries (a terminal 0.0 is appended);
// timesteps holds steps entries. Both are float32 (the pipeline boundary precision).
struct Schedule {
    std::vector<float> sigmas;    // [steps + 1]
    std::vector<float> timesteps; // [steps]
};

// mu = image_seq_len * m + b, the linear shift of the schedule in the latent token count, where
// m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len) and
// b = base_shift - m * base_image_seq_len.
[[nodiscard]] double calculate_shift(const SchedulerConstants& c, std::size_t image_seq_len) noexcept;

// Build the schedule for one job. The input sigmas are linspace(1.0, 1/steps, steps) in float32,
// rescaled elementwise by the exponential time shift at mu, stretched to the terminal
// (shift_terminal), and then timesteps = sigmas * num_train_timesteps with a terminal 0.0 sigma
// appended. A job's mu comes from calculate_shift on its geometry's token count.
//
// The arithmetic is unchanged from the float version it replaces: the token counts and the timestep
// count arrive as JSON integers and are promoted to double in exactly the positions the old doubles
// occupied, so every bit of the certified schedule is the same.
[[nodiscard]] Schedule build_schedule(const SchedulerConstants& c, std::size_t steps, double mu);

} // namespace infernix::models::qwen_image
