#pragma once

// Qwen-Image-2.1 denoise scheduler (FlowMatchEulerDiscrete), the host-side math only. The
// resolution-dependent shift (mu) and the sigma/timestep schedule are recomputed per job from the
// latent token count; the per-step Euler update is applied by the CUDA execution. The constants are
// the model's fixed scheduler config (artifact/scheduler/scheduler_config.json), not a per-job input.
// docs/maintainer/qwen-image-2-1-variables.md.

#include <cstddef>
#include <vector>

namespace infernix::models::qwen_image {

// The scheduler constants the supported path uses. The model config is exponential dynamic
// shifting with a terminal stretch and no Karras / beta / exponential-sigma / inverse variants.
struct SchedulerConfig {
    double base_image_seq_len  = 256.0;
    double max_image_seq_len   = 8192.0;
    double base_shift          = 0.5;
    double max_shift           = 0.9;
    double num_train_timesteps = 1000.0;
    double shift_terminal      = 0.02;
};

// The denoise schedule for one job. sigmas holds steps + 1 entries (a terminal 0.0 is appended);
// timesteps holds steps entries. Both are float32 (the pipeline boundary precision).
struct Schedule {
    std::vector<float> sigmas;    // [steps + 1]
    std::vector<float> timesteps; // [steps]
};

// mu = image_seq_len * m + b, the linear shift of the schedule in the latent token count, where
// m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len) and
// b = base_shift - m * base_image_seq_len.
[[nodiscard]] double calculate_shift(const SchedulerConfig& c, std::size_t image_seq_len) noexcept;

// Build the schedule for one job. The input sigmas are linspace(1.0, 1/steps, steps) in float32,
// rescaled elementwise by the exponential time shift at mu, stretched to the terminal
// (shift_terminal), and then timesteps = sigmas * num_train_timesteps with a terminal 0.0 sigma
// appended. A job's mu comes from calculate_shift on its geometry's token count.
[[nodiscard]] Schedule build_schedule(const SchedulerConfig& c, std::size_t steps, double mu);

} // namespace infernix::models::qwen_image
