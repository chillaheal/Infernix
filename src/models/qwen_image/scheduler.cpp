#include "models/qwen_image/scheduler.h"

#include <cmath>

namespace infernix::models::qwen_image {

double calculate_shift(const SchedulerConfig& c, std::size_t image_seq_len) noexcept {
    const double m = (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len);
    const double b = c.base_shift - m * c.base_image_seq_len;
    return static_cast<double>(image_seq_len) * m + b;
}

Schedule build_schedule(const SchedulerConfig& c, std::size_t steps, double mu) {
    // Input sigmas = linspace(1.0, 1/steps, steps); the value is computed in double and cast to the
    // float32 pipeline boundary (set_timesteps runs in float32).
    std::vector<float> sig_in(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        const double t = 1.0 + (1.0 / static_cast<double>(steps) - 1.0) *
                              static_cast<double>(i) / static_cast<double>(steps - 1);
        sig_in[i] = static_cast<float>(t);
    }

    // Dynamic exponential time shift, elementwise in float32: sigma -> exp(mu) / (exp(mu) + (1/sigma - 1)).
    const float emu = static_cast<float>(std::exp(mu));
    std::vector<float> shifted(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        const float t = sig_in[i];
        const float inv = 1.0F / t - 1.0F;
        shifted[i] = emu / (emu + inv);
    }

    // Stretch to the terminal (shift_terminal): rescale (1 - sigma) so the last entry reaches the
    // terminal, then append the terminal 0.0 sigma (invert_sigmas is off).
    const float one_minus_last = 1.0F - shifted[steps - 1];
    const float scale = one_minus_last / (1.0F - static_cast<float>(c.shift_terminal));
    std::vector<float> sigmas(steps + 1, 0.0F);
    for (std::size_t i = 0; i < steps; ++i) sigmas[i] = 1.0F - ((1.0F - shifted[i]) / scale);
    sigmas[steps] = 0.0F;

    std::vector<float> timesteps(steps);
    for (std::size_t i = 0; i < steps; ++i) timesteps[i] = sigmas[i] * static_cast<float>(c.num_train_timesteps);

    return Schedule{std::move(sigmas), std::move(timesteps)};
}

} // namespace infernix::models::qwen_image
