// Resolution-grid invariants of the production FlowMatchEulerDiscrete schedule (infernix_qwen_image).
// Host-only, weights-free, always runs (no oracle, no GPU): guards the variable-resolution gap that
// the single 512x512 pipeline oracle (test_image_scheduler_prod) cannot cover. These are A-PRIOI
// properties of the correct scheduler for ANY (height, width, steps), derived from its definition
// rather than mirroring the code's lines:
//   - geometry: pixel dims snapped DOWN to multiple_of=32; latent hl/width/16; tokens = hl*wl
//   - mu = calculate_shift is strictly increasing in the latent token count (linear, positive slope)
//   - sigmas has steps+1 entries, timesteps steps; sigmas[0]=1.0 and sigmas[steps]=0.0 exactly
//   - the stretch-to-terminal forces sigmas[steps-1] == shift_terminal (the last denoise sigma)
//   - sigmas strictly decreasing and within [0,1]; timesteps[i] == sigmas[i]*num_train_timesteps
// A wrong scale (the stretch) or a wrong shift slope would break one of these; none is narrowed to pass.

#include "models/qwen_image/image_job.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace qi = infernix::models::qwen_image;

int main() {
    int failures = 0;
    auto fail = [&](const char* msg, const char* detail) {
        std::printf("  FAIL: %s (%s)\n", msg, detail);
        ++failures;
    };

    const qi::SchedulerConfig cfg; // the real Qwen-Image-2.1 scheduler constants

    // mu must be strictly increasing in the latent token count (the resolution-dependent shift input).
    {
        double prev_mu = 0.0;
        std::uint32_t prev_tokens = 0;
        for (std::uint32_t s : {32u, 64u, 128u, 256u}) {
            const qi::ImageGeometry g = qi::ImageGeometry::derive(s, s);
            const double mu = qi::calculate_shift(cfg, g.tokens);
            if (g.tokens <= prev_tokens || mu <= prev_mu) {
                char b[96];
                std::snprintf(b, sizeof b, "tokens=%u prev=%u mu=%.6f prev_mu=%.6f", g.tokens, prev_tokens, mu, prev_mu);
                fail("mu not strictly increasing in token count", b);
            }
            prev_mu = mu;
            prev_tokens = g.tokens;
        }
    }

    const std::vector<std::uint32_t> sides = {256, 320, 384, 512, 640, 768, 544, 1024, 300, 500, 700};
    const std::vector<std::uint32_t> stepsList = {10, 20, 40, 50};

    for (std::uint32_t h : sides) {
        for (std::uint32_t w : sides) {
            for (std::uint32_t steps : stepsList) {
                const qi::ImageGeometry g = qi::ImageGeometry::derive(h, w);

                char b[160];
                std::snprintf(b, sizeof b, "h=%u w=%u steps=%u", h, w, steps);

                if (g.height % qi::ImageGeometry::multiple_of() != 0 ||
                    g.width % qi::ImageGeometry::multiple_of() != 0) {
                    std::snprintf(b, sizeof b, "snapped dims h=%u w=%u not multiple_of 32", g.height, g.width);
                    fail("snapped pixel dims not multiple_of 32", b);
                }
                if (g.hl != g.height / 16u || g.wl != g.width / 16u || g.tokens != g.hl * g.wl) {
                    std::snprintf(b, sizeof b, "hl=%u wl=%u tokens=%u (h=%u w=%u)", g.hl, g.wl, g.tokens, g.height, g.width);
                    fail("latent geometry inconsistent", b);
                }
                if (g.tokens == 0) {
                    fail("zero latent tokens", b);
                    continue;
                }

                const qi::ImageJob job{h, w, steps};
                const qi::Schedule sched = job.schedule(cfg);
                const std::size_t n = steps; // sigmas len = n+1, timesteps len = n

                if (sched.sigmas.size() != n + 1 || sched.timesteps.size() != n) {
                    std::snprintf(b, sizeof b, "sigmas=%zu timesteps=%zu", sched.sigmas.size(), sched.timesteps.size());
                    fail("schedule length", b);
                    continue;
                }

                if (sched.sigmas[0] != 1.0f) {
                    std::snprintf(b, sizeof b, "sigmas[0]=%.8f", sched.sigmas[0]);
                    fail("sigmas[0] != 1.0", b);
                }
                if (sched.sigmas[n] != 0.0f) {
                    std::snprintf(b, sizeof b, "sigmas[steps]=%.8f", sched.sigmas[n]);
                    fail("sigmas[steps] != 0.0 (terminal)", b);
                }
                // stretch-to-terminal: the last denoise sigma equals shift_terminal.
                {
                    const double st = cfg.shift_terminal;
                    const double got = sched.sigmas[n - 1];
                    if (std::abs(got - st) / st > 1e-4) {
                        std::snprintf(b, sizeof b, "sigmas[steps-1]=%.6f want shift_terminal=%.4f", got, st);
                        fail("sigmas[steps-1] != shift_terminal", b);
                    }
                }
                // strictly decreasing and within [0,1].
                for (std::size_t i = 0; i < n + 1; ++i) {
                    const float s = sched.sigmas[i];
                    if (!(s >= 0.0f && s <= 1.0f)) {
                        std::snprintf(b, sizeof b, "sigma[%zu]=%.8f", i, s);
                        fail("sigma out of [0,1]", b);
                        break;
                    }
                    if (i > 0 && !(sched.sigmas[i - 1] > sched.sigmas[i])) {
                        std::snprintf(b, sizeof b, "sigma[%zu]=%.8f <= sigma[%zu]=%.8f", i, s, i - 1, sched.sigmas[i - 1]);
                        fail("sigmas not strictly decreasing", b);
                        break;
                    }
                }
                // timesteps[i] == sigmas[i]*num_train_timesteps exactly (same f32 op the module performs).
                for (std::size_t i = 0; i < n; ++i) {
                    const float want = sched.sigmas[i] * static_cast<float>(cfg.num_train_timesteps);
                    if (sched.timesteps[i] != want) {
                        std::snprintf(b, sizeof b, "timesteps[%zu]=%.8f want %.8f", i, sched.timesteps[i], want);
                        fail("timesteps != sigmas*num_train_timesteps", b);
                        break;
                    }
                }
            }
        }
    }

    if (failures) {
        std::printf("IMAGE_SCHEDULE_GRID_TEST: FAIL (%d check(s))\n", failures);
        return 1;
    }
    std::printf("IMAGE_SCHEDULE_GRID_TEST: PASS (%zu x %zu sides, %zu steps: all invariants hold)\n", sides.size(), sides.size(),
                stepsList.size());
    return 0;
}
