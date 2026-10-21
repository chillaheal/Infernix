// Production scheduler qualification: the FlowMatchEulerDiscrete schedule built by
// infernix::models::qwen_image against the real 512x512 40-step pipeline oracle (the sigmas.npy +
// timesteps.npy in the tensors_pipeline dir named by INFERNIX_IMAGE_ORACLE_PIPELINE, dumped by
// image-spike/oracle/dump_pipeline.py running QwenImage21Pipeline line-for-line). Host-only and
// weights-free. Also pins the geometry's token count (the resolution-dependent shift input) and the
// mu for the 512 geometry. A-priori tol 2e-5 on the f32 schedule (the same precedent the f32-vs-f64
// scheduler oracle uses); the mu is exact to working precision. Never narrowed to pass.
// Skips (77) without INFERNIX_IMAGE_ORACLE_PIPELINE.

#include "models/qwen_image/image_job.h"
#include "qwen_image_oracle_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qo = qioracle;
namespace qi = infernix::models::qwen_image;

int main() {
    const char* root = std::getenv("INFERNIX_IMAGE_ORACLE_PIPELINE");
    if (root == nullptr || root[0] == '\0') {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE_PIPELINE not set\n");
        return 77;
    }
    const std::string S = std::string(root); // the env var names the tensors_pipeline dir directly

    const std::vector<double> sigmas_ref = qo::read_npy(qo::join(S, "sigmas.npy")).doubles();   // [41] f32
    const std::vector<double> ts_ref     = qo::read_npy(qo::join(S, "timesteps.npy")).doubles(); // [40] f32

    int failures = 0;
    auto check = [&](const char* name, double r, double tol) {
        const bool ok = (r < tol) && std::isfinite(r);
        std::printf("  %-14s rel=%.3e  tol=%.1e  %s\n", name, r, tol, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    };

    // Geometry: 512x512 -> latent 32x32 -> 1024 tokens (the resolution-dependent shift input).
    const qi::ImageGeometry g = qi::ImageGeometry::derive(512, 512);
    if (g.hl != 32 || g.wl != 32 || g.tokens != 1024) {
        std::printf("  %-14s FAIL (hl=%u wl=%u tokens=%u)\n", "geometry", g.hl, g.wl, g.tokens);
        ++failures;
    } else {
        std::printf("  %-14s hl=%u wl=%u tokens=%u  OK\n", "geometry", g.hl, g.wl, g.tokens);
    }

    const qi::SchedulerConfig cfg;
    const double mu = qi::calculate_shift(cfg, g.tokens);
    const double mu_ref = 0.5387096774193548; // schedule.json (the real pipeline schedule)
    std::printf("  mu = %.10f\n", mu);
    check("mu", std::abs(mu - mu_ref) / mu_ref, 1e-12);

    const qi::ImageJob job{512, 512, 40};
    const qi::Schedule sched = job.schedule(cfg);
    if (sched.sigmas.size() != 41 || sched.timesteps.size() != 40) {
        std::printf("  %-14s FAIL (sigmas=%zu timesteps=%zu)\n", "schedule_len", sched.sigmas.size(),
                    sched.timesteps.size());
        ++failures;
    }
    std::vector<double> sig_d(sched.sigmas.begin(), sched.sigmas.end());
    std::vector<double> ts_d(sched.timesteps.begin(), sched.timesteps.end());
    check("sigmas", qo::rel_fro_d(sig_d, sigmas_ref), 2e-5);
    check("timesteps", qo::rel_fro_d(ts_d, ts_ref), 2e-5);

    if (failures) {
        std::printf("IMAGE_SCHEDULER_PROD_TEST: FAIL (%d stage(s))\n", failures);
        return 1;
    }
    std::printf("IMAGE_SCHEDULER_PROD_TEST: PASS (512x512 steps=40 mu=%.6f)\n", mu);
    return 0;
}
