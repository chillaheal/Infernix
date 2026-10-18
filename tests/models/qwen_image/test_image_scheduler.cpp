// CPU numeric oracle for the Qwen-Image-2.1 flow-match sampler (FlowMatchEulerDiscreteScheduler).
//
// Weights-free, deterministic. Reproduces the full schedule + Euler sampling in C++ and qualifies it
// against the upstream dump in oracle/tensors_scheduler/: calculate_shift (mu), the exponential
// resolution-dependent sigma shift, the stretch-to-terminal, the timesteps = sigma * num_train_timesteps
// mapping, the trailing terminal sigma, and prev_sample = sample + (sigma_next - sigma) * model_output.
//
// The initial latent (traj[0]) and the per-step model_output (noise_pred) are the sampler's INPUTS and
// are read straight from the dump -- this oracle qualifies the step RULE and the schedule, not NumPy's
// RNG. The reference trajectory comes from the real diffusers scheduler, so the test is non-self-referential.
//
// Skips (77) when INFERNIX_IMAGE_ORACLE is unset.

#include "qwen_image_oracle_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qo = qioracle;

// Scheduler config, copied from artifact/scheduler/scheduler_config.json (recorded in
// oracle/tensors_scheduler/manifest.json). If the artifact config changes, the recomputed sigmas /
// timesteps below diverge from the committed dump and this test fails.
static const double kBaseSeq = 256.0, kMaxSeq = 8192.0, kBaseShift = 0.5, kMaxShift = 0.9;
static const double kNumTrain = 1000.0, kShiftTerminal = 0.02;
static const std::size_t kN = 50, kImageSeq = 4096, kD = 8;

int main() {
    const char* root_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (root_env == nullptr || root_env[0] == '\0') {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string S = qo::join(std::string(root_env), "tensors_scheduler");

    auto F8 = [&](const std::string& n) { return qo::read_npy(qo::join(S, n)).doubles(); };
    auto F32 = [&](const std::string& n) { return qo::read_npy(qo::join(S, n)).floats(); };

    const std::vector<double> sigmas_in_ref = F8("scheduler_sigmas_input.npy");   // [50] f64
    const std::vector<double> sigmas_ref = F8("scheduler_sigmas.npy");            // [51] f64
    const std::vector<float> timesteps_ref = F32("scheduler_timesteps.npy");      // [50] f32
    const std::vector<float> traj_ref = F32("scheduler_latent_traj.npy");         // [51,1,8] f32
    const std::vector<float> pred_ref = F32("scheduler_noise_pred.npy");          // [50,1,8] f32

    int failures = 0;
    auto check = [&](const char* name, double r, double tol) {
        const bool ok = (r < tol) && std::isfinite(r);
        std::printf("  %-16s rel=%.3e  tol=%.1e  %s\n", name, r, tol, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    };

    // ---- 1. calculate_shift (linear on image_seq_len) -> mu ----
    const double m = (kMaxShift - kBaseShift) / (kMaxSeq - kBaseSeq);
    const double b = kBaseShift - m * kBaseSeq;
    const double mu = static_cast<double>(kImageSeq) * m + b;
    // mu is baked into the schedule; the sigma/timestep comparison below is the real check.
    std::printf("  mu = %.10f\n", mu);

    // ---- 2. input sigmas: np.linspace(1, 1/N, N) ----
    std::vector<double> sig_in(kN);
    for (std::size_t i = 0; i < kN; ++i)
        sig_in[i] = 1.0 + (1.0 / static_cast<double>(kN) - 1.0) * static_cast<double>(i) / static_cast<double>(kN - 1);
    check("sigmas_input", qo::rel_fro_d(sig_in, sigmas_in_ref), 1e-12);

    // ---- 3. dynamic exponential shift + stretch-to-terminal, in float32 (the pipeline boundary) ----
    const float emu = static_cast<float>(std::exp(mu));
    std::vector<float> sig32(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        const float t = static_cast<float>(sig_in[i]);          // set_timesteps casts to float32
        const float inv = 1.0F / t - 1.0F;                       // (1/t - 1)**1.0
        sig32[i] = emu / (emu + inv);                            // time_shift_exponential(mu, 1.0, t)
    }
    // stretch_shift_to_terminal: scale = (1 - t[-1]) / (1 - shift_terminal)
    const float one_minus_last = 1.0F - sig32[kN - 1];
    const float scale = one_minus_last / (1.0F - static_cast<float>(kShiftTerminal));
    std::vector<float> sigmas(kN + 1, 0.0F);
    for (std::size_t i = 0; i < kN; ++i) sigmas[i] = 1.0F - ((1.0F - sig32[i]) / scale);
    sigmas[kN] = 0.0F;                                           // terminal sigma (invert_sigmas False)

    std::vector<float> timesteps(kN);
    for (std::size_t i = 0; i < kN; ++i) timesteps[i] = sigmas[i] * static_cast<float>(kNumTrain);

    std::vector<double> sigmas_d(kN + 1), ts_d(kN);
    for (std::size_t i = 0; i <= kN; ++i) sigmas_d[i] = static_cast<double>(sigmas[i]);
    for (std::size_t i = 0; i < kN; ++i) ts_d[i] = static_cast<double>(timesteps[i]);
    check("sigmas", qo::rel_fro_d(sigmas_d, sigmas_ref), 2e-5);
    std::vector<double> ts_ref_d(ts_d.size());
    for (std::size_t i = 0; i < kN; ++i) ts_ref_d[i] = static_cast<double>(timesteps_ref[i]);
    check("timesteps", qo::rel_fro_d(ts_d, ts_ref_d), 2e-5);

    // ---- 4. Euler step: prev = sample + (sigma_next - sigma) * model_output (float32) ----
    // traj[0] (init latent) and pred (model outputs) are the sampler's inputs, read from the dump.
    std::vector<float> x(traj_ref.begin(), traj_ref.begin() + kD);  // [1,8]
    std::vector<double> got_all, ref_all;
    for (std::size_t i = 0; i < kN; ++i) {
        const float dt = sigmas[i + 1] - sigmas[i];
        for (std::size_t d = 0; d < kD; ++d) {
            x[d] = x[d] + dt * pred_ref[i * kD + d];
            got_all.push_back(static_cast<double>(x[d]));
            ref_all.push_back(static_cast<double>(traj_ref[(i + 1) * kD + d]));
        }
    }
    check("latent_traj", qo::rel_fro_d(got_all, ref_all), 2e-4);

    if (failures) {
        std::printf("IMAGE_SCHEDULER_TEST: FAIL (%d stage(s))\n", failures);
        return 1;
    }
    std::printf("IMAGE_SCHEDULER_TEST: PASS (N=%zu D=%zu mu=%.6f)\n", kN, kD, mu);
    return 0;
}
