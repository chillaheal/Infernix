// Step-4 certification: the PRODUCTION image route, end to end. The job runs through the public
// Engine::run_image_interlude (frontier drain + VRAM yield + restoration, not the testing:: trio),
// its weights come from the REAL .infernix artifact through the production loader (bind_dit +
// bind_image_vae, one streamed-only plan), and its execution is the merged production entry
// run_image_job (40-step FlowMatchEuler over the streamed FP8 DiT + the VAE decode). The schedule
// is the production host module (ImageJob::schedule), cross-checked here against the oracle's
// timesteps.npy/sigmas.npy; the numeric gates are the diffusers references on the same
// 512x512 job the BF16 oracle route reproduced: FINAL_LATENT rel_fro < 0.5 vs latents_final.npy
// and FINAL_IMAGE rel_fro < 0.3 vs image.npy. The coexistence gate is the LLM: post-interlude ids
// == baseline, and the device-memory ledger prints free VRAM at every quiescent point. The step-0
// velocity gate lives with the BF16 route (test_image_interlude_dit_real.cu): run_image_job
// returns only the job's outputs, so this test certifies the whole-job end points.
//
// Needs INFERNIX_QWEN4_ARTIFACT (the LLM to yield from), INFERNIX_TEST_ARTIFACT (the image
// artifact) and INFERNIX_IMAGE_ORACLE ([INFERNIX_IMAGE_ORACLE_PIPELINE] for the references).
// Skips (77) without them; returns 2 with no CUDA device. A .cu for the cudaStream_t cast.
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   INFERNIX_TEST_ARTIFACT=out/Qwen-Image-2.1-FP8.infernix \
//   INFERNIX_IMAGE_ORACLE=.../oracle [INFERNIX_IMAGE_ORACLE_PIPELINE=.../tensors_pipeline] \
//   infernix_qwen4_exp_image_interlude_prod_real_test

#include "infernix/engine.h"

#include "artifact/reader.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "runtime/expert_offload/vram_monitor.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace qo  = qioracle;
namespace qi  = infernix::models::qwen_image;
namespace art = infernix::artifact;

namespace {

using T = std::vector<float>;

constexpr std::uint64_t kGiB = 1ULL << 30;

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((x >> 16) | 0x0040U);
    const std::uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}
float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}
std::vector<std::uint16_t> bits_of(const T& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}
T represent(const T& v) {
    T out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(to_bf16(v[i]));
    return out;
}
T floats_of(const std::vector<std::uint16_t>& bits) {
    T out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = from_bf16(bits[i]);
    return out;
}

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a long story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

std::uint64_t device_free_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return free_b;
}

void ledger(const char* point) {
    std::printf("LEDGER %s: device free %llu MiB\n", point,
                static_cast<unsigned long long>(device_free_bytes() >> 20));
}

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* qwen4 = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    const char* oracle = std::getenv("INFERNIX_IMAGE_ORACLE");
    const char* image_artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (qwen4 == nullptr || oracle == nullptr || std::string(oracle).empty() || image_artifact == nullptr
        || std::string(image_artifact).empty()) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT, INFERNIX_TEST_ARTIFACT and INFERNIX_IMAGE_ORACLE\n");
        return 77;
    }
    int result = 1;
    try {
        auto real = infernix::open_vram_budget_source(0);
        infernix::testing::set_vram_budget_source([&] { return real->query(); });
        infernix::runtime::expert_offload::testing::set_vram_grow_delay(3600.0);

        infernix::EngineOptions options;
        options.artifact_path         = qwen4;
        options.context_cache.enabled = false;
        options.enable_vision         = false;
        options.max_context           = 4096;
        options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes      = 8 * kGiB;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(engine.supports_image_interlude(), "the Engine offers the image interlude");
        check(infernix::models::qwen4_exp::testing::has_image_yield_program(), "a Program is loaded to yield from");

        const auto baseline = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(baseline.size() == 200, "baseline generated 200 tokens");

        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
            std::printf("NO_CUDA_DEVICE\n");
            return 2;
        }

        // ---- oracle references (the same 512x512 job the BF16 route reproduces) ----
        const std::string oroot(oracle);
        std::string TP = qo::join(oroot, "tensors_pipeline");
        if (const char* pe_pipeline = std::getenv("INFERNIX_IMAGE_ORACLE_PIPELINE"))
            if (*pe_pipeline) TP = pe_pipeline;

        qo::NpyArray pe_a = qo::read_npy(qo::join(TP, "prompt_embeds.npy"));
        qo::require(pe_a.shape.size() == 3 && pe_a.shape[0] == 1 && pe_a.shape[2] == 4096,
                    "prompt_embeds not [1,?,4096]");
        const T prompt_embeds = represent(pe_a.floats());
        qo::NpyArray img_a = qo::read_npy(qo::join(TP, "image.npy"));
        qo::require(img_a.shape.size() == 4 && img_a.shape[0] == 1 && img_a.shape[1] == 4 &&
                        (int)img_a.shape[2] % 16 == 0 && (int)img_a.shape[3] % 16 == 0,
                    "image.npy not [1,4,H,W] (H,W multiples of 16)");
        const int H = (int)img_a.shape[2], W = (int)img_a.shape[3];
        const std::size_t img_tokens = ((std::size_t)H / 16) * ((std::size_t)W / 16);
        const T img_ref = img_a.floats();
        qo::NpyArray li_a = qo::read_npy(qo::join(TP, "latents_init.npy"));
        qo::require(li_a.shape.size() == 3 && li_a.shape[1] == (int)img_tokens && li_a.shape[2] == 64,
                    "latents_init not [1,img_tokens,64]");
        const T lat_init = represent(li_a.floats());
        const T timesteps_ref = qo::read_npy(qo::join(TP, "timesteps.npy")).floats();
        const T sigmas_ref    = qo::read_npy(qo::join(TP, "sigmas.npy")).floats();
        const T lat_final_ref = qo::read_npy(qo::join(TP, "latents_final.npy")).floats();
        const T lat_mean      = qo::read_npy(qo::join(TP, "latents_mean.npy")).floats();
        const T lat_std       = qo::read_npy(qo::join(TP, "latents_std.npy")).floats();
        qo::require(timesteps_ref.size() == 40 && sigmas_ref.size() == 41, "timesteps/sigmas len mismatch");
        qo::require(lat_final_ref.size() == img_tokens * 64 && lat_mean.size() == 64 && lat_std.size() == 64,
                    "VAE reference sizes");

        // ---- production inputs: ImageJob geometry + the production scheduler ----
        qi::ImageJob job_desc;
        job_desc.height = (std::uint32_t)H;
        job_desc.width  = (std::uint32_t)W;
        job_desc.steps  = (std::uint32_t)timesteps_ref.size();
        const qi::ImageGeometry geom = job_desc.geometry();
        check(geom.height == (std::uint32_t)H && geom.width == (std::uint32_t)W,
              "the oracle job size is already on the geometry grid");
        check(geom.tokens == (std::uint32_t)img_tokens, "ImageGeometry tokens match the latent token count");
        const qi::Schedule sched = job_desc.schedule(qi::SchedulerConfig{});
        qo::require(sched.timesteps.size() == timesteps_ref.size() && sched.sigmas.size() == sigmas_ref.size(),
                    "production schedule has a different length than the oracle's");
        double dt = 0.0, ds = 0.0;
        for (std::size_t i = 0; i < sched.timesteps.size(); ++i)
            dt = std::max(dt, std::fabs((double)sched.timesteps[i] - (double)timesteps_ref[i]));
        for (std::size_t i = 0; i < sched.sigmas.size(); ++i)
            ds = std::max(ds, std::fabs((double)sched.sigmas[i] - (double)sigmas_ref[i]));
        std::printf("schedule max abs diff vs oracle: timesteps=%.3e sigmas=%.3e\n", dt, ds);
        check(dt == 0.0 && ds == 0.0, "the production schedule equals the oracle schedule at this resolution");
        std::printf("inputs: image %dx%d, %zu latent tokens, %zu steps\n", H, W, img_tokens,
                    timesteps_ref.size());

        qi::ImageJobInputs inputs;
        inputs.prompt_embeds = bits_of(prompt_embeds);
        inputs.init_latent   = bits_of(lat_init);
        inputs.timesteps     = sched.timesteps;
        inputs.sigmas        = sched.sigmas;
        inputs.latents_mean  = lat_mean;
        inputs.latents_std   = lat_std;
        inputs.height_px     = geom.height;
        inputs.width_px      = geom.width;

        // ---- production weights: real artifact, one streamed-only plan (device/pinned 0) ----
        art::Reader reader(image_artifact);  // non-const: run_image_job consumes Reader&
        art::Binder binder(reader);
        const qi::DiTWeights dit = qi::bind_dit(binder);
        const qi::VaeWeights vae = qi::bind_image_vae(binder);
        const art::MaterializationPlan plan = std::move(binder).finish();
        check(plan.device_objects.empty() && plan.device_capacity_bytes == 0 && plan.pinned_objects.empty()
                  && plan.pinned_capacity_bytes == 0,
              "the image job plan places nothing on device or pinned host before the interlude");

        ledger("baseline (LLM resident)");
        check(qi::image_device_bytes_live() == 0, "the image ledger is empty before any job");
        qi::reset_image_device_peak();
        const std::uint64_t free_full = device_free_bytes();

        // The serving layer admits or refuses on this answer before anything is yielded, so it has to be
        // the answer the interlude then acts on -- and asking it must not disturb a serving Engine.
        const infernix::ImageYieldProbe probe = engine.probe_image_yield(8 * kGiB, 0);
        check(probe.freed_bytes_expected > 0,
              "the probe predicts a yield of " + std::to_string(probe.freed_bytes_expected >> 20) + " MiB");
        check(probe.device_free_bytes + kGiB / 4 >= device_free_bytes(),
              "the probe reads the free memory the job would start from");
        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 0,
              "probing leaves the Program Running");
        check(infernix::models::qwen4_exp::testing::image_interlude_submit_open(),
              "probing never closes the LLM submit door");

        const std::uint64_t freed = engine.run_image_interlude(
            8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                check(ctx.freed_bytes > 0, "the interlude yield freed "
                                               + std::to_string(ctx.freed_bytes >> 20) + " MiB");
                check(ctx.image_stream != nullptr, "the interlude handed the job a dedicated stream");
                check(infernix::models::qwen4_exp::testing::image_interlude_state() == 3,
                      "the Program is Leased while the job runs");
                check(!infernix::models::qwen4_exp::testing::image_interlude_submit_open(),
                      "the LLM submit door is closed while the job runs");
                const std::uint64_t free_yield = device_free_bytes();
                check(free_yield - free_full > kGiB / 2, "driver free VRAM rose across the yield");
                ledger("inside the yield (before the job)");

                const qi::GeneratedImage out =
                    qi::run_image_job(reader, dit, vae, inputs, static_cast<cudaStream_t>(ctx.image_stream));

                ledger("inside the yield (after run_image_job)");
                // WHY THE GATE LOOKS AFTER A SYNC: the job's last frees are stream-ordered on the image
                // stream, so the host sees the driver short by the whole job peak the instant
                // run_image_job returns (measured 2026-10-22: 10063 MiB at the return, 11247 MiB after a
                // device-wide sync, with the default stream-ordered pool reserving 0 MiB before and
                // after -- nothing is held, the releases simply had not been processed yet). The route
                // that needs the bytes back, release_from_yield, takes device_.synchronize() before it
                // remaps anything, so production never allocates against that lag; the gate therefore
                // tests the contract where it can be observed, after the device has reached the frees.
                {
                    const std::uint64_t free_now = device_free_bytes();
                    (void)cudaDeviceSynchronize();
                    const std::uint64_t free_synced = device_free_bytes();
                    std::size_t reserved = 0, trimmed = 0;
                    cudaMemPool_t pool = nullptr;
                    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess && pool != nullptr) {
                        (void)cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
                        (void)cudaMemPoolTrimTo(pool, 0);
                        (void)cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &trimmed);
                    }
                    std::printf("RELEASE PROBE: driver free %llu MiB at the job's return, %llu MiB after a "
                                "device-wide sync; default pool reserved %llu MiB, %llu MiB after trim\n",
                                static_cast<unsigned long long>(free_now >> 20),
                                static_cast<unsigned long long>(free_synced >> 20),
                                static_cast<unsigned long long>(reserved >> 20),
                                static_cast<unsigned long long>(trimmed >> 20));
                }
                check(device_free_bytes() + kGiB / 10 >= free_yield,
                      "run_image_job returned with its device allocations freed");
                std::printf("IMAGE LEDGER: live %llu MiB, peak %llu MiB\n",
                            static_cast<unsigned long long>(qi::image_device_bytes_live() >> 20),
                            static_cast<unsigned long long>(qi::image_device_bytes_peak() >> 20));
                check(qi::image_device_bytes_live() == 0,
                      "the ledger holds zero image device bytes after the job (success path)");
                check(out.height == geom.height && out.width == geom.width, "the job returned its geometry");
                qo::require(out.latent.size() == img_tokens * 64 && out.channels.size() == (std::size_t)4 * H * W,
                            "job output sizes differ from the oracle");

                const double e_lat = qo::rel_fro(floats_of(out.latent), lat_final_ref);
                std::printf("FINAL_LATENT rel_fro=%.3e (vs latents_final.npy) tol=5e-1 %s\n", e_lat,
                            (e_lat < 0.5 ? "OK" : "FAIL"));
                check(std::isfinite(e_lat) && e_lat < 0.5,
                      "the production route's final latent matches the reference (rel_fro < 0.5)");

                T got_channels = out.channels;
                const double e_img = qo::rel_fro(got_channels, img_ref);
                double mn = 1e30, mx = -1e30;
                for (float v : got_channels) { mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
                std::printf("FINAL_IMAGE  rel_fro=%.3e (vs image.npy) tol=3e-1 %s  range [%.3f,%.3f]\n", e_img,
                            (e_img < 0.3 ? "OK" : "FAIL"), mn, mx);
                check(std::isfinite(e_img) && e_img < 0.3,
                      "the production route's decoded image matches the reference (rel_fro < 0.3)");

                // #17 release gate, THROW path: a job that fails after its first device
                // allocations must still leave the ledger empty (Buf destructors unwind it).
                qi::DiTWeights broken = dit;
                broken.txt_in_norm.binding.parts.clear();
                bool threw = false;
                try {
                    static_cast<void>(qi::run_image_job(reader, broken, vae, inputs,
                                                        static_cast<cudaStream_t>(ctx.image_stream)));
                } catch (const std::exception& error) {
                    threw = true;
                    std::printf("THROW-PATH LEDGER: %s -> live %llu MiB, peak %llu MiB\n", error.what(),
                                static_cast<unsigned long long>(qi::image_device_bytes_live() >> 20),
                                static_cast<unsigned long long>(qi::image_device_bytes_peak() >> 20));
                }
                check(threw, "a job whose weight reference binds no object throws");
                check(threw && qi::image_device_bytes_live() == 0,
                      "the ledger holds zero image device bytes after the job (throw path)");

                const char* ppm_env = std::getenv("INFERNIX_IMAGE_OUT_PPM");
                const char* ppath = (ppm_env && *ppm_env)
                                        ? ppm_env
                                        : "C:/Users/Micke/Documents/Kodprojekt/Qwen Flash Next/image-spike/interlude_prod_mine.ppm";
                FILE* pf = fopen(ppath, "wb");
                if (pf) {
                    const std::size_t HW = (std::size_t)out.height * out.width;
                    std::fprintf(pf, "P6\n%u %u\n255\n", out.width, out.height);
                    std::vector<unsigned char> px(HW * 3);
                    for (std::size_t idx = 0; idx < HW; ++idx)
                        for (int c = 0; c < 3; ++c) {
                            const double v = std::max(-1.0, std::min(1.0, (double)got_channels[(std::size_t)c * HW + idx]));
                            px[idx * 3 + c] = (unsigned char)std::lround(255.0 * 0.5 * (v + 1.0));
                        }
                    fwrite(px.data(), 1, px.size(), pf);
                    fclose(pf);
                    std::printf("WROTE_IMAGE %s (%ux%u)\n", ppath, out.width, out.height);
                } else {
                    std::printf("IMAGE_WRITE_SKIP %s\n", ppath);
                }
                return true;
            });
        check(freed > 0, "run_image_interlude reports " + std::to_string(freed >> 20) + " MiB freed and restored");
        // The yield can free LESS than the plan promises (a KV top some retained block pins raises the
        // limit rather than throwing), and never more: so a request refused on the probe's number was
        // refused for the right reason.
        check(freed <= probe.freed_bytes_expected + kGiB / 1024,
              "the yield freed no more than the probe predicted (" + std::to_string(freed >> 20) +
                  " MiB against " + std::to_string(probe.freed_bytes_expected >> 20) + " MiB)");
        ledger("after restoration (LLM pools back)");

        check(infernix::models::qwen4_exp::testing::image_interlude_state() == 0, "the Program returned to Running");
        check(infernix::models::qwen4_exp::testing::image_interlude_submit_open(), "the LLM submit door reopened");
        const auto after = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(after == baseline, "LLM generation is undisturbed after the production image job");

        infernix::testing::set_vram_budget_source({});
        result = failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        try { infernix::models::qwen4_exp::testing::end_image_quiesce(); } catch (...) {}
        return 1;
    }
    std::printf(result == 0 ? "qwen4_exp production image-interlude route certification passed\n" : "FAIL: %d checks failed\n",
                failures);
    return result;
}
