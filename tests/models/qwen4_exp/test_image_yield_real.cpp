// LLM<->image VRAM coexistence on the real artifact (coexistence track, design §19.3.7 / §19.3.11):
// the live Program's one-shot yield must ACTUALLY return physical device memory (cuMemUnmap of the
// expert-frame pool and the elastic KV pool), not just remap it, and giving it back must restore the
// pools so a later generation is undisturbed.
//
// The elastic monitor is parked (a fake source that never reports pressure + a huge grow delay) so the
// only device-memory movement in the measured window is the yield itself. cuMemGetInfo is the driver's
// own truth on physical free memory: the delta across the yield is the claim under test.
//
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact. The expert share is capped
// (--expert-ram-mib equivalent) so the Engine fits in Host RAM once the serving process is gone.
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] infernix_qwen4_exp_image_yield_real_test

#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "infernix/engine.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint64_t kGiB = 1ULL << 30;

std::uint64_t device_free_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { throw std::runtime_error("cudaMemGetInfo failed"); }
    return free_b;
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

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    int result = 1;
    try {
        // The source must report the REAL free VRAM at startup: Engine construction sizes the expert
        // frame plan from device_free (qwen4_exp_instance.cpp), so an inflated figure makes it plan
        // past the real card and the dense weights spill to system memory. Only the grow delay is
        // parked here — beyond the test — so the monitor cannot regrow the cache and confound the
        // device-memory movement the yield itself causes in the window below.
        auto real = infernix::open_vram_budget_source(0);
        infernix::testing::set_vram_budget_source([&] { return real->query(); });
        infernix::models::qwen4_exp::testing::set_vram_grow_delay(3600.0);

        infernix::EngineOptions options;
        options.artifact_path       = artifact;
        options.context_cache.enabled = false;
        options.enable_vision       = false;
        options.max_context         = 4096;
        options.kv_capacity         = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache            = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes    = 8 * kGiB; // lean SSD tier: fits Host RAM with the serving process gone
        // Engine construction opens the n-gram volume for every route (qwen4_exp_instance.cpp:182
        // derives "<artifact>.ngram", which is absent next to the numbered part). Point it at the
        // real serving n-gram so the open succeeds; speculation stays off, so it is only read on demand.
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(infernix::models::qwen4_exp::testing::has_image_yield_program(), "a Program is loaded to yield from");

        constexpr std::uint32_t kTokens = 200;
        const auto baseline = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(baseline.size() == kTokens, "baseline generated " + std::to_string(baseline.size()) + " tokens");

        const std::uint64_t free_full = device_free_bytes(); // serving state: expert cache full, driver free low
        std::uintptr_t lease          = 0;
        const std::uint64_t freed = infernix::models::qwen4_exp::testing::yield_image_vram(8 * kGiB, 0, lease);

        try {
            check(freed > 0, "the yield freed " + std::to_string(freed >> 20) + " MiB toward the image job");
            check(lease != 0, "the yield returned a lease to restore");
            const std::uint64_t free_peak = device_free_bytes(); // after unmap: driver free at its high point
            std::printf("  driver free VRAM rose by %llu MiB across the yield\n",
                        static_cast<unsigned long long>((free_peak - free_full) >> 20));
            check(free_peak > free_full && free_peak - free_full > kGiB / 2,
                  "cuMemGetInfo reports real free VRAM recovered (the unmap is physical)");

            infernix::models::qwen4_exp::testing::release_image_vram(lease); // give the LLM its VRAM back
            const std::uint64_t free_restored = device_free_bytes();         // remap should pull it back down
            const std::uint64_t recovered = free_peak > free_restored ? free_peak - free_restored : 0;
            std::printf("  driver free VRAM fell by %llu MiB across the release\n",
                        static_cast<unsigned long long>(recovered >> 20));
            check(recovered > kGiB / 2, "the release gave the memory back to the serving cache (driver free fell)");
        } catch (...) {
            infernix::models::qwen4_exp::testing::release_image_vram(lease); // always give it back
            throw;
        }
        const auto after = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(after == baseline, "ids after the yield/restore equal the undisturbed run");

        infernix::testing::set_vram_budget_source({});
        result = failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(result == 0 ? "qwen4_exp image-yield checks passed\n" : "FAIL: %d checks failed\n", failures);
    return result;
}
