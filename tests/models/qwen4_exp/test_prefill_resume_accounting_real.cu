// Prefill accounting across a prompt-paused resume on the real Engine.
//
// A long greedy request is submitted without prefix reuse on a worker thread. Once its prompt is
// observably progressing, the main thread starts a multi-GiB image interlude a short moment after a
// progress publish, inside the executing unit: the quiesce finds the walk active
// (interlude_yieldable() is false), and the #81 preemption path pauses the lane inside its prompt.
// The yield then unmaps the pool the prompt's KV lived
// in, and the resume rebinds the prompt from the deepest state the cache still holds -- here the
// root, because nothing was published. The rewalk therefore reprocesses tokens the Engine had
// already counted against the admitted suffix. Before the per-binding rebase that rewalk threw the
// engine invariant "prefill unit exceeded the admitted prompt suffix" and failed every active
// request; now the rewalk counts against the new binding's own suffix.
//
// Certified here: the scheduling shows a Paused and a Restored for the request (the preemption
// really happened), the request completes with its full output, the published prompt-progress
// positions never decrease and never exceed the prompt total, and the Engine stays available with
// a healthy follow-up generation.
//
// Needs INFERNIX_QWEN4_ARTIFACT ([INFERNIX_QWEN4_NGRAM]); skips (77) without it; returns 2 with no
// CUDA device. A .cu for the cudaStream_t cast. Run under the blackout (server stopped).
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   infernix_qwen4_exp_prefill_resume_accounting_real_test

#include "infernix/engine.h"

#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "runtime/expert_offload/vram_monitor.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kGiB          = 1ULL << 30;
// Above the Program's walk ceiling (kWalkMaxTokens = 65536): the layer-walk spans must cross unit
// boundaries, which is the only state the interlude preemption finds (a quiesce drains to a unit
// boundary, where a walk that fits inside one unit already left the ring back -- measured twice).
constexpr std::uint32_t kPromptTokens = 70000;
constexpr std::uint32_t kOutputTokens = 8;
// Large enough to make the interlude's yield real (the live crash ran a multi-GiB yield): the
// pause itself comes from landing mid-walk (interlude_yieldable() is false while walk_ is active),
// not from the size.
constexpr std::uint64_t kInterludeTarget = 17ULL * kGiB + kGiB / 2;

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

infernix::RequestOptions greedy(std::uint32_t tokens, bool reuse) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;
    return request;
}

// Small real device work on the interlude's stream: proves the stream is live without image weights.
void tiny_copy(void* image_stream) {
    const cudaStream_t stream = static_cast<cudaStream_t>(image_stream);
    float* device             = nullptr;
    float* pinned             = nullptr;
    bool   ok                 = cudaMalloc(&device, sizeof(float)) == cudaSuccess
                            && cudaMallocHost(&pinned, sizeof(float)) == cudaSuccess;
    if (ok) {
        *pinned = 1.0F;
        ok      = cudaMemcpyAsync(device, pinned, sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess
               && cudaStreamSynchronize(stream) == cudaSuccess;
        if (ok) {
            *pinned = 0.0F;
            ok      = cudaMemcpy(pinned, device, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess
                   && *pinned == 1.0F;
        }
    }
    if (device) { cudaFree(device); }
    if (pinned) { cudaFreeHost(pinned); }
    if (!ok) { throw std::runtime_error("the interlude stream did not carry a 4-byte copy"); }
}

struct AccountingSink : infernix::OutputSink {
    std::vector<std::uint32_t> positions;
    bool                       monotone      = true;
    bool                       within        = true;
    std::atomic<bool>          first_progress{false};

    void start(infernix::GenerationStart) override {}
    void progress(infernix::PromptProgress progress) override {
        if (!positions.empty() && progress.processed_prompt_tokens < positions.back()) { monotone = false; }
        if (progress.processed_prompt_tokens > progress.total_prompt_tokens) { within = false; }
        positions.push_back(progress.processed_prompt_tokens);
        first_progress = true;
    }
    void timing(infernix::GenerationTimingObservation) override {}
    void publish(infernix::OutputDelta) override {}
};

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* qwen4 = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (qwen4 == nullptr || std::string(qwen4).empty()) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        return 2;
    }
    try {
        auto source = infernix::open_vram_budget_source(0);
        infernix::testing::set_vram_budget_source([&] { return source->query(); });
        infernix::runtime::expert_offload::testing::set_vram_grow_delay(3600.0);

        infernix::EngineOptions options;
        options.artifact_path         = qwen4;
        options.context_cache.enabled = false;
        options.enable_vision         = false;
        options.max_context           = 73728;
        options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(73728);
        options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes      = 8 * kGiB;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(engine.supports_image_interlude(), "the Engine offers the image interlude");

        std::atomic<int> paused{0};
        std::atomic<int> restored{0};
        infernix::GenerationObservationOptions observation;
        observation.prompt_progress = true;
        observation.scheduling      = [&](const infernix::GenerationSchedulingObservation& event) {
            if (event.transition == infernix::GenerationSchedulingTransition::Paused) { ++paused; }
            if (event.transition == infernix::GenerationSchedulingTransition::Restored) { ++restored; }
        };

        AccountingSink sink;
        std::string                  error;
        std::vector<infernix::TokenId> ids;
        std::thread worker([&] {
            auto prompt = engine.prepare_tokens(synthetic_tokens(kPromptTokens, 7U), false);
            auto handle = engine.submit(std::move(prompt), greedy(kOutputTokens, /*reuse=*/false),
                                        infernix::OutputConsumerMode::Streaming, observation);
            try {
                ids = handle.wait(&sink).generated_token_ids;
            } catch (const std::exception& exception) {
                error = exception.what();
            }
            sink.first_progress = true; // wake the main thread even if the request died before publishing
        });

        // Wait for the prompt to be observably walking, then demand more VRAM than a walking lane
        // can lend: the quiesce must preempt this request inside its prompt to reach it.
        for (int i = 0; i < 3000 && !sink.first_progress; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        check(sink.first_progress, "the prompt started walking");
        std::uint64_t freed  = 0;
        bool          interlude = false;
        if (!error.empty()) {
            std::printf("worker died before the interlude: %s\n", error.c_str());
        } else if (!sink.positions.empty() && sink.positions.back() < kPromptTokens) {
            // The interlude must land while a unit is executing: a walk paused between units is
            // yieldable without a preemption (measured: lease_held=0, no pause). A unit of this
            // prefill takes seconds, so this short delay stays inside one.
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        } else if (!sink.positions.empty()) {
            freed     = engine.run_image_interlude(kInterludeTarget, 0,
                                                   [&](const infernix::ImageInterludeContext& ctx) -> bool {
                                                       tiny_copy(ctx.image_stream);
                                                       std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                                       return true;
                                                   });
            interlude = true;
        }
        worker.join();

        check(error.empty(), "the prompt-paused request survived its resume" + (error.empty() ? "" : ": " + error));
        check(interlude, "the interlude ran across the active prefill");
        check(freed > 0, "the interlude freed VRAM");
        check(paused.load() >= 1, "the Engine paused the request inside its prompt");
        check(restored.load() >= 1, "the Engine restored the request once the pools came back");
        check(ids.size() == kOutputTokens, "the request completed its full output");
        check(!sink.positions.empty() && sink.positions.back() == kPromptTokens,
              "the final progress position is the prompt end");
        check(sink.monotone, "prompt-progress positions never move backwards across the rewalk");
        check(sink.within, "prompt-progress positions never exceed the prompt total");

        check(engine.is_available(), "the Engine reports itself available");
        const auto after =
            engine.generate(engine.prepare_tokens(synthetic_tokens(64, 3U), false), greedy(16, false)).generated_token_ids;
        check(after.size() == 16, "a follow-up generation runs healthy");
    } catch (const std::exception& exception) {
        std::printf("EXCEPTION: %s\n", exception.what());
        return 1;
    }
    std::printf(failures == 0 ? "PREFILL_RESUME_ACCOUNTING_TEST: PASS\n" : "PREFILL_RESUME_ACCOUNTING_TEST: FAIL\n");
    return failures == 0 ? 0 : 1;
}
