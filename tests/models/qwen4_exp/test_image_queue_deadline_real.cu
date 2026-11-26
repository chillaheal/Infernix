// Admission-queue budget across an image interlude, on the real Engine (no image weights: the job here
// only holds the submit door). Two scenarios in order:
//   D1 a request that ARRIVES while an interlude holds the submit door closed keeps its whole queue
//      budget. The hold is credited back to it, so it is admitted when the door reopens and completes at
//      baseline ids even though the hold alone is far longer than pending_timeout_ms. Before the credit
//      ran, it expired with "inference request expired while waiting for admission": the worker loop's
//      door-held branch waits and continues, so the round body -- and with it the sweep that performs the
//      credit -- is never reached for the whole generation, and the hold was never dated at all.
//   D2 the budget stays real when nothing holds the door: with the single lane busy and the door open, a
//      request that cannot be admitted still expires at its deadline. Without this second half, D1 could
//      be satisfied by a queue that never expires.
//
// Needs INFERNIX_QWEN4_ARTIFACT ([INFERNIX_QWEN4_NGRAM]); skips (77) without it; returns 2 with no CUDA
// device. A .cu for the cudaStream_t cast. Run under the blackout (server stopped).
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   infernix_qwen4_exp_image_queue_deadline_real_test

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

constexpr std::uint64_t kGiB = 1ULL << 30;
// The queue budget is deliberately tiny and the door hold deliberately longer than it: the scenario is
// exactly the window where an uncredited hold kills a real chat request.
constexpr auto kPendingTimeout = std::chrono::milliseconds(1500);
constexpr auto kDoorHold       = std::chrono::milliseconds(4000);

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
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

// Small real device work on the interlude's stream: proves the stream is live without image weights.
void tiny_copy(void* image_stream) {
    const cudaStream_t stream = static_cast<cudaStream_t>(image_stream);
    float* device           = nullptr;
    float* pinned           = nullptr;
    bool   ok               = cudaMalloc(&device, sizeof(float)) == cudaSuccess
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

// Runs f, returns the thrown std::exception's message ("" when nothing threw).
template <typename F> std::string expect_throw(F&& f) {
    try {
        f();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

bool holds(const std::string& message, const char* needle) {
    const bool found = message.find(needle) != std::string::npos;
    if (!found) { std::printf("  (message was: %s)\n", message.c_str()); }
    return found;
}

struct LlmRun {
    std::thread thread;
    std::vector<infernix::TokenId> ids;
    std::string error;
    std::atomic<bool> done{false};
    std::chrono::milliseconds waited{0};
};

void start_llm_run(infernix::Engine& engine, LlmRun& run, std::uint32_t tokens) {
    run.thread = std::thread([&engine, &run, tokens] {
        const auto began = std::chrono::steady_clock::now();
        try {
            run.ids = engine.generate(engine.prepare(story()), greedy(tokens)).generated_token_ids;
        } catch (const std::exception& error) {
            run.error = error.what();
        }
        run.waited = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - began);
        run.done.store(true);
    });
}

namespace state_ns = infernix::models::qwen4_exp::testing;

// Waits (bounded) for the submit door to reach `open`, so a scenario never races the interlude's phases.
bool wait_for_door(bool open, std::chrono::milliseconds limit) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < until) {
        if (state_ns::image_interlude_submit_open() == open) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return state_ns::image_interlude_submit_open() == open;
}

// D2 needs a lane that is provably busy: the holder has to be RESIDENT before the second request is
// submitted. Submitted on the same instant, the two threads race and the loser is the one that expires.
bool wait_for_lane(infernix::Engine& engine, std::uint32_t resident, std::chrono::milliseconds limit) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < until) {
        if (engine.runtime_stats().running_requests >= resident) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return engine.runtime_stats().running_requests >= resident;
}

void check_llm_healthy(infernix::Engine& engine, const std::vector<infernix::TokenId>& baseline) {
    const auto after = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
    check(after == baseline, "a later generation still reproduces the baseline ids");
    check(engine.is_available(), "the Engine reports itself available");
}

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
        options.max_context           = 4096;
        options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
        options.expert_ram_bytes      = 8 * kGiB;
        options.max_concurrency       = 1; // D2 needs a lane that is provably busy
        options.pending_timeout_ms    = static_cast<std::uint32_t>(kPendingTimeout.count());
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

        infernix::Engine engine(options);
        check(engine.supports_image_interlude(), "the Engine offers the image interlude");
        check(infernix::models::qwen4_exp::testing::has_image_yield_program(), "a Program is loaded to yield from");

        const auto baseline = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
        check(baseline.size() == 200, "baseline generated 200 tokens");

        // ---- D1: a request parked behind a closed door outlives its own budget by the hold's length ----
        std::printf("[D1] pending request across a door hold of %lld ms (budget %lld ms)\n",
                    static_cast<long long>(kDoorHold.count()), static_cast<long long>(kPendingTimeout.count()));
        {
            std::string interlude_error;
            std::thread interlude([&] {
                try {
                    engine.run_image_interlude(
                        8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                            tiny_copy(ctx.image_stream);
                            std::this_thread::sleep_for(kDoorHold);
                            return true;
                        });
                } catch (const std::exception& error) {
                    interlude_error = error.what();
                }
            });
            check(wait_for_door(false, std::chrono::seconds(10)), "the interlude closed the submit door");

            LlmRun run;
            start_llm_run(engine, run, 200);
            std::this_thread::sleep_for(kPendingTimeout * 2);
            check(!run.done.load(), "the parked request was still alive past twice its queue budget");

            interlude.join();
            run.thread.join();
            check(interlude_error.empty(), "the interlude that held the door completed (" + interlude_error + ")");
            check(run.error.empty(), "the parked request did not expire: " + run.error);
            check(run.ids == baseline, "the parked request generated at baseline ids");
            std::printf("  parked request waited %lld ms\n", static_cast<long long>(run.waited.count()));
            check(wait_for_door(true, std::chrono::seconds(1)), "the submit door reopened");
        }
        check_llm_healthy(engine, baseline);

        // ---- D2: with the door open and the lane busy, the budget is still a budget ----
        std::printf("[D2] expired admission while the lane is busy and the door is open\n");
        {
            LlmRun busy;
            start_llm_run(engine, busy, 600); // holds the single lane for well over the budget
            check(wait_for_lane(engine, 1, std::chrono::seconds(10)),
                  "the 600-token request is resident on the lane before the second one is submitted" +
                  (busy.error.empty() ? std::string() : ", and the holder already failed: " + busy.error));
            const std::string message = expect_throw([&] {
                engine.generate(engine.prepare(story()), greedy(8));
            });
            check(holds(message, "expired"),
                  "a request that cannot be admitted while the door is open still expires");
            busy.thread.join();
            check(busy.error.empty(), "the lane-holding generation completed" +
                  (busy.error.empty() ? std::string() : ": " + busy.error));
            check(busy.ids.size() == 600, "the lane-holding generation produced its 600 tokens (got " +
                  std::to_string(busy.ids.size()) + ")");
        }
        check_llm_healthy(engine, baseline);

    } catch (const std::exception& error) {
        std::printf("EXCEPTION: %s\n", error.what());
        ++failures;
    }
    std::printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
