// C5b+C5c certification: the interlude failure-state machine and the submit-door coverage, on the
// real Engine (no image weights needed -- every job here is tiny real device work on the
// interlude's stream). Six scenarios in order:
//   S1 success route re-verified: yield frees, the Program is Leased with the door closed during
//      the job, restoration confirms, the door reopens, the LLM resumes with baseline ids.
//   S2 drain deadline: while a generation is mid-decode, a 1 ms drain deadline cannot settle the
//      frontier; the job is rejected, NOTHING moved, the LLM resumes and finishes at baseline.
//   S3 job throw with confirmed restoration: the job's OWN exception propagates unchanged; the
//      Program stays Running with the door open.
//   S4 single-flight: while one interlude holds the device, the second caller is told.
//   S5 door gate under load (C5c): a generation runs against an interlude whose job keeps the door
//      closed; the worker starts no execution body across the yield and the request still
//      completes at baseline ids (paused, never corrupted).
//   S6 restoration fault -> Degraded (terminal, LAST): the thrown error names BOTH failures, the
//      active request fails with the Degraded message, the door stays closed, is_available() is
//      false, further submissions are refused and a second interlude throws. Process restart is
//      the recovery unit -- this test must run last in the blackout batch.
//
// Needs INFERNIX_QWEN4_ARTIFACT ([INFERNIX_QWEN4_NGRAM]); skips (77) without it; returns 2 with no
// CUDA device. A .cu for the cudaStream_t cast. Run under the blackout (server stopped).
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   infernix_qwen4_exp_image_interlude_failures_real_test

#include "infernix/engine.h"

#include "core/vram_budget.h"
#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/program/vram_monitor.h"

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

std::uint64_t device_free_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    return free_b;
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
};

void start_llm_run(infernix::Engine& engine, LlmRun& run, std::uint32_t tokens) {
    run.thread = std::thread([&engine, &run, tokens] {
        try {
            run.ids = engine.generate(engine.prepare(story()), greedy(tokens)).generated_token_ids;
        } catch (const std::exception& error) {
            run.error = error.what();
        }
        run.done.store(true);
    });
}

void check_llm_healthy(infernix::Engine& engine, const std::vector<infernix::TokenId>& baseline) {
    const auto after = engine.generate(engine.prepare(story()), greedy(200)).generated_token_ids;
    check(after == baseline, "LLM generation is undisturbed");
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
        infernix::models::qwen4_exp::testing::set_vram_grow_delay(3600.0);

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

        namespace state_ns = infernix::models::qwen4_exp::testing;

        // ---- S1: the success route, end points only (full numerics live in the prod cert test) ----
        std::printf("[S1] success route\n");
        const std::uint64_t free_full = device_free_bytes();
        const std::uint64_t freed = engine.run_image_interlude(
            8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                check(ctx.freed_bytes > 0, "the yield freed " + std::to_string(ctx.freed_bytes >> 20) + " MiB");
                check(state_ns::image_interlude_state() == 3, "the Program is Leased while the job runs");
                check(!state_ns::image_interlude_submit_open(), "the submit door is closed while the job runs");
                tiny_copy(ctx.image_stream);
                check(device_free_bytes() - free_full > kGiB / 2, "driver free VRAM rose across the yield");
                return true;
            });
        check(freed > 0, "run_image_interlude reports the freed bytes");
        check(state_ns::image_interlude_state() == 0, "the Program returned to Running");
        check(state_ns::image_interlude_submit_open(), "the submit door reopened");
        check_llm_healthy(engine, baseline);

        // ---- S2: drain deadline -- the job fails, the LLM keeps serving, nothing moved ----
        std::printf("[S2] drain deadline unwind under live decode\n");
        {
            LlmRun run;
            start_llm_run(engine, run, 200);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            const std::string message = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                        tiny_copy(ctx.image_stream);
                        return true;
                    },
                    /*drain_deadline_ms=*/1);
            });
            check(holds(message, "did not drain within the deadline"),
                  "the deadline-missed interlude names the drain and rejects the job");
            check(state_ns::image_interlude_state() == 0, "the Program unwound to Running");
            check(state_ns::image_interlude_submit_open(), "the door reopened after the unwind");
            run.thread.join();
            check(run.error.empty(), "the interrupted-by-nothing LLM run completed");
            check(run.ids == baseline, "the LLM finished the deadline scenario at baseline ids");
        }
        check_llm_healthy(engine, baseline);

        // ---- S3: job throw + confirmed restoration -> the job's own exception, healthy ----
        std::printf("[S3] job throw rethrows the original exception\n");
        {
            const std::string message = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                        tiny_copy(ctx.image_stream);
                        throw std::runtime_error("injected image job failure 42");
                    });
            });
            check(message == "injected image job failure 42",
                  "the job's own exception propagates unchanged (job failed, engine healthy)");
            check(state_ns::image_interlude_state() == 0, "the Program is Running after the failed job");
            check(state_ns::image_interlude_submit_open(), "the door is open after the failed job");
            check_llm_healthy(engine, baseline);
        }

        // ---- S4: single-flight -- a second caller is told, never silently queued ----
        std::printf("[S4] concurrent interlude rejection\n");
        {
            std::string first_error;
            std::thread holder([&] {
                try {
                    engine.run_image_interlude(
                        8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                            tiny_copy(ctx.image_stream);
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                            return true;
                        });
                } catch (const std::exception& error) {
                    first_error = error.what();
                }
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            const std::string message = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool { return true; });
            });
            holder.join();
            check(first_error.empty(), "the holding interlude itself succeeded");
            check(holds(message, "already in flight"), "the concurrent caller is told another interlude is in flight");
            check(state_ns::image_interlude_state() == 0, "the Program is Running after the pair");
            check(state_ns::image_interlude_submit_open(), "the door is open after the pair");
            check_llm_healthy(engine, baseline);
        }

        // ---- S5: door gate under load -- the worker starts NO body across a live interlude ----
        std::printf("[S5] submit door gates a decoding LLM across the yield\n");
        {
            LlmRun run;
            start_llm_run(engine, run, 200);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            const std::string message = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                        std::this_thread::sleep_for(std::chrono::milliseconds(600));
                        tiny_copy(ctx.image_stream);
                        check(!state_ns::image_interlude_submit_open(), "the door stays closed while the job sleeps");
                        return true;
                    });
            });
            check(message.empty(), "the interlude across a decoding LLM drained, ran and restored");
            run.thread.join();
            check(run.error.empty(), "the LLM run that crossed the interlude completed");
            check(run.ids == baseline, "the interlude paused the LLM deterministically (ids == baseline)");
            check(state_ns::image_interlude_state() == 0, "the Program returned to Running");
            check(state_ns::image_interlude_submit_open(), "the door reopened");
            check_llm_healthy(engine, baseline);
        }

        // ---- S6: restoration fault -> Degraded (terminal; last scenario) ----
        std::printf("[S6] restoration fault parks the engine Degraded\n");
        {
            LlmRun run;
            start_llm_run(engine, run, 600);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            infernix::models::qwen4_exp::testing::set_image_restore_fault(1);
            const std::string message = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext& ctx) -> bool {
                        // Keep the door closed and the image stream live until the LLM run lands,
                        // so the restoration fault lands while a request is active on the device.
                        const auto start = std::chrono::steady_clock::now();
                        while (!run.done.load() &&
                               std::chrono::steady_clock::now() - start < std::chrono::seconds(30)) {
                            tiny_copy(ctx.image_stream);
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }
                        return true;
                    });
            });
            check(holds(message, "restoration failed"), "the thrown error names the restoration failure");
            check(holds(message, "Degraded until process restart"), "the thrown error names the Degraded landing");
            run.thread.join();
            if (!run.error.empty()) {
                check(holds(run.error, "Degraded"), "the active request failed with the Degraded engine message");
            } else {
                check(run.ids == baseline, "the request that finished before the fault was untouched");
            }
            check(state_ns::image_interlude_state() == 4, "the Program is Degraded (terminal)");
            check(!state_ns::image_interlude_submit_open(), "the door stays closed while Degraded");
            check(!engine.is_available(), "is_available() is false while Degraded");
            const std::string second = expect_throw([&] {
                engine.run_image_interlude(
                    8 * kGiB, 0, [&](const infernix::ImageInterludeContext&) -> bool { return true; });
            });
            check(!second.empty() && !holds(second, "already in flight"),
                  "a second interlude throws at the terminal state");
            const std::string submit = expect_throw([&] {
                engine.generate(engine.prepare(story()), greedy(8));
            });
            check(holds(submit, "unavailable"), "further submissions are refused");
        }

        infernix::testing::set_vram_budget_source({});
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "qwen4_exp interlude failure-state certification passed\n"
                              : "FAIL: %d checks failed\n",
                failures);
    return failures == 0 ? 0 : 1;
}
