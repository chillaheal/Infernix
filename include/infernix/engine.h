#pragma once

#include "infernix/types.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace infernix {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()), and the
    // distribution outputs options requests for those positions.
    [[nodiscard]] ScoreResult score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                           ScoreOptions options = {});

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] ModelMetadata model_metadata() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    // Which instruction positions the active chat template renders; prompt preparation lowers an
    // instruction turn that arrived in an unsupported position to a role the template does render.
    [[nodiscard]] ChatTemplateRoleSupport chat_template_role_support() const;
    [[nodiscard]] bool is_available() const;

    // LLM<->image VRAM coexistence: whether the loaded model supports an image interlude (today the
    // qwen4_exp Program). run_image_interlude suspends generation, yields physical VRAM toward the
    // image job's `target_bytes` plus a `margin_bytes` kept below the headroom, runs `job` with the
    // LLM unable to execute (it receives the ImageInterludeContext: the bytes actually freed and
    // its dedicated CUDA stream, and allocates the image's scratch in the freed region), then
    // restores the pools and resumes generation. The interlude's stream is destroyed when `job`
    // returns or throws. `drain_deadline_ms` bounds the wait for the LLM frontier to drain before
    // anything is moved; on a miss the job fails and the LLM keeps serving (0 waits forever).
    // Exception-safe: a failed job with confirmed restoration rethrows the job's own exception;
    // only an uncertain restoration parks the model Degraded (process restart is the recovery
    // unit). One interlude at a time; a concurrent caller is told another is in flight. Success is
    // published only after restoration is confirmed. Returns the bytes freed. Throws
    // std::logic_error when the Engine is moved from or its model supports no interlude.
    [[nodiscard]] bool supports_image_interlude() const noexcept;
    std::uint64_t run_image_interlude(std::uint64_t target_bytes, std::uint64_t margin_bytes,
                                      const std::function<bool(const ImageInterludeContext&)>& job,
                                      std::uint64_t drain_deadline_ms = 60000);

    // Whether an image job wanting `target_bytes` (plus `margin_bytes` kept below the headroom) would
    // fit right now. Runs the same pure plan the interlude applies against the live pools, WITHOUT
    // quiescing, draining, unmapping or resizing anything: a caller that gets `fits == false` refuses
    // the request before the submit door closes, so a job that cannot run never stalls the LLM. Legal
    // while the Engine is serving. Throws std::logic_error when the Engine is moved from or its model
    // supports no interlude.
    [[nodiscard]] ImageYieldProbe probe_image_yield(std::uint64_t target_bytes,
                                                   std::uint64_t margin_bytes) const;

    void reset_memory_peaks() noexcept;

    // Begins the orderly stop without waiting for it: new work is refused, and queued and active
    // generation requests end with an Unavailable error within one unit of work. A Generation
    // Engine then saves its prefix cache file, when configured. Destruction waits for the stop.
    // Idempotent and callable from any thread.
    void stop() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace infernix
