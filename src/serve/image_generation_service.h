#pragma once

// The synchronous POST /v1/images/generations execution. One validated OpenAI image request
// becomes ONE Engine image interlude (frontier drain + VRAM yield + restoration), one production
// image job on the yielded VRAM, and the OpenAI images response -- returned only after the LLM's
// restoration is confirmed. The lifecycle is recorded in the ImageJobStore
// (Queued -> WaitingForFrontier -> Running -> Restoring -> Succeeded / Failed /
// FailedEngineDegraded); the synchronous wire does not expose cancellation or retrieval.
//
// Host-testable seams: the interlude behind ImageInterludeRunner (production: infernix::Engine),
// the device job behind ImageJobExecutor (production: the artifact's streamed weights +
// run_image_job + PNG encode), and the prompt behind PromptRendererProvider (host: tokenizer + chat
// template) and PromptEmbedsProvider (device: the text tower forward). Every request is first sized
// against the model's own buffer inventory and probed against the yield it would need: a job that
// cannot fit is refused BEFORE the LLM quiesces. A second concurrent request is told the interlude is
// already in flight (429); a Degraded engine answers 503 until the process restarts (the recovery unit
// the C5 state machine defines).

#include "infernix/engine.h"
#include "models/qwen_image/image_weights.h"

#include <nlohmann/json.hpp>

#include "serve/image_job_store.h"
#include "serve/openai_images.h"
#include "serve/reference_image.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spdlog {
class logger;
}

namespace infernix::serve {

// The exact templated prompt the text tower will run: the rendered text, its full causal ids, and the
// leading system-prefix rows the DiT lane drops. Producing it is PURE HOST WORK (tokenizer + chat
// template), which is why it is its own seam: the service needs the token COUNT before it asks
// whether the job fits, and that count only exists once the prompt is rendered.
struct TowerPrompt {
    std::string rendered;
    std::vector<int> ids;
    std::size_t drop_rows = 0;

    // The rows the joint sequence actually carries.
    [[nodiscard]] std::size_t tokens() const noexcept { return ids.size() - drop_rows; }
};

// Thrown when another interlude already owns the VRAM yield. A type, not a message match: the HTTP
// surface (429 image_interlude_in_flight) must not depend on the words a Program happens to use.
struct ImageInterludeInFlight : std::runtime_error {
    ImageInterludeInFlight() : std::runtime_error("image interlude: another interlude is already in flight") {}
};

// Seam over Engine's image interlude; the host test runs the job inline on a fake context.
class ImageInterludeRunner {
public:
    virtual ~ImageInterludeRunner() = default;
    [[nodiscard]] virtual bool supports_interlude() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept       = 0;

    // Whether a job needing `target_bytes` (+ `margin_bytes`) fits, answered WITHOUT quiescing the LLM.
    // The service refuses on this before the submit door closes, and keeps `freed_bytes_expected` to
    // check what the yield actually delivered against what the plan promised.
    struct Fit {
        bool fits = false;
        std::uint64_t freed_bytes_expected = 0;
        std::uint64_t device_free_bytes    = 0;
        std::uint64_t headroom_bytes       = 0;
    };
    [[nodiscard]] virtual Fit probe(std::uint64_t target_bytes, std::uint64_t margin_bytes) = 0;

    virtual std::uint64_t
    run(std::uint64_t target_bytes, std::uint64_t margin_bytes,
        const std::function<bool(const infernix::ImageInterludeContext&)>& job) = 0;
};

// Production runner over the live Engine.
[[nodiscard]] std::unique_ptr<ImageInterludeRunner> make_engine_interlude_runner(infernix::Engine& engine);

// Produces the finished PNG bytes for one job, executed on the interlude's dedicated stream
// (`image_stream`, a cudaStream_t). `prompt_embeds` is the token-major BF16 bit payload.
// `reference` carries the host-prepared reference-image plane (#75) whenever the request had an
// `image` field; nullptr is plain text-to-image. `negative_embeds` carries the negative prompt's
// BF16 bits when `request.true_cfg_scale` is above 1 (true CFG); empty means the single-forward route.
// `progress` is the stored job's live progress state: the executor's pipeline reports each step
// through it and honors its stall flag; a fake executor may ignore it.
using ImageJobExecutor =
    std::function<std::vector<std::uint8_t>(const OpenAIImagesRequest& request, std::uint64_t seed,
                                            std::span<const std::uint16_t> prompt_embeds,
                                            std::span<const std::uint16_t> negative_embeds,
                                            const ReferencePixels* reference, void* image_stream,
                                            const std::shared_ptr<ImageJobProgress>& progress)>;

// prompt -> prompt_embeds, split in the two halves the VRAM handoff needs: rendering is host work the
// service does BEFORE it probes the yield, the tower forward needs device memory and runs inside it.
using PromptRendererProvider = std::function<TowerPrompt(const std::string& prompt)>;
using PromptEmbedsProvider   = std::function<std::vector<std::uint16_t>(const TowerPrompt& rendered)>;

// The bound DiT's dimensions, which the need calculation reads. Host-side (the artifact's component
// config), lazy and cached by the caller: it must be readable before any yield, so it can never come
// from the device-side weight binding.
using DiTDimsProvider = std::function<infernix::models::qwen_image::DiTDims()>;

// The production executor: lazy-binds the DiT + VAE weights and the VAE denormalization constants
// from one .infernix image artifact (one streamed-only plan) and runs the merged pipeline entry
// on the caller's stream. A bad artifact throws std::runtime_error on first use. `logger` receives
// the job's stage telemetry: the application logger owns the terminal panel, so its lines land
// above the stats panel instead of raw stderr text that would collide with the panel repaint.
[[nodiscard]] ImageJobExecutor make_artifact_image_job_executor(
    const std::string& image_artifact_path, std::shared_ptr<spdlog::logger> logger = nullptr);

// The host-side DiT dimensions of one .infernix image artifact, read from its component config through
// the same binding the job uses. Called OUTSIDE any interlude (it is host work the admission needs
// before the yield) and cached by the service, so the artifact opens once per process.
[[nodiscard]] DiTDimsProvider make_artifact_dit_dims_provider(const std::string& image_artifact_path);

// Standard-alphabet base64 (no line breaks), shared by the image route and its tests.
[[nodiscard]] std::string base64_encode_bytes(std::span<const std::uint8_t> bytes);

// Where a finished picture lives: the `images` collection under the engine's working directory, named
// after the stored job id (img-<created>-<sequence>). This is the single authority on that location --
// the writer here and the GET /v1/images/<id> route both resolve through it. An id that is not the
// job-id shape resolves to an empty path, so a request can never turn a name into a path.
[[nodiscard]] std::filesystem::path generated_image_path(std::string_view id);

class ImageGenerationService {
public:
    ImageGenerationService(std::unique_ptr<ImageInterludeRunner> runner, ImageJobStore& store,
                           PromptRendererProvider prompt_renderer, PromptEmbedsProvider prompt_embeds,
                           DiTDimsProvider dit_dims, ImageJobExecutor job_executor);

    // False when serve started without the image pipeline: the route answers a documented 503.
    [[nodiscard]] bool is_configured() const noexcept;

    // Synchronous generation. `now` is the store clock (epoch seconds), stamped as `created`.
    // `artifact_url_base` is the absolute URL prefix of this server's artifact route
    // (http://<host>/v1/images/); a request that asked for response_format "url" receives that prefix
    // followed by the stored job id, which is the picture's durable name on disk.
    // Returns the OpenAI images response object only after the interlude restored the LLM.
    // Every failure ends the stored job terminally and throws ApiException:
    //   503 image_generation_not_configured / image_interlude_unsupported / engine_unavailable
    //   429 image_interlude_in_flight          (another interlude owns the VRAM yield)
    //   503 image_engine_degraded              (Degraded: process restart is the recovery unit)
    //   503 image_interlude_failed             (the job never ran, e.g. a drain-deadline miss)
    //   500 image_generation_failed            (the job itself failed; restoration confirmed)
    //   400 image_invalid                      (the `image` field could not be prepared)
    RequestJson generate(const OpenAIImagesRequest& request, std::int64_t now,
                         std::string_view artifact_url_base = {});

    // Process-unique stored-job id: img-<created>-<sequence>.
    [[nodiscard]] std::string next_job_id(std::int64_t created);

    // The most recently accepted job's live progress state, or null before the first job. The
    // streaming chat handlers poll it from their HeartbeatTicker while the picture is drawn (the
    // hidden tool round runs before the response, so the request that triggered the job is the
    // one ticker this can ever name).
    [[nodiscard]] std::shared_ptr<ImageJobProgress> last_job_progress() const;

    // The GET /v1/images/<id> status object for a stored job: {"id", "status",
    // "progress_percent"}. Non-terminal jobs report "running"; terminals carry their stored
    // state name ("succeeded", "failed", ...). Null when the id names no job at all (the route
    // keeps its documented 404 for that). The picture itself still answers from the file for
    // succeeded jobs; this object is what an unfinished or failed job exposes.
    [[nodiscard]] nlohmann::json job_status_json(std::string_view id) const;

private:
    std::unique_ptr<ImageInterludeRunner> runner_;
    ImageJobStore& store_;
    PromptRendererProvider prompt_renderer_;
    PromptEmbedsProvider prompt_embeds_;
    DiTDimsProvider dit_dims_;
    ImageJobExecutor job_executor_;
    std::mutex dims_mutex_;
    infernix::models::qwen_image::DiTDims dims_cache_;
    std::mutex sequence_mutex_;
    std::uint64_t sequence_ = 0;
    mutable std::mutex progress_mutex_;
    std::shared_ptr<ImageJobProgress> last_progress_;
};

} // namespace infernix::serve
