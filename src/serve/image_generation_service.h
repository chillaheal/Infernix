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
// run_image_job, ending at the finished pixels), the prompt behind PromptRendererProvider (host:
// tokenizer + chat template) and PromptEmbedsProvider (device: the text tower forward). Every
// request is first sized against the model's own buffer inventory and probed against the yield it
// would need: a job that cannot fit is refused BEFORE the LLM quiesces. A second concurrent request
// is told the interlude is already in flight (429); a Degraded engine answers 503 until the process
// restarts (the recovery unit the C5 state machine defines).

#include "infernix/engine.h"
#include "models/qwen_image/image_device_need.h" // VisionLaneDims: the ledger's ViT lane
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/text_tower.h"        // ConditionedVision: the edit route's vision rows

#include <nlohmann/json.hpp>

#include "serve/image_job_store.h"
#include "serve/openai_images.h"
#include "serve/prompt_enhancer.h"
#include "serve/reference_image.h"
#include "serve/request_json.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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

// ---- the edit route's conditioning seams (declared here, like TowerPrompt above) -----------------
// The serving layer owns WHEN each half runs -- the render before the probe, the two forwards inside
// the yield -- so the types the route passes around belong to the route's header, not to
// serve/image_text_tower.h, which only builds them.
using ReferencePromptRendererProvider =
    std::function<TowerPrompt(const std::string& prompt,
                              const std::vector<std::uint64_t>& tokens_per_image)>;

struct ConditionedPromptEmbeds {
    std::vector<std::uint16_t> prompt_embeds; // [(S - drop_rows) * hidden] BF16 bits, pre-final-norm
    std::vector<std::uint8_t> image_pad_mask; // [S - drop_rows], 1 on the rows that carry the picture
};
using VisionRowsProvider =
    std::function<models::qwen_image::ConditionedVision(const ReferenceVisionPlane& plane)>;
using ConditionedEmbedsProvider =
    std::function<ConditionedPromptEmbeds(const TowerPrompt& rendered,
                                          const models::qwen_image::ConditionedVision& vision)>;

// Thrown when another interlude already owns the VRAM yield. A type, not a message match: the HTTP
// surface (429 image_interlude_in_flight) must not depend on the words a Program happens to use.
struct ImageInterludeInFlight : std::runtime_error {
    ImageInterludeInFlight() : std::runtime_error("image interlude: another interlude is already in flight") {}
};

// The finished picture as the device job leaves it: RGBA8 bytes on the host, exactly what the PNG
// encoder consumes. This is the boundary the LLM's pause ends at -- the encode, the base64 and the
// disk write are host work that costs the resident model nothing and therefore happen after the
// interlude has restored it (bundle B fas 1).
struct FinishedPicture {
    std::vector<std::uint8_t> rgba;
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
};

// The VAE of this checkpoint is a FOUR-channel image autoencoder -- encoder conv_in is [96,4,3,3] and
// decoder conv_out is [4,144,3,3] -- and the reference pipeline hands all four decoded channels to
// VaeImageProcessor.numpy_to_pil, which runs Image.fromarray on them uncut: the picture it returns is
// RGBA and its alpha IS the fourth channel, never a constant. The input side already carries alpha raw
// the same way (models/qwen_image/image_pipeline.h, ReferenceEncodeInputs). So the finished picture is
// composed from all four planes: `channels` is the job's channel-major [-1,1] plane, and the result is
// the RGBA8 `FinishedPicture` holds, with the reference's denormalize-then-round mapping.
[[nodiscard]] std::vector<std::uint8_t> rgba_from_generated_channels(const std::vector<float>& channels,
                                                                     std::uint32_t width,
                                                                     std::uint32_t height);

// What the server finally decided about the rewriter -- the answer to the question the capability
// document's `when_auto` clause raises. Until 2026-11-01 the four ways an `auto` request can end up
// drawn from the client's own words were indistinguishable from outside: the card-shortage one surfaced
// only because a probe timed two identical requests against a live server and a capability sentence
// turned out to be false. A skip is legitimate (delegating that judgement is what auto IS), so it is
// RECORDED and reported rather than merely allowed -- "skipped_no_rewriter" versus "skipped_no_device"
// is the difference between a deployment missing a sidecar and a deployment whose card is too tight.
// These strings are the wire values, and the capability document lists them by CALLING to_string, so the
// document can never advertise an outcome the server does not write.
enum class EnhancerOutcome : std::uint8_t {
    Off,                     // absent, false, or prompt_enhance: off
    SkippedNoReference,      // auto, but PE-I2I has no picture to read
    SkippedAlreadyRewritten, // auto, but the instruction already addresses <imageN>
    SkippedNoRewriter,       // auto, but no sidecar answers healthy()
    SkippedNoDevice,         // auto, but the yield cannot cover the rewriter's reserve
    Applied,                 // the rewrite ran; its words are what the picture was drawn from
};

// The wire value for one outcome. The default branch exists for a value no enumerator produces today:
// a reader gets "unknown" rather than being told "off", which would be a different claim.
//
// Inline HERE, not defined in the .cpp: a definition there lands inside that file's anonymous namespace,
// and a call from the same translation unit then matches two functions (C2668), the header's and the
// file-local one. One definition, at the one scope both the record and the capability document see.
[[nodiscard]] inline const char* to_string(EnhancerOutcome outcome) noexcept {
    switch (outcome) {
        case EnhancerOutcome::Off:                     return "off";
        case EnhancerOutcome::SkippedNoReference:      return "skipped_no_reference";
        case EnhancerOutcome::SkippedAlreadyRewritten: return "skipped_already_rewritten";
        case EnhancerOutcome::SkippedNoRewriter:       return "skipped_no_rewriter";
        case EnhancerOutcome::SkippedNoDevice:         return "skipped_no_device";
        case EnhancerOutcome::Applied:                 return "applied";
    }
    return "unknown";
}

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

// Produces one job's finished picture -- RGBA8 pixels and their size, `FinishedPicture` -- executed on
// the interlude's dedicated stream (`image_stream`, a cudaStream_t). It stops at the last byte the
// device touched: the encode, the base64 and the file are the service's work after the interlude.
// `prompt_embeds` is the token-major BF16 bit payload.
// `references` carries every host-prepared reference plane (#75), in request order, whenever the
// request had an `image` field; empty is plain text-to-image. `negative_embeds` carries the prompt's
// BF16 bits when `request.true_cfg_scale` is above 1 (true CFG); empty means the single-forward route.
// `progress` is the stored job's live progress state: the executor's pipeline reports each step
// through it and honors its stall flag; a fake executor may ignore it.
//
// `edit` is the edit route's half of the conditioning and is null for text-to-image, where every
// byte of the certified path stays as it is. With a reference the prompt embeds are NOT pre-computed
// (a plain text-tower forward would produce rows that have never seen the picture, which is the
// 2026-10-28 near-copy): the service hands over the two rendered ti2i prompts and the planes of the
// references, and the executor runs the vision tower and the conditioned text tower inside the yield
// and reads `prompt_image_pad_mask` off the result. `prompt_embeds`/`negative_embeds` are then empty.
// One ViT pass per reference; their rows are concatenated in the order they were sent, which is the
// order the ti2i template numbered them and the order the joint layout scatters their blocks.
struct EditConditioning {
    // The two rendered ti2i prompts. The references themselves travel as the executor's
    // `references` parameter: the VAE encode and the vision tower read the SAME planes, and a second
    // pointer to them here is a second thing to keep in sync with the first.
    const TowerPrompt* prompt          = nullptr; // the raw ti2i render, pads expanded
    const TowerPrompt* negative_prompt = nullptr; // same images, own text; null without true CFG
};
using ImageJobExecutor =
    std::function<FinishedPicture(const OpenAIImagesRequest& request, std::uint64_t seed,
                                   std::span<const std::uint16_t> prompt_embeds,
                                   std::span<const std::uint16_t> negative_embeds,
                                   std::span<const ReferencePlanes> references, void* image_stream,
                                   const std::shared_ptr<ImageJobProgress>& progress,
                                   const EditConditioning* edit)>;

// prompt -> prompt_embeds, split in the two halves the VRAM handoff needs: rendering is host work the
// service does BEFORE it probes the yield, the tower forward needs device memory and runs inside it.
using PromptRendererProvider = std::function<TowerPrompt(const std::string& prompt)>;
using PromptEmbedsProvider   = std::function<std::vector<std::uint16_t>(const TowerPrompt& rendered)>;

// The bound DiT's dimensions, which the need calculation reads. Host-side (the artifact's component
// config), lazy and cached by the caller: it must be readable before any yield, so it can never come
// from the device-side weight binding.
using DiTDimsProvider = std::function<infernix::models::qwen_image::DiTDims()>;

// The bound ViT's dimensions for the ledger's vision term, read the same host-side way (the artifact's
// component config, never the device-side binding). Optional: with no reference in the request the
// term is zero and the provider is never called, so the certified text-to-image route does not need
// one. A request WITH a reference and no provider is refused as unconfigured (503): an admission that
// cannot name the ViT lane's shape cannot claim it fits, and under-reserving OOMs the resident LLM.
using VisionLaneDimsProvider =
    std::function<infernix::models::qwen_image::VisionLaneDims()>;

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

// The host-side ViT dimensions of the same artifact, for the ledger's vision term. Same lazy, cached,
// outside-any-interlude contract as make_artifact_dit_dims_provider; only the edit route asks for it.
// Throws std::runtime_error when the artifact carries no vision component -- the caller must then
// refuse an `image` request rather than admit one it cannot size.
[[nodiscard]] VisionLaneDimsProvider
make_artifact_vision_dims_provider(const std::string& image_artifact_path);

// Standard-alphabet base64 (no line breaks), shared by the image route and its tests.
[[nodiscard]] std::string base64_encode_bytes(std::span<const std::uint8_t> bytes);

// Where a finished picture lives: the `images` collection under the engine's working directory, named
// after the stored job id (img-<created>-<sequence>). This is the single authority on that location --
// the writer here and the GET /v1/images/<id> route both resolve through it. An id that is not the
// job-id shape resolves to an empty path, so a request can never turn a name into a path.
[[nodiscard]] std::filesystem::path generated_image_path(std::string_view id);

// Reads a finished picture back off the same path. This is how the chat tool's `image` argument turns
// into a reference: the picture the model names is this server's own file, not something it can paste.
// Empty when `id` is not an id the artifact route serves, when the file cannot be read, or when it is
// larger than `kMaximumImageInputBytes` -- the caller cannot tell those apart and must not claim to:
// the picture is simply not available to draw from (never drawn here, or retired by the retention
// sweep), which is what the tool tells the model.
[[nodiscard]] std::vector<std::uint8_t> load_generated_image(std::string_view id);

// What one retention pass removed, for the log line the caller reports.
struct ImageRetentionResult {
    std::size_t files = 0;
    std::uint64_t bytes = 0;
};

// One pass of the retention policy over the same `images` collection the writer uses: a picture older
// than `retention` is deleted when the job that drew it is terminal, or when no job names it any more
// (its record aged out long before the file's URL was ever promised to expire). Three things keep the
// sweep from being a janitor with a broad broom: only names that are this route's own job-id shape are
// looked at, so a file another program put in the directory is not ours to delete; a job that is still
// running is never touched; and `retention` of zero or less means keep everything, which is what the
// server does when it starts with the retention switched off. A missing directory is the ordinary
// state before the first picture, not an error. `now` is the caller's clock so the policy is testable
// without waiting a day for the sun.
[[nodiscard]] ImageRetentionResult sweep_generated_images(
    ImageJobStore& store, std::chrono::seconds retention, std::chrono::system_clock::time_point now);

// Runs that sweep once when the server starts and then every hour, until the object goes away -- which
// is the server's own shutdown order, so no sweep is ever running against a store it reads. The hourly
// interval is part of the policy, not a knob: a picture's URL is promised for its retention, and an
// hour is how loosely that promise may then be kept.
class GeneratedImageRetention {
public:
    GeneratedImageRetention(ImageJobStore& store, std::chrono::seconds retention,
                            std::shared_ptr<spdlog::logger> logger);
    ~GeneratedImageRetention();

    GeneratedImageRetention(const GeneratedImageRetention&)            = delete;
    GeneratedImageRetention& operator=(const GeneratedImageRetention&) = delete;

    // The policy's own pass, exposed so a caller (and a test) may run it without owning a thread.
    [[nodiscard]] ImageRetentionResult sweep_now();

private:
    void run_loop();

    ImageJobStore& store_;
    std::chrono::seconds retention_;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool done_ = false;
    std::thread thread_;
};

class ImageGenerationService {
public:
    // `vision_dims` and `reference_prompt_renderer` are only needed by the edit route (a request that
    // carries an `image`); both default to null so every text-to-image caller and test keeps the exact
    // same admission as before. A request WITH a reference and either one missing is refused as
    // unconfigured: the renderer is what the admission sizes the ti2i prompt with, and guessing there
    // under-reserves the yield the resident LLM is resting in.
    // `enhancer` is the family's prompt enhancer sidecar (serve/prompt_enhancer.h). Null -- or set but
    // not answering -- keeps `prompt_enhance` on its documented 503, which is why it defaults: every
    // existing caller and host test gets the text-to-image and edit admission unchanged.
    ImageGenerationService(std::unique_ptr<ImageInterludeRunner> runner, ImageJobStore& store,
                           PromptRendererProvider prompt_renderer, PromptEmbedsProvider prompt_embeds,
                           DiTDimsProvider dit_dims, ImageJobExecutor job_executor,
                           VisionLaneDimsProvider vision_dims = {},
                           ReferencePromptRendererProvider reference_prompt_renderer = {},
                           std::shared_ptr<PromptEnhancer> enhancer = {});

    // False when serve started without the image pipeline: the route answers a documented 503.
    [[nodiscard]] bool is_configured() const noexcept;

    // How many pictures this process is drawing right now, and how long the oldest of them has been
    // going. /health carries both so a supervisor can tell a BUSY engine from a dead one: a chat probe
    // queued behind a two-minute picture times out, and a keepalive that reads that timeout as death
    // taskkills the serve and the picture with it -- which is what happened on 2026-11-01 22:30. The
    // count covers the whole of generate(), the rewriter's sampling included, because the card is held
    // for all of it. Zero means no picture is in flight and the age is reported as 0.
    [[nodiscard]] std::size_t in_flight_jobs() const;
    [[nodiscard]] std::int64_t oldest_in_flight_age_seconds(std::int64_t now) const;

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
    //   400 image_too_large / image_decode_failed (the `image` payload exceeded its budget, or is not
    //       a base64 PNG/JPEG the decoders accept; the codec's own words are never echoed)
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
    VisionLaneDimsProvider vision_dims_;
    ReferencePromptRendererProvider reference_prompt_renderer_;
    ImageJobExecutor job_executor_;
    std::shared_ptr<PromptEnhancer> enhancer_;
    std::mutex dims_mutex_;
    infernix::models::qwen_image::DiTDims dims_cache_;
    std::mutex sequence_mutex_;
    std::uint64_t sequence_ = 0;
    mutable std::mutex progress_mutex_;
    std::shared_ptr<ImageJobProgress> last_progress_;
    // One entry per job inside generate(): the token InFlightJob was built with, mapped to the second
    // it started. A map rather than a counter because the age has to survive jobs ending out of order,
    // and std::map's first key is the oldest token, which is the oldest job.
    mutable std::mutex in_flight_mutex_;
    std::map<std::uint64_t, std::int64_t> in_flight_;
    std::uint64_t in_flight_token_ = 0;
};

} // namespace infernix::serve
