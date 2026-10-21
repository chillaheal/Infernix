#include "serve/image_generation_service.h"

#include "artifact/reader.h"
#include "media/decode/decode.h"
#include "media/encode/png_encode.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/noise.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>

namespace infernix::serve {

namespace md = media::decode;

namespace {

namespace qi = models::qwen_image;

// The interlude hands the quiesced LLM's pools to the image job and restores them on return, so the
// job may take far more than the LLM's baseline free. The job's device footprint grows with
// latent-token count (~ width*height): the certified 512x512 production route peaked at 6183 MiB
// (interlude_s8dprod_test.out.log). Size the yield to the requested area with a margin for the
// attention's super-linear terms so 1024x1024 and larger get the headroom their activations need;
// clamp to [7 GiB, 26 GiB] on the 32 GiB card.
constexpr std::uint64_t kInterludeMarginBytes = 0;
std::uint64_t image_yield_target(std::int64_t width, std::int64_t height) {
    constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;
    constexpr std::uint64_t baseline_bytes = 6183ULL * kMiB;          // measured 512x512 peak
    constexpr std::uint64_t baseline_area  = 512ULL * 512ULL;
    constexpr std::uint64_t floor_bytes    = 7ULL * 1024ULL * kMiB;   // 7 GiB
    constexpr std::uint64_t cap_bytes      = 26ULL * 1024ULL * kMiB;  // 26 GiB
    const std::uint64_t area = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    std::uint64_t bytes = baseline_bytes * area / baseline_area;
    bytes = (bytes * 8ULL) / 5ULL; // +60% margin for super-linear attention terms
    if (bytes < floor_bytes) { bytes = floor_bytes; }
    if (bytes > cap_bytes) { bytes = cap_bytes; }
    return bytes;
}

std::uint16_t to_bf16(float value) {
    std::uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((x >> 16) | 0x0040U);
    const std::uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<std::uint16_t>(x >> 16);
}

float to_unit8(float signed_unit) {
    const double v = std::max(-1.0, std::min(1.0, static_cast<double>(signed_unit)));
    return static_cast<float>(std::lround(255.0 * 0.5 * (v + 1.0)));
}

class EngineInterludeRunner final : public ImageInterludeRunner {
public:
    explicit EngineInterludeRunner(infernix::Engine& engine) noexcept : engine_(engine) {}

    [[nodiscard]] bool supports_interlude() const noexcept override {
        return engine_.supports_image_interlude();
    }
    [[nodiscard]] bool is_available() const noexcept override { return engine_.is_available(); }

    std::uint64_t
    run(std::uint64_t target_bytes, std::uint64_t margin_bytes,
        const std::function<bool(const infernix::ImageInterludeContext&)>& job) override {
        return engine_.run_image_interlude(target_bytes, margin_bytes, job);
    }

private:
    infernix::Engine& engine_;
};

// The production job body: streamed artifact weights + the merged pipeline entry + PNG encode.
// One shared_ptr instance is lazy-bound on first use and reused for every later job (the certified
// route reads the 32 blocks' weights from the artifact once per job from this live reader).
class ArtifactImageJobExecutor {
public:
    explicit ArtifactImageJobExecutor(std::string artifact_path) : path_(std::move(artifact_path)) {}

    std::vector<std::uint8_t> operator()(const OpenAIImagesRequest& request, std::uint64_t seed,
                                         std::span<const std::uint16_t> prompt_embeds,
                                         std::span<const std::uint16_t> negative_embeds,
                                         const ReferencePixels* reference, void* image_stream,
                                         const std::shared_ptr<ImageJobProgress>& progress) {
        std::lock_guard lock(mutex_);
        ensure_locked();

        // The pipeline's units (steps + VAE) map onto the job's 0-100; the PNG encode's unit is
        // this executor's own, below. A null progress (a caller without a stored job) simply
        // certifies the route unchanged.
        struct ProgressAdapter final : qi::ImageJobProgressView {
            explicit ProgressAdapter(ImageJobProgress* state) : state_(state) {}
            [[nodiscard]] bool step(std::size_t units_done, std::size_t units_total) override {
                return state_->advance(static_cast<int>(units_done * 100 / units_total));
            }
            ImageJobProgress* state_;
        } view(progress.get());

        qi::ImageJob desc;
        desc.height     = request.height;
        desc.width      = request.width;
        desc.steps      = request.steps;
        const qi::ImageGeometry geom = desc.geometry();

        const qi::Schedule& sched = schedule_cache_for(geom.height, geom.width, request.steps);
        qi::ImageJobInputs inputs;
        inputs.prompt_embeds.assign(prompt_embeds.begin(), prompt_embeds.end());
        inputs.timesteps = sched.timesteps;
        inputs.sigmas    = sched.sigmas;
        inputs.init_latent.resize(static_cast<std::size_t>(geom.tokens) * dit_.dims.image_in);
        {
            std::vector<float> noise(inputs.init_latent.size());
            qi::fill_normal(seed, noise);
            for (std::size_t i = 0; i < noise.size(); ++i) inputs.init_latent[i] = to_bf16(noise[i]);
        }
        inputs.latents_mean = denorm_.mean;
        inputs.latents_std  = denorm_.std;
        inputs.height_px    = geom.height;
        inputs.width_px     = geom.width;
        if (!negative_embeds.empty()) {
            inputs.negative_prompt_embeds.assign(negative_embeds.begin(), negative_embeds.end());
            inputs.true_cfg_scale = static_cast<float>(request.true_cfg_scale);
        }

        // #75: the host-prepared reference plane becomes the constant reference block of the
        // joint sequence, encoded with the same VAE the job decodes with (inside the yielded VRAM).
        if (reference != nullptr) {
            qi::ReferenceEncodeInputs encode_in;
            encode_in.pixels       = reference->pixels;
            encode_in.latents_mean = denorm_.mean;
            encode_in.latents_std  = denorm_.std;
            encode_in.height_px    = reference->height_px;
            encode_in.width_px     = reference->width_px;
            const qi::EncodedReference encoded =
                qi::encode_reference_image(*reader_, vae_, encode_in,
                                           static_cast<cudaStream_t>(image_stream));
            inputs.ref_latents = std::move(encoded.latents);
            inputs.ref_grids.emplace_back(encoded.height_lat, encoded.width_lat);
        }

        const qi::GeneratedImage out = qi::run_image_job(
            *reader_, dit_, vae_, inputs, static_cast<cudaStream_t>(image_stream),
            progress ? &view : nullptr);

        // The certified route composites the decoded image's first three channels as RGB and
        // emits opaque alpha.
        const std::size_t pixels = static_cast<std::size_t>(out.height) * out.width;
        std::vector<std::uint8_t> rgba(4ULL * pixels);
        for (std::size_t i = 0; i < pixels; ++i) {
            rgba[4ULL * i + 0] = static_cast<std::uint8_t>(to_unit8(out.channels[0 * pixels + i]));
            rgba[4ULL * i + 1] = static_cast<std::uint8_t>(to_unit8(out.channels[1 * pixels + i]));
            rgba[4ULL * i + 2] = static_cast<std::uint8_t>(to_unit8(out.channels[2 * pixels + i]));
            rgba[4ULL * i + 3] = 255U;
        }
        return media::encode::encode_png_rgba8(out.width, out.height, rgba);
    }

private:
    void ensure_locked() {
        if (reader_ != nullptr) { return; }
        if (path_.empty()) { throw std::runtime_error("image job: no image artifact is configured"); }
        reader_ = std::make_unique<artifact::Reader>(path_);
        binder_ = std::make_unique<artifact::Binder>(*reader_);
        dit_    = qi::bind_dit(*binder_);
        vae_    = qi::bind_image_vae(*binder_);
        denorm_ = qi::read_vae_denorm(*binder_);
        if (denorm_.mean.size() != dit_.dims.image_in) {
            throw std::runtime_error(
                "image job: the artifact's VAE denormalization length differs from the DiT image_in");
        }
        (void)std::move(*binder_).finish();
    }

    // The resolution-dependent schedule is fixed per (height, width, steps); the certified
    // production scheduler is pure, so a tiny memoized cache keeps repeated jobs identical and cheap.
    const qi::Schedule& schedule_cache_for(std::uint32_t height_px, std::uint32_t width_px,
                                           std::uint32_t steps) {
        for (const auto& entry : schedules_) {
            if (entry.key_height == height_px && entry.key_width == width_px && entry.key_steps == steps) {
                return entry.schedule;
            }
        }
        qi::ImageJob desc;
        desc.height = height_px;
        desc.width  = width_px;
        desc.steps  = steps;
        schedules_.push_back(ScheduleCache{height_px, width_px, steps,
                                           desc.schedule(qi::SchedulerConfig{})});
        return schedules_.back().schedule;
    }

    struct ScheduleCache {
        std::uint32_t key_height = 0;
        std::uint32_t key_width  = 0;
        std::uint32_t key_steps  = 0;
        qi::Schedule schedule;
    };

    std::string path_;
    std::mutex mutex_;
    std::unique_ptr<artifact::Reader> reader_;
    std::unique_ptr<artifact::Binder> binder_;
    qi::DiTWeights dit_;
    qi::VaeWeights vae_;
    qi::VaeDenorm denorm_;
    std::vector<ScheduleCache> schedules_;
};

std::string base64_from(const std::uint8_t* data, std::size_t size) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (std::size_t i = 0; i < size; i += 3) {
        const std::size_t left = size - i;
        std::uint32_t group    = static_cast<std::uint32_t>(data[i]) << 16;
        if (left > 1) { group |= static_cast<std::uint32_t>(data[i + 1]) << 8; }
        if (left > 2) { group |= static_cast<std::uint32_t>(data[i + 2]); }
        out += alphabet[group >> 18 & 63];
        out += alphabet[group >> 12 & 63];
        out += left > 1 ? alphabet[group >> 6 & 63] : '=';
        out += left > 2 ? alphabet[group & 63] : '=';
    }
    return out;
}

} // namespace

std::unique_ptr<ImageInterludeRunner> make_engine_interlude_runner(infernix::Engine& engine) {
    return std::make_unique<EngineInterludeRunner>(engine);
}

ImageJobExecutor make_artifact_image_job_executor(const std::string& image_artifact_path) {
    auto executor = std::make_shared<ArtifactImageJobExecutor>(image_artifact_path);
    return [executor](const OpenAIImagesRequest& request, std::uint64_t seed,
                      std::span<const std::uint16_t> prompt_embeds,
                      std::span<const std::uint16_t> negative_embeds, const ReferencePixels* reference,
                      void* image_stream, const std::shared_ptr<ImageJobProgress>& progress) {
        return (*executor)(request, seed, prompt_embeds, negative_embeds, reference, image_stream,
                           progress);
    };
}

std::string base64_encode_bytes(std::span<const std::uint8_t> bytes) {
    return base64_from(bytes.data(), bytes.size());
}

// Generated images are collected in an `images` directory under the engine's working directory, so a
// run leaves the pictures on disk without the caller decoding the response body. The file name is the
// stored job id (img-<created>-<sequence>): unique per process and chronological.
std::filesystem::path generated_image_path(std::string_view id) {
    if (!is_image_artifact_id(id)) { return {}; }
    return std::filesystem::current_path() / "images" / (std::string(id) + ".png");
}

std::filesystem::path save_generated_image(const std::string& id, std::span<const std::uint8_t> png) {
    std::error_code error;
    const std::filesystem::path file      = generated_image_path(id);
    const std::filesystem::path directory = file.parent_path();
    std::filesystem::create_directories(directory, error);
    if (error || !std::filesystem::is_directory(directory)) {
        throw std::runtime_error("image save: cannot create " + directory.string() + " (" +
                                 (error ? error.message() : std::string("not a directory")) + ")");
    }
    std::ofstream output(file, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("image save: cannot open " + file.string() + " for writing"); }
    output.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    output.close();
    if (!output) { throw std::runtime_error("image save: the write of " + file.string() + " did not finish"); }
    return file;
}

ImageGenerationService::ImageGenerationService(std::unique_ptr<ImageInterludeRunner> runner,
                                               ImageJobStore& store,
                                               PromptEmbedsProvider prompt_embeds,
                                               ImageJobExecutor job_executor)
    : runner_(std::move(runner)),
      store_(store),
      prompt_embeds_(std::move(prompt_embeds)),
      job_executor_(std::move(job_executor)) {}

bool ImageGenerationService::is_configured() const noexcept {
    return runner_ != nullptr && prompt_embeds_ != nullptr && job_executor_ != nullptr;
}

std::string ImageGenerationService::next_job_id(std::int64_t created) {
    std::lock_guard lock(sequence_mutex_);
    return "img-" + std::to_string(created) + "-" + std::to_string(++sequence_);
}

std::shared_ptr<ImageJobProgress> ImageGenerationService::last_job_progress() const {
    const std::lock_guard lock(progress_mutex_);
    return last_progress_;
}

RequestJson ImageGenerationService::generate(const OpenAIImagesRequest& request, std::int64_t now,
                                             std::string_view artifact_url_base) {
    if (!is_configured()) {
        throw ApiException{ApiError{503, "server_error", "the image pipeline is not configured",
                                    "image_artifact", "image_generation_not_configured"}};
    }
    if (!runner_->supports_interlude()) {
        throw ApiException{ApiError{503, "server_error",
                                    "the loaded model supports no image interlude", "",
                                    "image_interlude_unsupported"}};
    }
    if (!runner_->is_available()) {
        throw ApiException{ApiError{503, "server_error", "the engine is not available", "",
                                    "engine_unavailable"}};
    }

    // #75 reference image: base64/decode/resize is host work and fails fast as a 400 before any
    // VRAM is yielded. The device encode of the prepared plane runs inside the job.
    std::optional<ReferencePixels> reference;
    if (!request.image.empty()) {
        try {
            reference = prepare_reference_image(request.image);
        } catch (const md::Error& error) {
            throw ApiException{ApiError{400, "invalid_request_error", error.what(), "image",
                                        error.kind() == md::ErrorKind::BudgetExceeded
                                            ? "image_too_large"
                                            : "image_invalid"}};
        } catch (const std::invalid_argument& error) {
            throw ApiException{ApiError{400, "invalid_request_error", error.what(), "image",
                                        "image_invalid"}};
        }
    }

    StoredImageJob record;
    record.id       = next_job_id(now);
    record.state    = ImageJobState::Queued;
    record.created  = now;
    record.progress = std::make_shared<ImageJobProgress>();
    record.request  = nlohmann::json{{"prompt", request.prompt},
                                     {"width", request.width},
                                     {"height", request.height},
                                     {"steps", request.steps}};
    store_.put(record);
    {
        const std::lock_guard lock(progress_mutex_);
        last_progress_ = record.progress;
    }

    std::atomic<bool> job_started{false};
    RequestJson response;
    try {
        store_.advance(record.id, JobUpdate{.to = ImageJobState::WaitingForFrontier, .now = now});
        runner_->run(image_yield_target(request.width, request.height), kInterludeMarginBytes,
                     [&](const infernix::ImageInterludeContext& context) -> bool {
                         store_.advance(record.id, JobUpdate{.to = ImageJobState::Running, .now = now});
                         job_started = true;
                         std::uint64_t seed = 0;
                         if (request.seed.has_value()) {
                             seed = *request.seed;
                         } else {
                             std::random_device random;
                             seed = (static_cast<std::uint64_t>(random()) << 32U) | random();
                         }
                         // The negative prompt goes through the same device text tower as the prompt,
                         // only when the request actually asked to steer with it.
                         const std::vector<std::uint16_t> negative =
                             request.true_cfg_scale > kMinimumImageCFGScale
                                 ? prompt_embeds_(request.negative_prompt)
                                 : std::vector<std::uint16_t>{};
                         const auto png = job_executor_(
                             request, seed, prompt_embeds_(request.prompt), negative,
                             reference.has_value() ? &*reference : nullptr, context.image_stream,
                             record.progress);
                         // The pipeline reported (steps + VAE)/100; the encoded PNG is the job's
                         // last unit and the picture is complete.
                         if (record.progress) { record.progress->advance(100); }
                         save_generated_image(record.id, png);
                         response = request.response_format == ImageResponseFormat::Url
                                      ? make_openai_image_url_response(
                                            now, {std::string(artifact_url_base) + record.id})
                                      : make_openai_image_response(now,
                                                                   {base64_encode_bytes(png)});
                         store_.advance(record.id, JobUpdate{.to = ImageJobState::Restoring, .now = now});
                         return true;
                     });
        store_.advance(record.id,
                       JobUpdate{.to = ImageJobState::Succeeded, .now = now,
                                 .response = nlohmann::json::parse(response.dump())});
        return response;
    } catch (const std::exception& error) {
        const std::string message  = error.what();
        const auto failure         = nlohmann::json{{"message", message}};
        const bool degraded        = !runner_->is_available();
        const bool in_flight_claim = message.find("already in flight") != std::string::npos;
        try {
            if (in_flight_claim) {
                store_.advance(record.id, JobUpdate{.to = ImageJobState::Failed, .now = now,
                                                    .error = failure});
            } else if (degraded) {
                // The C5 state machine: restoration could not be confirmed. Both failures are kept.
                store_.advance(record.id, JobUpdate{.to = ImageJobState::FailedEngineDegraded, .now = now,
                                                    .error = failure, .restoration_error = failure});
            } else {
                store_.advance(record.id, JobUpdate{.to = ImageJobState::Failed, .now = now,
                                                    .error = failure});
            }
        } catch (const ApiException&) {
            // A terminal race (an earlier cancellation): the stored terminal state wins.
        }
        if (in_flight_claim) {
            throw ApiException{ApiError{429, "rate_limit_error", message, "", "image_interlude_in_flight"}};
        }
        if (degraded) {
            throw ApiException{ApiError{503, "server_error", message, "", "image_engine_degraded"}};
        }
        if (job_started) {
            throw ApiException{ApiError{500, "server_error", message, "", "image_generation_failed"}};
        }
        throw ApiException{ApiError{503, "server_error", message, "", "image_interlude_failed"}};
    }
}

} // namespace infernix::serve
