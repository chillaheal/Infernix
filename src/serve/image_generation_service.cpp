#include "serve/image_generation_service.h"

// The edit route's device halves (the vision tower and the conditioned text tower forwards) are built
// here and run inside the interlude, which is the only place the device may be touched.
#include "serve/image_text_tower.h"

#include "artifact/reader.h"
#include "media/decode/decode.h"
#include "media/encode/png_encode.h"
#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/noise.h"

#include <spdlog/logger.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

// The free VRAM an image job leaves below what it takes. A yield that lands a hair short of the job's
// need is the failure mode that hurts: the allocation throws inside the interlude, and (depending on
// where) the restoration is the thing that fails. Half a GiB is the price of not finding out.
constexpr std::uint64_t kInterludeMarginBytes = 512ULL * kMiB;

// The job's device-memory need, from the pipeline's own buffer inventory (models/qwen_image). The
// serving layer asks for it before it probes the yield, so the reference block and the longer of the
// two CFG passes are already inside the number.
// `vision_tokens` / `vision` are the edit route's: the reference's vision token count (the
// <|image_pad|> rows already inside text_tokens, which the joint sequence carries four times each) and
// the bound ViT's shape. Both stay 0/zero for the certified text-to-image route, which keeps every
// term -- and total() -- bit-identical to what it was before the edit route existed.
std::uint64_t image_need_bytes(std::int64_t width, std::int64_t height, std::size_t text_tokens,
                              std::size_t negative_tokens, std::size_t ref_tokens,
                              const qi::DiTDims& dims, std::size_t vision_tokens = 0,
                              const qi::VisionLaneDims& vision = {}) {
    qi::ImageDeviceNeedRequest req;
    req.vision_tokens   = static_cast<std::uint32_t>(vision_tokens);
    req.vision          = vision;
    req.geometry        = qi::ImageGeometry::derive(static_cast<std::uint32_t>(height),
                                                   static_cast<std::uint32_t>(width));
    req.dims            = dims;
    req.text_tokens     = static_cast<std::uint32_t>(text_tokens);
    req.negative_tokens = static_cast<std::uint32_t>(negative_tokens);
    req.ref_tokens      = static_cast<std::uint32_t>(ref_tokens);
    req.blocks_resident = qi::kImageResidentBlocks; // the shipped route double-buffers two DiT block
                             // slots (copy prefetches i+1 while i computes); the same symbol the
                             // pipeline asserts and the cert requests, so the need can't drift from it
    return qi::image_device_need(req).total();
}

// The one refusal the request can be told about in its own words: what the job needs, what the model
// could give, and what was already free. A retry with a smaller size or without true CFG is the fix,
// so this is 503 (the answer changes with the device's state), not 400 (the size is never forbidden).
[[noreturn]] void throw_vram_insufficient(std::uint64_t need, const ImageInterludeRunner::Fit& fit) {
    throw ApiException{ApiError{503, "server_error",
                                "the image job needs " + std::to_string(need / kMiB) +
                                    " MiB; the model can yield " +
                                    std::to_string(fit.freed_bytes_expected / kMiB) + " MiB with " +
                                    std::to_string(fit.device_free_bytes / kMiB) +
                                    " MiB already free and " +
                                    std::to_string(fit.headroom_bytes / kMiB) +
                                    " MiB reserved below the headroom",
                                "size", "image_vram_insufficient"}};
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

    [[nodiscard]] Fit probe(std::uint64_t target_bytes, std::uint64_t margin_bytes) override {
        const infernix::ImageYieldProbe probe = engine_.probe_image_yield(target_bytes, margin_bytes);
        return Fit{.fits = probe.fits,
                   .freed_bytes_expected = probe.freed_bytes_expected,
                   .device_free_bytes    = probe.device_free_bytes,
                   .headroom_bytes       = probe.headroom_bytes};
    }

    std::uint64_t
    run(std::uint64_t target_bytes, std::uint64_t margin_bytes,
        const std::function<bool(const infernix::ImageInterludeContext&)>& job) override {
        // The Program's single-flight gate answers with a message; the HTTP contract must not depend on
        // its words, so it is translated to a type here, at the only place that reads the Engine.
        try {
            return engine_.run_image_interlude(target_bytes, margin_bytes, job);
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()).find("already in flight") != std::string_view::npos) {
                throw ImageInterludeInFlight{};
            }
            throw;
        }
    }

private:
    infernix::Engine& engine_;
};

// The production job body: streamed artifact weights + the merged pipeline entry + PNG encode.
// One shared_ptr instance is lazy-bound on first use and reused for every later job (the certified
// route reads the 32 blocks' weights from the artifact once per job from this live reader).
class ArtifactImageJobExecutor {
public:
    explicit ArtifactImageJobExecutor(std::string artifact_path,
                                      std::shared_ptr<spdlog::logger> logger = nullptr)
        : path_(std::move(artifact_path)), logger_(std::move(logger)) {}

    [[nodiscard]] std::function<void(const std::string&)> stage_logger() const {
        auto logger = logger_;
        if (logger == nullptr) { return {}; }
        return [logger](const std::string& line) { logger->info("{}", line); };
    }

    FinishedPicture operator()(const OpenAIImagesRequest& request, std::uint64_t seed,
                               std::span<const std::uint16_t> prompt_embeds,
                               std::span<const std::uint16_t> negative_embeds,
                               const ReferencePixels* reference, void* image_stream,
                               const std::shared_ptr<ImageJobProgress>& progress,
                               const EditConditioning* edit) {
        std::lock_guard lock(mutex_);
        ensure_locked();

        // The pipeline's units (steps + VAE) map onto the job's 0-100; the job's last unit is the
        // picture on disk, which the service claims after this returns. A null progress (a caller
        // without a stored job) simply certifies the route unchanged.
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

        // The edit route: the reference must ALSO reach the DiT as text rows that have seen the picture.
        // The vision tower and the conditioned 36-layer forward are the halves that need the yielded
        // VRAM, so they run here -- inside the interlude, after the probe said the lane fits -- on the
        // same artifact the job opened (shared_tower caches the tower, embedding table and vision bind
        // by path). The VAE block above is what the route had before this existed, and on its own it
        // returns the near pixel copy the M9 behavioral gate was written to catch: a reference the text
        // conditioning never saw cannot be edited by an instruction.
        if (edit != nullptr && edit->planes != nullptr && edit->prompt != nullptr) {
            if (!vision_rows_ || !conditioned_) {
                vision_rows_ = make_tower_vision_rows_provider(path_);
                conditioned_ = make_tower_conditioned_embeds_provider(path_);
            }
            const qi::ConditionedVision vision = vision_rows_(edit->planes->vision);
            const ConditionedPromptEmbeds embeds = conditioned_(*edit->prompt, vision);
            inputs.prompt_embeds          = embeds.prompt_embeds;
            inputs.prompt_image_pad_mask  = embeds.image_pad_mask;
            if (edit->negative_prompt != nullptr) {
                // Same reference, own text: the CFG joint differs only in the instruction rows, which is
                // what the reference family trains.
                const ConditionedPromptEmbeds negative = conditioned_(*edit->negative_prompt, vision);
                inputs.negative_prompt_embeds         = negative.prompt_embeds;
                inputs.negative_prompt_image_pad_mask = negative.image_pad_mask;
                inputs.true_cfg_scale = static_cast<float>(request.true_cfg_scale);
            }
        }

        const qi::GeneratedImage out = qi::run_image_job(
            *reader_, dit_, vae_, inputs, static_cast<cudaStream_t>(image_stream),
            progress ? &view : nullptr, stage_logger());

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
        return FinishedPicture{std::move(rgba), out.width, out.height};
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
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    std::unique_ptr<artifact::Reader> reader_;
    std::unique_ptr<artifact::Binder> binder_;
    qi::DiTWeights dit_;
    qi::VaeWeights vae_;
    qi::VaeDenorm denorm_;
    // The edit route's device halves, lazy-bound on the first job that carries a reference (they are
    // only ever called inside an interlude). Both share one tower and one vision bind with the prompt
    // providers of the same artifact path.
    VisionRowsProvider vision_rows_;
    ConditionedEmbedsProvider conditioned_;
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

ImageJobExecutor make_artifact_image_job_executor(const std::string& image_artifact_path,
                                                   std::shared_ptr<spdlog::logger> logger) {
    auto executor =
        std::make_shared<ArtifactImageJobExecutor>(image_artifact_path, std::move(logger));
    return [executor](const OpenAIImagesRequest& request, std::uint64_t seed,
                      std::span<const std::uint16_t> prompt_embeds,
                      std::span<const std::uint16_t> negative_embeds, const ReferencePixels* reference,
                      void* image_stream, const std::shared_ptr<ImageJobProgress>& progress,
                      const EditConditioning* edit) {
        return (*executor)(request, seed, prompt_embeds, negative_embeds, reference, image_stream,
                           progress, edit);
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

std::vector<std::uint8_t> load_generated_image(std::string_view id) {
    const std::filesystem::path file = generated_image_path(id);
    if (file.empty()) { return {}; }

    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    if (error || size == 0 || size > kMaximumImageInputBytes) { return {}; }

    std::ifstream input(file, std::ios::binary);
    if (!input) { return {}; }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!input) { return {}; }
    return bytes;
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

namespace {

// Where the sweep looks. Taken through `generated_image_path` rather than spelled out again, so the
// sweep cannot drift from the directory a picture is really written to.
[[nodiscard]] std::filesystem::path generated_image_directory() {
    return generated_image_path("img-0-0").parent_path();
}

constexpr std::chrono::seconds kGeneratedImageSweepInterval{std::chrono::hours(1).count()};

} // namespace

ImageRetentionResult sweep_generated_images(ImageJobStore& store, std::chrono::seconds retention,
                                            std::chrono::system_clock::time_point now) {
    ImageRetentionResult removed;
    if (retention <= std::chrono::seconds::zero()) { return removed; }

    const std::filesystem::path directory = generated_image_directory();
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) { return removed; }

    const auto cutoff = now - retention;
    std::vector<std::filesystem::directory_entry> entries;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) { entries.push_back(entry); }
    } catch (const std::filesystem::filesystem_error&) {
        // The directory vanished mid-listing (or was never readable): whatever was already seen is
        // still worth a pass, and the next hour will find the rest.
    }

    for (const auto& entry : entries) {
        const std::filesystem::path& file = entry.path();
        if (file.extension() != ".png") { continue; }
        const std::string id = file.stem().string();
        // The route's own id shape, in the route's own place. Anything else that lives in the
        // directory belongs to someone else and is left exactly where it is.
        if (generated_image_path(id) != file) { continue; }

        std::error_code time_error;
        const auto written = std::filesystem::last_write_time(file, time_error);
        if (time_error) { continue; }
        if (std::chrono::clock_cast<std::chrono::system_clock>(written) > cutoff) { continue; }

        // A job that has not ended still owns its file, however old the file looks: the writer stamps
        // it before the lifecycle reaches a terminal state.
        const std::shared_ptr<const StoredImageJob> record = store.get(id);
        if (record != nullptr && !is_terminal(record->state)) { continue; }

        std::uint64_t size = 0;
        {
            const std::uintmax_t bytes = std::filesystem::file_size(file, time_error);
            if (!time_error) { size = static_cast<std::uint64_t>(bytes); }
        }
        std::filesystem::remove(file, time_error);
        if (time_error) { continue; }
        // The record goes with the picture. The artifact route answers a missing file from the stored
        // job, so keeping a terminal record behind would leave the delivered URL answering a status
        // object forever instead of the 404 that tells a client the picture is gone; and a record of a
        // picture this server no longer has is not information worth keeping.
        (void)store.erase(id);
        ++removed.files;
        removed.bytes += size;
    }
    return removed;
}

GeneratedImageRetention::GeneratedImageRetention(ImageJobStore& store, std::chrono::seconds retention,
                                                 std::shared_ptr<spdlog::logger> logger)
    : store_(store), retention_(retention), logger_(std::move(logger)) {
    // Once at start-up: files left by a previous run of the server are already past due, and the
    // first hour of this one should not be spent carrying them.
    (void)sweep_now();
    thread_ = std::thread([this] { run_loop(); });
}

GeneratedImageRetention::~GeneratedImageRetention() {
    {
        std::lock_guard lock(mutex_);
        done_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
}

ImageRetentionResult GeneratedImageRetention::sweep_now() {
    const ImageRetentionResult removed =
        sweep_generated_images(store_, retention_, std::chrono::system_clock::now());
    if (removed.files != 0 && logger_ != nullptr) {
        logger_->info("image retention: removed {} picture file(s), {} MiB, past their retention of "
                      "{} s",
                      removed.files, removed.bytes / kMiB,
                      static_cast<long long>(retention_.count()));
    }
    return removed;
}

void GeneratedImageRetention::run_loop() {
    std::unique_lock lock(mutex_);
    while (!done_) {
        wake_.wait_for(lock, kGeneratedImageSweepInterval, [this] { return done_; });
        if (done_) { break; }
        lock.unlock();
        (void)sweep_now();
        lock.lock();
    }
}

ImageGenerationService::ImageGenerationService(std::unique_ptr<ImageInterludeRunner> runner,
                                               ImageJobStore& store,
                                               PromptRendererProvider prompt_renderer,
                                               PromptEmbedsProvider prompt_embeds,
                                               DiTDimsProvider dit_dims,
                                               ImageJobExecutor job_executor,
                                               VisionLaneDimsProvider vision_dims,
                                               ReferencePromptRendererProvider reference_prompt_renderer)
    : runner_(std::move(runner)),
      store_(store),
      prompt_renderer_(std::move(prompt_renderer)),
      prompt_embeds_(std::move(prompt_embeds)),
      dit_dims_(std::move(dit_dims)),
      vision_dims_(std::move(vision_dims)),
      reference_prompt_renderer_(std::move(reference_prompt_renderer)),
      job_executor_(std::move(job_executor)) {}

bool ImageGenerationService::is_configured() const noexcept {
    return runner_ != nullptr && prompt_renderer_ != nullptr && prompt_embeds_ != nullptr &&
           dit_dims_ != nullptr && job_executor_ != nullptr;
}

// The DiT dimensions from the artifact's component config: host work, opened once and cached. It is
// deliberately not the device-side binding -- the admission reads it before any VRAM is yielded.
DiTDimsProvider make_artifact_dit_dims_provider(const std::string& image_artifact_path) {
    auto cached = std::make_shared<std::optional<qi::DiTDims>>();
    auto guard  = std::make_shared<std::mutex>();
    return [path = image_artifact_path, cached, guard]() -> qi::DiTDims {
        const std::lock_guard lock(*guard);
        if (cached->has_value()) { return **cached; }
        artifact::Reader reader(path);
        artifact::Binder binder(reader);
        const qi::DiTDims dims = qi::bind_dit(binder).dims;
        (void)std::move(binder).finish();
        *cached = dims;
        return dims;
    };
}

// The same host-side read for the ViT lane: the ledger needs the vision block's and activation's
// shapes to reserve for the encode that runs inside the interlude, and it must get them from the
// artifact's component config, not from a constant that can drift from the shipped checkpoint.
VisionLaneDimsProvider make_artifact_vision_dims_provider(const std::string& image_artifact_path) {
    auto cached = std::make_shared<std::optional<qi::VisionLaneDims>>();
    auto guard  = std::make_shared<std::mutex>();
    return [path = image_artifact_path, cached, guard]() -> qi::VisionLaneDims {
        const std::lock_guard lock(*guard);
        if (cached->has_value()) { return **cached; }
        artifact::Reader reader(path);
        artifact::Binder binder(reader);
        const qi::VisionDims v = qi::bind_image_vision(binder).dims;
        (void)std::move(binder).finish();
        qi::VisionLaneDims out;
        out.hidden      = v.hidden;
        out.intermediate = v.intermediate;
        out.head_dim    = v.head_dim;
        out.patch       = v.patch;
        out.temporal    = v.temporal;
        out.merge       = v.merge;
        out.out_hidden  = v.out_hidden;
        *cached = out;
        return out;
    };
}

std::string ImageGenerationService::next_job_id(std::int64_t created) {
    std::lock_guard lock(sequence_mutex_);
    return "img-" + std::to_string(created) + "-" + std::to_string(++sequence_);
}

std::shared_ptr<ImageJobProgress> ImageGenerationService::last_job_progress() const {
    const std::lock_guard lock(progress_mutex_);
    return last_progress_;
}

nlohmann::json ImageGenerationService::job_status_json(std::string_view id) const {
    const auto record = store_.get(std::string(id));
    if (record == nullptr) { return nullptr; }
    const bool running = !is_terminal(record->state);
    return nlohmann::json{{"id", record->id},
                          {"status", running ? "running" : state_to_string(record->state)},
                          {"progress_percent",
                           record->progress ? record->progress->percent.load() : -1}};
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
    // Both planes of one decode and one resize: the raw-alpha VAE plane the encoder consumes and the
    // white-composited copy the Qwen3-VL tower was trained on. They come from the same call so the two
    // can never disagree about what picture is being edited.
    std::optional<ReferencePlanes> planes;
    if (!request.image.empty()) {
        // The codec's own words never reach the wire: they name decoders, byte offsets and internal
        // limits, which is both a leak and noise to a client that only wants to know its picture was
        // bad. The CODE distinguishes the cases a caller can act on.
        try {
            planes = prepare_reference_planes(request.image);
        } catch (const md::Error& error) {
            throw ApiException{ApiError{400, "invalid_request_error",
                                        error.kind() == md::ErrorKind::BudgetExceeded
                                            ? "the decoded reference image exceeds the accepted budget"
                                            : "the image is not a decodable PNG or JPEG",
                                        "image",
                                        error.kind() == md::ErrorKind::BudgetExceeded
                                            ? "image_too_large"
                                            : "image_decode_failed"}};
        } catch (const std::invalid_argument& error) {
            // Strict base64 rejected the payload, or the decoded bytes were not an image at all.
            throw ApiException{ApiError{400, "invalid_request_error",
                                        "the image is not a base64-encoded PNG or JPEG", "image",
                                        "image_decode_failed"}};
        }
    }

    // Admission, entirely before the LLM is touched. The reference's geometry comes first because the
    // prompt the admission sizes on is the one that carries the reference: the ti2i template with every
    // <|image_pad|> expanded into the vision rows, which is host work (tokenizer only) and therefore
    // runs here too. The probe then answers from the live pools without quiescing anything. A request
    // that cannot fit gets its 503 here -- no stored job, no closed submit door, no other request
    // waiting behind a job that could never run.
    //
    // The reference's latent tokens follow from the snapped plane prepare_reference_planes produced:
    // (height/16) * (width/16), the same count its VAE encode will return inside the job. Its vision
    // tokens are the <|image_pad|> rows the rendered prompt carries, and each of them becomes four of
    // those latent rows -- so 4 * vision != ref means OUR resize broke the x4 invariant the joint
    // assembly asserts (models/qwen_image/joint_layout.cpp:41-47). That is our bug, not the client's:
    // a 500 of our own making, raised host-side, before a single byte is yielded.
    std::size_t ref_tokens    = 0;
    std::size_t vision_tokens = 0;
    qi::VisionLaneDims vision{};
    if (planes.has_value()) {
        ref_tokens =
            static_cast<std::size_t>(planes->vae.height_px / qi::ImageGeometry::downscale()) *
            (static_cast<std::size_t>(planes->vae.width_px) / qi::ImageGeometry::downscale());
        vision_tokens = static_cast<std::size_t>(
            vision_token_count(planes->vae.height_px, planes->vae.width_px));
        if (4ULL * vision_tokens != ref_tokens) {
            throw ApiException{ApiError{500, "server_error",
                                        "the prepared reference disagrees with itself: 4 * " +
                                            std::to_string(vision_tokens) + " vision tokens != " +
                                            std::to_string(ref_tokens) + " latent tokens",
                                        "image", "image_conditioning_inconsistent"}};
        }
        // Two things an edit cannot be served without, both refused for the same reason the ledger
        // refuses a guess: the render is the only prompt whose token count is the one the job will run,
        // and a lane whose shape the artifact cannot name is a lane that cannot be reserved -- an
        // under-reserved yield OOMs the resident LLM, which outranks serving this request at all.
        if (!vision_dims_) {
            throw ApiException{ApiError{503, "server_error",
                                        "the image artifact carries no readable vision component, so a "
                                        "request with a reference cannot be sized",
                                        "image_artifact", "image_generation_not_configured"}};
        }
        if (!reference_prompt_renderer_) {
            throw ApiException{ApiError{503, "server_error",
                                        "no image text processor is configured, so the reference "
                                        "prompt cannot be rendered",
                                        "image_text_processor", "image_generation_not_configured"}};
        }
        vision = vision_dims_();
    }
    const auto render = [&](const std::string& text) {
        return planes.has_value() ? reference_prompt_renderer_(text, {vision_tokens})
                                  : prompt_renderer_(text);
    };
    const TowerPrompt prompt = render(request.prompt);
    const TowerPrompt negative_prompt =
        request.true_cfg_scale > kMinimumImageCFGScale ? render(request.negative_prompt)
                                                      : TowerPrompt{};
    const std::uint64_t need =
        image_need_bytes(request.width, request.height, prompt.tokens(), negative_prompt.tokens(),
                         ref_tokens, dit_dims_(), vision_tokens, vision);
    const auto fit = runner_->probe(need, kInterludeMarginBytes);
    if (!fit.fits) { throw_vram_insufficient(need, fit); }

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
    FinishedPicture finished;
    try {
        store_.advance(record.id, JobUpdate{.to = ImageJobState::WaitingForFrontier, .now = now});
        runner_->run(need, kInterludeMarginBytes,
                     [&](const infernix::ImageInterludeContext& context) -> bool {
                         // What the yield actually unmapped, against the same number the probe was
                         // asked about. A shortfall here stops the job BEFORE its first allocation:
                         // the LLM restores through the ordinary failed-job path and stays healthy,
                         // instead of the job discovering the shortage in a kernel.
                         if (context.freed_bytes + fit.device_free_bytes <
                             need + kInterludeMarginBytes + fit.headroom_bytes) {
                             throw_vram_insufficient(
                                 need, ImageInterludeRunner::Fit{.fits = false,
                                                                  .freed_bytes_expected =
                                                                      context.freed_bytes,
                                                                  .device_free_bytes =
                                                                      fit.device_free_bytes,
                                                                  .headroom_bytes = fit.headroom_bytes});
                         }
                         store_.advance(record.id, JobUpdate{.to = ImageJobState::Running, .now = now});
                         job_started = true;
                         std::uint64_t seed = 0;
                         if (request.seed.has_value()) {
                             seed = *request.seed;
                         } else {
                             std::random_device random;
                             seed = (static_cast<std::uint64_t>(random()) << 32U) | random();
                         }
                         // Both prompts were already rendered (host work, before the probe); only the
                         // tower forward -- the half that needs device memory -- runs here. With a
                         // reference that forward is the CONDITIONED one and the executor runs it inside
                         // the yield (it owns the artifact's vision and text towers), so no plain embeds
                         // are computed: a forward over rows that never saw the picture is not merely
                         // wasted work, it is the copy this route used to return.
                         const bool edit_route = planes.has_value();
                         const std::vector<std::uint16_t> positive =
                             edit_route ? std::vector<std::uint16_t>{} : prompt_embeds_(prompt);
                         const std::vector<std::uint16_t> negative =
                             edit_route || request.true_cfg_scale <= kMinimumImageCFGScale
                                 ? std::vector<std::uint16_t>{}
                                 : prompt_embeds_(negative_prompt);
                         const EditConditioning edit{
                             edit_route ? &*planes : nullptr, edit_route ? &prompt : nullptr,
                             (edit_route && request.true_cfg_scale > kMinimumImageCFGScale)
                                 ? &negative_prompt : nullptr};
                         finished = job_executor_(
                             request, seed, positive, negative,
                             edit_route ? &planes->vae : nullptr, context.image_stream,
                             record.progress, edit_route ? &edit : nullptr);
                         store_.advance(record.id, JobUpdate{.to = ImageJobState::Restoring, .now = now});
                         return true;
                     });
        // The paused window ends with the last byte the DEVICE had to touch. The PNG encode, the
        // base64 and the disk write are host work the yielded VRAM has no opinion about, so they run
        // here -- after the engine reported the LLM restored (bundle B fas 1). A failure in this block
        // is still a failed job whose restoration was confirmed: 500 image_generation_failed.
        const auto png = media::encode::encode_png_rgba8(finished.width, finished.height, finished.rgba);
        save_generated_image(record.id, png);
        // The pipeline reported (steps + VAE)/100; the written picture is the job's last unit.
        if (record.progress) { record.progress->advance(100); }
        response = request.response_format == ImageResponseFormat::Url
                     ? make_openai_image_url_response(now, {std::string(artifact_url_base) + record.id})
                     : make_openai_image_response(now, {base64_encode_bytes(png)});
        store_.advance(record.id,
                       JobUpdate{.to = ImageJobState::Succeeded, .now = now,
                                 .response = nlohmann::json::parse(response.dump())});
        return response;
    } catch (const std::exception& error) {
        const std::string message  = error.what();
        const auto failure         = nlohmann::json{{"message", message}};
        const bool degraded        = !runner_->is_available();
        // A refusal that already carries its own public code (the post-yield vram check) is the
        // client's answer as it stands: renaming it to "the job failed" would hide the fix.
        if (const auto* api = dynamic_cast<const ApiException*>(&error);
            api != nullptr && api->error().code == "image_vram_insufficient") {
            try {
                store_.advance(record.id,
                              JobUpdate{.to = ImageJobState::Failed, .now = now, .error = failure});
            } catch (const ApiException&) {
                // A terminal race: the stored terminal state wins, the client still gets the refusal.
            }
            throw;
        }
        const bool in_flight_claim = dynamic_cast<const ImageInterludeInFlight*>(&error) != nullptr;
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
