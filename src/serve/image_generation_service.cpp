#include "serve/image_generation_service.h"

// The edit route's device halves (the vision tower and the conditioned text tower forwards) are built
// here and run inside the interlude, which is the only place the device may be touched.
#include "serve/image_text_tower.h"

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "media/decode/decode.h"
#include "media/encode/png_encode.h"
#include "infernix/ops/attn_dit_joint.h"
#include "models/qwen_image/image_device_need.h"
#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/joint_layout.h"
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

// What the prompt enhancer is promised before it is asked to load: measured, not estimated. One real
// rewrite of the 5118-token kingfisher edit -- load, vision tower, prefill, 2473 tokens of thinking
// -- costs 19564 MiB above an idle card on this machine, of which 17947 MiB is the checkpoint's BF16
// weights, and the high-water mark the worker's own allocator reports for it is 19042 MiB reserved
// (image-edit/_p10_resident.json, worker_cost_mib and torch_peak_reserved_mib). The number below is
// that peak, so the promise the fit check makes -- this constant plus kInterludeMarginBytes plus the
// engine's headroom -- comes to the measured peak plus half a GiB, and the fit check still keeps its
// own margin on top: an allocation that lands a hair short is discovered here, not inside the worker.
// The constant used to be 20 GiB, which is 1438 MiB above the measured peak, and that excess is what
// made `prompt_enhance: "auto"` refuse itself: a request with 16000 MiB of yield and 4988 MiB already
// free had 20988 MiB against a floor of 20480 + 512 + headroom and skipped the rewriter by a few MiB,
// so the advertised auto never rewrote anything with the chat model resident. Over-reserving stays the
// safe direction -- a refused enhancer is a skip for auto and a 503 naming the numbers for true, while
// an under-reserved one allocates inside its own process while the resident LLM is quiesced and the
// shortage is discovered by whatever touches the card next -- but the promise is now the measurement
// plus the margin, not the measurement plus a second slack nobody re-measured.
constexpr std::uint64_t kEnhancerReserveBytes = 19042ULL * kMiB;

// Whether `prompt_enhance: "auto"` should spend an enhancer interlude on this request. The rule is
// the two facts that decide whether the edit rewriter can do anything at all -- not a judgement about
// the quality of the client's prose, which nothing outside a model could make here. PE-I2I reads the
// picture, so with no reference a plain text-to-image call hands it nothing to read and the rewrite
// buys ~68 s and ~19.5 GiB for nothing. And an instruction that already carries the tag the rewriter
// is required to address its inputs with (image-edit/refs/q21_system_prompt_edit.txt:57-61, the very
// tag serve/image_text_tower.cpp:229 renders) is a rewriter's own answer arriving back: running it
// through the rewriter again is how a good prompt drifts. Anything else gets the rewrite, including
// the one-phrase instruction -- which is exactly what the enhancer exists to be handed
// (image-edit/CLAUDE.md, "A short client instruction is the intended input"). A client that sent
// true never consults this: it asked for the rewrite, so it gets it or gets the documented refusal.
// What `auto` asks for, told apart BY REASON: the three are three different facts (nothing to read,
// already read, go and read it) and the stored record has to be able to say which one happened.
enum class AutoWish : std::uint8_t { Rewrite, SkipNoReference, SkipAlreadyRewritten };

[[nodiscard]] AutoWish wants_prompt_enhancement(const OpenAIImagesRequest& request) {
    if (request.images.empty()) { return AutoWish::SkipNoReference; }
    const std::string_view prompt = request.prompt;
    for (std::size_t at = prompt.find("<image"); at != std::string_view::npos;
         at = prompt.find("<image", at + 1)) {
        std::size_t end = at + 6;                        // past the literal "<image"
        while (end < prompt.size() && prompt[end] >= '0' && prompt[end] <= '9') { ++end; }
        if (end > at + 6 && end < prompt.size() && prompt[end] == '>') {
            return AutoWish::SkipAlreadyRewritten;
        }
    }
    return AutoWish::Rewrite;
}

// The enum and its wire words live in the header, where the capability document can reach them: the
// document lists these same strings by CALLING to_string, and a list kept in two places is a list that
// drifts. Defined there (inline) rather than here, so it is not inside this file's anonymous namespace.

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
// The canvas a too-long prompt WOULD have fitted on, so the refusal can name one.
//
// joint_token_count is `rows + 3 * vision + target` (models/qwen_image/joint_layout.h, its agreement with
// build_joint_layout pinned by a host test), and of those three terms only the target is the canvas: the
// rendered rows and the reference's vision tokens are fixed by this request. So the ask minus the target
// is what the canvas has to leave room for, and the largest canvas that fits is the asked one shrunk
// until its token count does -- longer side first, one 32 grid step at a time, never below one step,
// which keeps the aspect the client chose instead of trading it for a square.
//
// This exists because a refusal that says only "too long" sends the caller away to guess. Measured on
// the deployed engine: an edit asked for 2048x2048, was refused five rounds running with a message that
// named both numbers and no size, and only on the sixth did the model find 1536 by itself. Naming the
// size is the difference between one round and six.
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> largest_fitting_canvas(
    std::uint32_t width_px, std::uint32_t height_px, const long long target_budget) noexcept
{
    constexpr std::uint32_t step = qi::ImageGeometry::multiple_of();
    const auto tokens = [](const std::uint32_t w, const std::uint32_t h) {
        return static_cast<long long>(qi::ImageGeometry::derive(h, w).tokens);
    };
    std::uint32_t w = std::max(step, width_px);
    std::uint32_t h = std::max(step, height_px);
    while ((w > step || h > step) && tokens(w, h) > target_budget) {
        if (w >= h) { w = w > 2U * step ? w - step : step; }
        else        { h = h > 2U * step ? h - step : step; }
    }
    return {w, h};
}

// The 503 a client can act on. `canvas_is_the_lever` says whether shrinking the picture could have lowered
// the need at all: below roughly 1024 the need is dominated by a term that does not move with the canvas,
// and a refusal that then tells the client to ask for a smaller size sends it shrinking into the same
// refusal -- measured as five refusals in fifteen seconds while a model edited its own canvas down. The
// sentence is chosen from the arithmetic that decided it, never from a guess about the request.
[[noreturn]] void throw_vram_insufficient(std::uint64_t need, const ImageInterludeRunner::Fit& fit,
                                         bool canvas_is_the_lever) {
    throw ApiException{ApiError{503, "server_error",
                                "the image job needs " + std::to_string(need / kMiB) +
                                    " MiB; the model can yield " +
                                    std::to_string(fit.freed_bytes_expected / kMiB) + " MiB with " +
                                    std::to_string(fit.device_free_bytes / kMiB) +
                                    " MiB already free and " +
                                    std::to_string(fit.headroom_bytes / kMiB) +
                                    " MiB reserved below the headroom; " +
                                    (canvas_is_the_lever
                                         ? "a smaller canvas lowers this need"
                                         : "this need does not fall with the canvas, so a smaller size "
                                           "will not help"),
                                "size", "image_vram_insufficient"}};
}

// The smallest canvas the server will trade a refused request for (see the walk in generate()). The
// admission need is dominated by a term that does NOT shrink with the picture -- kImageWeightResidentBytes
// and the resident block slots -- so a card that can hold the job at all converges above this line, and
// below it the documented 503 with its two numbers is the honest answer rather than a thumbnail. It is a
// multiple of the geometry's own grid, so a canvas that reaches it is still a legal request.
inline constexpr std::uint32_t kVramCanvasFloorPx = 256U;

// How much the walk wants the probe's PROMISE to beat the need by before it is satisfied -- on top of
// the margin and the headroom the gate itself compares. The reason it is not enough to stop at the first
// promise that clears the gate: probe() is not exclusive. A draw already inside its interlude has the
// frame and KV pools unmapped, so a promise read at that instant can be gigabytes better than what THIS
// job's own yield delivers once the other one restores them -- and the net inside the interlude, which
// compares the delivered bytes against the same gate, refuses. Measured on the deployed engine: a
// 2016x2016 edit walked down to a canvas needing 15540 MiB, was promised enough, then delivered 15040
// with 855 MiB free and was refused by that net -- the walk found a size the card could not actually pay
// for. Uncontended a real promise beats the need by tens of GiB, so this slack costs nothing; it bites
// only in the band where a concurrent draw could push the job over.
inline constexpr std::uint64_t kVramWalkSlackBytes = 2048ULL * 1024ULL * 1024ULL;

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
                   .expert_pool_frozen_by_lease = probe.expert_pool_frozen_by_lease,
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
            const std::string_view words = error.what();
            if (words.find("already in flight") != std::string_view::npos) {
                throw ImageInterludeInFlight{};
            }
            // The other soft-fail: the LLM never reached a safe point inside the drain deadline and the
            // quiesce unwound itself. Nothing is broken and nothing moved, so this is a retry, not a
            // failure -- and the retry must be of THIS request, at the canvas that was asked for.
            if (words.find("no yieldable LLM frontier") != std::string_view::npos) {
                throw ImageInterludeBusy{};
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
                               std::span<const ReferencePlanes> references, void* image_stream,
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
        // The one place a request's `seed` becomes pixels: this engine's own reproducible N(0,1) stream
        // (models/qwen_image/noise.h), snapped to BF16 the way every other activation here is. A fixed-
        // latent GPU regression does not go through this line -- it hands run_image_job its own
        // init_latent, which is why the two are documented as separate contracts (image_pipeline.h).
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
        // #75: every host-prepared plane becomes part of the constant reference block of the joint
        // sequence, encoded with the same VAE the job decodes with (inside the yielded VRAM). The
        // blocks are concatenated in the order the client sent them, which is the order the ti2i
        // template numbered them and the order build_joint_layout scatters them into the sequence --
        // three lists that must agree, so all three are driven by this one loop over `references`.
        for (const ReferencePlanes& plane : references) {
            qi::ReferenceEncodeInputs encode_in;
            encode_in.pixels       = plane.vae.pixels;
            encode_in.latents_mean = denorm_.mean;
            encode_in.latents_std  = denorm_.std;
            encode_in.height_px    = plane.vae.height_px;
            encode_in.width_px     = plane.vae.width_px;
            const qi::EncodedReference encoded =
                qi::encode_reference_image(*reader_, vae_, encode_in,
                                           static_cast<cudaStream_t>(image_stream));
            inputs.ref_latents.insert(inputs.ref_latents.end(), encoded.latents.begin(),
                                      encoded.latents.end());
            inputs.ref_grids.emplace_back(encoded.height_lat, encoded.width_lat);
        }

        // The edit route: the reference must ALSO reach the DiT as text rows that have seen the picture.
        // The vision tower and the conditioned 36-layer forward are the halves that need the yielded
        // VRAM, so they run here -- inside the interlude, after the probe said the lane fits -- on the
        // same artifact the job opened (shared_tower caches the tower, embedding table and vision bind
        // by path). The VAE block above is what the route had before this existed, and on its own it
        // returns the near pixel copy the M9 behavioral gate was written to catch: a reference the text
        // conditioning never saw cannot be edited by an instruction.
        if (edit != nullptr && !references.empty() && edit->prompt != nullptr) {
            if (!vision_rows_ || !conditioned_) {
                vision_rows_ = make_tower_vision_rows_provider(path_);
                conditioned_ = make_tower_conditioned_embeds_provider(path_);
            }
            // One ViT pass per reference, their rows concatenated in request order.
            // ConditionedVision already describes the WHOLE picture set -- merger_rows is
            // [total pad rows, hidden] and grids are in pad order (models/qwen_image/text_tower.h) --
            // so N references are N runs appended, which is what the reference family hands the tower
            // when it is given a list. The merge constant and the deepstack level COUNT come from the
            // artifact, not the picture: a second reference answering with a different number of
            // levels means the tower disagrees with itself, which is our bug and a 500's worth of it.
            qi::ConditionedVision vision;
            bool first_reference = true;
            for (const ReferencePlanes& plane : references) {
                qi::ConditionedVision one = vision_rows_(plane.vision);
                if (first_reference) {
                    // The first answer IS the aggregate. Assigning the levels and then appending them
                    // again -- which is what the loop below does -- leaves every deepstack level at
                    // twice the pad rows while merger_rows stays correct, and the text tower's
                    // coverage check is the only thing that catches it.
                    vision.merge          = one.merge;
                    vision.merger_rows    = std::move(one.merger_rows);
                    vision.deepstack_rows = std::move(one.deepstack_rows);
                    vision.grids          = std::move(one.grids);
                    first_reference = false;
                    continue;
                }
                if (vision.deepstack_rows.size() != one.deepstack_rows.size()) {
                    throw std::runtime_error(
                        "image job: the vision tower returned a different number of deepstack levels "
                        "for two references of the same artifact");
                }
                vision.merger_rows.insert(vision.merger_rows.end(), one.merger_rows.begin(),
                                          one.merger_rows.end());
                for (std::size_t level = 0; level < one.deepstack_rows.size(); ++level) {
                    vision.deepstack_rows[level].insert(vision.deepstack_rows[level].end(),
                                                        one.deepstack_rows[level].begin(),
                                                        one.deepstack_rows[level].end());
                }
                vision.grids.insert(vision.grids.end(), one.grids.begin(), one.grids.end());
            }
            const ConditionedPromptEmbeds embeds = conditioned_(*edit->prompt, vision);
            inputs.prompt_embeds          = embeds.prompt_embeds;
            inputs.prompt_image_pad_mask  = embeds.image_pad_mask;
            if (edit->negative_prompt != nullptr) {
                // Same references, own text: the CFG joint differs only in the instruction rows,
                // which is what the reference family trains.
                const ConditionedPromptEmbeds negative = conditioned_(*edit->negative_prompt, vision);
                inputs.negative_prompt_embeds         = negative.prompt_embeds;
                inputs.negative_prompt_image_pad_mask = negative.image_pad_mask;
                inputs.true_cfg_scale = static_cast<float>(request.true_cfg_scale);
            }
        }

        const qi::GeneratedImage out = qi::run_image_job(
            *reader_, dit_, vae_, inputs, static_cast<cudaStream_t>(image_stream),
            progress ? &view : nullptr, stage_logger());

        // Four channels in, four channels out: the alpha is the VAE's own fourth decoded channel.
        return FinishedPicture{rgba_from_generated_channels(out.channels, out.width, out.height),
                               out.width, out.height};
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

// See the header. The tie case of the rounding was checked rather than assumed: a channel value that
// lands exactly on .5 of 255 must be of the form (2k-254)/255, and the only such BF16 value is 0,
// where numpy's round-half-to-even and lround agree on 128 -- so this stays lround.
std::vector<std::uint8_t> rgba_from_generated_channels(const std::vector<float>& channels,
                                                       std::uint32_t width,
                                                       std::uint32_t height) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (channels.size() != 4ULL * pixels) {
        throw std::runtime_error("image job: the decoded plane is not the four channels the VAE emits");
    }
    std::vector<std::uint8_t> rgba(4ULL * pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        for (std::size_t c = 0; c < 4ULL; ++c) {
            rgba[4ULL * i + c] = static_cast<std::uint8_t>(to_unit8(channels[c * pixels + i]));
        }
    }
    return rgba;
}

std::unique_ptr<ImageInterludeRunner> make_engine_interlude_runner(infernix::Engine& engine) {
    return std::make_unique<EngineInterludeRunner>(engine);
}

ImageJobExecutor make_artifact_image_job_executor(const std::string& image_artifact_path,
                                                   std::shared_ptr<spdlog::logger> logger) {
    auto executor =
        std::make_shared<ArtifactImageJobExecutor>(image_artifact_path, std::move(logger));
    return [executor](const OpenAIImagesRequest& request, std::uint64_t seed,
                      std::span<const std::uint16_t> prompt_embeds,
                      std::span<const std::uint16_t> negative_embeds,
                                            std::span<const ReferencePlanes> references, void* image_stream,
                      const std::shared_ptr<ImageJobProgress>& progress,
                      const EditConditioning* edit) {
        return (*executor)(request, seed, prompt_embeds, negative_embeds, references, image_stream,
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
                                               ReferencePromptRendererProvider reference_prompt_renderer,
                                               std::shared_ptr<PromptEnhancer> enhancer)
    : runner_(std::move(runner)),
      store_(store),
      prompt_renderer_(std::move(prompt_renderer)),
      prompt_embeds_(std::move(prompt_embeds)),
      dit_dims_(std::move(dit_dims)),
      vision_dims_(std::move(vision_dims)),
      reference_prompt_renderer_(std::move(reference_prompt_renderer)),
      job_executor_(std::move(job_executor)),
      enhancer_(std::move(enhancer)) {}

namespace {
// Registers one image job for the duration of its generate() and removes it on every way out.
// RAII rather than a counter incremented and decremented by hand: generate() leaves through a throw
// on almost every path, and a count left behind by a failed job would tell the keepalive that the
// card is busy forever and no healing would ever happen again -- the opposite failure of the one
// this exists to fix.
class InFlightJob {
public:
    InFlightJob(std::mutex& mutex, std::map<std::uint64_t, std::int64_t>& in_flight,
                std::uint64_t& token_source, std::int64_t now)
        : mutex_(mutex), in_flight_(in_flight) {
        const std::lock_guard lock(mutex_);
        token_        = ++token_source;
        in_flight_[token_] = now;
    }
    ~InFlightJob() {
        const std::lock_guard lock(mutex_);
        in_flight_.erase(token_);
    }
    InFlightJob(const InFlightJob&)            = delete;
    InFlightJob& operator=(const InFlightJob&) = delete;

private:
    std::mutex& mutex_;
    std::map<std::uint64_t, std::int64_t>& in_flight_;
    std::uint64_t token_ = 0;
};

std::int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
} // namespace

bool ImageGenerationService::is_configured() const noexcept {
    return runner_ != nullptr && prompt_renderer_ != nullptr && prompt_embeds_ != nullptr &&
           dit_dims_ != nullptr && job_executor_ != nullptr;
}

std::size_t ImageGenerationService::in_flight_jobs() const {
    const std::lock_guard lock(in_flight_mutex_);
    return in_flight_.size();
}

std::int64_t ImageGenerationService::oldest_in_flight_age_seconds(std::int64_t now) const {
    const std::lock_guard lock(in_flight_mutex_);
    if (in_flight_.empty()) { return 0; }
    return std::max<std::int64_t>(0, now - in_flight_.begin()->second);
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
    nlohmann::json status{{"id", record->id},
                          {"status", running ? "running" : state_to_string(record->state)},
                          {"progress_percent",
                           record->progress ? record->progress->percent.load() : -1}};
    // What the server decided about the rewriter, on the object a client can actually poll. A client that
    // sent `prompt_enhance: "auto"` is entitled to know which of the four skips it got: the capability
    // document describes that decision, and a described decision with no observable answer is exactly the
    // situation the 2026-11-01 auto 503 came out of. Omitted -- never sent as "" -- for a record written
    // before the key existed, so the document cannot promise an answer this particular job does not have.
    if (record->request.contains("prompt_enhance_outcome")) {
        status["prompt_enhance_outcome"] = record->request.at("prompt_enhance_outcome");
    }
    return status;
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

    // From here the card is ours until the function returns, so this is where the busy signal starts:
    // the rewriter's sampling, the yield, the diffusion and the encode all belong to it. /health reads
    // it so a supervisor waits instead of killing a serve that is only busy.
    const InFlightJob in_flight{in_flight_mutex_, in_flight_, in_flight_token_, unix_now()};

    // The draw's starting noise, decided at submit rather than inside the yield: it is host work, it
    // belongs to the request and not to the device, and the record below cannot carry it otherwise. A
    // client that named no seed gets one the engine picked -- and the only way it can ever ask for that
    // picture again is if the server remembers the number. It stays in the record; job_status_json
    // reports id, status and progress, so this is a diagnosis field and not a new response field. It is
    // picked BEFORE the enhancer because the enhancer samples on it too: the same request and the same
    // seed then reproduce the instruction and the noise together, which is the only pairing a diagnosis
    // of "the rewrite went wrong" can be written against.
    std::uint64_t seed = 0;
    if (request.seed.has_value()) {
        seed = *request.seed;
    } else {
        std::random_device random;
        seed = (static_cast<std::uint64_t>(random()) << 32U) | random();
    }

    // The family's prompt enhancer is a separate shipped model, not a template the server can put
    // on an existing tower: PE-I2I is a fine-tuned Qwen3.5-VL 9B that reads the instruction AND the
    // pictures and answers {rewritten_prompt, wh_ratio, ratio_follow}
    // (image-edit/refs/q21_prompt_rewrite_README.md:1-12, q21_system_prompt_edit.txt:175-183), and
    // the vision-language tower inside this image artifact is a different checkpoint that was never
    // trained against that system prompt -- running it would return the picture's embeddings, not a
    // better instruction. So it runs as its own process (tools/prompt_enhancer/pe_worker.py) behind
    // --image-prompt-enhancer-url, and a client that asked for enhancement while that is absent or
    // not answering is told so instead of getting a drawing made from words it believed had been
    // rewritten.
    //
    // Its ~20 GB of BF16 weights (refs/q21_prompt_rewrite_README.md:52-56) come onto the card through
    // an interlude of their own -- the same acquire/yield/restore the picture uses, one at a time --
    // because this card also holds the resident LLM and the DiT and may not hold all three at once.
    // The rewrite changes the INSTRUCTION. It may decide the canvas only where the request left the
    // canvas free: a client that named size, wh_ratio or ratio_follow keeps what it named, since the
    // enhancer's job is the wording and a server that overrode a client's explicit size would be
    // changing the picture the client asked for.
    OpenAIImagesRequest effective = request;
    // `auto` is the client's judgement delegated (openai_images.h PromptEnhanceMode). Two things can
    // make the rewrite impossible -- no rewriter answering, and no card to hand it -- and for auto
    // both are a skip, because the client asked the server to choose rather than promising itself a
    // rewrite. For true they stay the documented refusals: a client that believed its words were being
    // rewritten must never be handed a picture drawn from them as written. healthy() is asked only
    // when something was actually asked for, so a plain request still costs no sidecar round trip.
    const bool auto_mode = request.prompt_enhance == PromptEnhanceMode::Auto;
    EnhancerOutcome enhancer_outcome = EnhancerOutcome::Off;
    bool enhance = request.prompt_enhance != PromptEnhanceMode::Off;
    if (enhance && auto_mode) {
        // A client that sent true never consults the wish: it asked for the rewrite, so it gets it or
        // gets the documented refusal.
        const AutoWish wish = wants_prompt_enhancement(request);
        enhance = wish == AutoWish::Rewrite;
        if (!enhance) {
            enhancer_outcome = wish == AutoWish::SkipNoReference
                                   ? EnhancerOutcome::SkippedNoReference
                                   : EnhancerOutcome::SkippedAlreadyRewritten;
        }
    }
    bool enhancer_up = false;
    if (enhance) { enhancer_up = enhancer_ != nullptr && enhancer_->healthy(); }
    if (enhance && !enhancer_up) {
        if (!auto_mode) {
            throw ApiException{ApiError{503, "server_error",
                                        std::string("this build has no prompt enhancer running; the "
                                                    "family's is ") +
                                            std::string(kImagePromptEnhancerModel) +
                                            " (serve --image-prompt-enhancer-url, see "
                                            "tools/prompt_enhancer/pe_worker.py), and in its absence "
                                            "the client writes the instruction itself (see GET "
                                            "/v1/capabilities image_edit.instruction)",
                                    "prompt_enhance", "image_prompt_enhancer_unavailable"}};
        }
        enhance          = false;
        enhancer_outcome = EnhancerOutcome::SkippedNoRewriter;
    }
    // The rewrite itself, behind a lambda: the shortfall net inside the interlude measures itself
    // against the probe that promised this much, so the two have to live in one scope -- and auto is
    // allowed to decline after that probe, which must then skip the whole body rather than run it.
    // The decline that actually happens is the one BELOW, inside the interlude: the probe only promised
    // this much, and what it promised and what the yield unmapped are two different answers.
    bool enhancer_declined = false;
    const auto run_rewrite = [&](const ImageInterludeRunner::Fit& fit) {
        EnhancerAnswer answer;
        runner_->run(kEnhancerReserveBytes, kInterludeMarginBytes,
                     [&](const infernix::ImageInterludeContext& context) -> bool {
                         // The same shortfall net the picture runs: if the yield did not unmap what the
                         // enhancer was promised, say so BEFORE it allocates, rather than letting a
                         // foreign process discover the shortage in its own allocator while the LLM is
                         // quiesced.
                         if (context.freed_bytes + fit.device_free_bytes <
                             kEnhancerReserveBytes + kInterludeMarginBytes + fit.headroom_bytes) {
                             // The same rule the pre-probe applies, and for the same reason: for auto the
                             // card not being there is a SKIP and never a refusal, because the client
                             // delegated this judgement and the picture is then drawn from its own words.
                             // Only a client that named prompt_enhance: true may be refused over the
                             // enhancer -- it believed its words were being rewritten, and handing it the
                             // words as written would be the lie the document above forbids.
                             if (!auto_mode) {
                                 throw_vram_insufficient(
                                     kEnhancerReserveBytes,
                                     ImageInterludeRunner::Fit{.fits = false,
                                                               .freed_bytes_expected = context.freed_bytes,
                                                               .device_free_bytes = fit.device_free_bytes,
                                                               .headroom_bytes = fit.headroom_bytes},
                                     // The rewriter's reserve is a fixed amount the picture's size does not
                                     // touch, so a smaller canvas was never the answer here either.
                                     /*canvas_is_the_lever*/ false);
                             }
                             // Nothing was allocated, so nothing has to be released: leave before the
                             // Releaser is even constructed, and before the sidecar is asked for anything.
                             enhancer_declined  = true;
                             enhancer_outcome   = EnhancerOutcome::SkippedNoDevice;
                             return false;
                         }
                         // Released whatever the rewrite does to us: weights left resident in the
                         // sidecar would sit on the card the picture is about to yield for.
                         struct Releaser {
                             PromptEnhancer* enhancer;
                             ~Releaser() {
                                 try {
                                     enhancer->unload();
                                 } catch (...) { /* the idle timer in the worker is the backstop */ }
                             }
                         } releaser{enhancer_.get()};
                         enhancer_->load();
                         answer = enhancer_->rewrite(request.prompt, request.images, seed,
                                                     request.images.size());
                         return true;
                     });
        // A declined rewrite leaves `effective` exactly as the client spelled it. `answer` was never
        // filled, and copying it would blank the very instruction the picture is about to be drawn from.
        if (enhancer_declined) { return; }
        effective.prompt = answer.rewritten_prompt;
        if (!request.size_requested && !request.wh_ratio && !request.ratio_follow) {
            if (!answer.wh_ratio.empty()) { effective.wh_ratio = answer.wh_ratio; }
            if (!answer.ratio_follow.empty()) { effective.ratio_follow = answer.ratio_follow; }
        }
        enhancer_outcome = EnhancerOutcome::Applied;
    };
    if (enhance) {
        // A request the client wrote itself if this probe declines auto: `effective` is replaced only
        // inside run_rewrite, so a declined rewrite leaves it exactly as the client spelled it.
        const auto fit = runner_->probe(kEnhancerReserveBytes, kInterludeMarginBytes);
        // The same blindness as the picture's own walk: while the chat holds the residency lease the
        // promise is KV-only, and the interlude would take the ring back before loading the rewriter. So a
        // blind no here admits the attempt and lets the net over the DELIVERED bytes decline it (a skip for
        // auto, the documented refusal for a client that named prompt_enhance: true) rather than refusing
        // over a yield this job never asked for.
        const bool enhancer_probe_blind = fit.expert_pool_frozen_by_lease;
        if (!fit.fits && !enhancer_probe_blind) {
            if (!auto_mode) {
                // The rewriter's reserve does not shrink with the canvas: never send this client shrinking.
                throw_vram_insufficient(kEnhancerReserveBytes, fit, /*canvas_is_the_lever*/ false);
            }
            // The same judgement the interlude applies below, one level earlier: the pre-probe already
            // said the card cannot hold the rewriter, so auto skips it and says WHICH reason -- and a
            // client that named prompt_enhance: true keeps the documented refusal, because handing it its
            // own words back would be the lie the document above forbids. Left as Off, auto would report
            // "the client asked for no rewrite", which is what a client reading the answer would retry
            // forever over.
            enhancer_outcome = EnhancerOutcome::SkippedNoDevice;
        }
        // A blind no goes in too: the interlude is where the ring comes back, and the net inside it is the
        // one that knows what the yield delivered.
        if (fit.fits || enhancer_probe_blind) {
            try {
                run_rewrite(fit);
            } catch (const ApiException& exception) {
                // The two ways the rewriter can be impossible were both written down as a skip for
                // auto, but only one of them was ever checked: `healthy()` is a GET on a port that
                // answers, and the load that follows it is the ~19 GB that may not be there beside the
                // resident LLM. A sidecar that dies under that load is the same impossibility as a
                // refused connection -- the rewrite did not happen, and the client that delegated the
                // judgement gets its own words drawn rather than an error it can do nothing with. A
                // client that named prompt_enhance: true believed its words were rewritten, and keeps
                // the documented refusal; any other code is not this case and travels unchanged.
                if (!auto_mode || exception.error().code != "image_prompt_enhancer_unavailable") {
                    throw;
                }
                enhancer_outcome = EnhancerOutcome::SkippedNoRewriter;
            }
        }
    }

    // #75 reference image: base64/decode/resize is host work and fails fast as a 400 before any
    // VRAM is yielded. The device encode of the prepared plane runs inside the job.
    // Both planes of one decode and one resize: the raw-alpha VAE plane the encoder consumes and the
    // white-composited copy the Qwen3-VL tower was trained on. They come from the same call so the two
    // can never disagree about what picture is being edited.
    std::vector<ReferencePlanes> planes;
    for (const std::string& payload : request.images) {
        // The codec's own words never reach the wire: they name decoders, byte offsets and internal
        // limits, which is both a leak and noise to a client that only wants to know its picture was
        // bad. The CODE distinguishes the cases a caller can act on.
        try {
            planes.push_back(prepare_reference_planes(payload));
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

    // The canvas, decided once and here: an edit that named no size follows its reference's aspect
    // (openai_images.h resolve_canvas_geometry), filled to the deployment's configured area. The
    // reference it follows is the picture decoded just above, so the grid an edit is sampled onto and
    // the picture conditioning it cannot disagree -- a fixed square default used to make a portrait
    // reference generate a square, which is a conditioning mismatch the model was never trained on.
    // Host work, before the probe: the ledger below and the job both size on this answer. (`effective`
    // was made above, from the request and -- when enhancement ran -- from the enhancer's answer.)
    // Which reference the canvas follows: the one `ratio_follow` names, else the LAST one -- the
    // reference pipeline derives the geometry from image[-1]
    // (image-edit/refs/q21_pipeline_qwenimage21.py:630-636). A tag naming a picture the request did
    // not carry was refused at parse time, so the only states left here are "named one" and "none".
    std::size_t followed = planes.size();
    if (effective.ratio_follow) {
        (void)parse_image_reference_tag(*effective.ratio_follow, planes.size(), followed);
    } else if (!planes.empty()) {
        followed = planes.size() - 1;
    }
    resolve_canvas_geometry(effective,
                            followed < planes.size() ? planes[followed].source_width_px : 0U,
                            followed < planes.size() ? planes[followed].source_height_px : 0U,
                            effective.width, effective.height);

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
        std::vector<std::uint64_t> vision_per_image;  // one entry per reference, in request order
    qi::VisionLaneDims vision{};
    if (!planes.empty()) {
        // Every reference conditions the same forward, so the joint sequence and the ledger are sized
        // on the SUM over them: each one contributes 4 * its vision tokens of joint sequence and its
        // own latent tokens to all_img (models/qwen_image/joint_layout.h, image_device_need.h).
        // The x4 invariant is checked per plane -- one reference whose resize broke it is our bug,
        // and the sums would hide it in the other references' numbers.
        vision_per_image.reserve(planes.size());
        for (const ReferencePlanes& plane : planes) {
            const std::size_t ref =
                static_cast<std::size_t>(plane.vae.height_px / qi::ImageGeometry::downscale()) *
                (static_cast<std::size_t>(plane.vae.width_px) / qi::ImageGeometry::downscale());
            const std::size_t vis = static_cast<std::size_t>(
                vision_token_count(plane.vae.height_px, plane.vae.width_px));
            if (4ULL * vis != ref) {
                throw ApiException{ApiError{500, "server_error",
                                            "a prepared reference disagrees with itself: 4 * " +
                                                std::to_string(vis) + " vision tokens != " +
                                                std::to_string(ref) + " latent tokens",
                                            "image", "image_conditioning_inconsistent"}};
            }
            ref_tokens += ref;
            vision_tokens += vis;
            vision_per_image.push_back(vis);
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
        return planes.empty() ? prompt_renderer_(text)
                                  : reference_prompt_renderer_(text, vision_per_image);
    };
    const TowerPrompt prompt = render(effective.prompt);
    const TowerPrompt negative_prompt =
        request.true_cfg_scale > kMinimumImageCFGScale ? render(effective.negative_prompt)
                                                      : TowerPrompt{};

    // The joint sequence's LENGTH is a host fact once the prompt is rendered, so it is admitted here
    // and not discovered by the attention Op inside the interlude: every text row is one position and
    // every image slot four (models/qwen_image/joint_layout.h), and the Op refuses a sequence longer
    // than ops::kAttnDitJointMaxTokens. A request over that is the client's -- its prompt, its canvas --
    // and it gets the numbers and a 400 before the LLM is quiesced, instead of a job that yields the
    // card, throws and comes back a 500. The CFG negative pass is its own joint sequence over the SAME
    // references, so its rows are admitted on their own count.
    const qi::ImageGeometry canvas = qi::ImageGeometry::derive(effective.height, effective.width);
    const std::pair<const char*, std::size_t> lanes[] = {
        {"prompt", prompt.tokens()}, {"negative prompt", negative_prompt.tokens()}};
    for (const auto& [lane, rows] : lanes) {
        const long long joint = qi::joint_token_count(static_cast<long long>(rows),
                                                      static_cast<long long>(vision_tokens),
                                                      static_cast<long long>(canvas.tokens));
        if (joint > ops::kAttnDitJointMaxTokens) {
            // What the canvas may cost is the ceiling minus everything the request already fixed: the
            // joint count without the target term, taken from the same call that produced `joint` so the
            // two can never drift apart. Then name the biggest canvas that fits -- or, when not even the
            // smallest does, say plainly that no size helps and only a shorter lane does, rather than
            // sending the caller to shrink a canvas that was never the problem.
            const long long overhead = joint - static_cast<long long>(canvas.tokens);
            const auto [fit_w, fit_h] =
                largest_fitting_canvas(effective.width, effective.height,
                                       ops::kAttnDitJointMaxTokens - overhead);
            const std::string advice =
                qi::joint_token_count(static_cast<long long>(rows),
                                      static_cast<long long>(vision_tokens),
                                      static_cast<long long>(
                                          qi::ImageGeometry::derive(fit_h, fit_w).tokens)) <=
                        ops::kAttnDitJointMaxTokens
                    ? "a smaller size or a shorter " + std::string(lane) + " both work: " +
                          std::to_string(fit_w) + "x" + std::to_string(fit_h) + " fits"
                    : "no canvas fits this many " + std::string(lane) +
                          " rows, so only a shorter " + std::string(lane) + " works";
            throw ApiException{ApiError{
                400, "invalid_request_error",
                std::string("the ") + lane + " is too long for this canvas: it makes a joint sequence of " +
                    std::to_string(joint) + " tokens, and the model serves at most " +
                    std::to_string(ops::kAttnDitJointMaxTokens) + " (" + advice + ")",
                "prompt", "image_prompt_too_long"}};
        }
    }

    // What the card can hold is a function of the canvas, and the probe that answers for it is PURE --
    // no quiesce, no drain, no unmap, just the yield plan over the live pools -- so a refusal at this
    // canvas is an invitation to ask again one 32-px step smaller rather than the end of the request.
    // Measured on the deployed engine: an edit refused here for 768x768 was 508 MiB short of the gate,
    // and its own client redrew it at 512x512 one model round later (142 s lost, then 157 s drawn) and
    // got its picture. The server can make that judgement inside the round it already has, on the same
    // numbers it would have printed, and the walk is the one largest_fitting_canvas uses -- longer side
    // first, so the aspect the client chose survives. It does not stop at the FIRST promise that clears
    // the gate: a promise can be bought from a card another draw has temporarily yielded, and the net
    // inside this job's own interlude is what refuses then, so it wants kVramWalkSlackBytes of play and
    // remembers the largest canvas that merely cleared the gate for a card that never has it. It stops at
    // kVramCanvasFloorPx, below which the documented 503 with its two numbers is the honest answer. `effective` is what the job is submitted
    // as, so a canvas settled here is the canvas that is drawn.
    const std::uint32_t asked_width  = effective.width;
    const std::uint32_t asked_height = effective.height;
    const std::uint32_t canvas_step  = qi::ImageGeometry::multiple_of();
    std::uint64_t need = 0;
    ImageInterludeRunner::Fit fit{};
    // The largest canvas whose PLAIN gate held, remembered in case nothing on the way down clears the
    // bar above: drawing at it is what this code did before the slack existed, and it is a better answer
    // than a refusal when the card was never contended.
    bool have_plain_fit = false;
    std::uint32_t plain_w = 0U, plain_h = 0U;
    std::uint64_t plain_need = 0U;
    ImageInterludeRunner::Fit plain_fit{};
    // The need and the one step of the walk, named ONCE: the shortfall net inside the interlude below
    // makes this same walk again, on the bytes the yield actually delivered, and two copies of this
    // arithmetic is exactly how a canvas and a need would learn to disagree.
    const auto need_at = [&](std::uint32_t w, std::uint32_t h) {
        return image_need_bytes(w, h, prompt.tokens(), negative_prompt.tokens(), ref_tokens,
                                dit_dims_(), vision_tokens, vision);
    };
    const auto step_down = [canvas_step](std::uint32_t& side) {
        side = side > kVramCanvasFloorPx + canvas_step ? side - canvas_step : kVramCanvasFloorPx;
    };
    // Whether trading canvas for bytes is a real trade at this size. The need is dominated by a resident
    // term that does not move with the picture, so under roughly 1024 a smaller canvas asks for the same
    // memory: walking there costs a probe per step and finds the same number every time, and until
    // 2026-11-05 the refusal at the end of it advised exactly that walk. Measured: 768x768 and 640x640
    // both need 3701 MiB.
    const std::uint32_t asked_w = effective.width, asked_h = effective.height;
    const bool canvas_is_the_lever =
        need_at(kVramCanvasFloorPx, kVramCanvasFloorPx) < need_at(asked_w, asked_h);
    for (;;) {
        need = need_at(effective.width, effective.height);
        fit  = runner_->probe(need, kInterludeMarginBytes);
        if (fit.fits) {
            const std::uint64_t promise = fit.freed_bytes_expected + fit.device_free_bytes;
            const std::uint64_t gate    = need + kInterludeMarginBytes + fit.headroom_bytes;
            if (promise >= gate + kVramWalkSlackBytes) { break; } // the card can pay for this and spare
            if (!have_plain_fit) {                                // the largest canvas that cleared the gate
                have_plain_fit = true;
                plain_w        = effective.width;
                plain_h        = effective.height;
                plain_need     = need;
                plain_fit      = fit;
            }
        } else if (fit.expert_pool_frozen_by_lease && !have_plain_fit) {
            // The only "no" that is not evidence. The yield plan keeps the frame pool whole while the LLM
            // holds a residency lease, so this promise is KV-only and the shortfall is measured against a
            // yield this job would never ask for: an interlude begins by taking the ring back
            // (begin_image_quiesce) and yields the frames after that. Refusing here threw away pictures the
            // same server draws in 44 s once the chat is quiet -- measured, 18 refusals against 26
            // successes in one instance, every one of them 1-32 ms and none of them ever reaching the
            // quiesce that would have answered. So submit at the canvas that was ASKED and let the net
            // inside the interlude decide on the bytes the yield ACTUALLY delivered: it refuses, before the
            // first allocation, when even the floor is unpayable. It is the authority precisely because it
            // is not a snapshot of somebody else's yielded card.
            effective.width  = asked_w;
            effective.height = asked_h;
            need             = need_at(asked_w, asked_h);
            break;
        }
        if ((effective.width <= kVramCanvasFloorPx && effective.height <= kVramCanvasFloorPx) ||
            !canvas_is_the_lever) { // nothing left to trade: decide here, once
            if (have_plain_fit) { // nothing ever beat the gate by the slack: take the biggest that cleared it
                effective.width  = plain_w;
                effective.height = plain_h;
                need             = plain_need;
                fit              = plain_fit;
                break;
            }
            throw_vram_insufficient(need, fit, canvas_is_the_lever);
        }
        if (effective.width >= effective.height) { step_down(effective.width); }
        else                                      { step_down(effective.height); }
    }

    StoredImageJob record;
    record.id       = next_job_id(now);
    record.state    = ImageJobState::Queued;
    record.created  = now;
    record.progress = std::make_shared<ImageJobProgress>();
    // What the job actually ran, not what was asked: a stored canvas that disagreed with the picture
    // the client got back would make the record useless for a later diagnosis.
    // `prompt_enhance_outcome` is the reason, not the request field: the field would say what the client
    // asked and leave the four legitimate auto skips identical in the log.
    record.request  = nlohmann::json{{"prompt", effective.prompt},
                                     {"width", effective.width},
                                     {"height", effective.height},
                                     {"steps", effective.steps},
                                     {"seed", seed},
                                     {"prompt_enhance_outcome", to_string(enhancer_outcome)}};
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
                         // What the yield ACTUALLY gave this job, both terms measured after its pools
                         // moved -- not the number the probe above promised. The probe is a guess read off
                         // the live pools while some OTHER draw may be holding those pools yielded, and its
                         // promise can then be gigabytes better than what this job's own yield delivers.
                         // Measured on the deployed engine: a walk that stopped at 2016x2016 on a promise
                         // of 18092 MiB, was handed 14400 MiB plus 479 free, and printed a 503 for a
                         // picture this machine can paint.
                         //
                         // A shortfall is therefore not the end of the request. The card is yielded HERE,
                         // the ceiling is these bytes, and this job has allocated nothing yet -- so the
                         // canvas walks down to the largest one the delivered bytes hold and is DRAWN in
                         // this same interlude, rather than refused so the client can ask again one model
                         // round later. Only a canvas already at the floor the card still cannot pay for is
                         // a refusal, and what it names are the bytes the yield delivered. A shortfall that
                         // the walk cannot answer stops the job BEFORE its first allocation: the LLM
                         // restores through the ordinary failed-job path and stays healthy, instead of the
                         // job discovering the shortage in a kernel.
                         const std::uint64_t available = context.freed_bytes + context.device_free_bytes;
                         for (;;) {
                             if (available >= need + kInterludeMarginBytes + fit.headroom_bytes) { break; }
                             if (effective.width <= kVramCanvasFloorPx &&
                                 effective.height <= kVramCanvasFloorPx) {
                                 throw_vram_insufficient(
                                     need, ImageInterludeRunner::Fit{.fits = false,
                                                                      .freed_bytes_expected =
                                                                          context.freed_bytes,
                                                                      .device_free_bytes =
                                                                          context.device_free_bytes,
                                                                      .headroom_bytes = fit.headroom_bytes},
                                     canvas_is_the_lever);
                             }
                             if (effective.width >= effective.height) { step_down(effective.width); }
                             else                                      { step_down(effective.height); }
                             need = need_at(effective.width, effective.height);
                         }
                         store_.advance(record.id, JobUpdate{.to = ImageJobState::Running, .now = now});
                         job_started = true;
                         // `seed` is the submit-time number above: the same one the record carries, so a
                         // failed job's stored seed redraws the picture it did not produce.
                         // Both prompts were already rendered (host work, before the probe); only the
                         // tower forward -- the half that needs device memory -- runs here. With a
                         // reference that forward is the CONDITIONED one and the executor runs it inside
                         // the yield (it owns the artifact's vision and text towers), so no plain embeds
                         // are computed: a forward over rows that never saw the picture is not merely
                         // wasted work, it is the copy this route used to return.
                         const bool edit_route = !planes.empty();
                         const std::vector<std::uint16_t> positive =
                             edit_route ? std::vector<std::uint16_t>{} : prompt_embeds_(prompt);
                         const std::vector<std::uint16_t> negative =
                             edit_route || request.true_cfg_scale <= kMinimumImageCFGScale
                                 ? std::vector<std::uint16_t>{}
                                 : prompt_embeds_(negative_prompt);
                         const EditConditioning edit{
                             edit_route ? &prompt : nullptr,
                             (edit_route && request.true_cfg_scale > kMinimumImageCFGScale)
                                 ? &negative_prompt : nullptr};
                         finished = job_executor_(
                             effective, seed, positive, negative,
                             edit_route ? std::span<const ReferencePlanes>(planes)
                                        : std::span<const ReferencePlanes>{},
                             context.image_stream, record.progress, edit_route ? &edit : nullptr);
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
        // Additive on the OpenAI image response, and the reason a client that never polls still learns
        // which decision it got: the synchronous route is all some clients use, and the status object is
        // only what the artifact route serves while no picture file exists. Advertised in
        // /v1/capabilities image_edit.prompt_enhancer.reported_as, so nobody has to know it exists.
        response["prompt_enhance_outcome"] = to_string(enhancer_outcome);
        // The canvas the picture IS, named only when the card made it differ from the canvas the request
        // named -- so the ordinary answer stays the byte-exact OpenAI object, and a swapped one cannot be
        // mistaken for it. `finished` is the truth the pipeline handed back, not what was asked of it.
        // Advertised in /v1/capabilities image_generation.canvas.vram_downgrade.
        if (const std::string drawn = std::to_string(finished.width) + "x" + std::to_string(finished.height);
            drawn != std::to_string(asked_width) + "x" + std::to_string(asked_height)) {
            response["data"][0]["size"]       = drawn;
            response["data"][0]["size_asked"] = std::to_string(asked_width) + "x" + std::to_string(asked_height);
        }
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
        // A third party to the same contract: the card was never handed over because the LLM did not reach
        // a safe point in time. Nothing began, so nothing failed -- and a 500 here told the model the draw
        // was broken, which is how it ended up redrawing its own prompt instead of asking again.
        const bool busy_claim = dynamic_cast<const ImageInterludeBusy*>(&error) != nullptr;
        try {
            if (in_flight_claim || busy_claim) {
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
        if (busy_claim) {
            throw ApiException{ApiError{429, "rate_limit_error", message, "", "image_interlude_busy"}};
        }
        if (degraded) {
            throw ApiException{ApiError{503, "server_error", message, "", "image_engine_degraded"}};
        }
        if (job_started) {
            // The job ran and the LLM was restored, so the picture simply did not come -- but the family
            // still matters, and not only for the log: chat_image_tool.cpp hands this code and message to
            // the model as the tool's error result, and "try again" is the wrong advice for a kernel that
            // refused the geometry it was handed. A std::invalid_argument from inside the job can only
            // have come from an Op's own shape/range check -- every device failure is a runtime_error
            // (image_device_util.h cuda_ok) -- so it names a case the host admission should have refused
            // and never did: ours to fix, and retrying reproduces it.
            if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr) {
                throw ApiException{ApiError{500, "server_error", message, "image",
                                            "image_request_unservable"}};
            }
            if (dynamic_cast<const artifact::ArtifactError*>(&error) != nullptr) {
                // A weight could not be bound or streamed: the artifact lane, not the request.
                throw ApiException{ApiError{500, "server_error", message, "image_artifact",
                                            "image_artifact_failed"}};
            }
            throw ApiException{ApiError{500, "server_error", message, "", "image_generation_failed"}};
        }
        throw ApiException{ApiError{503, "server_error", message, "", "image_interlude_failed"}};
    }
}

} // namespace infernix::serve
