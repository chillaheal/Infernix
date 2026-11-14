// ImageGenerationService host contract: the synchronous generate() maps every interlude/job
// outcome onto the documented ApiException surface and one legal ImageJobStore terminal state.
// IMAGE_GENERATION_SERVICE_TEST: PASS
#include "serve/image_generation_service.h"

#include "artifact/schema.h"
#include "media/encode/png_encode.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix::serve;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n' << std::flush;
    if (!ok) { ++failures; }
}

class FakeRunner final : public ImageInterludeRunner {
public:
    enum class Mode { Run, InFlight, Busy, DrainFail, RestorationFault };

    explicit FakeRunner(Mode mode = Mode::Run) noexcept : mode_(mode) {}

    [[nodiscard]] bool supports_interlude() const noexcept override { return supports; }
    [[nodiscard]] bool is_available() const noexcept override { return available; }

    // The admission question, answered from fields the test sets. A refusal must be visible here
    // BEFORE run() is ever entered: that is the whole point of probing -- a job that cannot fit never
    // closes the LLM's submit door.
    [[nodiscard]] Fit probe(std::uint64_t target_bytes, std::uint64_t margin_bytes) override {
        ++probes;
        probed_target = target_bytes;
        probed_margin = margin_bytes;
        return probe_for ? probe_for(target_bytes) : probe_answer;
    }

    std::uint64_t
    run(std::uint64_t target_bytes, std::uint64_t margin_bytes,
        const std::function<bool(const infernix::ImageInterludeContext&)>& job) override {
        ++runs;
        probed_target = target_bytes; // the shortfall check below reads what the run was asked for
        run_margin    = margin_bytes;
        if (mode_ == Mode::InFlight) {
            // The seam's contract: the runner translates the Program's single-flight refusal into a
            // type, so the HTTP mapping cannot depend on the words the Program chose.
            throw ImageInterludeInFlight{};
        }
        if (mode_ == Mode::Busy) {
            // The seam again: the Program's soft-fail (the LLM never reached a safe point and the quiesce
            // unwound) must reach the HTTP layer as a TYPE, so a busy card answers 429 and never 500.
            throw ImageInterludeBusy{};
        }
        if (mode_ == Mode::DrainFail) {
            throw std::runtime_error("image interlude: the LLM frontier did not drain in time");
        }
        // A case that needs the yield to fall short for ONE of the two asks -- the rewriter's reserve --
        // and to hold for the other, the picture's own need, says so by target instead of by guesswork.
        const std::uint64_t delivered = freed_for ? freed_for(target_bytes) : freed;
        job(infernix::ImageInterludeContext{.freed_bytes = delivered, .device_free_bytes = device_free,
                                            .image_stream = nullptr});
        if (mode_ == Mode::RestorationFault) {
            available = false;
            throw std::runtime_error("image interlude: pool restoration was not confirmed");
        }
        return delivered;
    }

    bool supports = true;
    bool available = true;
    // Generous by default so the guard every existing case walks through stays satisfied: the
    // admission compares these against its own computed need plus the margin.
    static constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t freed      = 24ULL * kGiB;
    // The OTHER half of what the interlude reports, and the half that made the live engine refuse a
    // drawable picture: the free bytes on the card at the instant its own pools moved. The probe's
    // device_free_bytes is a reading of a DIFFERENT instant -- possibly of a card another draw had
    // yielded -- so a case that means to be contended sets this one and leaves the probe's generous.
    std::uint64_t device_free = 0;
    // When set, answers what the yield actually unmapped per requested target (see run()).
    std::function<std::uint64_t(std::uint64_t)> freed_for = {};
    // The same per-target seam on the OTHER side: a case that needs the admission to refuse the
    // rewriter's reserve and still admit the picture's own, smaller need. Unset, probe_answer is the
    // answer to every ask, which is right for every case but the one that is about the rewriter alone.
    std::function<Fit(std::uint64_t)> probe_for = {};
    Fit probe_answer{.fits = true, .freed_bytes_expected = 24ULL * kGiB, .device_free_bytes = 4ULL * kGiB};
    int probes       = 0;
    int runs         = 0;
    std::uint64_t probed_target = 0;
    std::uint64_t probed_margin = 0;
    std::uint64_t run_margin    = 0;

private:
    Mode mode_;
};

// Rendering is the host half of the tower seam: a fixed four-token prompt, with the drop-row count
// non-zero so a mistake (counting the dropped rows as conditioning) changes the need and shows up.
// A rewriter that answers without a sidecar listening on a port, so the service's DECISION about the
// enhancer -- run, skip, or refuse -- is checkable on the host. It counts what it was asked for, because
// the claim under test is usually that it was asked for nothing.
class FakeEnhancer final : public PromptEnhancer {
public:
    FakeEnhancer() : PromptEnhancer("http://127.0.0.1:9", std::chrono::milliseconds{50}) {}

    [[nodiscard]] bool healthy() const override { return healthy_result; }
    void load() const override {
        ++loads;
        // The way the sidecar actually died on this machine: /health answered, the load did not come
        // back, because the ~19 GB it stages cannot be had beside the resident LLM. This is the exact
        // ApiException the worker's POST failure produces in prompt_enhancer.cpp:217.
        if (load_throws) {
            throw ApiException{ApiError{503, "server_error",
                                        "the prompt enhancer at 127.0.0.1:9 is not answering at "
                                        "127.0.0.1:9 (Failed to read connection)", "prompt_enhance",
                                        "image_prompt_enhancer_unavailable"}};
        }
    }
    void unload() const override { ++unloads; }
    [[nodiscard]] EnhancerAnswer rewrite(const std::string& prompt,
                                         const std::vector<std::string>&,
                                         std::uint64_t, std::size_t) const override {
        ++rewrites;
        return EnhancerAnswer{.rewritten_prompt = "REWRITTEN: " + prompt};
    }

    bool healthy_result = true;
    bool load_throws    = false;
    mutable int loads   = 0;
    mutable int unloads = 0;
    mutable int rewrites = 0;
};

PromptRendererProvider fixed_renderer() {
    return [](const std::string& prompt) {
        TowerPrompt rendered;
        rendered.rendered   = prompt;
        rendered.ids        = std::vector<int>(8, 1);
        rendered.drop_rows  = 4;
        return rendered;
    };
}

PromptEmbedsProvider fixed_provider() {
    return [](const TowerPrompt&) { return std::vector<std::uint16_t>(4U, 0U); };
}

// The dimensions the need is built from: the shipped Qwen-Image recipe's numbers, without an artifact.
DiTDimsProvider fixed_dims() {
    return [] {
        infernix::models::qwen_image::DiTDims dims;
        dims.layers       = 32;
        dims.heads        = 32;
        dims.head_dim     = 128;
        dims.hidden       = 4096;
        dims.intermediate = 12288;
        dims.image_in     = 64;
        dims.image_out    = 64;
        return dims;
    };
}

// The ViT lane's shape for the ledger's vision term -- the shipped vision_config, without an artifact.
VisionLaneDimsProvider fixed_vision() {
    return [] {
        infernix::models::qwen_image::VisionLaneDims v;
        v.hidden       = 1152;
        v.intermediate = 4304;
        v.head_dim     = 72;
        v.patch        = 16;
        v.temporal     = 2;
        v.merge        = 2;
        v.out_hidden   = 4096;
        return v;
    };
}

// The edit route's host half: the ti2i render, counted the way the genuine renderer counts -- the
// text rows plus one run of tokens_per_image[0] <|image_pad|> rows. The admission sizes the joint
// sequence on THESE ids (the ledger subtracts the pad rows itself, image_device_need.h:120-125), so a
// fake that ignored the pads would hide a route that under-reserves the yield.
ReferencePromptRendererProvider fixed_reference_renderer() {
    return [](const std::string& prompt, const std::vector<std::uint64_t>& tokens_per_image) {
        TowerPrompt rendered;
        rendered.rendered      = prompt;
        const std::size_t pads = tokens_per_image.empty() ? 0ULL : tokens_per_image[0];
        rendered.ids.assign(8ULL + pads, 7);
        rendered.drop_rows = 2U;
        return rendered;
    };
}

// The picture a fake device job finishes with: RGBA8 pixels and their size, one solid shade. The
// encode, the base64 and the file are the SERVICE's work after the interlude, so a fake stops here
// and what belongs on disk is this encoder's output for these pixels.
[[nodiscard]] FinishedPicture fake_picture(std::uint8_t shade) {
    FinishedPicture picture;
    picture.width  = 2U;
    picture.height = 2U;
    picture.rgba.assign(4U * 2U * 2U, shade);
    for (std::size_t i = 0; i < 4U; ++i) picture.rgba[4U * i + 3] = 255U;
    return picture;
}

[[nodiscard]] std::vector<std::uint8_t> expected_png(const FinishedPicture& picture) {
    return infernix::media::encode::encode_png_rgba8(picture.width, picture.height, picture.rgba);
}

ImageJobExecutor fixed_executor() {
    return [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
              std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
              const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
        return fake_picture(0x01U);
    };
}

ImageJobExecutor throwing_executor() {
    return [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
              std::span<const std::uint16_t>, std::span<const ReferencePlanes>,
              void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) -> FinishedPicture {
        throw std::runtime_error("image job exploded");
    };
}

OpenAIImagesRequest sample_request() {
    OpenAIImagesRequest request;
    request.prompt = "a lighthouse at dusk";
    request.height = 64;
    request.width  = 64;
    request.steps  = 2;
    return request;
}

struct Attempt {
    bool threw   = false;
    int status   = 0;
    std::string code;
    std::string param;   // which field the refusal blames
    std::string message; // what it tells the caller to do about it
    RequestJson response;
};

Attempt attempt(const std::function<RequestJson()>& call) {
    Attempt outcome;
    try {
        outcome.response = call();
    } catch (const ApiException& exception) {
        outcome.threw  = true;
        outcome.status = exception.error().status;
        outcome.code   = exception.error().code;
        outcome.param    = exception.error().param;
        outcome.message  = exception.error().message;
    }
    return outcome;
}

std::shared_ptr<const StoredImageJob> only_record(ImageJobStore& store) {
    if (store.size() != 1U) { return nullptr; }
    for (const auto& id : {"img-1000-1"}) {
        if (auto record = store.get(id)) { return record; }
    }
    return nullptr;
}

void expect_precheck(const char* label, int status, const std::string& code,
                     ImageGenerationService& service) {
    const auto outcome = attempt([&] { return service.generate(sample_request(), 1000); });
    check(outcome.threw && outcome.status == status && outcome.code == code,
          std::string(label) + " -> " + std::to_string(status) + " " + code);
}

} // namespace

int main() {
    // base64 known vectors (RFC 4648 padding cases).
    const auto encode = [](std::vector<std::uint8_t> bytes) {
        return base64_encode_bytes(std::span<const std::uint8_t>(bytes));
    };
    check(encode({'M', 'a', 'n'}) == "TWFu", "base64 Man");
    check(encode({'M'}) == "TQ==", "base64 M");
    check(encode({'M', 'a'}) == "TWE=", "base64 Ma");
    check(encode({}) == "", "base64 empty");

    // Not configured: no runner, no provider, no executor -> no stored job at all.
    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(nullptr, store, PromptRendererProvider{}, PromptEmbedsProvider{}, DiTDimsProvider{}, ImageJobExecutor{});
        check(!service.is_configured(), "empty service is not configured");
        expect_precheck("not configured", 503, "image_generation_not_configured", service);
        check(store.size() == 0U, "not-configured stores nothing");
    }

    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(),
                                       fixed_executor());
        check(service.is_configured(), "full service is configured");
        const auto response = service.generate(sample_request(), 1000);
        check(response["created"] == 1000, "success stamps created");
        const std::string delivered = response["data"][0]["b64_json"].get<std::string>();
        check(response["data"].size() == 1U &&
                  delivered == base64_encode_bytes(expected_png(fake_picture(0x01U))),
              "success returns the base64 of the PNG the service encoded after the interlude");
        const auto record = only_record(store);
        check(record != nullptr && record->state == ImageJobState::Succeeded &&
                  record->id == "img-1000-1" && record->response.contains("data"),
              "success stores one Succeeded job with its response");
        check(service.next_job_id(1001) == "img-1001-2", "job ids increment per process");
        // The seed is decided at submit time and recorded, so the job record says which pixels it
        // drew: an explicit seed verbatim, an omitted one the engine drew (and only the engine knows).
        check(record != nullptr && record->request.contains("seed") &&
                  record->request.at("seed").is_number_unsigned(),
              "a request with no seed still records the seed the engine drew");
    }
    {
        // The seed is decided at submit time, so the record says which pixels this job drew: an
        // explicit seed verbatim, an omitted one the engine draw that nobody can recover after.
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(),
                                       fixed_executor());
        OpenAIImagesRequest pinned = sample_request();
        pinned.seed = 123456789ULL;
        check(!attempt([&] { return service.generate(pinned, 1000); }).threw,
              "a request with a pinned seed is served");
        const auto record = only_record(store);
        check(record && record->request.at("seed") == 123456789ULL,
              "an explicit seed reaches the record unchanged");
    }


    {
        auto runner = std::make_unique<FakeRunner>();
        runner->supports = false;
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        expect_precheck("interlude unsupported", 503, "image_interlude_unsupported", service);
        check(store.size() == 0U, "unsupported stores nothing");
    }

    {
        auto runner = std::make_unique<FakeRunner>();
        runner->available = false;
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        expect_precheck("engine unavailable", 503, "engine_unavailable", service);
        check(store.size() == 0U, "unavailable stores nothing");
    }

    // The job itself fails while the interlude restores cleanly.
    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(),
                                       throwing_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1000); });
        check(outcome.threw && outcome.status == 500 && outcome.code == "image_generation_failed",
              "job failure -> 500 image_generation_failed");
        const auto record = only_record(store);
        check(record != nullptr && record->state == ImageJobState::Failed &&
                  record->error["message"].get<std::string>().find("exploded") !=
                      std::string::npos,
              "job failure stores Failed with the job error");
    }

    // The taxonomy of a job that died AFTER it started. The chat tool forwards this code and message to
    // the model as the tool's error result, so "the draw failed" and "a kernel refused this geometry"
    // must not share a code: retrying the second reproduces it.
        {
        // Each family throws its own type, exactly as the device code does: an Op rejects a shape with
        // std::invalid_argument (image_device_util.h makes every CUDA failure a runtime_error).
        const ImageJobExecutor rejected_by_an_op = [](
            const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
            std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
            const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) -> FinishedPicture {
            throw std::invalid_argument("attn_dit_joint: S must be 1..20480");
        };
        const ImageJobExecutor artifact_unbound = [](
            const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
            std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
            const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) -> FinishedPicture {
            throw infernix::artifact::ArtifactError("text/embed_tokens is not bound");
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService refused_by_an_op(std::make_unique<FakeRunner>(), store,
                                                fixed_renderer(), fixed_provider(), fixed_dims(),
                                                rejected_by_an_op);
        const auto op_outcome =
            attempt([&] { return refused_by_an_op.generate(sample_request(), 1000); });
        check(op_outcome.threw && op_outcome.status == 500 &&
                  op_outcome.code == "image_request_unservable",
              "an Op refusing the job geometry -> 500 image_request_unservable, not a draw failure");

        ImageJobStore artifact_store(8, 1 << 20);
        ImageGenerationService artifact_failed(std::make_unique<FakeRunner>(), artifact_store,
                                               fixed_renderer(), fixed_provider(), fixed_dims(),
                                               artifact_unbound);
        const auto artifact_outcome =
            attempt([&] { return artifact_failed.generate(sample_request(), 1000); });
        check(artifact_outcome.threw && artifact_outcome.status == 500 &&
                  artifact_outcome.code == "image_artifact_failed",
              "an artifact bind failure -> 500 image_artifact_failed");
    }

    // A second concurrent request is told the interlude is already in flight (429).
    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(FakeRunner::Mode::InFlight),
                                       store, fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1000); });
        check(outcome.threw && outcome.status == 429 && outcome.code == "image_interlude_in_flight",
              "in flight -> 429 image_interlude_in_flight");
        const auto record = only_record(store);
        check(record != nullptr && record->state == ImageJobState::Failed,
              "in-flight request stores Failed");
    }

    // Drain-deadline miss: the job never ran, the LLM kept serving.
    {
        ImageJobStore store(8, 1 << 20);
    // The admission refuses before the interlude: a probe that answers "no" must never reach run(), so
    // the LLM's submit door stays open for everyone else.
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->probe_answer   = FakeRunner::Fit{.fits = false, .freed_bytes_expected = 1ULL << 30};
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), fixed_executor());
        const auto refusal = attempt([&] { return service.generate(sample_request(), 1700); });
        check(refusal.threw && refusal.status == 503 &&
                  refusal.code == "image_vram_insufficient",
              "a yield that cannot cover the need refuses the request");
        check(eyes->probes == 1 && eyes->runs == 0, "the refusal happens before anything is yielded");
    }

    // The blind no, measured on the deployed engine on 2026-11-05: 18 refusals of 768x768 in one instance,
    // every one 1-32 ms, every one while the chat held the residency lease, and the same canvas drawn in
    // 44 s once the chat was quiet. The probe had compared a KV-only promise against the gate because the
    // plan keeps the frame pool whole while a lease is out -- and the quiesce that takes the ring back was
    // never reached, because the refusal is what prevented it.
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->probe_answer   = FakeRunner::Fit{.fits = false,
                                               .expert_pool_frozen_by_lease = true,
                                               .freed_bytes_expected = 1ULL << 30};
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1750); });
        check(!outcome.threw, "a no from a frame pool the chat is holding admits the job");
        check(eyes->probes == 1 && eyes->runs == 1,
              "a blind no is not traded for canvas: one probe, then the interlude decides");
    }

    // The same blind no still ends in the honest refusal when the yield cannot pay even the floor: letting
    // the job in moved the decision, not the risk -- and the stop is still before the first allocation.
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->probe_answer   = FakeRunner::Fit{.fits = false,
                                               .expert_pool_frozen_by_lease = true,
                                               .freed_bytes_expected = 1ULL << 30};
        eyes->freed          = 1ULL << 20;
        bool executor_ran    = false;
        ImageJobExecutor watch =
            [&executor_ran](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
                            std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
                            const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
                executor_ran = true;
                return fake_picture(0x01U);
            };
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), watch);
        const auto refusal = attempt([&] { return service.generate(sample_request(), 1760); });
        check(refusal.threw && refusal.code == "image_vram_insufficient",
              "a blind no whose yield cannot pay the floor is still refused");
        check(eyes->runs == 1 && !executor_ran, "on the delivered bytes, before the first allocation");
        check(refusal.message.find("1 MiB already free") != std::string::npos ||
                  refusal.message.find("MiB already free") != std::string::npos,
              "the refusal names the bytes the yield delivered");
    }

    // A card that could not be handed over is contention, not a malfunction: the same request retried is
    // the fix, and a 500 tells the model to change something that was never the problem.
    {
        ImageJobStore store(8, 8ULL << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(FakeRunner::Mode::Busy), store,
                                       fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1770); });
        check(outcome.threw && outcome.status == 429 && outcome.code == "image_interlude_busy",
              "no yieldable frontier -> 429 image_interlude_busy, never 500");
    }

    // What the 503 may tell the client to do, decided by the arithmetic and not by a habit: a canvas whose
    // need genuinely falls may be traded, and one whose need does not must not be walked into the same
    // refusal a step at a time.
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->probe_answer   = FakeRunner::Fit{.fits = false, .freed_bytes_expected = 1ULL << 30};
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), fixed_executor());
        OpenAIImagesRequest big    = sample_request();
        big.width    = 1024;
        big.height   = 1024;
        const auto refusal = attempt([&] { return service.generate(big, 1780); });
        check(refusal.threw && refusal.code == "image_vram_insufficient",
              "a big canvas that never fits is refused");
        check(refusal.message.find("a smaller canvas lowers this need") != std::string::npos,
              "when the need does fall with the canvas, the refusal says so");
        check(eyes->probes > 1, "and the walk really walked its canvas down");
    }
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->probe_answer   = FakeRunner::Fit{.fits = false, .freed_bytes_expected = 1ULL << 30};
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), fixed_executor());
        OpenAIImagesRequest tiny  = sample_request();
        tiny.width   = 256;
        tiny.height  = 256;
        const auto refusal = attempt([&] { return service.generate(tiny, 1790); });
        check(refusal.threw && refusal.code == "image_vram_insufficient",
              "a floor canvas that does not fit is refused");
        check(refusal.message.find("will not help") != std::string::npos,
              "and the refusal stops advising a shrink that cannot lower the need");
        check(eyes->probes == 1, "with nothing to trade, the walk asks the probe once");
    }

    // What the yield actually unmapped is what the job trusts: a grant short of the computed need
    // stops the job before its first allocation, with the model healthy and the code intact.
    {
        ImageJobStore store(8, 8ULL << 20);
        auto          runner = std::make_unique<FakeRunner>();
        FakeRunner*   eyes   = runner.get();
        eyes->freed          = 1ULL << 30;
        eyes->probe_answer   = FakeRunner::Fit{.fits = true, .freed_bytes_expected = 24ULL << 30,
                                               .device_free_bytes = 0};
        bool executor_ran    = false;
        ImageJobExecutor watch =
            [&executor_ran](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
                            std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
                            const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
                executor_ran = true;
                return fake_picture(0x01U);
            };
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), watch);
        const auto refusal = attempt([&] { return service.generate(sample_request(), 1800); });
        check(refusal.threw && refusal.code == "image_vram_insufficient",
              "a short grant is reported as the same refusal, not as a failed draw");
        check(eyes->runs == 1 && !executor_ran, "the job stops before its first allocation");
        check(eyes->run_margin == 512ULL * 1024ULL * 1024ULL, "the yield is asked for a real margin");
    }

        ImageGenerationService service(std::make_unique<FakeRunner>(FakeRunner::Mode::DrainFail),
                                       store, fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1000); });
        check(outcome.threw && outcome.status == 503 && outcome.code == "image_interlude_failed",
              "drain miss -> 503 image_interlude_failed");
        const auto record = only_record(store);
        check(record != nullptr && record->state == ImageJobState::Failed,
              "drain miss stores Failed");
    }

    // Restoration not confirmed: the engine is Degraded and both failures are kept.
    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(
            std::make_unique<FakeRunner>(FakeRunner::Mode::RestorationFault), store,
            fixed_renderer(), fixed_provider(), fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return service.generate(sample_request(), 1000); });
        check(outcome.threw && outcome.status == 503 && outcome.code == "image_engine_degraded",
              "restoration fault -> 503 image_engine_degraded");
        const auto record = only_record(store);
        check(record != nullptr && record->state == ImageJobState::FailedEngineDegraded &&
                  record->error.contains("message") && record->restoration_error.contains("message"),
              "restoration fault stores FailedEngineDegraded with both errors");
    }

    // The busy signal /health carries. A supervisor has to be able to ask whether a picture is being
    // drawn, because a chat probe queued behind one times out and a keepalive that reads that timeout
    // as death kills a job that was succeeding (measured 2026-11-01 22:30). Read from INSIDE the
    // executor -- the only point that is certainly during the draw -- and again after it returns, so a
    // count left behind by a finished job cannot pass.
    {
        ImageJobStore store(8, 1ULL << 20);
        ImageGenerationService* self    = nullptr;
        std::size_t             during  = 0;
        std::int64_t            age_now = -1;
        std::int64_t            age_ahead = -1;
        ImageJobExecutor watch  =
            [&](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
                std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
                const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
                const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                during    = self->in_flight_jobs();
                age_now   = self->oldest_in_flight_age_seconds(static_cast<std::int64_t>(now));
                age_ahead = self->oldest_in_flight_age_seconds(static_cast<std::int64_t>(now) + 3600);
                return fake_picture(0x01U);
            };
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(),
                                       fixed_provider(), fixed_dims(), watch);
        self = &service;
        check(service.in_flight_jobs() == 0, "no job means nothing is in flight");
        (void)attempt([&] { return service.generate(sample_request(), 1000); });
        check(during == 1, "the executor sees its own job in flight");
        check(age_now >= 0 && age_now <= 2, "a job that just started reads an age of about zero");
        check(age_ahead >= 3598 && age_ahead <= 3602,
              "the age is the real elapsed time, not a constant: asked an hour ahead it says an hour");
        check(service.in_flight_jobs() == 0, "a finished job leaves no count behind");

        // The path that makes RAII the point: generate() leaves through a throw on almost every
        // failure, and a count surviving one of those would tell a supervisor the card is busy forever.
        ImageJobStore failing_store(8, 1ULL << 20);
        ImageGenerationService failing(std::make_unique<FakeRunner>(), failing_store, fixed_renderer(),
                                       fixed_provider(), fixed_dims(), throwing_executor());
        const auto failed = attempt([&] { return failing.generate(sample_request(), 1000); });
        check(failed.threw, "the throwing executor really threw");
        check(failing.in_flight_jobs() == 0, "a job that threw leaves no count behind either");
    }

    // #75 reference image: a valid base64 PNG reaches the executor as the prepared plane with the
    // reference geometry (PIL calculate_dimensions at 1024^2 area, 64x48 -> 1184x896), and the PNG's
    // alpha survives as the real fourth channel. An undecodable payload is a pre-interlude 400.
    {
        std::vector<std::uint8_t> rgba(64U * 48U * 4);
        for (unsigned i = 0; i < 64U * 48U; ++i) {
            rgba[4 * i + 0] = static_cast<std::uint8_t>(i);
            rgba[4 * i + 1] = static_cast<std::uint8_t>(i / 7U);
            rgba[4 * i + 2] = static_cast<std::uint8_t>(i / 13U);
            rgba[4 * i + 3] = static_cast<std::uint8_t>(i % 256U);
        }
        const std::vector<std::uint8_t> png = infernix::media::encode::encode_png_rgba8(64, 48, rgba);
        const std::string       payload     = base64_encode_bytes(png);

        bool        saw_reference  = false;
        std::size_t plane_floats   = 0;
        std::uint32_t plane_w      = 0, plane_h = 0;
        float       alpha_min = 0.F, alpha_max = 0.F;
        // The edit route's handoff: both planes, and the ti2i render whose pad rows ARE the reference.
        bool        saw_edit_planes = false, saw_edit_prompt = false;
        std::size_t edit_rows = 0, vision_rgb_bytes = 0;
        std::uint32_t vision_w = 0, vision_h = 0;
        ImageJobExecutor capture = [&](const OpenAIImagesRequest&, std::uint64_t,
                                       std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                       std::span<const ReferencePlanes> ref, void*,
                                       const std::shared_ptr<ImageJobProgress>&,
                                       const EditConditioning* edit)
            -> FinishedPicture {
            saw_reference = !ref.empty();
            if (edit != nullptr) {
                saw_edit_planes = !ref.empty() && edit->prompt != nullptr &&
                                  edit->negative_prompt == nullptr; // true_cfg_scale is 1 here
                if (!ref.empty()) {
                    vision_rgb_bytes = ref[0].vision.rgb.size();
                    vision_w         = ref[0].vision.width_px;
                    vision_h         = ref[0].vision.height_px;
                }
                if (edit->prompt != nullptr) { edit_rows = edit->prompt->tokens(); }
            }
            if (!ref.empty()) {
                plane_floats = ref[0].vae.pixels.size();
                plane_w      = ref[0].vae.width_px;
                plane_h      = ref[0].vae.height_px;
                const std::size_t pixels = static_cast<std::size_t>(ref[0].vae.width_px) * ref[0].vae.height_px;
                const float* alpha       = ref[0].vae.pixels.data() + 3 * pixels;
                const auto [lo, hi]      = std::minmax_element(alpha, alpha + pixels);
                alpha_min                = *lo;
                alpha_max                = *hi;
            }
            return fake_picture(0x01U);
        };

        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(),
                                       capture, fixed_vision(), fixed_reference_renderer());
        OpenAIImagesRequest request = sample_request();
        request.images              = {payload};
        service.generate(request, 1000);
        check(saw_reference && plane_w == 1184U && plane_h == 896U &&
                  plane_floats == 4ULL * 1184ULL * 896ULL,
              "valid PNG image -> 1184x896 RGBA plane at the executor");
        check(alpha_min < -0.5F && alpha_max > 0.5F, "reference alpha carried raw, not composited");
        // The edit route's handoff: the executor gets BOTH planes of the one snapped picture, and the
        // prompt it renders the conditioning from is the ti2i render -- its kept rows are the text plus
        // one vision row per pad, 4 * 1036 == the 4144 latent rows the VAE encode will return.
        check(saw_edit_planes && vision_w == 1184U && vision_h == 896U &&
                  vision_rgb_bytes == 3ULL * 1184ULL * 896ULL,
              "a reference hands the executor both planes of the same snapped picture");
        // 8 text rows + the 1036 pad rows, minus the 2 dropped system rows the t2i route drops too
        // (the pads sit after that line, so the prefix length does not move).
        check(edit_rows == 8ULL + 1036ULL - 2ULL,
              "a reference renders the ti2i prompt: 1036 vision pad rows on top of the text rows");

        ImageJobStore bad_store(8, 1 << 20);
        ImageGenerationService bad_service(std::make_unique<FakeRunner>(), bad_store,
                                           fixed_renderer(), fixed_provider(), fixed_dims(), capture);
        OpenAIImagesRequest bad = sample_request();
        bad.images              = {"not base64 at all!!"};
        const auto outcome      = attempt([&] { return bad_service.generate(bad, 1000); });
        // The codec's own words are never echoed to the client, so the CODE is the contract here: a
        // payload that is not base64 PNG/JPEG is image_decode_failed, not the generic image_invalid and
        // not a 500 (nothing about this request is the server's fault).
        check(outcome.threw && outcome.status == 400 && outcome.code == "image_decode_failed",
              "undecodable image -> 400 image_decode_failed");
        check(bad_store.size() == 0U, "invalid image stores no job (fails before the interlude)");
    }

    // The family's prompt enhancer is a SEPARATE shipped model (Qwen/Qwen-Image-2.1-PE-I2I, a
    // fine-tuned Qwen3.5-VL 9B that reads the instruction and the pictures and answers
    // {rewritten_prompt, wh_ratio, ratio_follow}); the tower inside the image artifact is a
    // different checkpoint and cannot stand in for it. None is installed in this build, so asking
    // for one is a refusal that names what is missing -- never a drawing made from words the client
    // believed had been rewritten. Nothing is stored or yielded for a request that cannot run.
    {
        ImageJobStore store(8, 1 << 20);
        auto runner = std::make_unique<FakeRunner>();
        FakeRunner* eyes = runner.get();
        ImageGenerationService service(
            std::move(runner), store, fixed_renderer(), fixed_provider(), fixed_dims(),
            [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
               std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
               const std::shared_ptr<ImageJobProgress>&,
               const EditConditioning*) { return fake_picture(0x01U); });
        OpenAIImagesRequest request = sample_request();
        request.prompt_enhance = PromptEnhanceMode::On;
        const auto outcome = attempt([&] { return service.generate(request, 1000); });
        check(outcome.threw && outcome.status == 503 &&
                  outcome.code == "image_prompt_enhancer_unavailable",
              "prompt_enhance without a rewriter -> 503 image_prompt_enhancer_unavailable");
        check(outcome.param == "prompt_enhance",
              "the refusal names the field the client asked about");
        check(eyes->probes == 0 && eyes->runs == 0,
              "the refusal never probes the yield and never runs a job");
        check(store.size() == 0U, "the refusal stores nothing");

        // The same request with the judgement delegated. auto told the server to choose, so a build
        // with no rewriter answers it by drawing from the client's own words rather than refusing --
        // the difference between auto and true is exactly this 503, and nothing else about the
        // request changed. Measured against the counters rather than against zero, so it still says
        // what it means after the request above touched them.
        const auto probes_before_auto = eyes->probes;
        const auto runs_before_auto = eyes->runs;
        request.prompt_enhance = PromptEnhanceMode::Auto;
        const auto automatic = attempt([&] { return service.generate(request, 1002); });
        check(!automatic.threw, "prompt_enhance \"auto\" without a rewriter is served, not refused");
        const auto probes_after_auto = eyes->probes;
        const auto runs_after_auto = eyes->runs;

        // And the same request without the switch is served: the refusal is about the switch, not
        // about the route.
        request.prompt_enhance = PromptEnhanceMode::Off;
        const auto served = attempt([&] { return service.generate(request, 1001); });
        check(!served.threw, "the same request without the switch is served");

        // The claim that matters is not "auto spent nothing" -- the picture's own interlude goes
        // through the same two counters, so an absolute zero here would be a claim about the fake.
        // What auto promised is that declining the rewriter costs the request nothing it would not
        // have paid anyway, and that is an equality against the identical request without the switch.
        check(probes_after_auto - probes_before_auto == eyes->probes - probes_after_auto &&
                  runs_after_auto - runs_before_auto == eyes->runs - runs_after_auto,
              "auto with no rewriter costs exactly what the identical plain request costs");
    }

    // The OTHER half of the auto promise, which no case above touches: not "no rewriter installed" but
    // "the card cannot be had for one". The pre-probe declines for auto already; what that probe merely
    // PROMISED is measured a second time inside the interlude, where the yield has actually unmapped
    // something -- and that second check threw for BOTH modes. So the server's own sentence, "the card
    // cannot be had for one, which for auto is a skip and never a 503", was false on the wire: a real
    // auto edit came back 503 image_vram_insufficient. The fake drives exactly the split that produces
    // it -- the probe says yes, the yield delivers enough for the picture and not for the rewriter.
    {
        const auto short_for_rewriter = [](std::uint64_t target_bytes) {
            // The rewriter asks for its whole reserve -- 19042 MiB, image_generation_service.cpp:67 --
            // and the picture asks for its own, far smaller, need. Short only for the first, so nothing
            // else in the request can be the reason for a refusal. The bound sits between the two and has
            // to be read against that constant: it was 20 GiB while the reserve was 20 GiB, and the
            // reserve dropped under it the day it became the measured peak instead of a second slack.
            return target_bytes >= 16ULL * FakeRunner::kGiB ? 8ULL * FakeRunner::kGiB
                                                            : 24ULL * FakeRunner::kGiB;
        };
        const std::string reference = base64_encode_bytes(
            infernix::media::encode::encode_png_rgba8(
                64, 48, std::vector<std::uint8_t>(64U * 48U * 4, 90U)));

        ImageJobStore store(8, 1 << 20);
        auto        runner  = std::make_unique<FakeRunner>();
        FakeRunner* eyes    = runner.get();
        eyes->freed_for     = short_for_rewriter;
        auto          enhancer = std::make_shared<FakeEnhancer>();
        std::string   drawn_prompt;
        ImageJobExecutor watch = [&drawn_prompt](const OpenAIImagesRequest& req, std::uint64_t,
                                                 std::span<const std::uint16_t>,
                                                 std::span<const std::uint16_t>,
                                                 std::span<const ReferencePlanes>, void*,
                                                 const std::shared_ptr<ImageJobProgress>&,
                                                 const EditConditioning*) {
            drawn_prompt = req.prompt;
            return fake_picture(0x02U);
        };
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), watch, fixed_vision(),
                                       fixed_reference_renderer(), enhancer);
        OpenAIImagesRequest request        = sample_request();
        request.images                     = {reference};
        request.prompt_enhance             = PromptEnhanceMode::Auto;
        const auto auto_outcome = attempt([&] { return service.generate(request, 1100); });
        check(!auto_outcome.threw, "auto is SERVED when the card cannot be had for the rewriter");
        check(enhancer->rewrites == 0 && enhancer->loads == 0 && enhancer->unloads == 0,
              "the declined rewrite never asked the sidecar for weights or for words");
        check(drawn_prompt == request.prompt,
              "a declined rewrite draws the client's own words, not the unfilled answer");
        check(eyes->runs == 2,
              "the request paid one interlude it declined and one for the picture");
        check(store.size() == 1U, "the served auto request is stored like any other");
        const auto stored = store.get("img-1100-1");
        check(stored && stored->request.at("prompt_enhance_outcome") == "skipped_no_device",
              "the record names WHICH auto skip this was, not only that one happened");
        // The two surfaces a client can read without a log. The status object is what GET /v1/images/<id>
        // serves while no picture file exists, which is the only time a client ever sees it -- so the
        // synchronous body is the one that reaches a client that does not poll.
        check(auto_outcome.response.contains("prompt_enhance_outcome") &&
                  auto_outcome.response.at("prompt_enhance_outcome") == "skipped_no_device",
              "the synchronous answer carries the reason, for a client that never polls");
        const auto polled = service.job_status_json("img-1100-1");
        check(!polled.is_null() && polled.contains("prompt_enhance_outcome") &&
                  polled.at("prompt_enhance_outcome") == "skipped_no_device",
              "the status object a polling client reads carries the same reason");

        // The asymmetry the parameter is defined by: the same shortage IS a refusal for a client that
        // named prompt_enhance: true, because handing it the words it believed had been rewritten is
        // the lie this field exists to prevent.
        auto refusing_runner         = std::make_unique<FakeRunner>();
        refusing_runner->freed_for   = short_for_rewriter;
        auto refusing_enhancer       = std::make_shared<FakeEnhancer>();
        ImageGenerationService refusing(std::move(refusing_runner), store, fixed_renderer(),
                                        fixed_provider(), fixed_dims(), watch, fixed_vision(),
                                        fixed_reference_renderer(), refusing_enhancer);
        OpenAIImagesRequest asked      = request;
        asked.prompt_enhance           = PromptEnhanceMode::On;
        const auto asked_outcome = attempt([&] { return refusing.generate(asked, 1101); });
        check(asked_outcome.threw && asked_outcome.status == 503 &&
                  asked_outcome.code == "image_vram_insufficient",
              "prompt_enhance: true keeps its 503 for the very shortage auto skips");
        check(refusing_enhancer->rewrites == 0, "the refusal never reached the sidecar either");
        check(store.get("img-1101-1") == nullptr,
              "the refused request stored no record for anyone to misread later");

        // The positive control, without which the two above only prove the fake is inert: handed a card,
        // true really pays the rewriter, hands the card back, and is drawn from the REWRITTEN words.
        auto generous_runner   = std::make_unique<FakeRunner>();
        auto generous_enhancer = std::make_shared<FakeEnhancer>();
        std::string rewritten_seen;
        ImageJobExecutor note  = [&rewritten_seen](const OpenAIImagesRequest& req, std::uint64_t,
                                                   std::span<const std::uint16_t>,
                                                   std::span<const std::uint16_t>,
                                                   std::span<const ReferencePlanes>, void*,
                                                   const std::shared_ptr<ImageJobProgress>&,
                                                   const EditConditioning*) {
            rewritten_seen = req.prompt;
            return fake_picture(0x03U);
        };
        ImageGenerationService generous(std::move(generous_runner), store, fixed_renderer(),
                                        fixed_provider(), fixed_dims(), note, fixed_vision(),
                                        fixed_reference_renderer(), generous_enhancer);
        const auto paid_outcome = attempt([&] { return generous.generate(asked, 1102); });
        check(!paid_outcome.threw && generous_enhancer->rewrites == 1 &&
                  generous_enhancer->loads == 1 && generous_enhancer->unloads == 1 &&
                  rewritten_seen == "REWRITTEN: " + asked.prompt,
              "given the card, true pays the rewriter, releases it, and draws the rewritten words");
        const auto paid = store.get("img-1102-1");
        check(paid && paid->request.at("prompt_enhance_outcome") == "applied",
              "the record says the rewrite was applied when it was");

        // The third way the rewriter can be impossible, and the only one no case covered: the sidecar
        // is healthy on /health and then DIES on the load the interlude asks of it -- what the card has
        // no room for, the host has no room to stage either. It reaches the service as the unavailable
        // 503 from the POST, not as a refused connection, and it used to escape the auto judgement
        // entirely: a real auto edit on this machine came back 503 image_prompt_enhancer_unavailable
        // after 7 s, while the document and the two branches above promise auto is always served.
        auto dead_runner     = std::make_unique<FakeRunner>();
        auto dead_enhancer   = std::make_shared<FakeEnhancer>();
        dead_enhancer->load_throws = true;
        std::string own_words;
        ImageJobExecutor remember  = [&own_words](const OpenAIImagesRequest& req, std::uint64_t,
                                                 std::span<const std::uint16_t>,
                                                 std::span<const std::uint16_t>,
                                                 std::span<const ReferencePlanes>, void*,
                                                 const std::shared_ptr<ImageJobProgress>&,
                                                 const EditConditioning*) {
            own_words = req.prompt;
            return fake_picture(0x05U);
        };
        ImageGenerationService dead(std::move(dead_runner), store, fixed_renderer(), fixed_provider(),
                                    fixed_dims(), remember, fixed_vision(),
                                    fixed_reference_renderer(), dead_enhancer);
        const auto dead_outcome = attempt([&] { return dead.generate(request, 1104); });
        check(!dead_outcome.threw, "a sidecar that dies on the load is served, not refused, for auto");
        check(own_words == request.prompt,
              "the picture is drawn from the client's own words, not from the answer that never arrived");
        const auto dead_record = store.get("img-1104-1");
        check(dead_record &&
                  dead_record->request.at("prompt_enhance_outcome") == "skipped_no_rewriter",
              "a sidecar that died mid-load is recorded as no rewriter, the same reason a refused one is");
        check(dead_enhancer->unloads == 1,
              "the weights a dying load had begun to take are handed back on the way out");

        // The refusal still belongs to the client that asked for the rewrite by name: the same dead
        // sidecar must still answer true with the documented 503, or this fix would have quietly
        // turned a promised rewrite into a silent one.
        auto dead_asked_runner  = std::make_unique<FakeRunner>();
        auto dead_asked         = std::make_shared<FakeEnhancer>();
        dead_asked->load_throws = true;
        ImageGenerationService dead_true(std::move(dead_asked_runner), store, fixed_renderer(),
                                         fixed_provider(), fixed_dims(), remember, fixed_vision(),
                                         fixed_reference_renderer(), dead_asked);
        OpenAIImagesRequest asked_dead = request;
        asked_dead.prompt_enhance      = PromptEnhanceMode::On;
        const auto dead_asked_outcome  = attempt([&] { return dead_true.generate(asked_dead, 1105); });
        check(dead_asked_outcome.threw && dead_asked_outcome.status == 503 &&
                  dead_asked_outcome.code == "image_prompt_enhancer_unavailable",
              "prompt_enhance: true keeps the 503 for the very same dead sidecar");
        check(store.get("img-1105-1") == nullptr,
              "and the refused request leaves no record a client could mistake for a picture");

        // The same skip one level EARLIER, which measured live said "off" instead. The block above
        // lets the probe promise the card and lets the yield fall short; this one is the pre-probe
        // itself saying no, before any interlude. That branch ran with enhancer_outcome still at Off,
        // so the answer told the client "no rewrite was asked for" when the truth was "the card cannot
        // hold one" -- the reader cannot tell a rewriter worth waiting for from one that never comes.
        {
            // The rewriter asks for its whole 19042 MiB reserve (image_generation_service.cpp:67) and
            // the picture for its own, far smaller need: refuse only the first, so the request that
            // survives is the picture and nothing else can be the reason for a refusal.
            const auto nofit_for_rewriter = [](std::uint64_t target_bytes) {
                if (target_bytes >= 16ULL * FakeRunner::kGiB) { return ImageInterludeRunner::Fit{}; }
                return ImageInterludeRunner::Fit{.fits = true,
                                                 .freed_bytes_expected = 24ULL * FakeRunner::kGiB,
                                                 .device_free_bytes  = 4ULL * FakeRunner::kGiB};
            };
            auto            nofit_runner   = std::make_unique<FakeRunner>();
            FakeRunner*     nofit          = nofit_runner.get();
            nofit->probe_for               = nofit_for_rewriter;
            auto            nofit_enhancer = std::make_shared<FakeEnhancer>();
            std::string     kept_prompt;
            ImageJobExecutor keep          = [&kept_prompt](const OpenAIImagesRequest& req, std::uint64_t,
                                                            std::span<const std::uint16_t>,
                                                            std::span<const std::uint16_t>,
                                                            std::span<const ReferencePlanes>, void*,
                                                            const std::shared_ptr<ImageJobProgress>&,
                                                            const EditConditioning*) {
                kept_prompt = req.prompt;
                return fake_picture(0x04U);
            };
            ImageGenerationService noservice(std::move(nofit_runner), store, fixed_renderer(),
                                             fixed_provider(), fixed_dims(), keep, fixed_vision(),
                                             fixed_reference_renderer(), nofit_enhancer);
            OpenAIImagesRequest ask          = request;
            ask.images                       = {reference};
            ask.prompt_enhance               = PromptEnhanceMode::Auto;
            const auto declined              = attempt([&] { return noservice.generate(ask, 1103); });
            check(!declined.threw,
                  "a pre-probe that cannot hold the rewriter still draws the picture for auto");
            check(nofit->probes == 2 && nofit->runs == 1,
                  "the declined rewrite was probed and never entered an interlude: the picture paid the only one");
            check(nofit_enhancer->loads == 0 && nofit_enhancer->rewrites == 0,
                  "the sidecar was never asked for weights or for words");
            check(kept_prompt == ask.prompt,
                  "the picture is drawn from the client's own words");
            const auto declined_record = store.get("img-1103-1");
            check(declined_record &&
                      declined_record->request.at("prompt_enhance_outcome") == "skipped_no_device",
                  "the pre-probe decline names skipped_no_device, not off");

            auto            nofit_refusing   = std::make_unique<FakeRunner>();
            nofit_refusing->probe_for         = nofit_for_rewriter;
            auto            nofit_refuse_enhancer = std::make_shared<FakeEnhancer>();
            ImageGenerationService nofit_refusal(std::move(nofit_refusing), store, fixed_renderer(),
                                                 fixed_provider(), fixed_dims(), keep, fixed_vision(),
                                                 fixed_reference_renderer(), nofit_refuse_enhancer);
            OpenAIImagesRequest believed = ask;
            believed.prompt_enhance      = PromptEnhanceMode::On;
            const auto refused           = attempt([&] { return nofit_refusal.generate(believed, 1104); });
            check(refused.threw && refused.status == 503 &&
                      refused.code == "image_vram_insufficient",
                  "the same pre-probe shortage stays a 503 for a client that believed its words were rewritten");
        }

        // The four endings that need no shortage at all, each with its OWN reason in the record. They
        // look identical from outside the server -- one picture, drawn from the client's own words --
        // and they are four different things a reader of the log must be able to tell apart: the client
        // did not ask, there was no picture to read, the instruction was already the rewriter's answer,
        // and no sidecar is answering. The last is a deployment gap; the others are ordinary requests.
        struct SkipCase {
            PromptEnhanceMode mode;
            bool              reference;
            bool              marked;
            bool              rewriter_up;
            const char*       outcome;
        };
        const SkipCase skips[] = {
            {PromptEnhanceMode::Off,  true,  false, true,  "off"},
            {PromptEnhanceMode::Auto, false, false, true,  "skipped_no_reference"},
            {PromptEnhanceMode::Auto, true,  true,  true,  "skipped_already_rewritten"},
            {PromptEnhanceMode::Auto, true,  false, false, "skipped_no_rewriter"},
        };
        for (const auto& c : skips) {
            ImageJobStore    skip_store(8, 1 << 20);
            auto             skip_runner   = std::make_unique<FakeRunner>();
            auto             skip_enhancer = std::make_shared<FakeEnhancer>();
            skip_enhancer->healthy_result  = c.rewriter_up;
            std::string      seen;
            ImageJobExecutor note2         = [&seen](const OpenAIImagesRequest& req, std::uint64_t,
                                                    std::span<const std::uint16_t>,
                                                    std::span<const std::uint16_t>,
                                                    std::span<const ReferencePlanes>, void*,
                                                    const std::shared_ptr<ImageJobProgress>&,
                                                    const EditConditioning*) {
                seen = req.prompt;
                return fake_picture(0x04U);
            };
            ImageGenerationService skip(std::move(skip_runner), skip_store, fixed_renderer(),
                                        fixed_provider(), fixed_dims(), note2, fixed_vision(),
                                        fixed_reference_renderer(), skip_enhancer);
            OpenAIImagesRequest r = sample_request();
            if (c.reference) { r.images = {reference}; }
            if (c.marked) { r.prompt = "On <image1>, make the ground green."; }
            r.prompt_enhance = c.mode;
            const auto served = attempt([&] { return skip.generate(r, 1200); });
            check(!served.threw && seen == r.prompt && skip_enhancer->rewrites == 0 &&
                      served.response.contains("prompt_enhance_outcome") &&
                      served.response.at("prompt_enhance_outcome") == c.outcome,
                  std::string(c.outcome) + ": served from the client's own words, rewriter untouched, "
                                            "and the body says so");
            const auto rec = skip_store.get("img-1200-1");
            check(rec && rec->request.at("prompt_enhance_outcome") == c.outcome,
                  std::string(c.outcome) + ": the record carries that reason");
        }
    }

    // P9 multi-reference: every reference conditions the SAME forward, so the contract is that all
    // of them arrive and none is quietly dropped or averaged. Three things must hold together, and
    // they are the three the executor's joint assembly cannot recover from getting wrong: the
    // renderer is handed one vision count per picture (the pad runs it expands, and therefore the
    // joint sequence the admission sizes on, is the SUM over them), the executor is handed one plane
    // per picture in request order, and the canvas follows the LAST reference when the client named
    // no size -- image[-1] in the reference pipeline
    // (image-edit/refs/q21_pipeline_qwenimage21.py:630-636).
    {
        const std::string wide = base64_encode_bytes(
            infernix::media::encode::encode_png_rgba8(
                64, 48, std::vector<std::uint8_t>(64U * 48U * 4, 90U)));   // -> 1184x896
        const std::string tall = base64_encode_bytes(
            infernix::media::encode::encode_png_rgba8(
                48, 64, std::vector<std::uint8_t>(48U * 64U * 4, 200U)));  // -> 896x1184

        std::vector<std::uint64_t> rendered_pads;
        ReferencePromptRendererProvider recorder = [&](const std::string& prompt,
                                                     const std::vector<std::uint64_t>& pads) {
            rendered_pads = pads;
            TowerPrompt rendered;
            rendered.rendered = prompt;
            std::size_t rows = 8ULL;
            for (const std::uint64_t pad : pads) { rows += pad; }
            rendered.ids.assign(rows, 7);
            rendered.drop_rows = 2U;
            return rendered;
        };

        std::size_t   planes_at_executor = 0;
        std::uint32_t first_w = 0, first_h = 0, last_w = 0, last_h = 0;
        std::uint32_t canvas_w = 0, canvas_h = 0;
        std::uint64_t probed_with_two = 0, probed_with_one = 0;
        ImageJobExecutor capture = [&](const OpenAIImagesRequest& effective, std::uint64_t,
                                       std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                       std::span<const ReferencePlanes> refs, void*,
                                       const std::shared_ptr<ImageJobProgress>&,
                                       const EditConditioning*) -> FinishedPicture {
            planes_at_executor = refs.size();
            if (!refs.empty()) {
                first_w = refs.front().vae.width_px;
                first_h = refs.front().vae.height_px;
                last_w  = refs.back().vae.width_px;
                last_h  = refs.back().vae.height_px;
            }
            canvas_w = effective.width;
            canvas_h = effective.height;
            return fake_picture(0x02U);
        };

        ImageJobStore store(8, 1 << 20);
        auto runner = std::make_unique<FakeRunner>();
        FakeRunner* eyes = runner.get();
        ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                       fixed_dims(), capture, fixed_vision(), recorder);

        OpenAIImagesRequest request = sample_request();
        request.size_requested = false;  // an edit that named no size: the reference decides the canvas
        request.images         = {wide, tall};
        service.generate(request, 1000);
        probed_with_two = eyes->probed_target;

        check(rendered_pads.size() == 2ULL && rendered_pads[0] == 1036ULL &&
                  rendered_pads[1] == 1036ULL,
              "two references render one pad run each, not one run for the pair");
        check(planes_at_executor == 2ULL && first_w == 1184U && first_h == 896U &&
                  last_w == 896U && last_h == 1184U,
              "both planes reach the executor, in the order the client sent them");
        check(canvas_w == 672U && canvas_h == 896U,
              "with no size named the canvas follows the LAST reference, as the pipeline does");

        // The same request naming <image1> is the client saying "the canvas is THAT picture": the
        // other reference still conditions the edit, and only the geometry it follows moves.
        request.ratio_follow = "<image1>";
        service.generate(request, 1001);
        check(canvas_w == 896U && canvas_h == 672U && planes_at_executor == 2ULL,
              "ratio_follow moves the canvas without dropping a reference");

        // The ledger is the resident LLM's protection, so the sum has to be in the NUMBER and not
        // only in the prompt: one reference of the same two is strictly cheaper.
        OpenAIImagesRequest one_ref = sample_request();
        one_ref.size_requested = false;
        one_ref.images         = {wide};
        service.generate(one_ref, 1002);
        probed_with_one = eyes->probed_target;
        check(probed_with_two > probed_with_one,
              "the second reference is reserved for, not just conditioned on");
    }

    // M7 admission with a reference. The ledger is what protects the resident LLM, so every case here
    // asserts the same thing twice: the refusal code AND that nothing was yielded (probes<=1, runs==0)
    // and nothing was stored.
    {
        std::vector<std::uint8_t> rgba(64U * 48U * 4, 90U);
        for (unsigned i = 0; i < 64U * 48U; ++i) rgba[4 * i + 3] = 255U;
        const std::vector<std::uint8_t> png = infernix::media::encode::encode_png_rgba8(64, 48, rgba);
        const std::string payload = base64_encode_bytes(png);

        // A request that carries a picture but no way to size the ViT encode is refused as
        // unconfigured BEFORE the probe: guessing the vision term is how the resident LLM gets OOMed.
        {
            ImageJobStore store(8, 8ULL << 20);
            auto runner = std::make_unique<FakeRunner>();
            FakeRunner* eyes = runner.get();
            ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                           fixed_dims(), fixed_executor());
            OpenAIImagesRequest request = sample_request();
            request.images = {payload};
            const auto outcome = attempt([&] { return service.generate(request, 1900); });
            check(outcome.threw && outcome.status == 503 &&
                      outcome.code == "image_generation_not_configured",
                  "a reference with no vision component -> 503 image_generation_not_configured");
            check(eyes->probes == 0 && eyes->runs == 0 && store.size() == 0U,
                  "the unsizable reference never reaches the probe or the yield");
        }

        // The same request WITH the lane: the probe is asked for more than the picture-less request
        // asked for (the vision lane plus the structural sequence terms), and a "no" still stops it
        // before a byte is yielded and before any job is stored.
        std::uint64_t asked_without = 0, asked_with = 0;
        {
            ImageJobStore store(8, 8ULL << 20);
            auto runner = std::make_unique<FakeRunner>();
            FakeRunner* eyes = runner.get();
            ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                           fixed_dims(), fixed_executor(), fixed_vision(),
                                           fixed_reference_renderer());
            (void)attempt([&] { return service.generate(sample_request(), 1910); });
            asked_without = eyes->probed_target;
            OpenAIImagesRequest request = sample_request();
            request.images = {payload};
            const auto outcome = attempt([&] { return service.generate(request, 1920); });
            asked_with = eyes->probed_target;
            check(!outcome.threw, "the sized reference request is admitted");
            check(asked_with > asked_without + 100ULL * 1024ULL * 1024ULL,
                  "a reference raises what the probe is asked for (the vision lane is in the number)");
        }
        {
            ImageJobStore store(8, 8ULL << 20);
            auto runner = std::make_unique<FakeRunner>();
            FakeRunner* eyes = runner.get();
            eyes->probe_answer = FakeRunner::Fit{.fits = false, .freed_bytes_expected = 1ULL << 30};
            bool executor_ran = false;
            ImageJobExecutor watch =
                [&executor_ran](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
                                std::span<const std::uint16_t>, std::span<const ReferencePlanes>, void*,
                                const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
                    executor_ran = true;
                    return fake_picture(0x01U);
                };
            ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                           fixed_dims(), watch, fixed_vision(),
                                           fixed_reference_renderer());
            OpenAIImagesRequest request = sample_request();
            request.images = {payload};
            const auto outcome = attempt([&] { return service.generate(request, 1930); });
            check(outcome.threw && outcome.status == 503 &&
                      outcome.code == "image_vram_insufficient",
                  "a reference whose ledger row does not fit -> 503 image_vram_insufficient");
            // The refusal is now REACHED by walking the canvas down to the floor, so more than one ask
            // happened. What still has to hold -- and what this case exists to pin -- is that none of
            // them yielded, ran or stored: a probe is pure arithmetic over the live pools, the yield is
            // what costs the resident model, and the door stayed shut.
            check(eyes->probes > 1 && eyes->runs == 0 && !executor_ran && store.size() == 0U,
                  "a reference the card cannot hold walks the canvas down at the probe alone: it yields "
                  "nothing, runs nothing and stores nothing");
        }
        // A canvas the card cannot hold is no longer the end of an edit. The same pure probe is asked
        // again one 32-px step smaller on the longer side until the plan fits, and the picture is DRAWN
        // at that canvas with both numbers reported. Measured reason: an edit refused at 768x768 on the
        // live engine was 508 MiB short of the gate while a draw of that size really holds about 2 GB,
        // and its own client redrew it at 512x512 one model round later. The server makes that call in
        // the round it already has, on the numbers it would have printed.
        {
            // What the walk has to be judged against is the service's OWN arithmetic, not a byte count
            // written down here: the admission need is image_device_need's certified-conservative sum, and
            // a constant chosen once would silently stop meaning "the card cannot hold 768" the day that
            // sum changes. So the need is READ from two runs against a generous fake -- one at the ask,
            // one at a canvas part-way down -- and the third run refuses everything above the smaller of
            // the two. The walk must then DRAW there instead of answering 503.
            auto need_at = [&](std::uint32_t side) {
                ImageJobStore measure_store(8, 8ULL << 20);
                auto measure_runner = std::make_unique<FakeRunner>(); // generous: every ask fits
                FakeRunner* measure_eyes = measure_runner.get();
                ImageGenerationService measure(std::move(measure_runner), measure_store, fixed_renderer(),
                                               fixed_provider(), fixed_dims(), fixed_executor(),
                                               fixed_vision(), fixed_reference_renderer());
                OpenAIImagesRequest r = sample_request();
                r.images         = {payload};
                r.size_requested = true;
                r.width          = side;
                r.height         = side;
                (void)measure.generate(r, 1900U + side);
                return measure_eyes->probed_target;
            };
            const std::uint32_t small_side   = 512U;
            const std::uint64_t need_asked   = need_at(768U);
            const std::uint64_t need_smaller = need_at(small_side);
            check(need_smaller < need_asked,
                  "the admission need shrinks with the canvas, which is the only reason a walk can work");

            ImageJobStore store(8, 8ULL << 20);
            auto runner = std::make_unique<FakeRunner>();
            FakeRunner* eyes = runner.get();
            const std::uint64_t budget = need_smaller; // admits 512x512, refuses everything above it
            eyes->probe_for = [budget](std::uint64_t target) {
                return FakeRunner::Fit{.fits = target <= budget, .freed_bytes_expected = budget};
            };
            std::uint32_t drew_w = 0;
            std::uint32_t drew_h = 0;
            ImageJobExecutor watch = [&](const OpenAIImagesRequest& request, std::uint64_t,
                                         std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                         std::span<const ReferencePlanes>, void*,
                                         const std::shared_ptr<ImageJobProgress>&,
                                         const EditConditioning*) {
                drew_w = request.width;
                drew_h = request.height;
                FinishedPicture picture; // the size the job was SUBMITTED at, as the real executor returns
                picture.width  = request.width;
                picture.height = request.height;
                picture.rgba.assign(4ULL * request.width * request.height, 0x02U);
                for (std::size_t i = 0; i < picture.rgba.size(); i += 4U) { picture.rgba[i + 3] = 255U; }
                return picture;
            };
            ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                           fixed_dims(), watch, fixed_vision(),
                                           fixed_reference_renderer());
            OpenAIImagesRequest request = sample_request();
            request.images         = {payload};
            request.size_requested = true; // the ask is 768x768, and nothing else decides it
            request.width          = 768;
            request.height         = 768;
            const auto outcome = attempt([&] { return service.generate(request, 1931); });
            check(!outcome.threw, "a canvas the card cannot hold is drawn rather than refused");
            check(eyes->runs == 1 && store.size() == 1U,
                  "the downgraded draw pays exactly the one interlude any other draw pays");
            check(drew_w != 0U && drew_w < 768U && drew_w >= 256U && drew_w == drew_h,
                  "the job is submitted at the largest square the card will take, never below the floor");
            // Walking to the floor would also be "drawn, not refused". What makes it the LARGEST that
            // fits is that the last ask the probe answered is exactly the card's whole budget: one step
            // more and the need is over it, which is the step the walk already took and was refused.
            check(eyes->probed_target == budget,
                  "the walk stopped at the largest canvas the card will take, not short of it");
            const auto data = outcome.response.find("data");
            check(data != outcome.response.end() && data->is_array() && !data->empty() &&
                      data->front().value("size_asked", std::string()) == "768x768" &&
                      data->front().value("size", std::string()) ==
                          std::to_string(drew_w) + "x" + std::to_string(drew_h),
                  "the answer names the canvas asked AND the canvas it drew, so nothing was traded in "
                  "silence");
            check(eyes->probes > 1U, "the size that fits was found by asking the probe again, not guessed");

            // The other half of the walk, and the half the live engine found on its own: probe() is NOT
            // exclusive. A draw already inside its interlude has the frame and KV pools unmapped, so a
            // promise read at that instant can be gigabytes better than what this job's own yield delivers
            // once the other one restores them -- and the net inside the interlude, which compares the
            // DELIVERED bytes against the same gate, refuses. Measured: a 2016x2016 edit walked down to a
            // canvas needing 15540 MiB, was promised enough, delivered 15040 with 855 free and was refused.
            // So stopping at the first promise that merely CLEARS the gate is not enough: the walk keeps
            // going while the promise does not beat it by kVramWalkSlackBytes, and remembers the largest
            // that did clear it so a tight-but-true card still gets the canvas it asked for.
            struct Seen {
                std::uint32_t w = 0U, h = 0U;
                int  probes        = 0;
                bool named_second  = false;
            };
            auto draw_with = [&](std::function<FakeRunner::Fit(std::uint64_t)> answer) {
                ImageJobStore s(8, 8ULL << 20);
                auto r      = std::make_unique<FakeRunner>();
                FakeRunner* eyes2 = r.get();
                eyes2->probe_for  = std::move(answer);
                Seen seen;
                ImageJobExecutor note = [&](const OpenAIImagesRequest& request, std::uint64_t,
                                            std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                            std::span<const ReferencePlanes>, void*,
                                            const std::shared_ptr<ImageJobProgress>&,
                                            const EditConditioning*) {
                    seen.w = request.width;
                    seen.h = request.height;
                    FinishedPicture picture;
                    picture.width  = request.width;
                    picture.height = request.height;
                    picture.rgba.assign(4ULL * request.width * request.height, 0x02U);
                    for (std::size_t i = 0; i < picture.rgba.size(); i += 4U) { picture.rgba[i + 3] = 255U; }
                    return picture;
                };
                ImageGenerationService service(std::move(r), s, fixed_renderer(), fixed_provider(),
                                               fixed_dims(), note, fixed_vision(),
                                               fixed_reference_renderer());
                OpenAIImagesRequest request = sample_request();
                request.images              = {payload};
                request.size_requested      = true;
                request.width               = 768U;
                request.height              = 768U;
                const auto out  = attempt([&] { return service.generate(request, 1932); });
                seen.probes     = eyes2->probes;
                const auto data = out.response.find("data");
                seen.named_second = data != out.response.end() && data->is_array() && !data->empty() &&
                                    data->front().contains("size_asked");
                return seen;
            };
            // The gate's own margin. Named here because the fake answers the promise the real probe would
            // have made; the run's own margin is pinned to this number by the run_margin check above.
            const std::uint64_t margin = 512ULL * 1024ULL * 1024ULL;
            const Seen barely = draw_with([margin](std::uint64_t target) {
                return FakeRunner::Fit{.fits = true, .freed_bytes_expected = target + margin};
            });
            check(barely.w == 768U && barely.h == 768U && barely.probes > 1 && !barely.named_second,
                  "a card that promises EXACTLY what the job needs still gets the canvas it asked for: the "
                  "walk keeps looking for room and falls back to the largest that cleared the gate");
            const Seen roomy_small = draw_with([margin, budget](std::uint64_t target) {
                // Room to spare only once the ask is small enough: that is the walk's cue to stop.
                const std::uint64_t spare = target <= budget ? 8ULL * FakeRunner::kGiB : 0ULL;
                return FakeRunner::Fit{.fits = true, .freed_bytes_expected = target + margin + spare};
            });
            check(roomy_small.w == small_side && roomy_small.h == small_side &&
                      roomy_small.probes > 1 && roomy_small.named_second,
                  "where the promise only equals the need the walk goes on down, and stops at the largest "
                  "canvas the card can both pay for and spare");

            // The refusal the live engine actually printed, and the one no pre-yield walk can prevent:
            // probe() is not exclusive, so the promise was read off a card another draw had yielded. The
            // walk therefore stopped at the ask, and the yield then handed over far less than it promised.
            // What the net inside the interlude decides on is what it was GIVEN -- and the largest canvas
            // those bytes hold is drawn THERE, in the interlude the job already opened. Measured: an edit
            // at 2016x2016 promised 18092 MiB, delivered 14400 plus 479 free, and answered 503 for a
            // picture this machine can paint.
            const std::uint64_t delivered = need_smaller + margin; // exactly what small_side costs
            const auto lier = [margin](std::uint64_t target) {
                return FakeRunner::Fit{.fits = true,
                                       .freed_bytes_expected = target + margin + 8ULL * FakeRunner::kGiB,
                                       .device_free_bytes    = 8ULL * FakeRunner::kGiB};
            };
            {
                ImageJobStore gave_store(8, 8ULL << 20);
                auto        gave_runner = std::make_unique<FakeRunner>();
                FakeRunner* gave        = gave_runner.get();
                gave->probe_for  = lier;
                gave->freed_for  = [delivered](std::uint64_t) { return delivered; };
                gave->device_free = 0ULL; // the other draw restored its pools; the card is not free at all
                std::uint32_t gave_w = 0, gave_h = 0;
                ImageJobExecutor gave_watch = [&](const OpenAIImagesRequest& request, std::uint64_t,
                                                  std::span<const std::uint16_t>,
                                                  std::span<const std::uint16_t>,
                                                  std::span<const ReferencePlanes>, void*,
                                                  const std::shared_ptr<ImageJobProgress>&,
                                                  const EditConditioning*) {
                    gave_w = request.width;
                    gave_h = request.height;
                    FinishedPicture picture;
                    picture.width  = request.width;
                    picture.height = request.height;
                    picture.rgba.assign(4ULL * request.width * request.height, 0x02U);
                    for (std::size_t i = 0; i < picture.rgba.size(); i += 4U) { picture.rgba[i + 3] = 255U; }
                    return picture;
                };
                ImageGenerationService gave_service(std::move(gave_runner), gave_store, fixed_renderer(),
                                                    fixed_provider(), fixed_dims(), gave_watch,
                                                    fixed_vision(), fixed_reference_renderer());
                OpenAIImagesRequest gave_request = sample_request();
                gave_request.images         = {payload};
                gave_request.size_requested = true;
                gave_request.width          = 768U;
                gave_request.height         = 768U;
                const auto gave_out = attempt([&] { return gave_service.generate(gave_request, 1933); });
                check(!gave_out.threw,
                      "a yield that delivers a fraction of what the probe promised is not a refusal");
                check(gave->probes == 1 && gave->runs == 1,
                      "the walk took the promise at its word and the shrinking happened inside the ONE "
                      "interlude the job opened: no second yield, no retry round");
                check(gave_w == small_side && gave_h == small_side,
                      "the picture is drawn at the largest canvas the DELIVERED bytes hold, not at the "
                      "canvas a borrowed promise looked good for");
                const auto gave_data = gave_out.response.find("data");
                check(gave_data != gave_out.response.end() && gave_data->is_array() &&
                          !gave_data->empty() &&
                          gave_data->front().value("size_asked", std::string()) == "768x768" &&
                          gave_data->front().value("size", std::string()) ==
                              std::to_string(gave_w) + "x" + std::to_string(gave_h),
                      "and the answer still names both canvases: a traded picture is never traded in silence");
            }
            {
                // What the walk inside the interlude bottoms out at is still a refusal, and the honest one:
                // it quotes the bytes the yield DELIVERED, because those are the ones the client's canvas
                // was measured against. A refusal that printed the promise would send a reader to fix a
                // number that was never the problem.
                ImageJobStore tight_store(8, 8ULL << 20);
                auto        tight_runner = std::make_unique<FakeRunner>();
                FakeRunner* tight        = tight_runner.get();
                tight->probe_for  = lier;
                tight->freed_for  = [](std::uint64_t) { return 1ULL << 30; }; // 1 GiB: nothing is drawable
                const auto tight_out = attempt([&] {
                    ImageGenerationService tight_service(std::move(tight_runner), tight_store,
                                                         fixed_renderer(), fixed_provider(), fixed_dims(),
                                                         fixed_executor(), fixed_vision(),
                                                         fixed_reference_renderer());
                    OpenAIImagesRequest tight_request = sample_request();
                    tight_request.images         = {payload};
                    tight_request.size_requested = true;
                    tight_request.width          = 768U;
                    tight_request.height         = 768U;
                    return tight_service.generate(tight_request, 1934);
                });
                check(tight_out.threw && tight_out.code == "image_vram_insufficient",
                      "a card that cannot hold even the floor canvas is still refused, not looped on");
                check(tight_out.message.find("can yield 1024 MiB") != std::string::npos,
                      "the refusal names the DELIVERED bytes (1 GiB), not the 24 GiB the probe promised");
            }
        }
    }

    // True CFG: with true_cfg_scale above 1 the negative prompt goes through the same embeds provider
    // and arrives at the executor; without it the executor gets no negative embeds and the (empty)
    // negative prompt never reaches the text tower.
    {
        std::vector<std::string> asked;
        PromptEmbedsProvider recording = [&](const TowerPrompt& rendered) {
            asked.push_back(rendered.rendered);
            return std::vector<std::uint16_t>(8U, 7U);
        };
        std::size_t neg_size = 0;
        double      seen_scale = 0.0;
        ImageJobExecutor capture = [&](const OpenAIImagesRequest& request, std::uint64_t,
                                       std::span<const std::uint16_t>,
                                       std::span<const std::uint16_t> negative,
                                       std::span<const ReferencePlanes>, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
            -> FinishedPicture {
            neg_size   = negative.size();
            seen_scale = request.true_cfg_scale;
            return fake_picture(0x01U);
        };

        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), recording, fixed_dims(), capture);
        OpenAIImagesRequest steered     = sample_request();
        steered.true_cfg_scale          = 3.5;
        steered.negative_prompt         = "blurry, low quality";
        service.generate(steered, 1000);
        check(neg_size == 8U && seen_scale == 3.5, "true_cfg_scale 3.5 -> negative embeds at executor");
        // Both prompts go through the tower exactly once; which one first is not part of the contract.
        const bool asked_prompt    = std::find(asked.begin(), asked.end(), "a lighthouse at dusk") !=
                                     asked.end();
        const bool asked_negative =
            std::find(asked.begin(), asked.end(), "blurry, low quality") != asked.end();
        check(asked.size() == 2U && asked_prompt && asked_negative,
              "both prompts pass the embeds provider exactly once");

        asked.clear();
        neg_size = 1;
        service.generate(sample_request(), 1000);
        check(neg_size == 0U && asked.size() == 1U,
              "true_cfg_scale 1 -> no negative embeds and no empty-prompt embeds");
    }

    {
        // Every generated image is also collected in an `images` directory under the working directory,
        // named after the stored job id (img-<created>-<sequence>), which is the served picture itself.
        PromptEmbedsProvider embeds = [](const TowerPrompt&) { return std::vector<std::uint16_t>(4U, 9U); };
        const FinishedPicture picture = fake_picture(0x89U);
        ImageJobExecutor writer = [&](const OpenAIImagesRequest&, std::uint64_t,
                                      std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                      std::span<const ReferencePlanes>, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
            -> FinishedPicture {
            return picture;
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), embeds, fixed_dims(), writer);

        const std::filesystem::path saved =
            std::filesystem::current_path() / "images" / "img-2000-1.png";
        std::filesystem::remove(saved);
        service.generate(sample_request(), 2000);

        std::vector<std::uint8_t> on_disk;
        if (std::filesystem::exists(saved)) {
            std::ifstream input(saved, std::ios::binary);
            on_disk.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        check(on_disk == expected_png(picture),
              "generated image is written to images/<job id>.png in the cwd");
        std::filesystem::remove(saved);
    }

    // response_format url: the response carries an address on this process, and the id inside it is the
    // collection file on disk, so the delivered link resolves without the response payload.
    {
        const FinishedPicture picture = fake_picture(0x0DU);
        ImageJobExecutor writer = [&](const OpenAIImagesRequest&, std::uint64_t,
                                      std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                      std::span<const ReferencePlanes>, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
            -> FinishedPicture {
            return picture;
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(), writer);

        OpenAIImagesRequest request = sample_request();
        request.response_format     = ImageResponseFormat::Url;
        const auto response = service.generate(request, 3000, "http://127.0.0.1:8080/v1/images/");

        const std::string base = "http://127.0.0.1:8080/v1/images/";
        const std::string url  = response["data"][0]["url"].get<std::string>();
        check(url == base + "img-3000-1", "url delivery is the base plus the job id");
        check(!response["data"][0].contains("b64_json"), "url delivery carries no inline payload");

        const std::filesystem::path file = generated_image_path(url.substr(base.size()));
        check(file.filename().string() == "img-3000-1.png",
              "the id in the delivered url is the collection name");
        std::vector<std::uint8_t> on_disk;
        if (std::filesystem::exists(file)) {
            std::ifstream input(file, std::ios::binary);
            on_disk.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        check(on_disk == expected_png(picture), "the delivered id resolves to the picture on disk");
        check(generated_image_path("generations").empty() &&
                  generated_image_path("img-3000-1/../x").empty() &&
                  generated_image_path("").empty(),
              "an id that is not an artifact id names no file at all");

        const auto record = store.size() == 1U ? store.get("img-3000-1") : nullptr;
        check(record != nullptr && record->state == ImageJobState::Succeeded &&
                  record->response["data"][0]["url"].get<std::string>() == url,
              "url delivery stores the response the client actually received");
        std::filesystem::remove(file);
    }

    // The job's live progress: the executor advances the shared state the record carries, the
    // service names the newest job's state for the streaming handlers' ticker, and a watchdog-
    // flagged job fails terminally with the pipeline's stall message.
    {
        ImageJobExecutor stepping = [](const OpenAIImagesRequest&, std::uint64_t,
                                       std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                       std::span<const ReferencePlanes>, void*,
                                       const std::shared_ptr<ImageJobProgress>& progress, const EditConditioning*)
            -> FinishedPicture {
            progress->advance(24);
            progress->advance(97);
            return fake_picture(0x01U);
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_renderer(), fixed_provider(), fixed_dims(),
                                       stepping);
        (void)service.generate(sample_request(), 4000);
        const auto live = service.last_job_progress();
        check(live != nullptr && live->percent.load() == 100,
              "the service does not name the finished job's progress at 100");
        const auto record = store.get("img-4000-1");
        check(record != nullptr && record->progress == live && live->percent.load() == 100,
              "the stored record does not share the job's progress state");

        ImageJobExecutor stalled = [](const OpenAIImagesRequest&, std::uint64_t,
                                      std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                      std::span<const ReferencePlanes>, void*,
                                      const std::shared_ptr<ImageJobProgress>& progress, const EditConditioning*)
            -> FinishedPicture {
            progress->stalled = true;
            if (!progress->advance(50)) {
                throw std::runtime_error("image job: stalled, flagged by the serving watchdog");
            }
            return fake_picture(0x01U);
        };
        ImageJobStore stalled_store(8, 1 << 20);
        ImageGenerationService stalled_service(std::make_unique<FakeRunner>(), stalled_store,
                                               fixed_renderer(), fixed_provider(), fixed_dims(), stalled);
        bool threw = false;
        try {
            (void)stalled_service.generate(sample_request(), 5000);
        } catch (const ApiException& error) {
            threw = error.error().code == "image_generation_failed";
        }
        const auto failed = stalled_store.get("img-5000-1");
        check(threw && failed != nullptr && failed->state == ImageJobState::Failed &&
                  failed->error["message"].get<std::string>().find("stalled") != std::string::npos,
              "a watchdog-flagged job does not fail terminally with its stall message");

        // The status object the GET route serves for stored jobs (the route's own wiring is the
        // same call; this is its single authority).
        const auto succeeded_status = service.job_status_json("img-4000-1");
        check(succeeded_status != nullptr &&
                  succeeded_status["status"].get<std::string>() == "succeeded" &&
                  succeeded_status["progress_percent"].get<int>() == 100,
              "a succeeded job's status object is not its stored terminal state at 100");
        const auto failed_status = stalled_service.job_status_json("img-5000-1");
        check(failed_status != nullptr && failed_status["status"].get<std::string>() == "failed",
              "a failed job's status object lost its state");
        check(service.job_status_json("img-9999-9").is_null(),
              "an unknown id named a status object (the route must still 404)");
        const auto stall_percent = failed->progress->percent.load();
        check(stall_percent == 50, "the failed job's stored percent is not its last step");
        const auto failed_status_percent = failed_status["progress_percent"].get<int>();
        check(failed_status_percent == stall_percent,
              "the status object's percent does not read the job's live state");
    }

    // Generated-image retention. The sweep is the only thing that ever removes a picture, so what it
    // keeps and what it deletes is the promise a delivered URL rests on: a file older than the
    // retention whose job has ended is gone, a fresh one and one an unfinished job still owns are not,
    // and a file this route never drew is not the route's to touch.
    {
        ImageJobStore store(8, 1 << 20);
        const std::filesystem::path directory = std::filesystem::current_path() / "images";
        std::filesystem::create_directories(directory);
        const std::chrono::seconds retention{14 * 24 * 60 * 60};
        const auto now        = std::chrono::system_clock::now();

        auto draw = [&](const std::string& id, std::chrono::seconds old) {
            const std::filesystem::path file = generated_image_path(id);
            {
                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out << "picture-bytes";
            }
            std::filesystem::last_write_time(file, std::chrono::file_clock::now() - old);
        };
        auto succeed = [&](const std::string& id, std::int64_t stamp) {
            StoredImageJob job;
            job.id = id;
            store.put(job);
            (void)store.advance(id, {ImageJobState::WaitingForFrontier, stamp});
            (void)store.advance(id, {ImageJobState::Running, stamp});
            (void)store.advance(id, {ImageJobState::Restoring, stamp});
            (void)store.advance(id, {ImageJobState::Succeeded, stamp});
        };

        draw("img-10-1", std::chrono::days(20));
        succeed("img-10-1", 10);
        draw("img-11-1", std::chrono::days(1));
        draw("img-12-1", std::chrono::days(20));   // terminal record missing: aged out long ago
        draw("img-13-1", std::chrono::days(20));
        {
            StoredImageJob running;
            running.id = "img-13-1";
            store.put(running);
            (void)store.advance("img-13-1", {ImageJobState::WaitingForFrontier, 1});
            (void)store.advance("img-13-1", {ImageJobState::Running, 1});
        }
        // Two things the sweep must not decide about: a name that is not an artifact id, and a file
        // that merely shares the directory.
        { std::ofstream other(directory / "notes.txt", std::ios::trunc); other << "not a picture"; }
        { std::ofstream shape(directory / "img-not-an-id.png", std::ios::binary | std::ios::trunc);
          shape << "wrong shape"; }
        std::filesystem::last_write_time(directory / "notes.txt",
                                         std::chrono::file_clock::now() - std::chrono::days(60));
        std::filesystem::last_write_time(directory / "img-not-an-id.png",
                                         std::chrono::file_clock::now() - std::chrono::days(60));

        const ImageRetentionResult first = sweep_generated_images(store, retention, now);
        check(first.files == 2 && first.bytes == 2 * 13,
              "the sweep took exactly the two pictures past their retention");
        check(!std::filesystem::exists(generated_image_path("img-10-1")),
              "a terminal job older than its retention lost its file");
        check(store.get("img-10-1") == nullptr && store.get("img-13-1") != nullptr,
              "a swept picture loses its record too, so its URL answers 404 instead of a status "
              "object forever, while the job still protecting its picture keeps both");
        check(std::filesystem::exists(generated_image_path("img-11-1")),
              "a picture inside its retention keeps its file, so its URL still answers");
        check(!std::filesystem::exists(generated_image_path("img-12-1")),
              "a picture no job record names any more is swept once past its retention");
        check(std::filesystem::exists(generated_image_path("img-13-1")),
              "a picture a job that has not ended still owns is never swept out from under it");
        check(std::filesystem::exists(directory / "notes.txt") &&
                  std::filesystem::exists(directory / "img-not-an-id.png"),
              "files this route never drew stay exactly where they are, id-shaped name or not");

        // Retention off: the server that asked to keep everything keeps everything.
        const ImageRetentionResult off = sweep_generated_images(store, std::chrono::seconds::zero(),
                                                                now + std::chrono::days(365));
        check(off.files == 0 && std::filesystem::exists(generated_image_path("img-11-1")),
              "retention 0 removes nothing, however old the pictures are");

        // A second pass over what survives finds nothing left to do (no double counting, no churn).
        check(sweep_generated_images(store, retention, now).files == 0,
              "a second sweep of the same directory removes nothing more");

        // And no directory at all -- the state before this server drew its first picture -- is a
        // no-op, not a throw. Everything the earlier checks left in the collection is emptied out
        // first, since that is exactly what the end of this test does to it anyway.
        std::error_code unused;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) { std::filesystem::remove(entry.path(), unused); }
        }
        std::filesystem::remove(directory, unused);
        check(!std::filesystem::exists(directory) &&
                  sweep_generated_images(store, retention, now).files == 0,
              "a missing images directory is an ordinary empty sweep");
    }

    // The joint sequence is admitted on the host, before the probe: a request whose rendered prompt and
    // canvas cannot be assembled is a 400 that names its own numbers, not a job that yields the card and
    // then dies inside the attention Op with a 500. 2048x2048 is 16384 target tokens, so 4096 prompt
    // rows sit exactly on the 20480 ceiling (admitted) and 4097 is the first refusal.
    {
        const auto rows_renderer = [](std::size_t rows) {
            return [rows](const std::string&) {
                TowerPrompt rendered;
                rendered.ids       = std::vector<int>(rows, 1);
                rendered.drop_rows = 0;
                return rendered;
            };
        };
        const auto big_canvas = [] {
            OpenAIImagesRequest request = sample_request();
            request.width  = 2048U;
            request.height = 2048U;
            return request;
        };
        ImageJobStore store(8, 1 << 20);
        auto at_ceiling = std::make_unique<FakeRunner>();
        FakeRunner& ceiling_runner = *at_ceiling;
        ImageGenerationService at_limit(std::move(at_ceiling), store, rows_renderer(4096),
                                        fixed_provider(), fixed_dims(), fixed_executor());
        check(!attempt([&] { return at_limit.generate(big_canvas(), 1000); }).threw,
              "a joint sequence of exactly the ceiling is served");
        check(ceiling_runner.probes == 1, "the admitted request reached the probe");

        ImageJobStore refused_store(8, 1 << 20);
        auto over = std::make_unique<FakeRunner>();
        FakeRunner& over_runner = *over;
        ImageGenerationService past_limit(std::move(over), refused_store, rows_renderer(4097),
                                          fixed_provider(), fixed_dims(), fixed_executor());
        const auto outcome = attempt([&] { return past_limit.generate(big_canvas(), 1000); });
        check(outcome.threw && outcome.status == 400 && outcome.code == "image_prompt_too_long",
              "one row past the ceiling is a 400 image_prompt_too_long");
        check(over_runner.probes == 0 && refused_store.size() == 0U,
              "the refusal never probes the device and stores no job");

        // Naming the numbers is not enough to act on. Measured on the deployed engine: an edit asked
        // for 2048x2048, was refused five rounds running with a message quoting both counts and no
        // size, and only on the sixth did the model settle on 1536 by itself. The ceiling less this
        // request's fixed 4097 rows leaves 16383 target tokens -- one short of the 16384 a 2048x2048
        // costs -- so the honest answer is the asked canvas with its longer side stepped down one 32
        // grid step, and the message has to say that instead of "a smaller size".
        check(outcome.message.find("2016x2048 fits") != std::string::npos,
              "the refusal names the largest canvas that fits this prompt");
        check(outcome.message.find("2048x2048") == std::string::npos,
              "and never offers back the canvas it just refused");

        // When the rows alone overflow the ceiling, no canvas helps and "a smaller size" is a lie that
        // costs the caller another round. 20477 rows leave 3 tokens for the target, and the smallest
        // canvas there is -- 32x32 -- costs 4.
        ImageJobStore hopeless_store(8, 1 << 20);
        ImageGenerationService hopeless(std::make_unique<FakeRunner>(), hopeless_store,
                                        rows_renderer(20477), fixed_provider(), fixed_dims(),
                                        fixed_executor());
        const auto hopeless_outcome = attempt([&] { return hopeless.generate(big_canvas(), 1000); });
        check(hopeless_outcome.threw && hopeless_outcome.code == "image_prompt_too_long" &&
                  hopeless_outcome.message.find("no canvas fits") != std::string::npos &&
                  hopeless_outcome.message.find("only a shorter prompt works") != std::string::npos,
              "a prompt no canvas can carry is refused as one only shortening fixes");
    }

    // The four-channel composition the reference does: alpha is the VAE's fourth decoded plane, not a
    // constant 255. The tie case (a channel of exactly 0 -> 127.5) is in here because it is the only
    // value where the rounding rule could be seen to differ from numpy's.
    {
        constexpr std::size_t kPixels = 6ULL;
        std::vector<float> channels(4ULL * kPixels);
        for (std::size_t i = 0; i < kPixels; ++i) {
            channels[0 * kPixels + i] = -1.0F; // red   -> 0
            channels[1 * kPixels + i] =  1.0F; // green -> 255
            channels[2 * kPixels + i] =  0.0F; // blue  -> 128
            channels[3 * kPixels + i] = -0.5F; // alpha -> 64
        }
        const std::vector<std::uint8_t> rgba = rgba_from_generated_channels(channels, 3U, 2U);
        bool pixels_ok = rgba.size() == 4ULL * kPixels;
        for (std::size_t i = 0; i < kPixels && pixels_ok; ++i) {
            pixels_ok = rgba[4 * i + 0] == 0U && rgba[4 * i + 1] == 255U &&
                        rgba[4 * i + 2] == 128U && rgba[4 * i + 3] == 64U;
        }
        check(pixels_ok, "all four planes map through the reference's denormalize-round, alpha included");
        check(rgba_from_generated_channels(std::vector<float>(4ULL * 4, 1.0F), 2U, 2U) ==
                  std::vector<std::uint8_t>(4ULL * 4, 255U),
              "an opaque decode still comes back opaque");
        bool refused = false;
        try {
            (void)rgba_from_generated_channels(std::vector<float>(3ULL * 4, 0.0F), 2U, 2U);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        check(refused, "a plane that is not the VAE's four channels is refused, not read past its end");
    }

    // Every check that reached a job wrote its picture into the working directory; leave nothing behind.
    if (const std::filesystem::path directory = std::filesystem::current_path() / "images";
        std::filesystem::is_directory(directory)) {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            const std::string name = entry.path().filename().string();
            if (entry.is_regular_file() && name.rfind("img-", 0) == 0 && entry.path().extension() == ".png") {
                std::filesystem::remove(entry.path());
            }
        }
        std::error_code unused;
        std::filesystem::remove(directory, unused);
    }

    std::cout << (failures == 0 ? "IMAGE_GENERATION_SERVICE_TEST: PASS"
                                : "IMAGE_GENERATION_SERVICE_TEST: FAIL")
              << " (" << (failures == 0 ? "all" : std::to_string(failures) + " failed") << ")\n";
    return failures == 0 ? 0 : 1;
}
