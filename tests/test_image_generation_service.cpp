// ImageGenerationService host contract: the synchronous generate() maps every interlude/job
// outcome onto the documented ApiException surface and one legal ImageJobStore terminal state.
// IMAGE_GENERATION_SERVICE_TEST: PASS
#include "serve/image_generation_service.h"

#include "media/encode/png_encode.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    std::cout << (ok ? "ok: " : "FAIL: ") << what << '\n';
    if (!ok) { ++failures; }
}

class FakeRunner final : public ImageInterludeRunner {
public:
    enum class Mode { Run, InFlight, DrainFail, RestorationFault };

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
        return probe_answer;
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
        if (mode_ == Mode::DrainFail) {
            throw std::runtime_error("image interlude: the LLM frontier did not drain in time");
        }
        job(infernix::ImageInterludeContext{.freed_bytes = freed, .image_stream = nullptr});
        if (mode_ == Mode::RestorationFault) {
            available = false;
            throw std::runtime_error("image interlude: pool restoration was not confirmed");
        }
        return freed;
    }

    bool supports = true;
    bool available = true;
    // Generous by default so the guard every existing case walks through stays satisfied: the
    // admission compares these against its own computed need plus the margin.
    static constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t freed      = 24ULL * kGiB;
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
              std::span<const std::uint16_t>, const ReferencePixels*, void*,
              const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
        return fake_picture(0x01U);
    };
}

ImageJobExecutor throwing_executor() {
    return [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
              std::span<const std::uint16_t>, const ReferencePixels*,
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
                            std::span<const std::uint16_t>, const ReferencePixels*, void*,
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
                                       const ReferencePixels* ref, void*,
                                       const std::shared_ptr<ImageJobProgress>&,
                                       const EditConditioning* edit)
            -> FinishedPicture {
            saw_reference = ref != nullptr;
            if (edit != nullptr) {
                saw_edit_planes = edit->planes != nullptr && edit->prompt != nullptr &&
                                  edit->negative_prompt == nullptr; // true_cfg_scale is 1 here
                if (edit->planes != nullptr) {
                    vision_rgb_bytes = edit->planes->vision.rgb.size();
                    vision_w         = edit->planes->vision.width_px;
                    vision_h         = edit->planes->vision.height_px;
                }
                if (edit->prompt != nullptr) { edit_rows = edit->prompt->tokens(); }
            }
            if (ref != nullptr) {
                plane_floats = ref->pixels.size();
                plane_w      = ref->width_px;
                plane_h      = ref->height_px;
                const std::size_t pixels = static_cast<std::size_t>(ref->width_px) * ref->height_px;
                const float* alpha       = ref->pixels.data() + 3 * pixels;
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
        request.image               = payload;
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
        bad.image               = "not base64 at all!!";
        const auto outcome      = attempt([&] { return bad_service.generate(bad, 1000); });
        // The codec's own words are never echoed to the client, so the CODE is the contract here: a
        // payload that is not base64 PNG/JPEG is image_decode_failed, not the generic image_invalid and
        // not a 500 (nothing about this request is the server's fault).
        check(outcome.threw && outcome.status == 400 && outcome.code == "image_decode_failed",
              "undecodable image -> 400 image_decode_failed");
        check(bad_store.size() == 0U, "invalid image stores no job (fails before the interlude)");
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
            request.image = payload;
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
            request.image = payload;
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
                                std::span<const std::uint16_t>, const ReferencePixels*, void*,
                                const std::shared_ptr<ImageJobProgress>&, const EditConditioning*) {
                    executor_ran = true;
                    return fake_picture(0x01U);
                };
            ImageGenerationService service(std::move(runner), store, fixed_renderer(), fixed_provider(),
                                           fixed_dims(), watch, fixed_vision(),
                                           fixed_reference_renderer());
            OpenAIImagesRequest request = sample_request();
            request.image = payload;
            const auto outcome = attempt([&] { return service.generate(request, 1930); });
            check(outcome.threw && outcome.status == 503 &&
                      outcome.code == "image_vram_insufficient",
                  "a reference whose ledger row does not fit -> 503 image_vram_insufficient");
            check(eyes->probes == 1 && eyes->runs == 0 && !executor_ran && store.size() == 0U,
                  "the refused reference yields nothing, runs nothing and stores nothing");
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
                                       const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
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
                                      const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
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
                                      const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&, const EditConditioning*)
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
                                       const ReferencePixels*, void*,
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
                                      const ReferencePixels*, void*,
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
