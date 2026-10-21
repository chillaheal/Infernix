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

    std::uint64_t
    run(std::uint64_t, std::uint64_t,
        const std::function<bool(const infernix::ImageInterludeContext&)>& job) override {
        if (mode_ == Mode::InFlight) {
            throw std::runtime_error("image interlude: another interlude is already in flight");
        }
        if (mode_ == Mode::DrainFail) {
            throw std::runtime_error("image interlude: the LLM frontier did not drain in time");
        }
        job(infernix::ImageInterludeContext{.freed_bytes = 8ULL << 30, .image_stream = nullptr});
        if (mode_ == Mode::RestorationFault) {
            available = false;
            throw std::runtime_error("image interlude: pool restoration was not confirmed");
        }
        return 8ULL << 30;
    }

    bool supports = true;
    bool available = true;

private:
    Mode mode_;
};

PromptEmbedsProvider fixed_provider() {
    return [](const std::string&) { return std::vector<std::uint16_t>(4U, 0U); };
}

ImageJobExecutor fixed_executor() {
    return [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
              std::span<const std::uint16_t>, const ReferencePixels*, void*,
              const std::shared_ptr<ImageJobProgress>&) {
        return std::vector<std::uint8_t>{0x01U, 0x02U};
    };
}

ImageJobExecutor throwing_executor() {
    return [](const OpenAIImagesRequest&, std::uint64_t, std::span<const std::uint16_t>,
              std::span<const std::uint16_t>, const ReferencePixels*,
              void*, const std::shared_ptr<ImageJobProgress>&) -> std::vector<std::uint8_t> {
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
        ImageGenerationService service(nullptr, store, PromptEmbedsProvider{}, ImageJobExecutor{});
        check(!service.is_configured(), "empty service is not configured");
        expect_precheck("not configured", 503, "image_generation_not_configured", service);
        check(store.size() == 0U, "not-configured stores nothing");
    }

    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_provider(),
                                       fixed_executor());
        check(service.is_configured(), "full service is configured");
        const auto response = service.generate(sample_request(), 1000);
        check(response["created"] == 1000, "success stamps created");
        check(response["data"].size() == 1U && response["data"][0]["b64_json"] == "AQI=",
              "success returns the base64 PNG");
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
        ImageGenerationService service(std::move(runner), store, fixed_provider(), fixed_executor());
        expect_precheck("interlude unsupported", 503, "image_interlude_unsupported", service);
        check(store.size() == 0U, "unsupported stores nothing");
    }

    {
        auto runner = std::make_unique<FakeRunner>();
        runner->available = false;
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::move(runner), store, fixed_provider(), fixed_executor());
        expect_precheck("engine unavailable", 503, "engine_unavailable", service);
        check(store.size() == 0U, "unavailable stores nothing");
    }

    // The job itself fails while the interlude restores cleanly.
    {
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_provider(),
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
                                       store, fixed_provider(), fixed_executor());
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
        ImageGenerationService service(std::make_unique<FakeRunner>(FakeRunner::Mode::DrainFail),
                                       store, fixed_provider(), fixed_executor());
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
            fixed_provider(), fixed_executor());
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
        ImageJobExecutor capture = [&](const OpenAIImagesRequest&, std::uint64_t,
                                       std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                       const ReferencePixels* ref, void*,
                                       const std::shared_ptr<ImageJobProgress>&)
            -> std::vector<std::uint8_t> {
            saw_reference = ref != nullptr;
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
            return std::vector<std::uint8_t>{0x01U};
        };

        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_provider(),
                                       capture);
        OpenAIImagesRequest request = sample_request();
        request.image               = payload;
        service.generate(request, 1000);
        check(saw_reference && plane_w == 1184U && plane_h == 896U &&
                  plane_floats == 4ULL * 1184ULL * 896ULL,
              "valid PNG image -> 1184x896 RGBA plane at the executor");
        check(alpha_min < -0.5F && alpha_max > 0.5F, "reference alpha carried raw, not composited");

        ImageJobStore bad_store(8, 1 << 20);
        ImageGenerationService bad_service(std::make_unique<FakeRunner>(), bad_store,
                                           fixed_provider(), capture);
        OpenAIImagesRequest bad = sample_request();
        bad.image               = "not base64 at all!!";
        const auto outcome      = attempt([&] { return bad_service.generate(bad, 1000); });
        check(outcome.threw && outcome.status == 400 && outcome.code == "image_invalid",
              "undecodable image -> 400 image_invalid");
        check(bad_store.size() == 0U, "invalid image stores no job (fails before the interlude)");
    }

    // True CFG: with true_cfg_scale above 1 the negative prompt goes through the same embeds provider
    // and arrives at the executor; without it the executor gets no negative embeds and the (empty)
    // negative prompt never reaches the text tower.
    {
        std::vector<std::string> asked;
        PromptEmbedsProvider recording = [&](const std::string& prompt) {
            asked.push_back(prompt);
            return std::vector<std::uint16_t>(8U, 7U);
        };
        std::size_t neg_size = 0;
        double      seen_scale = 0.0;
        ImageJobExecutor capture = [&](const OpenAIImagesRequest& request, std::uint64_t,
                                       std::span<const std::uint16_t>,
                                       std::span<const std::uint16_t> negative,
                                       const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&)
            -> std::vector<std::uint8_t> {
            neg_size   = negative.size();
            seen_scale = request.true_cfg_scale;
            return std::vector<std::uint8_t>{0x01U};
        };

        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, recording, capture);
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
        PromptEmbedsProvider embeds = [](const std::string&) { return std::vector<std::uint16_t>(4U, 9U); };
        const std::vector<std::uint8_t> picture{0x89U, 'P', 'N', 'G', 0x0AU};
        ImageJobExecutor writer = [&](const OpenAIImagesRequest&, std::uint64_t,
                                      std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                      const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&)
            -> std::vector<std::uint8_t> {
            return picture;
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, embeds, writer);

        const std::filesystem::path saved =
            std::filesystem::current_path() / "images" / "img-2000-1.png";
        std::filesystem::remove(saved);
        service.generate(sample_request(), 2000);

        std::vector<std::uint8_t> on_disk;
        if (std::filesystem::exists(saved)) {
            std::ifstream input(saved, std::ios::binary);
            on_disk.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        check(on_disk == picture, "generated image is written to images/<job id>.png in the cwd");
        std::filesystem::remove(saved);
    }

    // response_format url: the response carries an address on this process, and the id inside it is the
    // collection file on disk, so the delivered link resolves without the response payload.
    {
        const std::vector<std::uint8_t> picture{0x89U, 'P', 'N', 'G', 0x0DU};
        ImageJobExecutor writer = [&](const OpenAIImagesRequest&, std::uint64_t,
                                      std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                                      const ReferencePixels*, void*, const std::shared_ptr<ImageJobProgress>&)
            -> std::vector<std::uint8_t> {
            return picture;
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_provider(), writer);

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
        check(on_disk == picture, "the delivered id resolves to the picture on disk");
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
                                       const std::shared_ptr<ImageJobProgress>& progress)
            -> std::vector<std::uint8_t> {
            progress->advance(24);
            progress->advance(97);
            return std::vector<std::uint8_t>{0x01U};
        };
        ImageJobStore store(8, 1 << 20);
        ImageGenerationService service(std::make_unique<FakeRunner>(), store, fixed_provider(),
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
                                      const std::shared_ptr<ImageJobProgress>& progress)
            -> std::vector<std::uint8_t> {
            progress->stalled = true;
            if (!progress->advance(50)) {
                throw std::runtime_error("image job: stalled, flagged by the serving watchdog");
            }
            return std::vector<std::uint8_t>{0x01U};
        };
        ImageJobStore stalled_store(8, 1 << 20);
        ImageGenerationService stalled_service(std::make_unique<FakeRunner>(), stalled_store,
                                               fixed_provider(), stalled);
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
