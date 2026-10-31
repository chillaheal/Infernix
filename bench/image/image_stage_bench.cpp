// Bundle B fas 0: the production image route's stage baseline.
//
// One process, one Engine, one image artifact, the real ImageGenerationService route (the same
// seams serve wires in apps/serve/main.cpp: the Engine interlude runner, the tower prompt renderer
// and embeds provider, the artifact DiT dims provider and the artifact job executor). Two numbers are
// printed per row, always together, because bundle B is judged on both:
//
//   total_ms  the whole service call: probe, prompt render, tower embeds, the interlude with the job,
//             PNG/base64 and the stored record.
//   pause_ms  the interlude itself (the runner's run() wall): the window the LLM is paused and the
//             submit door closed.
//   device_ms the job executor's wall inside that window. Since fas 1 the executor stops at the
//             finished pixels, so this is DiT loop + VAE only; the PNG/base64/disk work is counted in
//             total_ms and no longer in pause_ms.
//
// Inside the job the pipeline reports its own stage lines (ledger, driver free, pool reserved, and the
// block-upload share of the loop wall) through the logger handed to the executor; they land on stderr
// with timestamps, which is what attributes the numbers above.
//
// Sizes: the ledger cert's matrix (512, 768, 1024) plus a reference and a true-CFG row, each run cold
// (first use of that size, weights lazy-bound on the first job) and warm. The reference is a picture
// this same run just produced, read back off disk and re-encoded -- the route's own `image` field, no
// fixture files.
//
// Skips (exit 77) without INFERNIX_QWEN4_ARTIFACT, INFERNIX_TEST_ARTIFACT or
// INFERNIX_IMAGE_TEXT_PROCESSOR; returns 2 with no CUDA device. Run on a GPU with nothing else on it:
// the server must be stopped (the pause number is meaningless beside a live LLM).
//
//   INFERNIX_QWEN4_ARTIFACT=part1.infernix [INFERNIX_QWEN4_NGRAM=..ngram] \
//   INFERNIX_TEST_ARTIFACT=Qwen-Image-2.1-FP8-v2.infernix \
//   INFERNIX_IMAGE_TEXT_PROCESSOR=<Qwen3-VL processor dir> \
//   [INFERNIX_BENCH_MAX_CONTEXT=65536] [INFERNIX_BENCH_EXPERT_RAM_GIB=8] \
//   infernix_image_stage_bench

#include "serve/image_generation_service.h"
#include "serve/image_job_progress.h"
#include "serve/image_text_tower.h"
#include "serve/openai_images.h"
#include "product/logging/logging.h"
#include "serve/reference_image.h"

#include <infernix/engine.h>


#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace std::chrono_literals;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

const char* env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : fallback;
}

// The interlude window, measured where it happens: the production runner, timed around run().
class TimingRunner final : public infernix::serve::ImageInterludeRunner {
public:
    explicit TimingRunner(std::unique_ptr<ImageInterludeRunner> inner) : inner_(std::move(inner)) {}
    [[nodiscard]] bool supports_interlude() const noexcept override { return inner_->supports_interlude(); }
    [[nodiscard]] bool is_available() const noexcept override { return inner_->is_available(); }
    [[nodiscard]] Fit probe(std::uint64_t target_bytes, std::uint64_t margin_bytes) override {
        const auto t0 = std::chrono::steady_clock::now();
        Fit fit       = inner_->probe(target_bytes, margin_bytes);
        probe_ms      = ms_since(t0);
        ++probes;
        return fit;
    }
    std::uint64_t run(std::uint64_t target_bytes, std::uint64_t margin_bytes,
                      const std::function<bool(const infernix::ImageInterludeContext&)>& job) override {
        const auto t0 = std::chrono::steady_clock::now();
        const std::uint64_t freed = inner_->run(target_bytes, margin_bytes, job);
        pause_ms += ms_since(t0);
        return freed;
    }

    double pause_ms = 0.0;  // the LLM's paused window across the measured rows
    double probe_ms = 0.0;
    std::size_t probes = 0;

private:
    std::unique_ptr<ImageInterludeRunner> inner_;
};

// The device job alone, measured inside the interlude: the encode, the base64 and the file are the
// service's work after it, so they land in total_ms and not in pause_ms.
class TimingExecutor {
public:
    TimingExecutor(infernix::serve::ImageJobExecutor inner, TimingRunner* runner)
        : inner_(std::move(inner)), runner_(runner) {}

    infernix::serve::FinishedPicture operator()(const infernix::serve::OpenAIImagesRequest& request,
                                               std::uint64_t seed,
                                               std::span<const std::uint16_t> prompt_embeds,
                                               std::span<const std::uint16_t> negative_embeds,
                                               const infernix::serve::ReferencePixels* reference,
                                               void* image_stream,
                                               const std::shared_ptr<infernix::serve::ImageJobProgress>& progress,
                                               const infernix::serve::EditConditioning* edit) {
        const auto t0 = std::chrono::steady_clock::now();
        auto picture  = inner_(request, seed, prompt_embeds, negative_embeds, reference, image_stream,
                               progress, edit);
        last_device_ms = ms_since(t0);
        return picture;
    }

    double last_device_ms = 0.0;

private:
    infernix::serve::ImageJobExecutor inner_;
    TimingRunner* runner_;
};

std::string base64_encode(const std::vector<std::uint8_t>& bytes) {
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t chunk = static_cast<std::uint32_t>(bytes[i]) << 16 |
                                    (i + 1 < bytes.size() ? static_cast<std::uint32_t>(bytes[i + 1]) << 8 : 0) |
                                    (i + 2 < bytes.size() ? static_cast<std::uint8_t>(bytes[i + 2]) : 0);
        out.push_back(table[(chunk >> 18) & 0x3F]);
        out.push_back(table[(chunk >> 12) & 0x3F]);
        out.push_back(i + 1 < bytes.size() ? table[(chunk >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < bytes.size() ? table[chunk & 0x3F] : '=');
    }
    return out;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The id the route just stored, from its own url (".../v1/images/img-...").
std::string id_from_url(const std::string& url) {
    const auto slash = url.find_last_of('/');
    return slash == std::string::npos ? std::string() : url.substr(slash + 1);
}

struct Row {
    std::uint32_t size;
    const char* kind;
    bool reference = false;
    bool cfg       = false;
};

void run_row(infernix::serve::ImageGenerationService& images, TimingRunner& runner,
             TimingExecutor& executor, const Row& row, std::string& reference_b64,
             const std::string& image_model) {
    infernix::serve::OpenAIImagesRequest request;
    request.model            = image_model;
    request.prompt           = "a red fox standing on a green hill at sunset, painted";
    request.width = request.height = row.size;
    request.steps                  = 40;
    request.response_format        = infernix::serve::ImageResponseFormat::Url;
    if (row.cfg) {
        request.true_cfg_scale  = 2.0;
        request.negative_prompt = "blurry, low quality, watermark";
    }
    if (row.reference) {
        if (reference_b64.empty()) {
            std::printf("SKIP row %ux%u %s: no reference picture was produced earlier\n", row.size, row.size,
                        row.kind);
            return;
        }
        request.image = reference_b64;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        const auto response = images.generate(request, static_cast<std::int64_t>(
                                                           std::chrono::duration_cast<std::chrono::seconds>(
                                                               std::chrono::system_clock::now().time_since_epoch())
                                                               .count()),
                                             "http://127.0.0.1:8888/v1/images/");
        const double total_ms = ms_since(t0);
        std::string url;
        if (response.contains("data") && !response["data"].empty() && response["data"][0].contains("url")) {
            url = response["data"][0]["url"].get<std::string>();
        }
        std::printf("RESULT size=%ux%u kind=%s ref=%d cfg=%d total_ms=%.0f pause_ms=%.0f device_ms=%.0f "
                    "probe_ms=%.0f probes=%zu id=%s\n",
                    row.size, row.size, row.kind, row.reference ? 1 : 0, row.cfg ? 1 : 0, total_ms,
                    runner.pause_ms, executor.last_device_ms, runner.probe_ms, runner.probes, url.c_str());
        std::fflush(stdout);
        // The first 768 picture becomes the reference row's input: the route's own file, off disk.
        if (row.size == 768 && !row.reference && reference_b64.empty() && !url.empty()) {
            const auto png = read_file(std::filesystem::current_path() / "images" / (id_from_url(url) + ".png"));
            if (!png.empty()) { reference_b64 = base64_encode(png); }
        }
    } catch (const std::exception& e) {
        std::printf("RESULT size=%ux%u kind=%s ref=%d cfg=%d FAILED: %s\n", row.size, row.size, row.kind,
                    row.reference ? 1 : 0, row.cfg ? 1 : 0, e.what());
        std::fflush(stdout);
    }
    runner.pause_ms = 0.0;  // per row from here on
    runner.probes   = 0;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* qwen4   = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    const char* image   = std::getenv("INFERNIX_TEST_ARTIFACT");
    const char* tower   = std::getenv("INFERNIX_IMAGE_TEXT_PROCESSOR");
    if (qwen4 == nullptr || image == nullptr || tower == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT, INFERNIX_TEST_ARTIFACT and "
                    "INFERNIX_IMAGE_TEXT_PROCESSOR\n");
        return 77;
    }
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        return 2;
    }

    const std::string image_artifact  = image;
    const std::string image_model     = env_or("INFERNIX_BENCH_IMAGE_MODEL", "qwen-image-2-1");
    const std::uint32_t max_context   = static_cast<std::uint32_t>(std::atoi(env_or("INFERNIX_BENCH_MAX_CONTEXT", "65536")));
    const std::uint64_t expert_ram    = static_cast<std::uint64_t>(
                                            std::atoi(env_or("INFERNIX_BENCH_EXPERT_RAM_GIB", "8"))) * 1024ULL * 1024ULL * 1024ULL;

    infernix::EngineOptions options;
    options.artifact_path         = qwen4;
    options.context_cache.enabled = false;
    options.enable_vision         = false;
    options.max_context           = max_context;
    options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(max_context);
    options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
    options.expert_ram_bytes      = expert_ram;
    if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }

    infernix::product::LoggingOptions logging_options;
    logging_options.logger_name  = "image_stage_bench";
    logging_options.level        = infernix::product::LogLevel::Info;
    logging_options.presentation = infernix::product::LogPresentation::Tool;
    infernix::product::LoggingRuntime logging(logging_options);
    auto logger = logging.logger();

    infernix::Engine engine(options);

    infernix::serve::ImageJobStore jobs(64, 256ULL << 20);
    auto runner = std::make_unique<TimingRunner>(
        infernix::serve::make_engine_interlude_runner(engine));
    TimingRunner* runner_ptr = runner.get();
    TimingExecutor executor(infernix::serve::make_artifact_image_job_executor(image_artifact, logger),
                            runner_ptr);
    infernix::serve::ImageGenerationService images(
        std::move(runner), jobs,
        infernix::serve::make_tower_prompt_renderer(tower),
        infernix::serve::make_tower_prompt_embeds_provider(image_artifact),
        infernix::serve::make_artifact_dit_dims_provider(image_artifact),
        infernix::serve::ImageJobExecutor(
            [executor_ptr = &executor](const infernix::serve::OpenAIImagesRequest& request, std::uint64_t seed,
                                        std::span<const std::uint16_t> prompt_embeds,
                                        std::span<const std::uint16_t> negative_embeds,
                                        const infernix::serve::ReferencePixels* reference, void* image_stream,
                                        const std::shared_ptr<infernix::serve::ImageJobProgress>& progress) {
                return (*executor_ptr)(request, seed, prompt_embeds, negative_embeds, reference, image_stream,
                                       progress);
            }));

    std::printf("BENCH engine max_context=%u expert_ram_gib=%llu image_artifact=%s\n", max_context,
                static_cast<unsigned long long>(expert_ram / (1024ULL * 1024ULL * 1024ULL)),
                image_artifact.c_str());

    // INFERNIX_BENCH_ROWS narrows the matrix (comma-separated kind names: cold,warm,ref,cfg) so one row
    // can be re-measured without the full run. The ref row feeds back the first 768 picture, so a filter
    // that leaves out the 768 cold run makes it print SKIP instead of inventing an input.
    const std::string only = env_or("INFERNIX_BENCH_ROWS", "cold,warm,ref,cfg");
    const auto wanted = [&only](const char* kind) { return only.find(kind) != std::string::npos; };
    std::string reference_b64;
    for (const Row& row : std::vector<Row>{{512, "cold"},  {512, "warm"},
                                           {768, "cold"},  {768, "warm"},
                                           {768, "ref", /*reference=*/true},
                                           {768, "cfg", /*reference=*/false, /*cfg=*/true},
                                           {1024, "cold"}, {1024, "warm"}}) {
        if (!wanted(row.kind)) { continue; }
        run_row(images, *runner_ptr, executor, row, reference_b64, image_model);
    }
    return 0;
}
