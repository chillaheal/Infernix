#include "serve/image_startup.h"

#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/image_generation_service.h"
#include "serve/image_job_store.h"
#include "serve/image_text_tower.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <chrono>
#include <memory>
#include <utility>

namespace infernix::serve {

std::unique_ptr<ImageLane> start_image_lane(HttpServer& server, GenerationService& service,
                                            const ServeOptions& options,
                                            const std::shared_ptr<spdlog::logger>& logger) {
    // The lane's own early return, and the only one: with no image artifact this process draws no
    // picture, so nothing below is created, no route is attached and every picture request answers
    // its documented 503 -- exactly what the inline code in main.cpp did. The `if` that follows is
    // HEAD's line kept byte for byte and reads the very same option, so the two cannot disagree.
    if (options.image_artifact_path.empty()) { return nullptr; }

        // The synchronous image pipeline: the runner yields the LLM's VRAM through the Engine's
        // interlude and the executor binds the image artifact's weights on first job. The raw-prompt
        // route additionally needs --image-text-processor (the Qwen3-VL processor directory that
        // turns the prompt string into prompt embeds); without either the route answers the
        // documented 503 image_generation_not_configured.
        std::unique_ptr<infernix::serve::ImageJobStore> image_jobs;
        std::unique_ptr<infernix::serve::ImageGenerationService> images;
        // Declared after the store it reads, so the shutdown order is sweep-stopped then store-gone.
        std::unique_ptr<infernix::serve::GeneratedImageRetention> image_retention;
        if (!options.image_artifact_path.empty()) {
            infernix::serve::PromptRendererProvider prompt_renderer;
            infernix::serve::PromptEmbedsProvider prompt_embeds;
            infernix::serve::ReferencePromptRendererProvider reference_prompt_renderer;
            if (!options.image_text_processor_path.empty()) {
                // Two halves of one route: rendering the prompt is host work the service does before it
                // probes the yield; the tower forward needs the yielded VRAM. The edit route's renderer
                // is the same host half for the RAW ti2i template: the admission sizes on the ids the
                // conditioned tower will actually forward, pads included.
                prompt_renderer = infernix::serve::make_tower_prompt_renderer(
                    options.image_text_processor_path);
                prompt_embeds   = infernix::serve::make_tower_prompt_embeds_provider(
                    options.image_artifact_path);
                reference_prompt_renderer = infernix::serve::make_tower_reference_prompt_renderer(
                    options.image_text_processor_path);
            }
            image_jobs = std::make_unique<infernix::serve::ImageJobStore>(64, 256ULL << 20);
            // The family's prompt enhancer is a separate checkpoint in a separate process (it wants
            // about 20 GB of the card the LLM and the DiT also live on), and this is the handle on it.
            // A configured sidecar that is not answering is deliberately NOT a startup failure: the
            // route answers its documented 503 for prompt_enhance and every other request keeps
            // serving, so an operator who starts the worker late -- or not at all -- loses one field.
            std::shared_ptr<infernix::serve::PromptEnhancer> enhancer;
            if (!options.image_prompt_enhancer_url.empty()) {
                enhancer = std::make_shared<infernix::serve::PromptEnhancer>(
                    options.image_prompt_enhancer_url, std::chrono::minutes(15));
                logger->info("image prompt enhancer configured at {} ({})",
                             options.image_prompt_enhancer_url,
                             enhancer->healthy() ? "answering" : "not answering now");
            }
            images     = std::make_unique<infernix::serve::ImageGenerationService>(
                infernix::serve::make_engine_interlude_runner(service.engine()), *image_jobs,
                std::move(prompt_renderer), std::move(prompt_embeds),
                infernix::serve::make_artifact_dit_dims_provider(options.image_artifact_path),
                infernix::serve::make_artifact_image_job_executor(options.image_artifact_path,
                                                                  logger),
                // The edit route's two remaining providers. The ViT lane's shape comes from the
                // artifact's own component config so the ledger can reserve for an encode it has not
                // run yet; without it a reference request is refused rather than guessed, because an
                // under-reserved interlude takes the resident LLM down with it.
                infernix::serve::make_artifact_vision_dims_provider(options.image_artifact_path),
                std::move(reference_prompt_renderer), std::move(enhancer),
                // The third provider that reads this artifact's own declarations: the VAE decoder geometry
                // the VRAM ledger prices every job's decode from (the same topology the binding verifies
                // against every stored decoder weight). Unlike the ViT lane's, it is needed by EVERY image
                // job, so it goes to the same artifact the DiT and the VAE are streamed from and a serve
                // whose artifact has no readable "vae" component refuses jobs as unconfigured -- there is
                // no compiled-in decoder geometry standing in for the checkpoint's.
                infernix::serve::make_artifact_vae_topology_provider(options.image_artifact_path));
            server.attach_images(*images);
            // A picture's URL is the only handle a chat reply hands out, so the file behind it is kept
            // for a stated number of days and then swept; 0 switches the cleanup off, and the server
            // keeps everything it ever drew.
            if (options.image_retention_days > 0) {
                image_retention = std::make_unique<infernix::serve::GeneratedImageRetention>(
                    *image_jobs, std::chrono::days(options.image_retention_days), logger);
            }
        }
    // The three handles move into the lane in the order HEAD created them, which is the order they
    // were declared in there, so the destruction contract is untouched: ImageLane dies reverse order,
    // which stops the sweep, then takes the service, then the store the sweep read.
    auto lane = std::make_unique<ImageLane>();
    lane->jobs      = std::move(image_jobs);
    lane->service   = std::move(images);
    lane->retention = std::move(image_retention);
    return lane;
}

// Empty by design: the shutdown order is ImageLane's member declaration order and nothing else.
ImageLane::~ImageLane() = default;

} // namespace infernix::serve
