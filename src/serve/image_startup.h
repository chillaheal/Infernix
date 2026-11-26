#pragma once

#include <memory>

namespace spdlog {
class logger;
}

namespace infernix::serve {

class GenerationService;
class HttpServer;
class ImageGenerationService;
class ImageJobStore;
class GeneratedImageRetention;
struct ServeOptions;  // (it lives in serve/serve_options.h -- include it in the .cpp)

// The startup of the image lane: the job store, the artifact's provider factories, the prompt
// enhancer handle, the generation service, server.attach_images and the retention sweep all live in
// this file (serve/image_startup.cpp), so an upstream change to apps/serve/main.cpp never has to
// carry image text or image rules. main.cpp reaches the lane through the single typed call below;
// there is no registry and no name-driven dispatch.

// Owns what start_image_lane created. Destroyed by its owner after listen() returns, in the order
// the members are declared -- that order is the shutdown contract (sweep stops, then the service,
// then the store) and must not change.
struct ImageLane {
    std::unique_ptr<ImageJobStore>            jobs;
    std::unique_ptr<ImageGenerationService>   service;
    std::unique_ptr<GeneratedImageRetention>  retention;

    // Declared here and defined in the .cpp, where the three types above are complete: the lane owns
    // their headers and main.cpp must not need them, and a unique_ptr can only destroy what it can
    // see. An empty definition, so the contract above is still plain member order -- reverse
    // declaration, which is retention, then the service, then the store.
    ~ImageLane();
};

// What the inline startup in main.cpp always did, in the same order: the store, then the enhancer
// handle and the service whose executor binds the image artifact's weights on the first job, then
// server.attach_images, then the sweep when --image-retention-days is above zero. nullptr when this
// process has no image artifact, exactly as the inline code did -- which is also when no image route
// is attached and every picture request answers its documented 503. A sidecar that is configured but
// not answering stays a startup success, as before. `logger` is main.cpp's own application logger
// (const std::shared_ptr<spdlog::logger> at main.cpp:252), handed to the two callees that take it.
// The caller keeps the handle alive until after listen() returns.
[[nodiscard]] std::unique_ptr<ImageLane>
start_image_lane(HttpServer& server, GenerationService& service, const ServeOptions& options,
                 const std::shared_ptr<spdlog::logger>& logger);

} // namespace infernix::serve
