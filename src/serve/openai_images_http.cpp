// POST /v1/images/generations: the synchronous image route. Wire parsing and the model
// value-equality gate run before the ImageGenerationService, which owns the interlude, the job
// lifecycle and the failure mapping; the response is the OpenAI images object.
#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/image_generation_service.h"
#include "serve/openai_common.h"
#include "serve/openai_images.h"

namespace infernix::serve {

void HttpServer::handle_images_generations(const httplib::Request& req, httplib::Response& res) {
    OpenAIImagesRequest request;
    try {
        request = parse_openai_images_request(parse_json_body(req));
        // Either public id names this process: the image resource's own id, or the chat id every
        // client already sends (one process, one routeable image pipeline).
        validate_openai_model(request.model, {public_model_id_, image_model_id()});
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    if (images_ == nullptr) {
        write_openai_error(res, ApiError{.status  = 503,
                                         .type    = "server_error",
                                         .message = "the image pipeline is not configured",
                                         .code    = "image_generation_not_configured"});
        return;
    }
    try {
        const RequestJson response =
            images_->generate(request, unix_time_now(), artifact_url_base(req, options_));
        res.set_content(response.dump(-1, ' ', false, RequestJson::error_handler_t::replace),
                        "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    }
}

} // namespace infernix::serve
