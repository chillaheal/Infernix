#pragma once

#include "serve/serve_options.h"

#include <nlohmann/json.hpp>
#include <string_view>

namespace infernix::serve {

// The /v1/capabilities surface of the image lane: the image_generation block, the image_edit block,
// the prompt_enhancer image_generation copies out of it, and the two image entries of the endpoints
// table all live in this file (serve/image_capabilities.cpp), so an upstream change to
// serve/http_server.cpp's capabilities document never has to carry image text. http_server.cpp's
// make_capabilities_json still builds and returns the document itself and reaches the lane through
// the typed call below; there is no registry and no name-driven dispatch.

// Adds image_generation, image_edit and the two image endpoint entries of /v1/capabilities.
// A null image block means this process has no image model, exactly as before. The document's type
// is the one make_capabilities_json builds it with (http_server.cpp:660, `using Json =
// nlohmann::json;`), spelled out here because that alias is file-local to http_server.cpp: it is
// nlohmann's key-sorted default container -- not serve/request_json.h's ordered_json -- so the order
// these keys are added in cannot move a byte of the dumped document.
void add_image_capabilities(nlohmann::json& payload, const ServeOptions& options,
                            std::string_view image_model_id);

} // namespace infernix::serve
