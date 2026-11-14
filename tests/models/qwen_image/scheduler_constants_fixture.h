// The scheduler constants the certified schedule guards qualify the production code against, read from
// the COMMITTED artifact document (tests/convert/fixtures/qwen_image_scheduler_config.json -- the
// converter's verbatim copy of the reference checkpoint's own scheduler/scheduler_config.json, pinned
// by SHA-256 in tests/convert/test_convert_qwen_image_metadata.py) through the PRODUCTION reader
// (src/models/qwen_image/artifact_constants.cpp), not from literals in a test.
//
// Why: the shipped image artifact carries that document as its "scheduler" component and the serve route
// builds every schedule from it (DiTWeights::scheduler). A guard that carried its own copy of the six
// numbers would keep passing on hand-written values while the artifact said something else, which is the
// exact failure this lane closes. Reading the committed file keeps the guard's numbers identical (the
// artifact declares precisely those values -- test_image_artifact_constants.cpp pins that equality, the
// certified mu and the schedule bounds) while making the guard qualify what actually ships.
//
// Header-only, host tests only: it needs the artifact-facing reader, which is compiled at C++23 where
// std::expected exists (the CUDA lane stays at C++20 and does not use this header). The target must be
// registered with NEEDS_SOURCE_DIR so the fixture is found from the source tree.

#pragma once

#include "models/qwen_image/artifact_constants.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infernix::tests::qwen_image {

// The one path a guard may read scheduler constants from. Never a fallback value: an unreadable or
// refusing document is a hard failure of the guard, because a guard that silently drew its own
// schedule would certify nothing.
[[nodiscard]] inline infernix::models::qwen_image::SchedulerConstants scheduler_constants_from_artifact() {
#ifndef INFERNIX_SOURCE_DIR
    throw std::runtime_error(
        "the scheduler constants come from the committed artifact document, which this target cannot "
        "locate: register it with NEEDS_SOURCE_DIR");
#else
    const std::string path = std::string(INFERNIX_SOURCE_DIR) +
                             "/tests/convert/fixtures/qwen_image_scheduler_config.json";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open the committed scheduler document " + path +
                                 ", which the supported denoise schedule is read from");
    }
    std::ostringstream text;
    text << in.rdbuf();
    const nlohmann::json document = nlohmann::json::parse(text.str(), nullptr, /*allow_exceptions=*/false);
    if (document.is_discarded()) {
        throw std::runtime_error("the committed scheduler document " + path + " is not valid JSON");
    }
    const auto constants =
        infernix::models::qwen_image::read_scheduler_config(document);
    if (!constants.has_value()) {
        throw std::runtime_error("the committed scheduler document is refused by the reader: " +
                                 constants.error());
    }
    return *constants;
#endif
}

} // namespace infernix::tests::qwen_image
