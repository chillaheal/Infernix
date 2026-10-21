#include "serve/openai_images.h"

#include "serve/request_validation.h"

#include <charconv>
#include <filesystem>
#include <initializer_list>
#include <string_view>

namespace infernix::serve {
namespace {

// The exact set of top-level parameters this endpoint accepts. Anything outside it is
// rejected rather than silently ignored, so a client never mistakes a dropped quality,
// style or input_fidelity for honored behavior.
constexpr std::string_view kAllowedKeys[] = {"model", "prompt",   "n",
                                             "size",  "response_format", "user", "seed", "steps",
                                             "image", "negative_prompt", "true_cfg_scale"};

[[nodiscard]] bool is_allowed_key(std::string_view key) {
    for (std::string_view allowed : kAllowedKeys) {
        if (allowed == key) { return true; }
    }
    return false;
}

// Parse a strictly-decimal token into a uint32 with overflow detection. Returns false on
// empty, non-digit, or out-of-range input.
[[nodiscard]] bool parse_dimension(std::string_view token, std::uint32_t& out) {
    if (token.empty()) { return false; }
    std::uint64_t value = 0;
    const char* first   = token.data();
    const char* last    = token.data() + token.size();
    const auto result   = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) { return false; }
    if (value > 0xFFFFFFFFULL) { return false; }
    out = static_cast<std::uint32_t>(value);
    return true;
}

void parse_size(std::string_view size, std::uint32_t& width, std::uint32_t& height) {
    const std::size_t separator = size.find('x');
    if (separator == std::string_view::npos || size.find('x', separator + 1) != std::string_view::npos) {
        bad_request("size must be formatted as <width>x<height>", "size", "image_size_invalid");
    }
    std::uint32_t w = 0;
    std::uint32_t h = 0;
    if (!parse_dimension(size.substr(0, separator), w) ||
        !parse_dimension(size.substr(separator + 1), h)) {
        bad_request("size dimensions must be positive integers", "size", "image_size_invalid");
    }
    auto validate_dimension = [](std::uint32_t dimension, std::string_view side) {
        if (dimension < kMinimumImageDimension || dimension > kMaximumImageDimension) {
            bad_request("size " + std::string(side) + " must be between " +
                            std::to_string(kMinimumImageDimension) + " and " +
                            std::to_string(kMaximumImageDimension) + " pixels",
                        "size", "image_size_invalid");
        }
        if (dimension % 32 != 0) {
            bad_request("size " + std::string(side) +
                            " must be a multiple of 32 for this model",
                        "size", "image_size_invalid");
        }
    };
    validate_dimension(w, "width");
    validate_dimension(h, "height");
    width  = w;
    height = h;
}

} // namespace

OpenAIImagesRequest parse_openai_images_request(const RequestJson& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }

    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!is_allowed_key(it.key())) {
            bad_request("unsupported parameter: " + it.key(), it.key(), "unsupported_parameter");
        }
    }

    OpenAIImagesRequest output;

    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    output.model = body.at("model").get<std::string>();

    if (!body.contains("prompt") || body.at("prompt").is_null()) {
        bad_request("missing required field: prompt", "prompt");
    }
    if (!body.at("prompt").is_string()) { bad_request("prompt must be a string", "prompt"); }
    output.prompt = body.at("prompt").get<std::string>();
    if (output.prompt.empty()) { bad_request("prompt must not be empty", "prompt"); }
    if (output.prompt.size() > kMaximumImagePromptBytes) {
        bad_request("prompt exceeds maximum length", "prompt", "string_above_max_length");
    }

    if (body.contains("n") && !body.at("n").is_null()) {
        const RequestJson& n = body.at("n");
        if (!n.is_number_integer() || n.get<int>() < 1) {
            bad_request("n must be a positive integer", "n");
        }
        if (n.get<int>() != 1) {
            bad_request("this model generates one image per request", "n", "image_n_not_supported");
        }
    }

    if (body.contains("size") && !body.at("size").is_null()) {
        if (!body.at("size").is_string()) { bad_request("size must be a string", "size"); }
        parse_size(body.at("size").get<std::string_view>(), output.width, output.height);
    }

    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        if (!body.at("response_format").is_string()) {
            bad_request("response_format must be a string", "response_format");
        }
        const std::string_view format = body.at("response_format").get<std::string_view>();
        if (format == "b64_json") {
            output.response_format = ImageResponseFormat::Base64;
        } else if (format == "url") {
            output.response_format = ImageResponseFormat::Url;
        } else {
            bad_request("response_format must be b64_json or url", "response_format",
                        "response_format_not_supported");
        }
    }

    if (body.contains("user") && !body.at("user").is_null()) {
        if (!body.at("user").is_string()) { bad_request("user must be a string", "user"); }
        output.user = body.at("user").get<std::string>();
    }

    if (body.contains("seed") && !body.at("seed").is_null()) {
        const RequestJson& seed = body.at("seed");
        if (!seed.is_number_integer() || seed.get<std::int64_t>() < 0) {
            bad_request("seed must be a non-negative integer", "seed");
        }
        output.seed = static_cast<std::uint64_t>(seed.get<std::int64_t>());
    }

    if (body.contains("steps") && !body.at("steps").is_null()) {
        const RequestJson& steps = body.at("steps");
        if (!steps.is_number_integer() || steps.get<std::int64_t>() < 0) {
            bad_request("steps must be a non-negative integer", "steps");
        }
        const std::int64_t requested = steps.get<std::int64_t>();
        if (requested < static_cast<std::int64_t>(kMinimumImageSteps) ||
            requested > static_cast<std::int64_t>(kMaximumImageSteps)) {
            bad_request("steps must be between " + std::to_string(kMinimumImageSteps) + " and " +
                            std::to_string(kMaximumImageSteps) + " for this model",
                        "steps", "image_steps_invalid");
        }
        output.steps = static_cast<std::uint32_t>(requested);
    }

    if (body.contains("true_cfg_scale") && !body.at("true_cfg_scale").is_null()) {
        const RequestJson& scale = body.at("true_cfg_scale");
        if (!scale.is_number()) { bad_request("true_cfg_scale must be a number", "true_cfg_scale"); }
        const double value = scale.get<double>();
        // The negated comparison also rejects a NaN, which no guidance scale can use.
        if (!(value >= kMinimumImageCFGScale && value <= kMaximumImageCFGScale)) {
            bad_request("true_cfg_scale must be between " + std::to_string(kMinimumImageCFGScale) +
                            " and " + std::to_string(kMaximumImageCFGScale) + " for this model",
                        "true_cfg_scale", "image_cfg_scale_invalid");
        }
        output.true_cfg_scale = value;
    }

    if (body.contains("negative_prompt") && !body.at("negative_prompt").is_null()) {
        if (!body.at("negative_prompt").is_string()) {
            bad_request("negative_prompt must be a string", "negative_prompt");
        }
        output.negative_prompt = body.at("negative_prompt").get<std::string>();
        if (output.negative_prompt.size() > kMaximumImagePromptBytes) {
            bad_request("negative_prompt exceeds maximum length", "negative_prompt",
                        "string_above_max_length");
        }
    }

    // Steering without a negative prompt would silently mean "no steering" while the client asked for
    // it, so the pair is validated here rather than degraded in the engine.
    if (output.true_cfg_scale > kMinimumImageCFGScale && output.negative_prompt.empty()) {
        bad_request("negative_prompt is required when true_cfg_scale is above 1", "negative_prompt",
                    "image_negative_prompt_required");
    }

    if (body.contains("image") && !body.at("image").is_null()) {
        if (!body.at("image").is_string()) { bad_request("image must be a string", "image"); }
        std::string value = body.at("image").get<std::string>();
        if (value.starts_with("http://") || value.starts_with("https://")) {
            bad_request("image must be a base64-encoded image, not a URL", "image",
                        "image_url_not_supported");
        }
        if (value.starts_with("data:")) {
            const std::size_t comma = value.find(',');
            const std::string_view prefix(value.data(),
                                          comma == std::string_view::npos ? value.size() : comma);
            if (comma == std::string_view::npos || prefix.find("base64") == std::string_view::npos) {
                bad_request("image data: URIs must be base64", "image", "image_data_uri_not_base64");
            }
            value.erase(0, comma + 1);
        }
        // Cheap pre-gate on the base64 length (raw grows 4 bytes per 3); the exact decoded
        // byte and pixel budgets are enforced where the image is decoded.
        if (value.size() > (kMaximumImageInputBytes / 3) * 4 + 4) {
            bad_request("image exceeds the maximum size", "image", "image_too_large");
        }
        output.image = std::move(value);
    }

    return output;
}

RequestJson make_openai_image_response(std::int64_t created,
                                       const std::vector<std::string>& b64_images,
                                       const std::optional<std::string>& revised_prompt) {
    RequestJson data = RequestJson::array();
    for (const std::string& b64 : b64_images) {
        RequestJson item{{"b64_json", b64}};
        if (revised_prompt) { item["revised_prompt"] = *revised_prompt; }
        data.push_back(std::move(item));
    }
    return RequestJson{{"created", created}, {"data", std::move(data)}};
}

RequestJson make_openai_image_url_response(std::int64_t created,
                                           const std::vector<std::string>& artifact_urls) {
    RequestJson data = RequestJson::array();
    for (const std::string& url : artifact_urls) { data.push_back(RequestJson{{"url", url}}); }
    return RequestJson{{"created", created}, {"data", std::move(data)}};
}

std::string image_model_id_from_artifact(std::string_view artifact_path) {
    if (artifact_path.empty()) { return {}; }
    return std::filesystem::path(artifact_path).stem().string();
}

bool is_image_artifact_id(std::string_view id) noexcept {
    const auto digits = [](std::string_view text) {
        if (text.empty()) { return false; }
        for (const char symbol : text) {
            if (symbol < '0' || symbol > '9') { return false; }
        }
        return true;
    };
    constexpr std::string_view prefix = "img-";
    if (!id.starts_with(prefix)) { return false; }
    const std::size_t dash = id.find('-', prefix.size());
    if (dash == std::string_view::npos) { return false; }
    return digits(id.substr(prefix.size(), dash - prefix.size())) && digits(id.substr(dash + 1));
}

} // namespace infernix::serve
