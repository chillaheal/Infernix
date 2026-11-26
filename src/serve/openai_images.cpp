#include "serve/openai_images.h"

#include "serve/request_validation.h"

#include <charconv>
#include <cmath>
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
                                             "image", "negative_prompt", "true_cfg_scale",
                                             "wh_ratio", "ratio_follow", "prompt_enhance"};

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

int snap_to_32(double pixels) {
    // Python round(): banker's rounding of v/32, back up by 32. Exact .5 fractions are the only
    // divergence from round-half-away; matching them keeps boundary geometries oracle-identical.
    // This is the rounding in the reference pipeline's calculate_dimensions, and the reference plane
    // and the canvas must agree on it -- see reference_image.h.
    const double q = pixels / 32.0;
    const double floor_q = std::floor(q);
    const double rest = q - floor_q;
    double snapped = 0.0;
    if (rest > 0.5) { snapped = floor_q + 1.0; }
    else if (rest < 0.5) { snapped = floor_q; }
    else { snapped = (std::fmod(floor_q, 2.0) == 0.0) ? floor_q : floor_q + 1.0; }
    if (snapped < 1.0) { return 0; }
    return static_cast<int>(snapped) * 32;
}

double parse_image_aspect_ratio(std::string_view wh_ratio) {
    const std::size_t separator = wh_ratio.find(':');
    if (separator == std::string_view::npos ||
        wh_ratio.find(':', separator + 1) != std::string_view::npos) {
        bad_request("wh_ratio must be formatted as <width>:<height>", "wh_ratio", "image_ratio_invalid");
    }

    std::uint32_t w = 0;
    std::uint32_t h = 0;
    if (!parse_dimension(wh_ratio.substr(0, separator), w) ||
        !parse_dimension(wh_ratio.substr(separator + 1), h) || w == 0 || h == 0) {
        bad_request("wh_ratio sides must be positive integers", "wh_ratio", "image_ratio_invalid");
    }
    const double ratio = static_cast<double>(w) / static_cast<double>(h);
    if (!(ratio >= kMinimumImageAspectRatio && ratio <= kMaximumImageAspectRatio)) {
        bad_request("wh_ratio must be between 1:64 and 64:1 for this model", "wh_ratio",
                    "image_ratio_invalid");
    }
    return ratio;
}

bool parse_image_reference_tag(std::string_view tag, std::size_t reference_count, std::size_t& index) {
    const std::string_view prefix = kImageRatioFollowPrefix;
    if (!tag.starts_with(prefix) || tag.size() < prefix.size() + 2 || tag.back() != '>') {
        return false;
    }
    const std::string_view digits(tag.data() + prefix.size(),
                                 tag.size() - prefix.size() - 1);  // strip "<image" and the '>'
    std::uint64_t value = 0;
    const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (result.ec != std::errc{} || result.ptr != digits.data() + digits.size() || value < 1 ||
        value > reference_count) {
        return false;  // a 1-based tag naming no picture this request carries
    }
    index = static_cast<std::size_t>(value - 1);
    return true;
}

void resolve_canvas_geometry(const OpenAIImagesRequest& request, std::uint32_t reference_width,
                            std::uint32_t reference_height, std::uint32_t& width,
                            std::uint32_t& height) {
    if (request.size_requested) { return; }  // an explicit size is already the canvas, exactly

    double aspect = 0.0;
    if (request.wh_ratio) {
        aspect = parse_image_aspect_ratio(*request.wh_ratio);
    } else if (reference_width > 0 && reference_height > 0) {
        // ratio_follow, and the no-size edit: the reference pipeline derives the canvas from the
        // reference's own aspect when the caller names neither size nor ratio.
        aspect = static_cast<double>(reference_width) / static_cast<double>(reference_height);
    }
    if (!(aspect > 0.0)) { return; }  // text-to-image with no aspect: the default canvas stands

    if (aspect < kMinimumImageAspectRatio) { aspect = kMinimumImageAspectRatio; }
    if (aspect > kMaximumImageAspectRatio) { aspect = kMaximumImageAspectRatio; }

    // calculate_dimensions(kCanvasTargetArea, aspect): both sides from the area, then snapped. The
    // area (not the long side) is what is held constant, which is what keeps the joint sequence the
    // same cost in every direction.
    int w = snap_to_32(std::sqrt(kCanvasTargetArea * aspect));
    int h = snap_to_32(std::sqrt(kCanvasTargetArea / aspect));
    if (w < static_cast<int>(kMinimumImageDimension)) { w = static_cast<int>(kMinimumImageDimension); }
    if (h < static_cast<int>(kMinimumImageDimension)) { h = static_cast<int>(kMinimumImageDimension); }
    if (w > static_cast<int>(kMaximumImageDimension)) { w = static_cast<int>(kMaximumImageDimension); }
    if (h > static_cast<int>(kMaximumImageDimension)) { h = static_cast<int>(kMaximumImageDimension); }
    width  = static_cast<std::uint32_t>(w);
    height = static_cast<std::uint32_t>(h);
}

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

    // Three ways to name a canvas, and only ever one of them. An explicit size is an order; an aspect
    // is a shape the engine fills to its configured area; following the reference is what the reference
    // pipeline does when the caller names nothing. Silently preferring one over another would let a
    // client that sent both believe the one it cared about was honored.
    const int canvas_keys = static_cast<int>(body.contains("size") && !body.at("size").is_null()) +
                            static_cast<int>(body.contains("wh_ratio") && !body.at("wh_ratio").is_null()) +
                            static_cast<int>(body.contains("ratio_follow") &&
                                             !body.at("ratio_follow").is_null());
    if (canvas_keys > 1) {
        bad_request("give at most one of size, wh_ratio and ratio_follow", "size",
                    "image_size_conflict");
    }

    if (body.contains("size") && !body.at("size").is_null()) {
        if (!body.at("size").is_string()) { bad_request("size must be a string", "size"); }
        parse_size(body.at("size").get<std::string_view>(), output.width, output.height);
        output.size_requested = true;
    }

    if (body.contains("wh_ratio") && !body.at("wh_ratio").is_null()) {
        if (!body.at("wh_ratio").is_string()) { bad_request("wh_ratio must be a string", "wh_ratio"); }
        const std::string_view ratio = body.at("wh_ratio").get<std::string_view>();
        (void)parse_image_aspect_ratio(ratio);  // validated now, so a bad ratio is never a silent fallback
        output.wh_ratio = std::string(ratio);
    }

    if (body.contains("ratio_follow") && !body.at("ratio_follow").is_null()) {
        if (!body.at("ratio_follow").is_string()) {
            bad_request("ratio_follow must be a string", "ratio_follow");
        }
        const std::string_view follow = body.at("ratio_follow").get<std::string_view>();
        // The form is checked here; whether the tag names a reference this request actually carries
        // can only be answered once the references are parsed, so that half runs at the end.
        if (!follow.starts_with(kImageRatioFollowPrefix) || follow.empty() || follow.back() != '>') {
            bad_request("ratio_follow must be <imageN>, e.g. <image1>", "ratio_follow",
                        "image_ratio_follow_unsupported");
        }
        output.ratio_follow = std::string(follow);
            }
        
            if (body.contains("prompt_enhance") && !body.at("prompt_enhance").is_null()) {
                const RequestJson& enhance = body.at("prompt_enhance");
                if (enhance.is_boolean()) {
                    output.prompt_enhance =
                        enhance.get<bool>() ? PromptEnhanceMode::On : PromptEnhanceMode::Off;
                } else if (enhance.is_string() && enhance.get<std::string_view>() == "auto") {
                    // Accepted here, decided there: what auto runs on is a property of the whole
                    // request and of what the enhancer costs, so the rule lives beside the numbers in
                    // image_generation_service.cpp rather than in the parser.
                    output.prompt_enhance = PromptEnhanceMode::Auto;
                } else {
                    bad_request("prompt_enhance must be a boolean or \"auto\"", "prompt_enhance");
                }
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
        // The family takes a LIST of references and every one of them conditions the forward; this
        // route accepts up to kMaximumImageReferences (whose comment says what bounds the number and
        // where). The single-string form stays legal, so neither a one-picture nor a many-picture
        // client has to special-case us. Past the cap it is a COUNT error, not a type error: the
        // payloads are pictures, the request just asks for more than the route can reserve.
        const RequestJson& field = body.at("image");
        std::vector<std::string> payloads;
        const auto accept_one = [&payloads](const RequestJson& item) {
            if (!item.is_string()) {
                bad_request("image must be a base64 string, or an array of up to " +
                                std::to_string(kMaximumImageReferences) + " of them", "image",
                            "image_count_invalid");
            }
            payloads.push_back(item.get<std::string>());
        };
        if (field.is_string()) {
            accept_one(field);
        } else if (field.is_array()) {
            if (field.empty() || field.size() > kMaximumImageReferences) {
                bad_request("image must carry between 1 and " +
                                std::to_string(kMaximumImageReferences) + " reference images",
                            "image", "image_count_invalid");
            }
            for (const RequestJson& item : field) { accept_one(item); }
        } else {
            bad_request("image must be a base64 string, or an array of up to " +
                            std::to_string(kMaximumImageReferences) + " of them", "image",
                        "image_count_invalid");
        }
        for (std::string value : payloads) {
            if (value.empty()) {
                bad_request("image must carry a non-empty base64 reference", "image",
                            "image_count_invalid");
            }
            if (value.starts_with("http://") || value.starts_with("https://")) {
                bad_request("image must be a base64-encoded image, not a URL", "image",
                            "image_url_not_supported");
            }
            if (value.starts_with("data:")) {
                const std::size_t comma = value.find(',');
                const std::string_view prefix(value.data(),
                                              comma == std::string_view::npos ? value.size() : comma);
                if (comma == std::string_view::npos ||
                    prefix.find("base64") == std::string_view::npos) {
                    bad_request("image data: URIs must be base64", "image",
                                "image_data_uri_not_base64");
                }
                value.erase(0, comma + 1);
            }
            // Cheap pre-gate on the base64 length (raw grows 4 bytes per 3); the exact decoded byte
            // and pixel budgets are enforced where each image is decoded. Per reference, because each
            // one is decoded on its own -- four half-size pictures are not one big picture.
            if (value.size() > (kMaximumImageInputBytes / 3) * 4 + 4) {
                bad_request("image exceeds the maximum size", "image", "image_too_large");
            }
            output.images.push_back(std::move(value));
        }
    }

    // A ratio_follow tag is only meaningful against the references actually sent: naming <image3>
    // in a two-picture request is a client mistake, and quietly following picture 2 instead would
    // answer a request the client did not make.
    if (output.ratio_follow) {
        std::size_t followed = 0;
        if (!parse_image_reference_tag(*output.ratio_follow, output.images.size(), followed)) {
            bad_request("ratio_follow must name a reference this request carries, as <imageN>"
                            " (1-based) up to <image" + std::to_string(output.images.size()) + ">",
                        "ratio_follow", "image_ratio_follow_unsupported");
        }
    }

    return output;
}

OpenAIImagesRequest parse_openai_images_edit_request(const RequestJson& body) {
    OpenAIImagesRequest request = parse_openai_images_request(body);
    // Empty is both halves of 'no reference': the field was absent, or it was there and blank.
    // Either way an edit has nothing to edit, and the generations default (draw from text alone)
    // is not what this route was asked for.
    if (request.images.empty()) {
        bad_request("this is the edit route: image is required, as base64 in JSON or as a multipart "
                    "file part named image",
                    "image", "image_reference_required");
    }
    return request;
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
