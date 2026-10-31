#pragma once

// OpenAI Images wire objects: POST /v1/images/generations request parsing and the
// image-response object builder. Serve-internal: this module translates the external
// OpenAI contract only; it does not acquire media or own model/device state.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// How the finished picture is delivered. `url` points at this server's own artifact route
// (GET /v1/images/<id>), which serves the picture written to the on-disk `images` collection.
enum class ImageResponseFormat : std::uint8_t {
    Base64,
    Url,
};

// Parsed, validated subset of the OpenAI image-generation request. Protocol-only
// fields we cannot honor are rejected during parse, never silently dropped.
struct OpenAIImagesRequest {
    std::string model;                 // presence/type validated here; value membership is route-side
    std::string prompt;                // non-empty, <= kMaximumImagePromptBytes
    std::uint32_t width  = 768;        // default until variable size lands; fits the interlude yield on a 32 GiB card
    std::uint32_t height = 768;        // snapped from "WxH"; multiple of 32, in [32, 2048] (1024x1024 OOM'd the 8 GiB yield)
    std::uint32_t steps  = 40;         // internal default; no OpenAI counterpart
    double true_cfg_scale = 1.0;       // product extension; > 1 steers with a second DiT forward
    std::string negative_prompt;       // required non-empty when true_cfg_scale > 1
    std::optional<std::uint64_t> seed; // product extension; absent => engine-chosen
    std::string user;                  // opaque client tag; never echoed back
    std::string image;                 // #75 product extension: base64 reference image (PNG/JPEG),
                                       // optional data: URI prefix, given as the string or as an array
                                       // of exactly one such string (kMaximumImageReferences); an
                                       // empty string means plain text-to-image
    ImageResponseFormat response_format = ImageResponseFormat::Base64;
};

// Path prefix of the route that serves a finished picture: GET <prefix><artifact id>.
inline constexpr std::string_view kImageArtifactRoutePrefix = "/v1/images/";

// Inclusive range for the true-CFG guidance scale. 1.0 means off: one DiT forward per step, the route
// certified before CFG existed. Above 1 each step also runs the negative prompt and the velocity
// becomes neg + scale * (cond - neg), so the wall clock of a step roughly doubles.
inline constexpr double kMinimumImageCFGScale = 1.0;
inline constexpr double kMaximumImageCFGScale = 20.0;

// Upper bound on the decoded reference-image payload accepted in the `image` field (bytes).
inline constexpr std::size_t kMaximumImageInputBytes = 16ULL * 1024 * 1024;

// How many references one request may carry. The Qwen-Image family allows up to 10; this build renders
// exactly one, because M7's admission ledger sizes the joint sequence for a single reference on the
// 1024^2-area cap (4 * vision tokens == the reference's latent tokens) and nothing has measured what a
// second one costs. The WIRE accepts the list form at this cap so a client written against the family
// does not special-case us; raising the cap is a ledger change first (M7), never a parse change alone.
inline constexpr std::size_t kMaximumImageReferences = 1;

// Upper bound on prompt length accepted at the images boundary (bytes).
inline constexpr std::size_t kMaximumImagePromptBytes = 4096;

// Inclusive range for a requested image dimension (pixels).
inline constexpr std::uint32_t kMinimumImageDimension = 32;
inline constexpr std::uint32_t kMaximumImageDimension = 2048;

// Inclusive range for the denoise-step count (num_inference_steps == the number of DiT forwards).
// The floor is 2, not 1: the resolution-dependent schedule (build_schedule) divides by
// (steps - 1) and is defined only for two or more steps. The ceiling bounds one request's
// wall-clock/interlude hold so a client cannot pin the shared LLM's VRAM for an unbounded run.
inline constexpr std::uint32_t kMinimumImageSteps = 2;
inline constexpr std::uint32_t kMaximumImageSteps = 100;

// The public id of the image resource this process serves: the image artifact's file stem, which is
// the name of the model actually loaded. Empty when no image artifact is configured. It is the id
// listed in /v1/models next to the chat model and the value a client sends as `model` here.
[[nodiscard]] std::string image_model_id_from_artifact(std::string_view artifact_path);

// True for the id shape the artifact route serves: img-<created>-<sequence>. The stored job id is
// the only legal way to name a picture, so a caller can never steer this route at a path.
[[nodiscard]] bool is_image_artifact_id(std::string_view id) noexcept;

// Parse and validate the JSON body of POST /v1/images/generations. Throws ApiException
// (HTTP 400) for every violation of the accepted subset.
OpenAIImagesRequest parse_openai_images_request(const RequestJson& body);

// Build the synchronous image-generation response object:
// { "created": <epoch>, "data": [ { "b64_json": <b64> [, "revised_prompt": <str>] }, ... ] }
RequestJson make_openai_image_response(std::int64_t created,
                                       const std::vector<std::string>& b64_images,
                                       const std::optional<std::string>& revised_prompt = {});

// The same object for response_format:"url": { "created", "data": [ { "url": <absolute> } ] }.
// `artifact_urls` are the absolute URLs of the pictures this server already wrote to disk.
RequestJson make_openai_image_url_response(std::int64_t created,
                                           const std::vector<std::string>& artifact_urls);

} // namespace infernix::serve
