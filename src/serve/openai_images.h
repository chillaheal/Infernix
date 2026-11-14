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
#include <utility>
#include <vector>

namespace infernix::serve {

// How the finished picture is delivered. `url` points at this server's own artifact route
// (GET /v1/images/<id>), which serves the picture written to the on-disk `images` collection.
enum class ImageResponseFormat : std::uint8_t {
    Base64,
    Url,
};

// `prompt_enhance` as the wire sends it: absent or false is Off, true is On, and the string "auto"
// hands the decision to the server (image_generation_service.cpp wants_prompt_enhancement). Off and
// On are the only two states this parameter ever had; Auto is additive, so no client that works
// today sends it and none can be broken by it.
enum class PromptEnhanceMode : std::uint8_t {
    Off,
    On,
    Auto,
};

// Parsed, validated subset of the OpenAI image-generation request. Protocol-only
// fields we cannot honor are rejected during parse, never silently dropped.
struct OpenAIImagesRequest {
    std::string model;                 // presence/type validated here; value membership is route-side
    std::string prompt;                // non-empty, <= kMaximumImagePromptBytes
    std::uint32_t width  = 768;        // the canvas: from `size`, or from the aspect below, or this
    std::uint32_t height = 768;        // default (which is also the text-to-image canvas)
    // How the canvas was asked for. Exactly one of these three may be present; the reference's own
    // aspect is the fourth, implicit answer (an edit with no size follows its reference, as the
    // reference pipeline does). `size` wins its own path and sets size_requested.
    bool size_requested = false;
    std::optional<std::string> wh_ratio;      // "W:H"; the canvas keeps that ratio at kCanvasTargetArea
    std::optional<std::string> ratio_follow;  // "<imageI>"; the canvas keeps that reference's aspect
    std::uint32_t steps  = 40;         // internal default; no OpenAI counterpart
    double true_cfg_scale = 1.0;       // product extension; > 1 steers with a second DiT forward
    std::string negative_prompt;       // required non-empty when true_cfg_scale > 1
    std::optional<std::uint64_t> seed; // product extension; absent => engine-chosen
    std::string user;                  // opaque client tag; never echoed back
    // #75 product extension: the base64 reference images (PNG/JPEG), in the order the client sent
    // them. Each accepts the plain string or a data: URI prefix; the JSON field takes one string or an
    // array of up to kMaximumImageReferences (the family's list form). Empty means plain
    // text-to-image. The LAST entry is what the canvas follows when no size is named -- the reference
    // pipeline derives the geometry from image[-1] (image-edit/refs/q21_pipeline_qwenimage21.py:630-636).
    std::vector<std::string> images;
    // Ask the server to rewrite the instruction with the family's prompt enhancer before drawing.
    // That enhancer is a SEPARATE shipped model, not a template: Qwen/Qwen-Image-2.1-PE-I2I is a
    // fine-tuned Qwen3.5-VL 9B that reads the instruction AND the source images and answers
    // {rewritten_prompt, wh_ratio, ratio_follow} (image-edit/refs/q21_prompt_rewrite_README.md:1-12,
    // q21_system_prompt_edit.txt:175-183). The vision-language tower inside the image artifact is a
    // different checkpoint, never trained against that system prompt, so it cannot stand in for it.
    // serve's --image-prompt-enhancer-url points the route at the sidecar that runs it; without that
    // flag, or with the sidecar not answering, asking is the 503 image_prompt_enhancer_unavailable
    // that says so -- never a drawing made from words the client believed had been rewritten. When it
    // does run, the rewrite replaces `prompt`, and its answer may decide the canvas ONLY where the
    // client left the canvas free: a request that named size, wh_ratio or ratio_follow keeps the
    // client's geometry, because the enhancer's job is the instruction. The key is whitelisted rather
    // than left unknown so a client is never told "unsupported key" about a parameter that exists
    // upstream. See capabilities image_edit.instruction for what to send without it.
    //
    // "auto" is that parameter's third value, for a client that does not want to decide per request.
    // It differs from true in one way that matters to a client that knows nothing about this server:
    // when no rewriter is running, or the card cannot be had for one, true is the documented 503 -- a
    // client that asked for a rewrite must never be handed a picture drawn from words it believed had
    // been rewritten -- while auto has already delegated that judgement and is served the client's own
    // words instead. What it runs on is wants_prompt_enhancement, not a guess about the prose.
    PromptEnhanceMode prompt_enhance = PromptEnhanceMode::Off;
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

// How many references one request may carry. The family allows up to 10, and every one of them
// conditions the same forward, so this is a route policy and not a model limit. What really bounds it
// is measured per request, not here: each reference costs 4 * its vision tokens of joint sequence
// (models/qwen_image/joint_layout.h joint_token_count, admitted against
// ops::kAttnDitJointMaxTokens before any VRAM is yielded) plus its latent tokens in the reservation
// (image_device_need.h: all_img == ref_tokens + target). Four 1024^2-area references reach roughly
// half the joint ceiling with room left for the instruction and a CFG pass; a fifth, or any set that
// does not fit, is the admission's refusal -- never a silent drop. The joint attention, the rope walk
// and the host template were built for N from the start (build_joint_layout scatters the blocks in
// order, render_ti2i_template loops image_count, expand_image_pads takes tokens_per_image); it was
// the serve plumbing that held one of each.
inline constexpr std::size_t kMaximumImageReferences = 4;

// Upper bound on prompt length accepted at the images boundary (bytes). This is a WIRE bound -- what one
// request may carry -- and deliberately not the model's bound: the model's limit is the joint sequence
// (image_generation_service.h and models/qwen_image/joint_layout.h joint_token_count against
// ops::kAttnDitJointMaxTokens), which depends on the canvas and the reference and is therefore admitted
// per request after the prompt is rendered. The two used to be the same 4096, which refused a
// text-rendering prompt the model would have served fine: Qwen-Image-2.1's own prompt enhancer writes
// an instruction of thousands of characters, and 4096 bytes is a few hundred CJK characters. Nothing on
// this route truncates a prompt -- over the byte bound is a 400, under it is the whole text.
inline constexpr std::size_t kMaximumImagePromptBytes = 16ULL * 1024;

// Inclusive range for a requested image dimension (pixels).
inline constexpr std::uint32_t kMinimumImageDimension = 32;
inline constexpr std::uint32_t kMaximumImageDimension = 2048;

// The area an aspect-only request is filled to. This is a DEPLOYMENT policy, not a request semantic:
// 768^2 is the canvas the interlude yield on this card affords (1024^2 OOM'd it), and it keeps the
// joint sequence roughly token-neutral however the canvas is rotated -- `ratio_follow` turns a
// landscape into a portrait, it does not make it bigger. Raising it is P14, which also has to raise
// the joint-attention token ceiling and add VAE tiling; it is not a constant change on its own.
inline constexpr double kCanvasTargetArea = 768.0 * 768.0;

// Inclusive range for an aspect ratio (width / height) a client may name. The reference geometry
// already stops at about this aspect, and past it the snap leaves a side below kMinimumImageDimension.
inline constexpr double kMinimumImageAspectRatio = 1.0 / 64.0;
inline constexpr double kMaximumImageAspectRatio = 64.0;

// A `ratio_follow` tag names one reference by the same <imageN> the engine renders in the ti2i
// template (image_text_tower.cpp kTi2iImageBlock), so the wire and the template agree. N is 1-based
// and must name a reference the request actually carries: following a picture nobody sent is the
// client's error, never a fallback to "whichever one we like". parse_image_reference_tag is the one
// parser of the form.
inline constexpr std::string_view kImageRatioFollowPrefix = "<image";

// The enhancer this route's prompt_enhance parameter names. One string, so the refusal, the
// capability block and the docs cannot disagree about what is missing.
inline constexpr std::string_view kImagePromptEnhancerModel = "Qwen/Qwen-Image-2.1-PE-I2I";

// The reference pipeline's calculate_dimensions rounding: the nearest multiple of 32, with Python
// round()'s banker's rounding on an exact .5 so boundary geometries match the oracle. ONE
// implementation, used for the reference plane and for the canvas: two roundings that can drift is
// exactly the class of bug this geometry has.
[[nodiscard]] int snap_to_32(double pixels);

// "W:H" -> width/height. Throws HTTP 400 image_ratio_invalid for a malformed pair, a zero side, or an
// aspect outside [kMinimumImageAspectRatio, kMaximumImageAspectRatio].
[[nodiscard]] double parse_image_aspect_ratio(std::string_view wh_ratio);

// "<imageN>" -> the 0-based index into the request's references. False for any other shape (a bare
// "image1", a zero index, a number past what was sent, trailing text): the caller turns that into a
// 400 rather than guessing which picture the client meant.
[[nodiscard]] bool parse_image_reference_tag(std::string_view tag, std::size_t reference_count,
                                             std::size_t& index);

// Decide the canvas. `size` (size_requested) is honored exactly and nothing else is consulted.
// Otherwise the aspect comes from `wh_ratio`, from the reference named by `ratio_follow` (whose
// decoded size is `reference_width`/`reference_height`), or from the reference when an edit names
// none -- then that aspect is filled to kCanvasTargetArea and both sides snapped to 32. A request
// with no reference and no aspect keeps the 768x768 default. `reference_width`/`reference_height`
// may be 0 for text-to-image. Throws HTTP 400 image_size_conflict if more than one of the three
// keys was given, before any of them is applied.
void resolve_canvas_geometry(const OpenAIImagesRequest& request, std::uint32_t reference_width,
                             std::uint32_t reference_height, std::uint32_t& width,
                             std::uint32_t& height);

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

// The edit route's parser: the generations parser, plus the one thing that makes this route an
// edit -- a reference is required. Falling through to text-to-image instead would answer a client
// that mis-attached its picture with an unrelated drawing, which is a worse failure than a
// refusal. Code: 400 image_reference_required.
OpenAIImagesRequest parse_openai_images_edit_request(const RequestJson& body);

// POST /v1/images/edits also arrives as the multipart/form-data an OpenAI SDK posts: a file part
// named image plus text fields. This is the httplib-free view of that form (text parts and file
// parts, each in the order the client sent them), and edit_body_from_multipart turns it into
// EXACTLY the JSON document parse_openai_images_request validates -- one validation for both wire
// shapes, because a rule that lives on only one of them is a bug waiting to be filed.
struct MultipartEditParts {
    std::vector<std::pair<std::string, std::string>> fields;  // text parts: name, value
    std::vector<std::pair<std::string, std::string>> files;   // file parts: name, bytes
};

// Build that document. n, seed and steps become numbers when their text is all digits, and
// true_cfg_scale when it is a number, so a bad value gets the ordinary range error instead of a
// type error; a text part named image is taken as the base64 reference (some clients send it that
// way). A file part named image is base64-encoded here. More than kMaximumImageReferences
// references is 400 image_count_invalid; a mask part is 400 image_mask_not_supported (this
// pipeline has no mask input, and quietly ignoring one would draw an unmasked picture); any other
// file part name is 400 image_unsupported_file_field naming it. Absent reference is NOT an error
// here -- that is parse_openai_images_edit_request's call, with its own code.
[[nodiscard]] RequestJson edit_body_from_multipart(const MultipartEditParts& parts);

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
