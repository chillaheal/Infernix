#pragma once

// The text tower's PromptEmbedsProvider: a prompt STRING becomes the DiT's token-major BF16
// prompt conditioning. Per call: render the processor's own chat_template.jinja for the single user
// turn, encode with the processor's genuine Qwen3-VL tokenizer, and run the production text tower
// (models/qwen_image/text_tower.h) on the yielded interlude VRAM -- the provider executes INSIDE
// the image job, exactly where the device may be touched. First use lazy-binds the artifact's text
// tower and dequantizes the FP8 embedding table ONCE (~1.2 GiB host, cached for the process). The
// provider is the seam the service documents; an unset artifact or processor dir throws on first
// use (the service 503s before then via the empty-provider configuration gate).

#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/text_tower.h"
#include "serve/image_generation_service.h"
#include "serve/reference_image.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// `TowerPrompt` (the rendered text, its templated ids, the dropped system rows) is declared in
// serve/image_generation_service.h: the service needs it for the token COUNT before it probes the
// yield, so it is part of the route's seam rather than of this module.

// The device-free prompt half of the production route: the tokenizer and chat template only, no
// device and no artifact. The service calls it before it asks whether the job fits.
// `processor_dir` holds tokenizer.json, tokenizer_config.json and chat_template.jinja (the HF
// Qwen3-VL processor directory). Lazy: the files load on the first prompt.
[[nodiscard]] PromptRendererProvider make_tower_prompt_renderer(const std::string& processor_dir);

// `image_artifact_path` is the same .infernix artifact the job executor streams (its "text"
// component supplies the tower). Takes the already-rendered prompt: the forward is the half that
// needs the yielded VRAM.
[[nodiscard]] PromptEmbedsProvider make_tower_prompt_embeds_provider(const std::string& image_artifact_path);

// The edit route's RAW template strings, byte-identical to the diffusers pipeline's constants
// (pipeline_qwenimage21.py prompt_template_t2i / prompt_template_ti2i). The pipeline formats these
// strings and hands them straight to the processor; it does NOT run them through
// chat_template.jinja -- the two tokenize differently. render_t2i_template must stay
// byte-identical to what the production chat-template renderer produces today (the certified
// t2i route); the ti2i form carries one "<image{i}><|vision_start|><|image_pad|><|vision_end|>"
// block per condition image, the literal <imageN> being plain text. Only <|image_pad|> is
// expanded by the processor (host-side reproduction gated by test_image_edit_template).
[[nodiscard]] std::string render_t2i_template(std::string_view prompt);
[[nodiscard]] std::string render_ti2i_template(std::string_view prompt, std::size_t image_count);

// ---- the edit route's host conditioning (the M4/M5 seam) ----------------------------------------
//
// A reference never reaches the DiT as pixels: the processor turns its white-composited plane into
// the vision tower's patch rows, and the tokenizer turns the template's single <|image_pad|> into
// one id per vision token. Both are pure host work that has to finish BEFORE the interlude asks for
// VRAM, so they live here rather than in the model layer, and the host gate grades them against the
// processor's own tensors (image-spike/oracle/tensors_edit_vision/gradient_*).

// What the processor's patchify does to one plane -- the shipped Qwen3VL image processor inherits
// Qwen2VLImageProcessor.patchify, verified in that file's source: rows arrive in
// (block_h, block_w, merge_h, merge_w) order over the (h/patch, w/patch) grid, and each row is
// channel-major over [channel, temporal, patch, patch] with the single frame DUPLICATED along the
// temporal axis; every value is float32 (v / 255 - mean[c]) / std[c] rounded to BF16 -- a DIVISION by
// 255 as the processor does, never a multiply by the reciprocal, which is one float32 ULP off and
// flips the BF16 rounding of ~1 value in 256. Production
// takes the constants from the artifact's own VisionDims; the explicit form lets the host gate run
// on the processor's numbers without opening the artifact.
struct VisionPatchGeometry {
    std::uint32_t patch    = 0; // vision_config.patch_size
    std::uint32_t temporal = 0; // vision_config.temporal_patch_size
    std::uint32_t merge    = 0; // vision_config.spatial_merge_size
    float mean[3]    = {0.5F, 0.5F, 0.5F}; // preprocessor_config.json image_mean
    float std_dev[3] = {0.5F, 0.5F, 0.5F}; // preprocessor_config.json image_std
};

[[nodiscard]] std::vector<std::uint16_t> pack_vision_patches(const ReferenceVisionPlane& plane,
                                                              const VisionPatchGeometry& geometry);
[[nodiscard]] std::vector<std::uint16_t> pack_vision_patches(const ReferenceVisionPlane& plane,
                                                              const models::qwen_image::VisionDims& dims);

// The processor's in-place expansion of the template's one <|image_pad|> per reference: pad #i
// becomes tokens_per_image[i] copies, in template order. Throws when the ids do not carry exactly
// one pad per reference (the x4 invariant itself is checked where the planes are built).
[[nodiscard]] std::vector<int> expand_image_pads(const std::vector<int>& ids,
                                                 const std::vector<std::uint64_t>& tokens_per_image);

// The edit route's prompt half: the RAW ti2i template for tokens_per_image.size() references,
// encoded with the genuine tokenizer and every pad expanded. drop_rows is the t2i route's: the pads
// sit after the dropped system line, so the prefix length does not move.
using ReferencePromptRendererProvider =
    std::function<TowerPrompt(const std::string& prompt, const std::vector<std::uint64_t>& tokens_per_image)>;
[[nodiscard]] ReferencePromptRendererProvider
make_tower_reference_prompt_renderer(const std::string& processor_dir);

// The edit route's device halves, counterpart to make_tower_prompt_embeds_provider. Both lazy-bind
// and share one reader, tower, embedding table and vision bind with the t2i provider for the same
// artifact path; both must run inside the interlude.
struct ConditionedPromptEmbeds {
    std::vector<std::uint16_t> prompt_embeds; // [(S - drop_rows) * hidden] BF16 bits, pre-final-norm
    std::vector<std::uint8_t> image_pad_mask; // [S - drop_rows], 1 on the rows that carry the picture
};
using VisionRowsProvider =
    std::function<models::qwen_image::ConditionedVision(const ReferenceVisionPlane& plane)>;
using ConditionedEmbedsProvider =
    std::function<ConditionedPromptEmbeds(const TowerPrompt& rendered,
                                          const models::qwen_image::ConditionedVision& vision)>;

[[nodiscard]] VisionRowsProvider make_tower_vision_rows_provider(const std::string& image_artifact_path);
[[nodiscard]] ConditionedEmbedsProvider
make_tower_conditioned_embeds_provider(const std::string& image_artifact_path);

} // namespace infernix::serve
