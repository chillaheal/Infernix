#pragma once

// The text tower's PromptEmbedsProvider: a prompt STRING becomes the DiT's token-major BF16
// prompt conditioning. Per call: render the processor's own chat_template.jinja for the single user
// turn, encode with the processor's genuine Qwen3-VL tokenizer, and run the production text tower
// (models/qwen_image/text_tower.h) on the yielded interlude VRAM -- the provider executes INSIDE
// the image job, exactly where the device may be touched. First use lazy-binds the artifact's text
// tower and dequantizes the FP8 embedding table ONCE (~1.2 GiB host, cached for the process). The
// provider is the seam the service documents; an unset artifact or processor dir throws on first
// use (the service 503s before then via the empty-provider configuration gate).

#include "serve/image_generation_service.h"

#include <cstddef>
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

} // namespace infernix::serve
