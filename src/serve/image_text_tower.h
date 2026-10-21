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

#include <functional>
#include <string>
#include <vector>

namespace infernix::serve {

// The device-free prompt half of the production route, shared with the provider: the exact rendered
// text the tokenizer sees, its full templated ids (causal), and the leading system-prefix rows the
// DiT lane drops. Exposed so the host gate proves the production rendering and tokenization against
// the oracle without touching the device.
struct TowerPrompt {
    std::string rendered;
    std::vector<int> ids;
    std::size_t drop_rows = 0;
};

// `processor_dir` holds tokenizer.json, tokenizer_config.json and chat_template.jinja (the HF
// Qwen3-VL processor directory). Lazy: the files load on the first prompt.
[[nodiscard]] std::function<TowerPrompt(const std::string&)> make_tower_prompt_renderer(const std::string& processor_dir);

// `image_artifact_path` is the same .infernix artifact the job executor streams (its "text"
// component supplies the tower).
[[nodiscard]] PromptEmbedsProvider make_tower_prompt_embeds_provider(const std::string& image_artifact_path,
                                                                     const std::string& processor_dir);

} // namespace infernix::serve
