#include "serve/image_text_tower.h"

#include "artifact/reader.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/text_tower.h"
#include "models/qwen_image/vision_tower.h"
#include "ops/common/canonical_math.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace infernix::serve {
namespace {

namespace qi = models::qwen_image;
namespace fi = models::qwen3_5::frontend;

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("image text tower: cannot read " + path); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string join(const std::string& dir, const std::string& name) {
    if (dir.empty()) { throw std::runtime_error("image text tower: empty processor directory"); }
    return (dir.back() == '/' || dir.back() == '\\') ? dir + name : dir + "/" + name;
}

// The Qwen-Image text-conditioning route ALWAYS prepends this fixed system turn (the HF pipeline
// passes it as the system message); the tower then drops exactly its tokenized prefix. The prompt
// renderer gate proves the rendered conversation reproduces the processor oracle byte-for-byte.
constexpr const char* kImageSysPrompt = "Comprehend and analyze the provided prompt.";

// The production prompt half: the processor's own chat template and genuine Qwen3-VL tokenizer.
// The drop prefix is prompt-independent (system turn only), so its token count is computed once
// (the tokenization gate proves it equals the oracle drop length and leads the full encode).
class PromptRenderer {
public:
    explicit PromptRenderer(std::string processor_dir) : processor_(std::move(processor_dir)) {}

    TowerPrompt operator()(const std::string& prompt) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        std::vector<fi::ChatMessage> messages(2);
        messages[0].role       = ChatRole::System;
        messages[0].parts.push_back(fi::ChatPart::text_part(kImageSysPrompt));
        messages[1].role       = ChatRole::User;
        messages[1].parts.push_back(fi::ChatPart::text_part(prompt));
        TowerPrompt out;
        out.rendered = template_->render(messages).text;
        out.ids      = tokenizer_->encode(out.rendered);
        if (drop_rows_ == 0) {
            const std::size_t user_pos = out.rendered.find("<|im_start|>user");
            if (user_pos == std::string::npos) {
                throw std::runtime_error("image text tower: the rendered prompt has no user turn");
            }
            drop_rows_ = tokenizer_->encode(out.rendered.substr(0, user_pos)).size();
        }
        out.drop_rows = drop_rows_locked();
        return out;
    }

    // The edit route's prompt: the pipeline's RAW ti2i template (NOT chat_template.jinja -- the two
    // tokenize differently, which is exactly what the M1 gate measures) encoded with the genuine
    // tokenizer, with every <|image_pad|> expanded to its own reference's vision-token count. The
    // expansion is the processor's own: it replaces the single placeholder per image in place.
    TowerPrompt reference(const std::string& prompt, const std::vector<std::uint64_t>& tokens_per_image) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        TowerPrompt out;
        out.rendered  = render_ti2i_template(prompt, tokens_per_image.size());
        out.ids       = expand_image_pads(tokenizer_->encode(out.rendered), tokens_per_image);
        out.drop_rows = drop_rows_locked();
        return out;
    }

private:
    // The number of leading rows the tower drops: the tokenized system line. Prompt-independent, so
    // measured once from an empty-user render -- the same prefix the certified t2i route measured on
    // its own string, which the tokenization gate pins to the oracle's drop length.
    std::size_t drop_rows_locked() {
        if (drop_rows_ != 0) { return drop_rows_; }
        std::vector<fi::ChatMessage> probe(2);
        probe[0].role       = ChatRole::System;
        probe[0].parts.push_back(fi::ChatPart::text_part(kImageSysPrompt));
        probe[1].role       = ChatRole::User;
        probe[1].parts.push_back(fi::ChatPart::text_part(""));
        const std::string text    = template_->render(probe).text;
        const std::size_t user_pos = text.find("<|im_start|>user");
        if (user_pos == std::string::npos) {
            throw std::runtime_error("image text tower: the rendered prompt has no user turn");
        }
        drop_rows_ = tokenizer_->encode(text.substr(0, user_pos)).size();
        return drop_rows_;
    }

    void ensure_locked() {
        if (tokenizer_ != nullptr) { return; }
        const std::string tok_json = read_file(join(processor_, "tokenizer.json"));
        const std::string cfg_json = read_file(join(processor_, "tokenizer_config.json"));
        const std::string gen_json = std::string("{\"eos_token_id\": 151645}"); // encode() never reads stop ids
        tokenizer_ = std::make_unique<fi::Tokenizer>(fi::TokenizerResources{tok_json, cfg_json, gen_json});

        // The special tokens tokenizer_config.json declares, the same set the LLM frontend hands
        // to resolve.
        nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
        const auto config             = nlohmann::json::parse(cfg_json);
        for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                                "cls_token", "mask_token"}) {
            if (!config.contains(key) || config[key].is_null()) continue;
            const auto& value = config[key];
            if (value.is_string())
                tokens[key] = value;
            else if (value.is_object() && value.contains("content"))
                tokens[key] = value.at("content");
        }
        if (config.contains("additional_special_tokens"))
            tokens["additional_special_tokens"] = config["additional_special_tokens"];
        template_ = fi::CompiledChatTemplate::resolve(read_file(join(processor_, "chat_template.jinja")),
                                                      "processor:chat_template.jinja", std::move(tokens));
    }

    std::string processor_;
    std::mutex mutex_;
    std::unique_ptr<fi::Tokenizer> tokenizer_;
    std::optional<fi::CompiledChatTemplate> template_;
    std::size_t drop_rows_ = 0;
};

// The device half of the tower route: an ALREADY RENDERED prompt (the service rendered it on the host,
// before it probed the yield) becomes the DiT's conditioning. The forward is what needs device memory,
// so this is the half that runs inside the interlude.
class TowerProvider {
public:
    explicit TowerProvider(std::string artifact_path) : path_(std::move(artifact_path)) {}

    std::vector<std::uint16_t> operator()(const TowerPrompt& rendered) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        return qi::run_text_tower(*reader_, tower_, embed_table_, rendered.ids, rendered.drop_rows);
    }

    // The reference plane -> the merger rows that condition the text tower (and the deepstack taps it
    // adds onto them). The pack is host work; only the forward touches the device, and the caller
    // (the service) has already yielded VRAM by the time this runs. Vision binds on first use so a
    // text-only artifact -- one without the vision/* names -- can never break the certified t2i
    // route: it never gets here.
    qi::ConditionedVision run_vision(const ReferenceVisionPlane& plane) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        if (vision_.dims.depth == 0) {
            auto vision_binder = std::make_unique<artifact::Binder>(*reader_);
            vision_            = qi::bind_image_vision(*vision_binder);
            (void)std::move(*vision_binder).finish();
        }
        const std::vector<std::uint16_t> patches = pack_vision_patches(plane, vision_.dims);
        const qi::VisionItemGeometry grid{1U, plane.height_px / vision_.dims.patch,
                                          plane.width_px / vision_.dims.patch};
        qi::VisionTowerOutput rows = qi::run_vision_tower(*reader_, vision_, patches, grid);
        qi::ConditionedVision out;
        out.merger_rows    = std::move(rows.merger_rows);
        out.deepstack_rows = std::move(rows.deepstack_rows);
        // The same grid the ViT just consumed travels with the rows: the text tower's M-RoPE places
        // the picture's rows in three axes from it, and an out-of-band copy could silently disagree
        // with the pixels that produced them.
        out.grids = {std::array<int, 3>{static_cast<int>(grid.temporal), static_cast<int>(grid.height),
                                        static_cast<int>(grid.width)}};
        out.merge = vision_.dims.merge;
        return out;
    }

    // The rendered prompt + the merger rows -> the DiT conditioning and the mask of the kept rows
    // that carry the picture. The tower entry is the same one the device-qualified M4 gate runs.
    ConditionedPromptEmbeds run_conditioned(const TowerPrompt& rendered, const qi::ConditionedVision& vision) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        const qi::ConditionedTextTowerOutput out = qi::run_conditioned_text_tower(
            *reader_, tower_, embed_table_, rendered.ids, rendered.drop_rows, vision);
        return ConditionedPromptEmbeds{std::move(out.prompt_embeds), std::move(out.image_pad_mask)};
    }

private:
    void ensure_locked() {
        if (reader_ != nullptr) { return; }
        if (path_.empty()) { throw std::runtime_error("image text tower: no image artifact is configured"); }
        reader_ = std::make_unique<artifact::Reader>(path_);
        binder_ = std::make_unique<artifact::Binder>(*reader_);
        tower_  = qi::bind_text_encoder(*binder_);
        (void)std::move(*binder_).finish();
        dequantize_embed_table();
    }

    // The FP8 RowScale embedding table decoded once into a BF16 host cache. The decode itself is the
    // model module's (image_weights.h dequantize_embed_table) so the device-qualified tower test gates
    // the very bytes this route feeds the gather.
    void dequantize_embed_table() { embed_table_ = qi::dequantize_embed_table(*reader_, tower_); }

    std::string path_;
    std::mutex mutex_;
    std::unique_ptr<artifact::Reader> reader_;
    std::unique_ptr<artifact::Binder> binder_;
    qi::TextTowerWeights tower_;
    qi::VisionWeights vision_; // bound lazily by run_vision (dims.depth == 0 means unbound)
    std::vector<std::byte> embed_table_;
};

} // namespace

namespace {

// The pipeline's raw template constants (pipeline_qwenimage21.py lines 217-226), "{}" marking the
// prompt slot and the ti2i block being the single expandable placeholder.
constexpr const char* kTemplateT2i =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n";
constexpr const char* kTi2iImageBlock = "<image1><|vision_start|><|image_pad|><|vision_end|>";
constexpr const char* kTemplateTi2i =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n";

std::string format_into_template(std::string_view tmpl, std::string_view prompt) {
    const std::size_t slot = tmpl.find("{}");
    if (slot == std::string_view::npos) { throw std::runtime_error("image template has no prompt slot"); }
    std::string out;
    out.reserve(tmpl.size() + prompt.size());
    out.append(tmpl, 0, slot);
    out.append(prompt);
    out.append(tmpl, slot + 2, tmpl.size() - slot - 2);
    return out;
}

} // namespace

std::string render_t2i_template(std::string_view prompt) { return format_into_template(kTemplateT2i, prompt); }

std::string render_ti2i_template(std::string_view prompt, std::size_t image_count) {
    if (image_count == 0) { throw std::runtime_error("ti2i template needs at least one image"); }
    // The pipeline replaces the single placeholder block with one block per image, numbered,
    // space-separated (pipeline_qwenimage21.py lines 259-264).
    std::string replace = kTi2iImageBlock;
    for (std::size_t i = 2; i <= image_count; ++i) {
        replace += " <image" + std::to_string(i) + "><|vision_start|><|image_pad|><|vision_end|>";
    }
    const std::string_view tmpl{kTemplateTi2i};
    const std::size_t at = tmpl.find(kTi2iImageBlock);
    if (at == std::string_view::npos) { throw std::runtime_error("ti2i template has no image block"); }
    std::string merged;
    merged.reserve(tmpl.size() + replace.size());
    merged.append(tmpl, 0, at);
    merged.append(replace);
    merged.append(tmpl, at + std::strlen(kTi2iImageBlock), tmpl.size() - at - std::strlen(kTi2iImageBlock));
    return format_into_template(merged, prompt);
}

// ---- the edit route's host conditioning (M4/M5 seam) --------------------------------------------

// One white-composited plane -> the vision tower's patch rows, reproducing the shipped processor
// exactly. The layout is copied from Qwen2VLImageProcessor.patchify (image-spike/gpu_diffusers/
// transformers/models/qwen2_vl/image_processing_qwen2_vl.py:165-198 -- the class Qwen3VLProcessor
// actually instantiates for images): reshape [B, C, gh/merge, merge, patch, gw/merge, merge, patch],
// permute to [B, gh/merge, gw/merge, merge_h, merge_w, C, patch, patch], unsqueeze+expand a
// temporal axis of `temporal` (the SINGLE frame duplicated), flatten to
// [gh*gw, C*temporal*patch*patch]. So row (r) walks (block_h, block_w, merge_h, merge_w) and column
// j walks (channel, frame, py, px). The value is float32 (v * rescale - mean[c]) / std[c] in that
// operation order, then BF16-rounded -- the same arithmetic the processor performs, so a mismatch
// the host gate reports is a real ordering or normalisation bug and never rounding.
std::vector<std::uint16_t> pack_vision_patches(const ReferenceVisionPlane& plane,
                                               const VisionPatchGeometry& geometry) {
    const std::uint32_t patch  = geometry.patch;
    const std::uint32_t merge  = geometry.merge;
    const std::uint32_t frames = geometry.temporal;
    if (patch == 0U || merge == 0U || frames == 0U) {
        throw std::runtime_error("vision pack: the patch geometry is not declared");
    }
    const std::size_t pixels = static_cast<std::size_t>(plane.height_px) * plane.width_px;
    if (plane.height_px == 0U || plane.width_px == 0U || plane.rgb.size() != pixels * 3ULL) {
        throw std::runtime_error("vision pack: the reference plane is not RGB8 of its own size");
    }
    const std::uint32_t step = patch * merge; // the processor's resize factor
    if (plane.height_px % step != 0U || plane.width_px % step != 0U) {
        throw std::runtime_error("vision pack: the reference plane is not on the processor's " +
                                 std::to_string(step) + "-pixel grid");
    }
    const std::size_t grid_h = plane.height_px / patch, grid_w = plane.width_px / patch;
    const std::size_t row_width = 3ULL * frames * patch * patch;
    std::vector<std::uint16_t> out(grid_h * grid_w * row_width);

    // The processor's rescale is a DIVISION by 255.0, not a multiplication by the pre-rounded
    // reciprocal: float32 (1.0F/255.0F) is not 1/255, so multiplying differs from dividing by one
    // float32 ULP, and ~1 value in 256 sits close enough to a BF16 midpoint for that ULP to flip the
    // rounding. The host gate (test_image_edit_template) measures exactly this: dividing matches the
    // processor's pixel_values bit-for-bit on all 718848 values of the gradient plane, multiplying
    // misses 2368 of them.
    const float rescale_divisor = 255.0F;
    std::size_t row = 0;
    for (std::size_t block_h = 0; block_h < grid_h / merge; ++block_h) {
        for (std::size_t block_w = 0; block_w < grid_w / merge; ++block_w) {
            for (std::size_t merge_h = 0; merge_h < merge; ++merge_h) {
                for (std::size_t merge_w = 0; merge_w < merge; ++merge_w, ++row) {
                    const std::size_t y0 = (block_h * merge + merge_h) * patch;
                    const std::size_t x0 = (block_w * merge + merge_w) * patch;
                    std::uint16_t* dst   = out.data() + row * row_width;
                    std::size_t col      = 0;
                    for (std::size_t channel = 0; channel < 3ULL; ++channel) {
                        for (std::size_t frame = 0; frame < frames; ++frame) {
                            (void)frame; // duplicated axis: the same pixels every time
                            for (std::size_t py = 0; py < patch; ++py) {
                                for (std::size_t px = 0; px < patch; ++px, ++col) {
                                    const float v = static_cast<float>(
                                        plane.rgb[((y0 + py) * plane.width_px + (x0 + px)) * 3ULL + channel]);
                                    dst[col] = ops::canon::f32_to_bf16_rn(
                                        (v / rescale_divisor - geometry.mean[channel]) / geometry.std_dev[channel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

std::vector<std::uint16_t> pack_vision_patches(const ReferenceVisionPlane& plane,
                                               const models::qwen_image::VisionDims& dims) {
    if (dims.patch_width != 3U * dims.temporal * dims.patch * dims.patch) {
        throw std::runtime_error("vision pack: the artifact's patch_width is not 3*temporal*patch*patch");
    }
    VisionPatchGeometry geometry;
    geometry.patch    = dims.patch;
    geometry.temporal = dims.temporal;
    geometry.merge    = dims.merge;
    return pack_vision_patches(plane, geometry);
}

std::vector<int> expand_image_pads(const std::vector<int>& ids, const std::vector<std::uint64_t>& tokens_per_image) {
    for (const std::uint64_t count : tokens_per_image) {
        if (count == 0ULL) {
            throw std::runtime_error("image pad expansion: a reference produced no vision tokens");
        }
    }
    std::vector<int> out;
    out.reserve(ids.size());
    std::size_t seen = 0;
    for (const int id : ids) {
        if (static_cast<std::uint32_t>(id) != qi::kImagePadTokenId) {
            out.push_back(id);
            continue;
        }
        if (seen >= tokens_per_image.size()) {
            throw std::runtime_error("image pad expansion: more <|image_pad|> tokens than reference images");
        }
        out.insert(out.end(), static_cast<std::size_t>(tokens_per_image[seen]),
                   static_cast<int>(qi::kImagePadTokenId));
        ++seen;
    }
    if (seen != tokens_per_image.size()) {
        throw std::runtime_error("image pad expansion: the template carries " + std::to_string(seen) +
                                 " <|image_pad|> tokens for " + std::to_string(tokens_per_image.size()) +
                                 " reference images");
    }
    return out;
}

// One bound tower per artifact path, shared by every provider of that path: the FP8 embedding table
// it caches is ~1.2 GiB of host memory and the vision bind reads the same names, so a second copy
// for the edit route would be pure waste. Weak entries so an unused route does not pin the mapping.
std::shared_ptr<TowerProvider> shared_tower(const std::string& path) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<TowerProvider>> registry;
    const std::lock_guard lock(registry_mutex);
    for (auto it = registry.begin(); it != registry.end();) {
        if (it->second.expired()) {
            it = registry.erase(it);
        } else {
            ++it;
        }
    }
    if (auto found = registry.find(path); found != registry.end()) {
        if (auto state = found->second.lock()) { return state; }
    }
    auto state = std::make_shared<TowerProvider>(path);
    registry.emplace(path, state);
    return state;
}

PromptRendererProvider make_tower_prompt_renderer(const std::string& processor_dir) {
    auto renderer = std::make_shared<PromptRenderer>(processor_dir);
    return [renderer](const std::string& prompt) { return (*renderer)(prompt); };
}

ReferencePromptRendererProvider make_tower_reference_prompt_renderer(const std::string& processor_dir) {
    auto renderer = std::make_shared<PromptRenderer>(processor_dir);
    return [renderer](const std::string& prompt, const std::vector<std::uint64_t>& tokens_per_image) {
        return renderer->reference(prompt, tokens_per_image);
    };
}

PromptEmbedsProvider make_tower_prompt_embeds_provider(const std::string& image_artifact_path) {
    auto state = shared_tower(image_artifact_path);
    return [state](const TowerPrompt& rendered) { return (*state)(rendered); };
}

VisionRowsProvider make_tower_vision_rows_provider(const std::string& image_artifact_path) {
    auto state = shared_tower(image_artifact_path);
    return [state](const ReferenceVisionPlane& plane) { return state->run_vision(plane); };
}

ConditionedEmbedsProvider make_tower_conditioned_embeds_provider(const std::string& image_artifact_path) {
    auto state = shared_tower(image_artifact_path);
    return [state](const TowerPrompt& rendered, const models::qwen_image::ConditionedVision& vision) {
        return state->run_conditioned(rendered, vision);
    };
}

} // namespace infernix::serve
