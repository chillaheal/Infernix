#include "serve/image_text_tower.h"

#include "artifact/reader.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/text_tower.h"
#include "ops/common/canonical_math.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
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
        out.drop_rows = drop_rows_;
        return out;
    }

private:
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

    // The FP8 RowScale embedding table decoded once into a BF16 host cache: value = e4m3 code * the
    // row's BF16 scale, rounded to BF16 (the production route's quantization error, disclosed with
    // the tower's device gate).
    void dequantize_embed_table() {
        const artifact::ParameterReference& ref = tower_.embed_tokens;
        if (ref.binding.parts.size() != 1) {
            throw std::runtime_error("image text tower: embed_tokens binds more than one object");
        }
        const auto object = ref.binding.parts.front().object;
        const WeightGeometry& g    = reader_->geometry(object);
        const std::size_t vocab    = static_cast<std::size_t>(tower_.dims.vocab);
        const std::size_t hidden   = static_cast<std::size_t>(tower_.dims.hidden);
        if (g.format != QType::FP8_E4M3FN_ROW_BF16 || g.layout != QuantLayout::RowScale ||
            g.code_bytes != vocab * hidden || g.scale_bytes != vocab * 2 ||
            g.scale_offset < g.code_bytes) {
            throw std::runtime_error("image text tower: embed_tokens is not a plain [codes][scales] FP8 plane");
        }
        const std::vector<std::byte> obj = reader_->read_object(object);
        if (obj.size() < g.scale_offset + g.scale_bytes) {
            throw std::runtime_error("image text tower: embed_tokens object is shorter than its geometry");
        }
        const auto* codes  = reinterpret_cast<const std::uint8_t*>(obj.data());
        const auto* scales = reinterpret_cast<const std::uint16_t*>(obj.data() + g.scale_offset);
        embed_table_.resize(vocab * hidden * 2);
        auto* table = reinterpret_cast<std::uint16_t*>(embed_table_.data());
        for (std::size_t row = 0; row < vocab; ++row) {
            const float scale = ops::canon::bf16_to_f32(scales[row]);
            for (std::size_t col = 0; col < hidden; ++col) {
                const std::uint8_t byte = codes[row * hidden + col];
                const float value       = ops::canon::e4m3_value(byte & 0x7FU) *
                                          ((byte & 0x80U) ? -scale : scale);
                table[row * hidden + col] = ops::canon::f32_to_bf16_rn(value);
            }
        }
    }

    std::string path_;
    std::mutex mutex_;
    std::unique_ptr<artifact::Reader> reader_;
    std::unique_ptr<artifact::Binder> binder_;
    qi::TextTowerWeights tower_;
    std::vector<std::byte> embed_table_;
};

} // namespace

PromptRendererProvider make_tower_prompt_renderer(const std::string& processor_dir) {
    auto renderer = std::make_shared<PromptRenderer>(processor_dir);
    return [renderer](const std::string& prompt) { return (*renderer)(prompt); };
}

PromptEmbedsProvider make_tower_prompt_embeds_provider(const std::string& image_artifact_path) {
    auto state = std::make_shared<TowerProvider>(image_artifact_path);
    return [state](const TowerPrompt& rendered) { return (*state)(rendered); };
}

} // namespace infernix::serve
