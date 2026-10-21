// Host loader gate for the Qwen-Image-2.1 production weights (task #13 step 2): bind_dit and
// bind_text_encoder must resolve EVERY logical parameter the converter recipe names (tools/
// convert/qwen_image.py) against the REAL artifact's Directory bindings, at shapes derived from the
// artifact's own component config, at the recipe's exact representations (the 17 quantized role
// families FP8 E4M3 RowScale, everything else direct BF16), and the combined Binder must finish to a
// MaterializationPlan with ZERO device and pinned bytes: every image weight streams. The dims are
// asserted against the qualified pipeline's architecture (32 blocks x 32 heads x 128 = 4096, MLP 3x,
// 64-dim packed image io, context 4096, eps 1e-6; 36-layer Qwen3-VL tower GQA 32/8) -- an artifact
// that no longer matches the route the ops were qualified on must fail HERE, not at oracle time.
// Malformed demands are negative-tested: an unknown parameter name, a wrong shape, or a wrong
// representation each throws before any device work. Skips (77) without INFERNIX_TEST_ARTIFACT /
// when absent. No GPU, no server.
#include "artifact/reader.h"
#include "models/qwen_image/image_weights.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::models::qwen_image;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_eq(std::uint64_t actual, std::uint64_t expected, const std::string& what) {
    if (actual != expected) {
        throw std::runtime_error(what + ": got " + std::to_string(actual) + ", expected " +
                                std::to_string(expected));
    }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& what) {
    bool thrown = false;
    try {
        fn();
    } catch (const ArtifactError&) {
        thrown = true;
    }
    require(thrown, "malformed demand accepted: " + what);
}

std::uint64_t object_bytes(const Binder& binder, const ParameterReference& ref) {
    std::uint64_t total = 0;
    for (const auto& part : ref.binding.parts) { total += binder.reader().geometry(part.object).bytes; }
    return total;
}

QType object_format(const Binder& binder, const ParameterReference& ref) {
    return binder.reader().geometry(ref.binding.parts.front().object).format;
}

void require_role(const Binder& binder, const ParameterReference& ref, QType format, const std::string& what) {
    require(ref.residency == Residency::Streamed, what + ": role is not Streamed");
    require(object_format(binder, ref) == format, what + ": role representation differs from the recipe");
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("INFERNIX_TEST_ARTIFACT");
        if (env == nullptr || std::string_view(env).empty()) {
            std::cout << "INFERNIX_TEST_ARTIFACT unset; skipping weight-loader gate\n";
            return 77;
        }
        const std::filesystem::path path(env);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "artifact absent; skipping weight-loader gate: " << env << "\n";
            return 77;
        }

        const Reader reader(path);
        Binder binder(reader);

        const DiTWeights dit = bind_dit(binder);
        require_eq(dit.dims.layers, 32, "dit layers");
        require_eq(dit.dims.heads, 32, "dit heads");
        require_eq(dit.dims.head_dim, 128, "dit head_dim");
        require_eq(dit.dims.hidden, 4096, "dit hidden");
        require_eq(dit.dims.intermediate, 12288, "dit intermediate (mlp_ratio 3)");
        require_eq(dit.dims.image_in, 64, "dit image_in");
        require_eq(dit.dims.image_out, 64, "dit image_out");
        require_eq(dit.dims.context_in, 4096, "dit context_in");
        require(dit.dims.eps == 1e-6F, "dit eps differs from 1e-6");
        require(dit.blocks.size() == 32, "bound block count differs from the config");

        require_role(binder, dit.img_in, QType::BF16, "dit/img_in");
        require_role(binder, dit.proj_out, QType::BF16, "dit/proj_out");
        require_role(binder, dit.modulation, QType::BF16, "dit/modulation");
        require_role(binder, dit.norm_out, QType::BF16, "dit/norm_out");
        require_role(binder, dit.txt_in_in, QType::FP8_E4M3FN_ROW_BF16, "dit/txt_in/in");
        require_role(binder, dit.txt_in_out, QType::FP8_E4M3FN_ROW_BF16, "dit/txt_in/out");
        require_role(binder, dit.txt_in_norm, QType::BF16, "dit/txt_in/norm");
        require_role(binder, dit.time_linear_1, QType::BF16, "dit/time_embed/linear_1");
        require_role(binder, dit.time_linear_2, QType::BF16, "dit/time_embed/linear_2");
        for (std::size_t l = 0; l < dit.blocks.size(); ++l) {
            const auto& b = dit.blocks[l];
            const std::string w = "block " + std::to_string(l);
            require_role(binder, b.q, QType::FP8_E4M3FN_ROW_BF16, w + " q");
            require_role(binder, b.k, QType::FP8_E4M3FN_ROW_BF16, w + " k");
            require_role(binder, b.v, QType::FP8_E4M3FN_ROW_BF16, w + " v");
            require_role(binder, b.query_norm, QType::BF16, w + " query_norm");
            require_role(binder, b.key_norm, QType::BF16, w + " key_norm");
            require_role(binder, b.attention_output, QType::FP8_E4M3FN_ROW_BF16, w + " attn out");
            require_role(binder, b.mlp_gate, QType::FP8_E4M3FN_ROW_BF16, w + " gate");
            require_role(binder, b.mlp_up, QType::FP8_E4M3FN_ROW_BF16, w + " up");
            require_role(binder, b.mlp_down, QType::FP8_E4M3FN_ROW_BF16, w + " down");
        }
        // The recipe's FP8 framing, a priori: one block's largest role is a 12288 x 4096 code plane
        // plus one BF16 scale per row (the stream-host test proves the framing on every object).
        require_eq(dit.max_block_object_bytes, 12288ULL * 4096 + 12288 * 2, "block streaming bound");

        const TextTowerWeights text = bind_text_encoder(binder);
        require_eq(text.dims.layers, 36, "text layers");
        require_eq(text.dims.hidden, 4096, "text hidden");
        require_eq(text.dims.heads, 32, "text heads");
        require_eq(text.dims.kv_heads, 8, "text kv heads");
        require_eq(text.dims.head_dim, 128, "text head_dim");
        require_eq(text.dims.intermediate, 12288, "text intermediate");
        require_eq(text.dims.vocab, 151936, "text vocab");
        require(text.dims.rope_theta == 5000000.0F, "text rope_theta differs from 5e6");
        require(text.dims.eps == 1e-6F, "text eps differs from 1e-6");
        require(text.layers.size() == 36, "bound text layer count differs from the config");
        require_role(binder, text.embed_tokens, QType::FP8_E4M3FN_ROW_BF16, "text/embed_tokens");
        require_role(binder, text.final_norm, QType::BF16, "text/final_norm");
        for (std::size_t l = 0; l < text.layers.size(); ++l) {
            const auto& r = text.layers[l];
            const std::string w = "text layer " + std::to_string(l);
            require_role(binder, r.input_norm, QType::BF16, w + " input_norm");
            require_role(binder, r.post_attention_norm, QType::BF16, w + " post_attention_norm");
            require_role(binder, r.q, QType::FP8_E4M3FN_ROW_BF16, w + " q");
            require_role(binder, r.k, QType::FP8_E4M3FN_ROW_BF16, w + " k");
            require_role(binder, r.v, QType::FP8_E4M3FN_ROW_BF16, w + " v");
            require_role(binder, r.query_norm, QType::BF16, w + " query_norm");
            require_role(binder, r.key_norm, QType::BF16, w + " key_norm");
            require_role(binder, r.attention_output, QType::FP8_E4M3FN_ROW_BF16, w + " attn out");
            require_role(binder, r.mlp_gate, QType::FP8_E4M3FN_ROW_BF16, w + " gate");
            require_role(binder, r.mlp_up, QType::FP8_E4M3FN_ROW_BF16, w + " up");
            require_role(binder, r.mlp_down, QType::FP8_E4M3FN_ROW_BF16, w + " down");
        }
        // The whole artifact's largest single object is the FP8 embedding table (the streaming bound
        // the host-staging tests measured at ~594 MiB): codes + one BF16 scale per row.
        require_eq(object_bytes(binder, text.embed_tokens), 151936ULL * 4096 + 151936 * 2,
                   "embed_tokens RowScale bytes");
        require(text.max_object_bytes == object_bytes(binder, text.embed_tokens),
                "tower streaming bound is not the embedding table");

        // Malformed demands must be refused at open time, each on its own Binder.
        require_throws([&] { (void)Binder(reader).parameter("dit/layers/32/attention/query", {4096, 4096},
                                                           Residency::Streamed); },
                       "unknown parameter name");
        require_throws([&] { (void)Binder(reader).parameter("dit/img_in", {4097, 64}, Residency::Streamed,
                                                           QType::BF16); },
                       "wrong shape");
        require_throws([&] { (void)Binder(reader).parameter("dit/modulation", {4 * 4096ULL, 4096},
                                                           Residency::Streamed, QType::FP8_E4M3FN_ROW_BF16); },
                       "BF16 role demanded as FP8");
        require_throws([&] { (void)Binder(reader).parameter("dit/layers/0/attention/query", {4096, 4096},
                                                           Residency::Streamed, QType::BF16); },
                       "FP8 role demanded as BF16");

        // Everything streams: the combined plan places nothing on device and pins nothing.
        const MaterializationPlan plan = std::move(binder).finish();
        require(plan.source == &reader, "plan source is not the artifact Reader");
        require(plan.device_objects.empty() && plan.device_capacity_bytes == 0,
                "image weight plan places bytes on device");
        require(plan.pinned_objects.empty() && plan.pinned_capacity_bytes == 0,
                "image weight plan pins host bytes");
        // 9 shared + 32 x 9 block DiT roles and 2 shared + 36 x 11 tower roles, one object each.
        require_eq(plan.streamed_objects.size(), 9 + 32 * 9 + 2 + 36 * 11, "streamed object count");

        std::cout << "image weight-loader gate passed: dit blocks=" << dit.blocks.size()
                  << " hidden=" << dit.dims.hidden << " intermediate=" << dit.dims.intermediate
                  << " block_max=" << dit.max_block_object_bytes << " text layers=" << text.layers.size()
                  << " embed=" << text.max_object_bytes << " streamed=" << plan.streamed_objects.size()
                  << " device=0 pinned=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "image weight-loader gate FAILED: " << error.what() << "\n";
        return 1;
    }
}
