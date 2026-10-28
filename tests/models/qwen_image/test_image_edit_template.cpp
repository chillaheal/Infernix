// Instruction-guided image edit (M1): the host template/geometry gate for the edit route.
//
// Gates, against oracle_edit_template.json (produced by image-edit/dump_edit_template_oracle.py
// from the GENUINE Qwen3-VL processor on the RAW ti2i template; synthetic solid plane, no photos):
//   * the engine's raw template constants and renderers are byte-identical to the pipeline's, for
//     N=1 and the N=2 user-turn shape;
//   * encoding the raw ti2i string with the engine's tokenizer and expanding every 151655 by the
//     processor's grid tokens (grid_t*h*w / merge^2) reproduces the processor's input_ids
//     token-for-token -- the expansion the C++ edit path must own;
//   * the vision-run sits exactly between <|vision_start|> (151652) and <|vision_end|> (151653),
//     and the runtime drop (system-only prefix) equals the oracle drop_idx;
//   * the production chat-template renderer STILL renders the raw t2i template byte-for-byte --
//     the certified text-to-image route must not move;
//   * the x4 invariant on the snapped grid: vision_token_count(832,1248)=1014 and 4x = the VAE's
//     4056 (oracle_ref_encode/meta.json), small plane (96,160) = 15 -> 60 (meta.json);
//   * composite_rgba_over_white == PIL's white.paste(img, mask=alpha) byte-for-byte on an 8x8
//     alpha-sweep vector.
//
// Host-only, no GPU. Skips (77) without INFERNIX_QWENVL_PROCESSOR; the oracle JSON is read from
// the source tree (NEEDS_SOURCE_DIR).

#include "models/qwen3_5/frontend/tokenizer.h"
#include "serve/image_text_tower.h"
#include "serve/reference_image.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Tokenizer = infernix::models::qwen3_5::frontend::Tokenizer;
using TokenizerResources = infernix::models::qwen3_5::frontend::TokenizerResources;

void die(const std::string& message) { std::printf("  ERROR %s\n", message.c_str()); std::exit(1); }

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-34s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { die("cannot open " + path); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// The oracle vectors are decoded with the SHIPPED decoder, not a copy of it: rgba_b64 below is a
// length class that a fixed 16/8/0 group shift decodes with a zero last byte, so grading the oracle
// against a local twin of the same decoder would have passed while the service stayed broken.
std::vector<std::uint8_t> decode_vector(const std::string& text) {
    try {
        return infernix::serve::base64_decode(text);
    } catch (const std::exception& e) {
        die(std::string("oracle base64: ") + e.what());
    }
    return {};  // unreachable: die exits
}

std::vector<int> as_ids(const json& arr) {
    std::vector<int> out;
    out.reserve(arr.size());
    for (const auto& v : arr) { out.push_back(v.get<int>()); }
    return out;
}

} // namespace

// The oracle is a repo-relative fixture, so its path comes from NEEDS_SOURCE_DIR's compiled-in source
// dir (the idiom tests/models/qwen4_exp uses), NOT an env var: gating it on the environment made the
// whole M1 suite skip 77 under a plain ctest run, which reads as green.
#ifndef INFERNIX_SOURCE_DIR
#    define INFERNIX_SOURCE_DIR "."
#endif

int main() {
    const char* proc_env = std::getenv("INFERNIX_QWENVL_PROCESSOR");
    if (!proc_env) {
        std::printf("SKIP: INFERNIX_QWENVL_PROCESSOR not set\n");
        return 77;
    }
    const std::string oracle_path =
        std::string(INFERNIX_SOURCE_DIR) + "/tests/models/qwen_image/oracle_edit_template.json";
    const json o = json::parse(read_file(oracle_path));

    const std::string prompt   = o.at("prompt").get<std::string>();
    const std::string tmpl_t2i = o.at("template_t2i").get<std::string>();
    const std::string tmpl_ti2i = o.at("template_ti2i").get<std::string>();
    const std::string ti2i_text = o.at("ti2i_text").get<std::string>();
    const std::string t2i_text  = o.at("t2i_text").get<std::string>();
    const std::vector<int> ti2i_ids = as_ids(o.at("ti2i_ids"));
    const std::vector<int> t2i_ids  = as_ids(o.at("t2i_ids"));
    const int img_id    = o.at("image_pad_id").get<int>();
    const int vs_id     = o.at("vision_start_id").get<int>();
    const int ve_id     = o.at("vision_end_id").get<int>();
    const long long drop_idx     = o.at("drop_idx").get<long long>();
    const long long vision_tokens = o.at("vision_tokens").get<long long>();
    const auto grid = o.at("image_grid_thw").get<std::vector<long long>>();
    const long long merge = o.at("merge_size").get<long long>();
    const long long plane_w = o.at("plane").at("width_px").get<long long>();
    const long long plane_h = o.at("plane").at("height_px").get<long long>();
    const long long vae_tokens = o.at("plane").at("vae_tokens").get<long long>();

    // 1) Raw template strings, byte-identical to the pipeline constants.
    {
        std::string expect_t2i = tmpl_t2i;
        const auto slot = expect_t2i.find("{}");
        if (slot == std::string::npos) { die("oracle t2i template has no {} slot"); }
        expect_t2i.replace(slot, 2, "X");
        check("raw t2i template", infernix::serve::render_t2i_template("X") == expect_t2i,
              "engine render_t2i_template matches the oracle template string byte-for-byte");
        const std::string expect_ti2i = [&] {
            const std::string block = "<image1><|vision_start|><|image_pad|><|vision_end|>";
            std::string t           = tmpl_ti2i;
            const auto at           = t.find(block);
            if (at == std::string::npos) { die("oracle ti2i template has no image block"); }
            t.replace(at, block.size(), block);  // N=1 keeps the block itself
            const auto s = t.find("{}");
            t.replace(s, 2, prompt);
            return t;
        }();
        check("raw ti2i template constant", expect_ti2i == ti2i_text,
              "oracle template formats to oracle ti2i_text (sanity of the JSON itself)");
    }
    const std::string ti2i_rendered = infernix::serve::render_ti2i_template(prompt, 1);
    check("raw ti2i render N=1", ti2i_rendered == ti2i_text,
          ti2i_rendered == ti2i_text ? "byte-identical to the processor input string"
                                     : "engine text differs from oracle ti2i_text");

    // 2) N=2 user turn: "<image1>...<|vision_end|> <image2>...<|vision_end|>" before the prompt.
    {
        const std::string two = infernix::serve::render_ti2i_template(prompt, 2);
        const std::string blocks =
            "<image1><|vision_start|><|image_pad|><|vision_end|>"
            " <image2><|vision_start|><|image_pad|><|vision_end|>";
        const std::string expect =
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n" + blocks + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        check("raw ti2i render N=2", two == expect, "numbered blocks, space-separated, literal <imageN>");
    }

    // 3) Tokenize + expand reproduces the processor's ti2i ids token-for-token.
    const std::string tok_json = read_file(std::string(proc_env) + "/tokenizer.json");
    const std::string cfg_json = read_file(std::string(proc_env) + "/tokenizer_config.json");
    const std::string gen_json = std::string("{\"eos_token_id\": 151645}");
    std::optional<Tokenizer> tokz;
    try {
        tokz.emplace(TokenizerResources{tok_json, cfg_json, gen_json});
    } catch (const std::exception& e) {
        die(std::string("tokenizer construction failed: ") + e.what());
    }

    const std::vector<int> single = tokz->encode(ti2i_rendered);
    const long long grid_tokens = grid[0] * grid[1] * grid[2] / (merge * merge);
    std::vector<int> expanded;
    expanded.reserve(single.size() + static_cast<std::size_t>(grid_tokens));
    long long pads_seen = 0;
    for (const int id : single) {
        if (id == img_id) {
            expanded.insert(expanded.end(), static_cast<std::size_t>(grid_tokens), img_id);
            ++pads_seen;
        } else {
            expanded.push_back(id);
        }
    }
    check("expansion grid tokens", grid_tokens == vision_tokens,
          "grid " + std::to_string(grid[0]) + "x" + std::to_string(grid[1]) + "x" +
          std::to_string(grid[2]) + " / merge^2 = " + std::to_string(grid_tokens));
    bool same = expanded.size() == ti2i_ids.size();
    std::size_t first_bad = 0;
    for (std::size_t i = 0; same && i < ti2i_ids.size(); ++i) {
        if (expanded[i] != ti2i_ids[i]) { same = false; first_bad = i; }
    }
    check("ti2i ids exact", same,
          same ? "engine expansion of the raw string == processor input_ids (" +
                     std::to_string(ti2i_ids.size()) + " tokens)"
               : "first diff at " + std::to_string(first_bad));

    // 4) t2i ids from the raw string match the processor.
    {
        const std::vector<int> t2i_engine = tokz->encode(t2i_text);
        bool t2i_same = t2i_engine.size() == t2i_ids.size();
        for (std::size_t i = 0; t2i_same && i < t2i_ids.size(); ++i) {
            if (t2i_engine[i] != t2i_ids[i]) { t2i_same = false; }
        }
        check("t2i ids exact", t2i_same,
              t2i_same ? "raw t2i string == processor input_ids" : "t2i ids moved");
    }

    // 5) The vision run sits between vision_start/vision_end; pad count == vision tokens.
    {
        long long count = 0;
        for (const int id : ti2i_ids) { if (id == img_id) { ++count; } }
        check("pad count == vision tokens", count == vision_tokens,
              std::to_string(count) + " pads");
        const auto first = std::find(ti2i_ids.begin(), ti2i_ids.end(), img_id);
        const bool guarded = first != ti2i_ids.begin() && *(first - 1) == vs_id &&
                             *(first + vision_tokens) == ve_id;
        check("vision_start/end guard", guarded, "run of pads bracketed by 151652/151653");
    }

    // 6) Runtime drop: the system-only prefix encodes to drop_idx and leads the full ids.
    {
        const std::string user_marker = "<|im_start|>user";
        const std::size_t user_pos = ti2i_rendered.find(user_marker);
        if (user_pos == std::string::npos) { die("rendered ti2i has no user turn"); }
        const std::vector<int> sys = tokz->encode(ti2i_rendered.substr(0, user_pos));
        bool prefix_ok = static_cast<long long>(sys.size()) == drop_idx;
        for (std::size_t i = 0; prefix_ok && i < sys.size(); ++i) {
            if (sys[i] != ti2i_ids[i]) { prefix_ok = false; }
        }
        check("drop_idx derived", prefix_ok,
              "engine prefix len " + std::to_string(sys.size()) + " oracle " + std::to_string(drop_idx) +
              " (no hardcoded literal)");
    }

    // 7) The production jinja renderer still renders the raw t2i template byte-for-byte.
    {
        auto renderer = infernix::serve::make_tower_prompt_renderer(proc_env);
        const auto tp = renderer(prompt);
        const std::string expect = infernix::serve::render_t2i_template(prompt);
        check("jinja t2i unchanged", tp.rendered == expect,
              tp.rendered == expect ? "certified t2i route untouched by the edit template work"
                                     : "production render != raw t2i template");
    }

    // 8) x4 geometry invariant on the snapped grid.
    {
        const auto v1014 = infernix::serve::vision_token_count(
            static_cast<std::uint32_t>(plane_h), static_cast<std::uint32_t>(plane_w));
        check("portrait vision count", v1014 == static_cast<std::uint64_t>(vision_tokens) && v1014 == 1014,
              std::to_string(v1014) + " == grid tokens == 1014 (1248x832)");
        check("x4 == VAE tokens", v1014 * 4 == static_cast<std::uint64_t>(vae_tokens) && vae_tokens == 4056,
              "4 * 1014 == 4056 (oracle_ref_encode meta.json)");
        const auto small = infernix::serve::vision_token_count(96, 160);
        check("small plane x4", small == 15 && small * 4 == 60,
              "160x96 (oracle_ref_encode_small meta.json): 15 vision, 60 VAE");
    }

    // 9) PIL alpha-over-white parity on an 8x8 alpha sweep.
    {
        // RFC 4648's three padding cases: the 1- and 2-byte tails are the lengths that a group shift
        // assuming four significant characters decodes as a zero last byte.
        const std::vector<std::uint8_t> m3 = decode_vector("TWFu"), m1 = decode_vector("TQ=="),
                                        m2 = decode_vector("TWE=");
        check("base64 padding cases", m3 == std::vector<std::uint8_t>{'M', 'a', 'n'} &&
                                       m1 == std::vector<std::uint8_t>{'M'} &&
                                       m2 == std::vector<std::uint8_t>{'M', 'a'},
              "\"TWFu\"/\"TQ==\"/\"TWE=\" -> Man/M/Ma");
        const std::vector<std::uint8_t> rgba = decode_vector(o.at("composite").at("rgba_b64").get<std::string>());
        const std::vector<std::uint8_t> want =
            decode_vector(o.at("composite").at("rgb_over_white_b64").get<std::string>());
        const std::size_t pixels = static_cast<std::size_t>(o.at("composite").at("width").get<long long>()) *
                                   static_cast<std::size_t>(o.at("composite").at("height").get<long long>());
        check("shipped base64 lengths", rgba.size() == pixels * 4 && want.size() == pixels * 3,
              "rgba " + std::to_string(rgba.size()) + " == 4px, rgb " + std::to_string(want.size()) +
              " == 3px (a short decode means a dropped pad byte)");
        const std::vector<std::uint8_t> got = infernix::serve::composite_rgba_over_white(rgba.data(), pixels);
        std::string comp_bad;
        for (std::size_t b = 0; b < want.size() && b < got.size(); ++b) {
            if (got[b] != want[b]) {
                comp_bad = "byte " + std::to_string(b) + " (pixel " + std::to_string(b / 3) + " channel " +
                           std::to_string(b % 3) + "): got " + std::to_string(got[b]) + ", want " +
                           std::to_string(want[b]);
                break;
            }
        }
        if (comp_bad.empty() && got.size() != want.size()) {
            comp_bad = "length: got " + std::to_string(got.size()) + ", want " + std::to_string(want.size());
        }
        check("composite over white", comp_bad.empty(),
              comp_bad.empty() ? "engine composite == PIL white.paste(mask=alpha) byte-for-byte" : comp_bad);
    }

    // 10) Record the processor contract the device path relies on.
    check("processor emitted tensors", o.at("has_pixel_values").get<bool>() &&
                                           o.at("has_mm_token_type_ids").get<bool>(),
          "pixel_values " + o.at("pixel_values_shape").dump() + " + mm_token_type_ids present");

    if (!g_ok) { std::printf("IMAGE_EDIT_TEMPLATE_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_EDIT_TEMPLATE_TEST: PASS (raw ti2i/t2i templates, processor expansion, "
                "drop, x4 geometry and alpha-over-white all match the genuine processor)\n");
    return 0;
}
