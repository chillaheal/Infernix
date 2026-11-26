// Step 8 (8C): the PRODUCTION prompt renderer gate.
//
// serve/image_text_tower.h's renderer is what POST /v1/images/generations runs for a raw prompt
// string: it renders the processor's chat_template.jinja through the engine's compiled Jinja (with
// add_generation_prompt, unlike the tokenize test's manual "{}" substitution) and encodes with the
// genuine VL tokenizer. This gate proves that production path reproduces the Qwen3-VL processor's
// oracle EXACTLY: the rendered text equals the manifest template with the prompt inserted, the ids
// equal input_ids.npy token-for-token, and the computed drop length equals the manifest drop_idx
// with the runtime slice at embed_len. Host-only, no GPU. Skips (77) without
// INFERNIX_QWENVL_PROCESSOR / INFERNIX_IMAGE_ORACLE.

#include "serve/image_text_tower.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using infernix::serve::TowerPrompt;

void die(const std::string& message) { std::printf("  ERROR %s\n", message.c_str()); std::exit(1); }

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (a.back() == '/' || a.back() == '\\') ? a + b : a + "/" + b;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { die("cannot open " + path); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Minimal int64 .npy reader (the shared qio::read_npy rejects non-float descr).
std::vector<long long> read_int64_npy(const std::string& path) {
    const std::string raw = read_file(path);
    std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
    if (bytes.size() < 13) { die("npy too small: " + path); }
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (std::memcmp(bytes.data(), magic, 6) != 0) { die("not a npy file: " + path); }
    const std::uint8_t major = bytes[6];
    std::size_t header_len = 0, data_off = 0;
    if (major == 1) {
        header_len = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8);
        data_off = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8) |
                     (static_cast<std::size_t>(bytes[10]) << 16) | (static_cast<std::size_t>(bytes[11]) << 24);
        data_off = 12 + header_len;
    } else { die("unsupported npy version: " + path); }
    if (bytes.size() < data_off) { die("npy truncated: " + path); }
    const std::string header(reinterpret_cast<const char*>(bytes.data() + (major == 1 ? 10 : 12)), header_len);
    if (header.find("'fortran_order': False") == std::string::npos) { die("npy not C-contiguous: " + path); }
    if (header.find("'<i8'") == std::string::npos) { die("npy descr is not <i8: " + path); }
    const std::size_t n = (bytes.size() - data_off) / 8;
    std::vector<long long> out(n);
    for (std::size_t i = 0; i < n; ++i) { std::memcpy(&out[i], bytes.data() + data_off + i * 8, 8); }
    return out;
}

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-28s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

void gate_one(const std::function<TowerPrompt(const std::string&)>& render, const std::string& label,
              const std::string& tower_dir) {
    const json manifest = json::parse(read_file(join(tower_dir, "manifest.json")));
    const std::string prompt     = manifest.at("prompt").get<std::string>();
    const std::string template_s = manifest.at("template").get<std::string>();
    const long long drop_idx     = manifest.at("drop_idx").get<long long>();
    const long long L            = manifest.at("L").get<long long>();
    const long long embed_len    = manifest.at("embed_len").get<long long>();

    std::string expected_text = template_s;
    const std::size_t slot = expected_text.find("{}");
    if (slot == std::string::npos) die("template has no {} placeholder in " + tower_dir);
    expected_text.replace(slot, 2, prompt);

    TowerPrompt p;
    try {
        p = render(prompt);
    } catch (const std::exception& e) {
        die(label + " production render failed: " + e.what());
    }

    check(label + " rendered text", p.rendered == expected_text,
          p.rendered == expected_text
              ? "production Jinja render equals the oracle template byte-for-byte"
              : "got " + std::to_string(p.rendered.size()) + " bytes want " +
                    std::to_string(expected_text.size()) + " (first diff at " +
                    std::to_string(p.rendered.find_first_not_of(expected_text)) + ")");

    const std::vector<long long> expected = read_int64_npy(join(tower_dir, "input_ids.npy"));
    bool identical = p.ids.size() == expected.size();
    long long first_bad = -1;
    for (std::size_t i = 0; identical && i < expected.size(); ++i) {
        if (static_cast<long long>(p.ids[i]) != expected[i]) { identical = false; first_bad = static_cast<long long>(i); }
    }
    std::string detail;
    if (identical) {
        detail = "all " + std::to_string(expected.size()) + " tokens match input_ids.npy";
    } else if (first_bad < 0) {
        detail = "length mismatch (got " + std::to_string(p.ids.size()) + " want " + std::to_string(expected.size()) + ")";
    } else {
        detail = "first mismatch at [" + std::to_string(first_bad) + "]: engine=" + std::to_string(p.ids[first_bad]) +
                 " oracle=" + std::to_string(expected[first_bad]);
    }
    check(label + " ids exact match", identical, detail);

    check(label + " drop_rows == drop_idx", static_cast<long long>(p.drop_rows) == drop_idx,
          "engine=" + std::to_string(p.drop_rows) + " oracle=" + std::to_string(drop_idx));
    check(label + " runtime slice len",
          static_cast<long long>(p.ids.size()) - static_cast<long long>(p.drop_rows) == embed_len,
          std::to_string(p.ids.size()) + " - " + std::to_string(p.drop_rows) + " want " + std::to_string(embed_len));
}

} // namespace

int main() {
    const char* proc_env   = std::getenv("INFERNIX_QWENVL_PROCESSOR");
    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (!proc_env || !oracle_env) {
        std::printf("SKIP: INFERNIX_QWENVL_PROCESSOR / INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }

    const auto render = infernix::serve::make_tower_prompt_renderer(proc_env);
    gate_one(render, "ascii", join(oracle_env, "tensors_text_tower"));
    gate_one(render, "marks", join(oracle_env, "tensors_text_tower_marks"));

    if (!g_ok) { std::printf("IMAGE_TOWER_PROMPT_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_TOWER_PROMPT_TEST: PASS (production chat-template render + encode reproduces "
                "the Qwen3-VL processor oracle text and ids exactly)\n");
    return 0;
}
