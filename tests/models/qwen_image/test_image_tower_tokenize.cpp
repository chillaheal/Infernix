// Variable-input image generation (#37): the host-side text-tower tokenization gate.
//
// Loads the GENUINE Qwen3-VL text-processor tokenizer.json / tokenizer_config.json into the engine's
// Qwen tokenizer and gates that encode() reproduces the Qwen3-VL processor's input_ids token-for-
// token for the oracle t2i template -- for BOTH the ASCII "a black cat" oracle (tensors_text_tower/)
// and a Devanagari prompt whose Unicode Marks survive NFC (tensors_text_tower_marks/). This is the
// host tokenize slice the text-tower input depends on; the tower forward is qualified on-device
// (test_image_texttower_prod_cuda.cu). Prompt/template/L/drop_idx/embed_len are read from each
// oracle's manifest.json, so there is no transcription.
//
// WHY THE GENUINE FILE NOW LOADS (the change this test protects): validate_pipeline previously
// required the loaded pre_tokenizer regex to equal the engine's mark-INCLUSIVE kQwenSplitPattern and
// rejected the Qwen3-VL mark-EXCLUSIVE descriptor -- construction threw, so the earlier version of
// this test had to substitute the mark-inclusive form. validate_pipeline now accepts both forms.
// That is not a loosened gate: the two are BEHAVIORALLY IDENTICAL for the Qwen vocab -- 0 of 151669
// merge tokens contain a combining mark (the corpus was NFC-normalized before BPE training), so a
// mark always falls to isolated byte-level tokens and the two word-groupings yield identical ids for
// every input. The marks oracle exercises exactly such a mark-containing prompt against the
// processor's own ids, so a regression that re-narrows the guard (or breaks mark handling) fails.
//
// Host-only, no GPU. Skips (77) without INFERNIX_QWENVL_PROCESSOR (the processor dir holding
// tokenizer.json / tokenizer_config.json) or INFERNIX_IMAGE_ORACLE (holding the oracle dirs).

#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using json = nlohmann::json;
using Tokenizer = infernix::models::qwen3_5::frontend::Tokenizer;
using TokenizerResources = infernix::models::qwen3_5::frontend::TokenizerResources;

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

// Minimal int64 .npy reader (v1/v2 header, C-contiguous, little-endian '<i8'). The shared
// qio::read_npy rejects non-float descr, so the token ids are parsed here.
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
    std::printf("  %-22s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

// One oracle directory: build the templated text, encode it, and gate token count, exact token-for-token
// identity vs input_ids.npy, the slice arithmetic against the manifest's L/drop_idx/embed_len, and the
// _drop_idx the engine computes at runtime from the system-only turn (vs the oracle's prompt-independent
// drop_idx). All oracle values come from the manifest, never from a transcribed literal.
void gate_one(const Tokenizer& tokz, const std::string& label, const std::string& tower_dir) {
    const json manifest = json::parse(read_file(join(tower_dir, "manifest.json")));
    const std::string prompt     = manifest.at("prompt").get<std::string>();
    const std::string template_s = manifest.at("template").get<std::string>();
    const long long drop_idx     = manifest.at("drop_idx").get<long long>();
    const long long L            = manifest.at("L").get<long long>();
    const long long embed_len    = manifest.at("embed_len").get<long long>();

    std::string text = template_s;
    const std::size_t slot = text.find("{}");
    if (slot == std::string::npos) die("template has no {} placeholder in " + tower_dir);
    text.replace(slot, 2, prompt);

    std::vector<int> ids;
    try {
        ids = tokz.encode(text);
    } catch (const std::exception& e) {
        die(label + " encode failed: " + e.what());
    }

    const std::vector<long long> expected = read_int64_npy(join(tower_dir, "input_ids.npy"));
    check(label + " count == L", static_cast<long long>(ids.size()) == L,
          "got " + std::to_string(ids.size()) + " want " + std::to_string(L));

    bool identical = ids.size() == expected.size();
    long long first_bad = -1;
    for (std::size_t i = 0; identical && i < expected.size(); ++i) {
        if (static_cast<long long>(ids[i]) != expected[i]) { identical = false; first_bad = static_cast<long long>(i); }
    }
    std::string detail;
    if (identical) {
        detail = "all " + std::to_string(expected.size()) + " tokens match input_ids.npy";
    } else if (first_bad < 0) {
        detail = "length mismatch (got " + std::to_string(ids.size()) + " want " + std::to_string(expected.size()) + ")";
    } else {
        detail = "first mismatch at [" + std::to_string(first_bad) + "]: engine=" + std::to_string(ids[first_bad]) +
                 " oracle=" + std::to_string(expected[first_bad]);
    }
    check(label + " exact match", identical, detail);

    check(label + " slice L-drop", L - drop_idx == embed_len,
          std::to_string(L) + " - " + std::to_string(drop_idx) + " = " + std::to_string(L - drop_idx) +
          " want " + std::to_string(embed_len));

    // Runtime _drop_idx: at generation time there is no oracle, so the engine must compute it itself as the
    // token length of the system-only turn (every template char before the user turn). Cross-check that this
    // computed count equals the oracle's drop_idx AND that those ids are exactly the leading ids -- together
    // they prove the [drop_idx:] slice the tower consumes is well-defined from the prompt string alone.
    const std::size_t user_pos = template_s.find("<|im_start|>user");
    if (user_pos == std::string::npos) die("template has no user turn in " + tower_dir);
    const std::vector<int> sys_ids = tokz.encode(template_s.substr(0, user_pos));
    const long long drop_idx_engine = static_cast<long long>(sys_ids.size());
    check(label + " drop_idx computed", drop_idx_engine == drop_idx,
          "engine=" + std::to_string(drop_idx_engine) + " oracle=" + std::to_string(drop_idx));
    bool prefix_ok = static_cast<long long>(sys_ids.size()) == drop_idx;
    for (long long i = 0; prefix_ok && i < drop_idx; ++i) {
        if (sys_ids[i] != ids[i]) prefix_ok = false;
    }
    check(label + " drop_idx == prefix", prefix_ok,
          "leading " + std::to_string(drop_idx) + " ids equal the system-only encode (slice is exact)");
    check(label + " runtime slice len",
          static_cast<long long>(ids.size()) - drop_idx_engine == embed_len,
          std::to_string(ids.size()) + " - " + std::to_string(drop_idx_engine) +
          " = " + std::to_string(static_cast<long long>(ids.size()) - drop_idx_engine) +
          " want " + std::to_string(embed_len));
}

} // namespace

int main() {
    const char* proc_env   = std::getenv("INFERNIX_QWENVL_PROCESSOR");
    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (!proc_env || !oracle_env) {
        std::printf("SKIP: INFERNIX_QWENVL_PROCESSOR / INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string proc = proc_env, oracle = oracle_env;

    // The genuine Qwen3-VL descriptor, loaded WITHOUT substitution: this construction succeeding is
    // itself the gate that validate_pipeline accepts the mark-exclusive pre_tokenizer.
    const std::string tok_json = read_file(join(proc, "tokenizer.json"));
    const std::string cfg_json = read_file(join(proc, "tokenizer_config.json"));
    const std::string gen_json = std::string("{\"eos_token_id\": 151645}"); // encode() never reads stop ids

    std::optional<Tokenizer> tokz;
    try {
        tokz.emplace(TokenizerResources{tok_json, cfg_json, gen_json});
    } catch (const std::exception& e) {
        die(std::string("constructing the genuine Qwen3-VL tokenizer failed (guard rejected it?): ") + e.what());
    }
    check("VL descriptor loaded", true, "validate_pipeline accepted the mark-exclusive pre_tokenizer");

    gate_one(*tokz, "ascii", join(oracle, "tensors_text_tower"));
    gate_one(*tokz, "marks", join(oracle, "tensors_text_tower_marks"));

    if (!g_ok) { std::printf("IMAGE_TOWER_TOKENIZE_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_TOWER_TOKENIZE_TEST: PASS (engine reproduces the Qwen3-VL processor input_ids "
                "token-for-token from the genuine VL descriptor, for ASCII and \\p{M}-mark prompts)\n");
    return 0;
}
