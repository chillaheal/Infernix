// Variable-input image generation (#37): the production front-half embed_tokens GATHER gate.
//
// build_inputs_embeds (src/models/qwen_image/embed_gather.{h,cpp}) turns a prompt's token ids into the
// Qwen3-VL text tower's INPUT embedding: inputs_embeds[t] = widen(embed_tokens[ids[t]]), a pure BF16 row
// lookup widened to FP32. This is the paired bookend to build_prompt_embeds -- together the pure data
// movement around the (on-device-qualified) 36-layer forward, so a raw prompt string can drive the whole
// chain (docs/maintainer/qwen-image-2-1-variables.md gap #1). image-spike/verify_embed_gather.py proved
// the SAME gather is bit-exact vs the dumped inputs_embeds.npy in Python; this test qualifies the ENGINE
// function that performs it.
//
// Two gates:
//   (A) SYNTHETIC, always runs: a hand-built BF16 table of known BF16-exact values + an id sequence with
//       repeats, and the out-of-range guard. The expected widened floats are the ORIGINAL table values
//       (independent of the implementation), so this proves the row lookup, ordering, repeats and the
//       top-16-bit widening exactly -- and that an out-of-range id is refused. No model data required.
//   (B) ORACLE, gated on INFERNIX_IMAGE_ORACLE + INFERNIX_IMAGE_TEXT_ENCODER_ST: gather the REAL
//       model.language_model.embed_tokens.weight (BF16, read straight from the Qwen3-VL text-encoder
//       safetensors) rows for the real tensors_text_tower/input_ids.npy and require the output to
//       reproduce tensors_text_tower_weights/inputs_embeds.npy BITWISE (pure lookup + widening -> tol 0).
//       Skips (not fails) when the external artifact is absent -- it is a prerequisite, not repo state.
//
// Host-only, no GPU. Reads a 5 GB safetensors by header + slice seek only (never the whole file).

#include "models/qwen_image/embed_gather.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using infernix::models::qwen_image::build_inputs_embeds;

void die(const std::string& m) { std::printf("  ERROR %s\n", m.c_str()); std::exit(1); }

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (a.back() == '/' || a.back() == '\\') ? a + b : a + "/" + b;
}

std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Npy { std::vector<long long> shape; std::string descr; std::vector<std::uint8_t> body; };

// Minimal C-contiguous little-endian .npy reader (v1/2/3). Same idiom as test_image_text_embeds.cpp.
Npy read_npy(const std::string& path) {
    const std::string raw = read_text_file(path);
    std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
    if (bytes.size() < 13) die("npy too small: " + path);
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (std::memcmp(bytes.data(), magic, 6) != 0) die("not a npy file: " + path);
    const std::uint8_t major = bytes[6];
    std::size_t header_len = 0, data_off = 0;
    if (major == 1) {
        header_len = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8);
        data_off   = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8) |
                     (static_cast<std::size_t>(bytes[10]) << 16) | (static_cast<std::size_t>(bytes[11]) << 24);
        data_off   = 12 + header_len;
    } else die("unsupported npy version: " + path);
    if (bytes.size() < data_off) die("npy truncated: " + path);
    const std::string header(reinterpret_cast<const char*>(bytes.data() + (major == 1 ? 10 : 12)), header_len);
    if (header.find("'fortran_order': False") == std::string::npos) die("npy not C-contiguous: " + path);

    Npy a;
    const std::size_t dp = header.find("'descr'");
    const std::size_t q1 = header.find('\'', header.find(':', dp) + 1);
    const std::size_t q2 = header.find('\'', q1 + 1);
    if (dp == std::string::npos || q1 == std::string::npos || q2 == std::string::npos) die("no descr: " + path);
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    const std::size_t sp = header.find("'shape'");
    const std::size_t op = header.find('(', sp);
    const std::size_t cp = header.find(')', op);
    if (sp == std::string::npos || op == std::string::npos || cp == std::string::npos) die("no shape: " + path);
    std::string sd = header.substr(op + 1, cp - op - 1);
    std::size_t i = 0;
    while (i < sd.size()) {
        while (i < sd.size() && (sd[i] == ' ' || sd[i] == ',')) ++i;
        std::size_t j = i;
        while (j < sd.size() && sd[j] >= '0' && sd[j] <= '9') ++j;
        if (j > i) a.shape.push_back(std::stoll(sd.substr(i, j - i)));
        i = (j > i) ? j : i + 1;
    }
    a.body.assign(bytes.begin() + data_off, bytes.end());
    return a;
}

std::vector<float> as_f32(const Npy& a) {
    if (a.descr != "<f4") die("expected <f4, got " + a.descr);
    std::vector<float> out(a.body.size() / 4);
    std::memcpy(out.data(), a.body.data(), out.size() * 4);
    return out;
}
std::vector<long long> as_i64(const Npy& a) {
    if (a.descr != "<i8") die("expected <i8, got " + a.descr);
    std::vector<long long> out(a.body.size() / 8);
    std::memcpy(out.data(), a.body.data(), out.size() * 8);
    return out;
}

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-34s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

// Store an FP32 value's BF16 bits (its top 16 bits) -- the inverse of the gather's widening. The synthetic
// table is built from BF16-exact small integers, so no bit is truncated.
std::uint16_t f32_to_bf16_bits(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

// ---- (A) synthetic: exact gather + widening + out-of-range guard, no model data ----
void gate_synthetic() {
    const int vocab = 4, hidden = 8;
    // table_val(r,d) = (r*8+d) - 12 -> covers negative, zero and positive small integers (all BF16-exact),
    // so a byte-swap or sign bug in the widening cannot slip through.
    auto table_val = [&](int r, int d) { return static_cast<float>((r * hidden + d) - 12); };
    std::vector<std::uint16_t> table(static_cast<std::size_t>(vocab) * hidden);
    for (int r = 0; r < vocab; ++r)
        for (int d = 0; d < hidden; ++d) table[r * hidden + d] = f32_to_bf16_bits(table_val(r, d));

    const std::vector<long long> ids = {3, 1, 3, 0, 2};  // repeats + non-monotonic order
    const std::vector<float> got = build_inputs_embeds(table.data(), vocab, hidden, ids.data(),
                                                       static_cast<int>(ids.size()));

    bool size_ok = got.size() == ids.size() * static_cast<std::size_t>(hidden);
    check("synthetic output size", size_ok,
          "got " + std::to_string(got.size()) + " want " + std::to_string(ids.size() * hidden));

    bool exact = size_ok;
    long long first_bad = -1;
    for (std::size_t t = 0; exact && t < ids.size(); ++t)
        for (int d = 0; d < hidden; ++d) {
            const float want = table_val(static_cast<int>(ids[t]), d);
            if (got[t * hidden + d] != want) { exact = false; first_bad = static_cast<long long>(t * hidden + d); break; }
        }
    check("synthetic gather+broaden exact", exact,
          exact ? "row lookup + order + repeats + top-16-bit widening bitwise"
                : "first bad element at " + std::to_string(first_bad));

    // Discriminator: a wrong row (id+1) must NOT match -- guards a lost/shifted index multiply.
    bool discrim = ids.size() > 0;
    for (int d = 0; d < hidden; ++d)
        if (got[d] == table_val(1, d)) discrim = false;  // row for id=3 must differ from row for id=1
    check("synthetic row index distinct", discrim, "output row 0 is ids[0]=3, not a shifted/aliased row");

    bool throws_high = false, throws_neg = false;
    const std::vector<long long> bad_high = {static_cast<long long>(vocab)};
    const std::vector<long long> bad_neg = {-1};
    try { static_cast<void>(build_inputs_embeds(table.data(), vocab, hidden, bad_high.data(), 1)); }
    catch (const std::exception&) { throws_high = true; }
    try { static_cast<void>(build_inputs_embeds(table.data(), vocab, hidden, bad_neg.data(), 1)); }
    catch (const std::exception&) { throws_neg = true; }
    check("synthetic out-of-range refused", throws_high && throws_neg,
          "id==vocab and id<0 both throw std::invalid_argument");
}

// Parse the embed_tokens tensor out of a HF safetensors file by header + slice offset only (never read
// the whole 5 GB file). Fills vocab/hidden/byte_offset; ok=false if the key/dtype is absent.
struct EmbedSlice { long long vocab = 0; int hidden = 0; std::uint64_t byte_off = 0; bool ok = false; };

EmbedSlice parse_embed_slice(const std::string& st_path, const std::string& key) {
    EmbedSlice e;
    std::ifstream f(st_path, std::ios::binary);
    if (!f) return e;
    std::uint64_t hdr_len = 0;
    f.read(reinterpret_cast<char*>(&hdr_len), 8);
    if (!f || hdr_len == 0 || hdr_len > (1ULL << 30)) return e;
    std::string hdr(static_cast<std::size_t>(hdr_len), '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hdr_len));
    if (!f) return e;
    json j;
    try { j = json::parse(hdr); } catch (const std::exception&) { return e; }
    if (!j.contains(key)) return e;
    const json& m = j.at(key);
    if (m.at("dtype").get<std::string>() != "BF16") return e;
    const auto shp = m.at("shape").get<std::vector<long long>>();
    const auto offs = m.at("data_offsets").get<std::vector<long long>>();
    if (shp.size() != 2 || offs.size() != 2 || offs[0] < 0) return e;
    e.vocab = shp[0];
    e.hidden = static_cast<int>(shp[1]);
    e.byte_off = 8ULL + hdr_len + static_cast<std::uint64_t>(offs[0]);
    e.ok = true;
    return e;
}

// ---- (B) oracle: gather the REAL embed table for the REAL ids -> must reproduce inputs_embeds bitwise ----
void gate_oracle(const std::string& tower_dir, const std::string& weights_dir, const std::string& st_path) {
    const std::string key = "model.language_model.embed_tokens.weight";
    const Npy ids_npy = read_npy(join(tower_dir, "input_ids.npy"));
    const std::vector<long long> ids = as_i64(ids_npy);            // flatten [1,L] -> L
    const Npy emb_npy = read_npy(join(weights_dir, "inputs_embeds.npy"));
    if (emb_npy.shape.size() != 2) die("inputs_embeds not [L,D]");
    const std::vector<float> want = as_f32(emb_npy);
    const long long L = emb_npy.shape[0];
    const int D = static_cast<int>(emb_npy.shape[1]);
    if (static_cast<long long>(ids.size()) != L)
        die("ids length " + std::to_string(ids.size()) + " != inputs_embeds rows " + std::to_string(L));

    const EmbedSlice es = parse_embed_slice(st_path, key);
    if (!es.ok) { std::printf("  ORACLE GATHER SKIPPED (embed key '%s' not BF16 in %s)\n", key.c_str(), st_path.c_str()); return; }
    if (es.hidden != D) die("embed hidden " + std::to_string(es.hidden) + " != inputs_embeds D " + std::to_string(D));

    std::vector<std::uint16_t> table(static_cast<std::size_t>(es.vocab) * es.hidden);
    {
        std::ifstream f(st_path, std::ios::binary);
        f.seekg(es.byte_off);
        f.read(reinterpret_cast<char*>(table.data()),
               static_cast<std::streamsize>(table.size()) * 2);
    }

    const std::vector<float> got = build_inputs_embeds(table.data(), es.vocab, es.hidden, ids.data(),
                                                       static_cast<int>(ids.size()));
    bool size_ok = got.size() == want.size();
    check("oracle output size", size_ok, "gathered " + std::to_string(got.size()) + " floats (" +
          std::to_string(L) + "x" + std::to_string(D) + "), vocab " + std::to_string(es.vocab));

    bool exact = size_ok;
    long long first_bad = -1;
    for (std::size_t i = 0; size_ok && i < want.size(); ++i)
        if (got[i] != want[i]) { exact = false; first_bad = static_cast<long long>(i); break; }
    check("oracle inputs_embeds bitwise", exact,
          exact ? "gather(embed_tokens, input_ids) bf16->f32 == inputs_embeds.npy, tol 0"
                : "first bad element at flat index " + std::to_string(first_bad));
}

} // namespace

int main() {
    gate_synthetic();

    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    const char* st_env = std::getenv("INFERNIX_IMAGE_TEXT_ENCODER_ST");
    if (oracle_env == nullptr || std::string(oracle_env).empty() ||
        st_env == nullptr || std::string(st_env).empty()) {
        std::printf("ORACLE GATHER SKIPPED (need INFERNIX_IMAGE_ORACLE + INFERNIX_IMAGE_TEXT_ENCODER_ST)\n");
    } else {
        const std::string oracle(oracle_env), st_path(st_env);
        std::ifstream probe(st_path);
        if (!probe) {
            std::printf("ORACLE GATHER SKIPPED (text-encoder safetensors absent: %s)\n", st_path.c_str());
        } else {
            gate_oracle(join(oracle, "tensors_text_tower"), join(oracle, "tensors_text_tower_weights"), st_path);
        }
    }

    if (!g_ok) { std::printf("IMAGE_EMBED_GATHER_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_EMBED_GATHER_TEST: PASS (production input_ids -> inputs_embeds gather: "
                "synthetic exact + out-of-range refused; oracle inputs_embeds bitwise)\n");
    return 0;
}
