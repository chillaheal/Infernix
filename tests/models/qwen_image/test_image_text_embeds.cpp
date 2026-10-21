// Variable-input image generation (#37): the production post-tower text-conditioning gate.
//
// build_prompt_embeds (src/models/qwen_image/text_embeds.{h,cpp}) turns the text tower's pre_final-RMSNorm
// hidden state into the DiT's prompt_embeds / encoder_attention_mask: per batch item, gather the VALID
// token rows, drop the leading drop_idx of them, right-pad to the longest kept length. This is the pure
// data-movement step the reference pipeline does after the tower (image-spike/oracle/dump_text_tower.py);
// the tower forward itself is qualified on-device (infernix_image_texttower_prod_cuda_test).
//
// Two gates:
//   (A) SYNTHETIC, always runs: a deterministic multi-item batch with LEFT padding on one item and
//       RIGHT padding on another, distinct drop lengths. This exercises the general variable case the
//       single-item oracle cannot: the drop must remove the first drop_idx VALID rows (mask-aware), so a
//       left-padded item drops its first real token, not the pad. The synthetic expected arrays are
//       built by hand (independent of the implementation), so this is a real reference, not a mirror.
//   (B) ORACLE, gated on INFERNIX_IMAGE_ORACLE: run the production function on the real ASCII oracle's
//       pre_norm_hidden.npy [1,25,4096] + attention_mask.npy + manifest drop_idx, and require the output
//       to reproduce the diffusers prompt_embeds.npy [1,11,4096] and encoder_attention_mask.npy exactly
//       (pure selection/padding of the same f32 rows -> bitwise, tol 0). Skips (77) when absent.
//
// Host-only, no GPU.

#include "models/qwen_image/text_embeds.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using infernix::models::qwen_image::build_prompt_embeds;
using infernix::models::qwen_image::TextEmbeds;

void die(const std::string& m) { std::printf("  ERROR %s\n", m.c_str()); std::exit(1); }

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (a.back() == '/' || a.back() == '\\') ? a + b : a + "/" + b;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Npy { std::vector<long long> shape; std::string descr; std::vector<std::uint8_t> body; };

// Minimal C-contiguous little-endian .npy reader (v1/v2/v3 header). Returns the raw body; the caller
// reinterprets as f32 or i64 after checking descr.
Npy read_npy(const std::string& path) {
    const std::string raw = read_file(path);
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
    // Parse 'descr': '<f4' / '<i8'.
    const std::size_t dp = header.find("'descr'");
    const std::size_t q1 = header.find('\'', header.find(':', dp) + 1);
    const std::size_t q2 = header.find('\'', q1 + 1);
    if (dp == std::string::npos || q1 == std::string::npos || q2 == std::string::npos) die("no descr: " + path);
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    // Parse 'shape': (d0, d1, ...).
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
    std::printf("  %-30s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

// ---- (A) synthetic general batch: left-pad + right-pad + distinct drop lengths ----
void gate_synthetic() {
    const int B = 2, L = 6, D = 3, drop = 1;
    // value at (b,l,d) = b*100 + l*10 + d, so every source row is uniquely recognizable.
    auto val = [&](int b, int l, int d) { return static_cast<float>(b * 100 + l * 10 + d); };
    std::vector<float> h(static_cast<std::size_t>(B) * L * D);
    for (int b = 0; b < B; ++b)
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < D; ++d) h[(b * L + l) * D + d] = val(b, l, d);
    const int mask[2][6] = {{0, 1, 1, 1, 1, 1},   // b0: left-pad 1, 5 valid -> drop first valid (l=1) -> kept l=2,3,4,5
                            {1, 1, 1, 1, 0, 0}};  // b1: right-pad 2, 4 valid -> drop first valid (l=0) -> kept l=1,2,3
    TextEmbeds out = build_prompt_embeds(h.data(), B, L, D, &mask[0][0], drop);

    check("synthetic dims", out.batch == 2 && out.max_len == 4 && out.dim == 3,
          "batch=" + std::to_string(out.batch) + " max_len=" + std::to_string(out.max_len) +
          " dim=" + std::to_string(out.dim));

    // Hand-built expected (independent of the implementation).
    // b0 kept: l=2,3,4,5 ; pad slot 3.  b1 kept: l=1,2,3 ; pad slot 3.
    auto row = [&](int b, int l) { return std::vector<float>{val(b, l, 0), val(b, l, 1), val(b, l, 2)}; };
    const int kept_l[2][4] = {{2, 3, 4, 5}, {1, 2, 3, -1}};  // -1 = pad slot
    bool embed_ok = out.prompt_embeds.size() == static_cast<std::size_t>(B) * 4 * D &&
                    out.encoder_attention_mask.size() == static_cast<std::size_t>(B) * 4;
    int first_bad = -1;
    if (embed_ok) {
        for (int b = 0; b < B && first_bad < 0; ++b)
            for (int s = 0; s < 4; ++s) {
                const int l = kept_l[b][s];
                std::vector<float> want = (l < 0) ? std::vector<float>{0, 0, 0} : row(b, l);
                for (int d = 0; d < D; ++d) {
                    const float got = out.prompt_embeds[(b * 4 + s) * D + d];
                    if (got != want[d]) { embed_ok = false; first_bad = b * 4 + s; break; }
                }
                const int want_mask = (l < 0) ? 0 : 1;
                if (embed_ok && out.encoder_attention_mask[b * 4 + s] != want_mask) {
                    embed_ok = false; first_bad = b * 4 + s;
                }
            }
    }
    check("synthetic select+pad+mask", embed_ok,
          first_bad < 0 ? "mask-aware drop, left/right pad, all rows exact"
                        : "first bad slot b*max_len+s=" + std::to_string(first_bad));

    // Discriminator: the mask-aware drop must NOT keep the row it should have dropped.
    // b0 must not contain l=1 ([10,11,12]); b1 must not contain l=0 ([0,1,2]).
    bool no_b0_l1 = true, no_b1_l0 = true;
    for (std::size_t i = 0; i + 2 < out.prompt_embeds.size() / D; ++i) {
        const int b = static_cast<int>(i / 4);
        const std::size_t o = i * D;
        if (out.encoder_attention_mask[i] == 0) continue;
        if (b == 0 && out.prompt_embeds[o] == 10.0F) no_b0_l1 = false;
        if (b == 1 && out.prompt_embeds[o] == 0.0F)  no_b1_l0 = false;
    }
    check("synthetic drop skips pad", no_b0_l1, "left-padded item dropped its first VALID row (l=1), not the pad");
    check("synthetic drops first valid", no_b1_l0, "right-padded item dropped its first VALID row (l=0)");
}

// ---- (B) oracle: real pre_norm_hidden -> must reproduce diffusers prompt_embeds exactly ----
void gate_oracle(const std::string& tower_dir) {
    const json manifest = json::parse(read_file(join(tower_dir, "manifest.json")));
    const long long drop_idx = manifest.at("drop_idx").get<long long>();

    Npy h_npy = read_npy(join(tower_dir, "pre_norm_hidden.npy"));
    Npy m_npy = read_npy(join(tower_dir, "attention_mask.npy"));
    if (h_npy.shape.size() != 3 || h_npy.shape[0] != 1) die("pre_norm_hidden not [1,L,D]");
    const int B = static_cast<int>(h_npy.shape[0]), L = static_cast<int>(h_npy.shape[1]),
              D = static_cast<int>(h_npy.shape[2]);
    const std::vector<float> h = as_f32(h_npy);
    std::vector<int> mask_i64_as_int;
    const std::vector<long long> mask_i64 = as_i64(m_npy);
    for (long long v : mask_i64) mask_i64_as_int.push_back(static_cast<int>(v));

    TextEmbeds out = build_prompt_embeds(h.data(), B, L, D, mask_i64_as_int.data(),
                                         static_cast<int>(drop_idx));

    Npy pe_npy = read_npy(join(tower_dir, "prompt_embeds.npy"));
    const std::vector<float> want_pe = as_f32(pe_npy);
    Npy em_npy = read_npy(join(tower_dir, "encoder_attention_mask.npy"));
    const std::vector<long long> want_em = as_i64(em_npy);

    const long long want_len = pe_npy.shape.size() >= 2 ? pe_npy.shape[1] : -1;
    check("oracle max_len == embed_len", out.max_len == want_len,
          "got " + std::to_string(out.max_len) + " oracle " + std::to_string(want_len));

    bool pe_ok = out.prompt_embeds.size() == want_pe.size();
    long long first_bad = -1;
    double max_abs = 0.0;
    for (std::size_t i = 0; pe_ok && i < want_pe.size(); ++i) {
        const double dd = std::fabs(static_cast<double>(out.prompt_embeds[i]) - want_pe[i]);
        if (dd > max_abs) max_abs = dd;
        if (out.prompt_embeds[i] != want_pe[i] && first_bad < 0) first_bad = static_cast<long long>(i);
    }
    check("oracle prompt_embeds exact", pe_ok && first_bad < 0,
          pe_ok ? ("pure selection/pad -> bitwise (max abs diff " + std::to_string(max_abs) + ")")
                : "size mismatch got " + std::to_string(out.prompt_embeds.size()) + " want " +
                      std::to_string(want_pe.size()));

    bool em_ok = out.encoder_attention_mask.size() == want_em.size();
    for (std::size_t i = 0; em_ok && i < want_em.size(); ++i)
        if (out.encoder_attention_mask[i] != want_em[i]) em_ok = false;
    check("oracle enc_mask exact", em_ok && out.encoder_attention_mask.size() == want_em.size(),
          "1 kept / 0 pad matches encoder_attention_mask.npy");
}

} // namespace

int main() {
    gate_synthetic();
    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (oracle_env == nullptr || std::string(oracle_env).empty()) {
        std::printf("ORACLE SKIPPED (INFERNIX_IMAGE_ORACLE not set)\n");
    } else {
        gate_oracle(join(std::string(oracle_env), "tensors_text_tower"));
    }
    if (!g_ok) { std::printf("IMAGE_TEXT_EMBEDS_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_TEXT_EMBEDS_TEST: PASS (production prompt_embeds prep: mask-aware drop + pad; "
                "synthetic general batch exact, oracle prompt_embeds bitwise)\n");
    return 0;
}
