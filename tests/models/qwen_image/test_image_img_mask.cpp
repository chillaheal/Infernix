// Variable-input image generation (#37): the production joint img_mask gate.
//
// build_img_mask (src/models/qwen_image/img_mask.{h,cpp}) reproduces dump_pipeline.py:124 -- the DiT joint
// img_mask is the text encoder's image_pad_mask_base row-concatenated with target_slots ones, where
// target_slots = (Hl*Wl)//4 is the 2x2-packed target image-token count. The base widens with the N
// condition images' image_pad slots, so the same rule generalizes text-only -> N condition images. This
// replaces the fixed img_mask.npy the interlude reads today.
//
// Two gates:
//   (A) SYNTHETIC, always runs: a multi-item batch whose base already carries condition-image 1-slots, +
//       a target_slots=0 edge. The expected output is the base copied verbatim followed by target_slots
//       ones -- a hand-built reference (independent of the implementation), exercising the N-condition
//       case the single text-only oracle cannot, and confirming a base 0 stays 0 (never overwritten by the
//       appended ones).
//   (B) ORACLE, gated on INFERNIX_IMAGE_ORACLE: read the real image_pad_mask_base.npy [1,11] + meta.json
//       tokens (target_slots = tokens//4 = 64 at 256x256), build, and require the output to reproduce the
//       diffusers img_mask.npy [1,75] EXACTLY (uint8, tol 0) with length == meta img_mask_len. Skips
//       (77) when absent.
//
// Host-only, no GPU.

#include "models/qwen_image/img_mask.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using infernix::models::qwen_image::build_img_mask;

void die(const std::string& m) { std::printf("  ERROR %s\n", m.c_str()); std::exit(1); }

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (a.back() == '/' || a.back() == '\\') ? a + b : a + "/" + b;
}
bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return f.good(); }
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Npy { std::vector<long long> shape; std::string descr; std::vector<std::uint8_t> body; };

// Minimal C-contiguous little-endian .npy reader (v1/v2/v3 header) for uint8 (<u1) masks.
Npy read_npy_u8(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() < 13) die("npy too small: " + path);
    const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (std::memcmp(bytes.data(), magic, 6) != 0) die("not a npy file: " + path);
    const std::uint8_t major = bytes[6];
    std::size_t header_len = 0, data_off = 0;
    if (major == 1) { header_len = bytes[8] | (std::size_t(bytes[9]) << 8); data_off = 10 + header_len; }
    else if (major == 2 || major == 3) {
        header_len = bytes[8] | (std::size_t(bytes[9]) << 8) | (std::size_t(bytes[10]) << 16) |
                     (std::size_t(bytes[11]) << 24);
        data_off = 12 + header_len;
    } else die("unsupported npy version: " + path);
    if (bytes.size() < data_off) die("npy truncated: " + path);
    const std::string header(reinterpret_cast<const char*>(bytes.data() + (major == 1 ? 10 : 12)), header_len);
    if (header.find("'fortran_order': False") == std::string::npos) die("npy not C-contiguous: " + path);
    Npy a;
    const std::size_t dp = header.find("'descr'");
    const std::size_t q1 = header.find('\'', header.find(':', dp) + 1), q2 = header.find('\'', q1 + 1);
    if (dp == std::string::npos || q1 == std::string::npos || q2 == std::string::npos) die("no descr: " + path);
    a.descr = header.substr(q1 + 1, q2 - q1 - 1);
    const std::size_t sp = header.find("'shape'"), op = header.find('(', sp), cp = header.find(')', op);
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
    // numpy writes uint8 as "|u1" (byte-order '|', not applicable to 1 byte); "<u1" accepted too.
    if (a.descr != "|u1" && a.descr != "<u1") die("expected |u1/<u1 mask, got " + a.descr);
    return a;
}

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-32s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

// ---- (A) synthetic multi-condition batch + zero-target edge ----
void gate_synthetic() {
    const int B = 2, base_len = 5, target_slots = 3;
    // base already carries condition-image image_pad slots (1s):
    const std::uint8_t base[2][5] = {{0, 1, 1, 0, 0}, {0, 0, 1, 1, 1}};
    const std::vector<std::uint8_t> out = build_img_mask(&base[0][0], B, base_len, target_slots);
    const int len = base_len + target_slots;
    check("synthetic size", out.size() == static_cast<std::size_t>(B) * len,
          "batch=" + std::to_string(B) + " len=" + std::to_string(len));

    // hand-built expected: base verbatim + target_slots ones
    const std::uint8_t want[2][8] = {
        {0, 1, 1, 0, 0, 1, 1, 1},
        {0, 0, 1, 1, 1, 1, 1, 1}};
    bool exact = out.size() == 16;
    int first_bad = -1;
    for (int b = 0; b < B && exact; ++b)
        for (int p = 0; p < len; ++p)
            if (out[b * len + p] != want[b][p]) { exact = false; first_bad = b * len + p; break; }
    check("synthetic concat exact", exact,
          first_bad < 0 ? "base verbatim + target ones"
                        : "first bad b*len+p=" + std::to_string(first_bad));

    // target_slots = 0 -> base only (no append)
    const std::vector<std::uint8_t> z = build_img_mask(&base[0][0], 1, base_len, 0);
    bool zok = z.size() == static_cast<std::size_t>(base_len);
    for (int p = 0; p < base_len && zok; ++p) if (z[p] != base[0][p]) zok = false;
    check("synthetic zero-target", zok, "target_slots=0 yields the base row unchanged");
}

// ---- (B) oracle: real base + target_slots -> must reproduce img_mask.npy exactly ----
void gate_oracle(const std::string& pd) {
    const std::string base_path = join(pd, "image_pad_mask_base.npy");
    const std::string mask_path = join(pd, "img_mask.npy");
    if (!file_exists(base_path) || !file_exists(mask_path)) {
        std::printf("  ORACLE SKIPPED (img_mask references absent)\n");
        return;
    }
    const json meta = json::parse(read_file(join(pd, "meta.json")));
    const int target_slots = static_cast<int>(meta.at("tokens").get<long long>() / 4);

    Npy base = read_npy_u8(base_path);
    if (base.shape.size() != 2) die("image_pad_mask_base not 2-D");
    const int B = static_cast<int>(base.shape[0]), base_len = static_cast<int>(base.shape[1]);
    const std::vector<std::uint8_t> got = build_img_mask(base.body.data(), B, base_len, target_slots);

    Npy want = read_npy_u8(mask_path);
    check("oracle length", got.size() == want.body.size() && want.shape.size() == 2 &&
                               want.shape[1] == base_len + target_slots,
          "got len " + std::to_string(got.size() / B) + " base " + std::to_string(base_len) +
              " + target " + std::to_string(target_slots));

    bool exact = got.size() == want.body.size();
    long long first_bad = -1;
    for (std::size_t i = 0; exact && i < want.body.size(); ++i)
        if (got[i] != want.body[i]) { exact = false; first_bad = static_cast<long long>(i); break; }
    check("oracle img_mask exact", exact && first_bad < 0,
          exact ? "base + target ones, bitwise vs img_mask.npy"
                : "first bad flat idx " + std::to_string(first_bad));

    const long long meta_len = meta.at("img_mask_len").get<long long>();
    check("oracle meta len match", want.shape[1] == meta_len,
          "img_mask shape[1] " + std::to_string(want.shape[1]) + " == meta " + std::to_string(meta_len));
}

} // namespace

int main() {
    gate_synthetic();
    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (oracle_env == nullptr || std::string(oracle_env).empty()) {
        std::printf("  ORACLE SKIPPED (INFERNIX_IMAGE_ORACLE not set)\n");
    } else {
        gate_oracle(join(std::string(oracle_env), "tensors_pipeline"));
    }
    if (!g_ok) { std::printf("IMAGE_IMG_MASK_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_IMG_MASK_TEST: PASS (production img_mask: base + packed target slots; "
                "synthetic multi-condition exact, oracle img_mask bitwise)\n");
    return 0;
}
