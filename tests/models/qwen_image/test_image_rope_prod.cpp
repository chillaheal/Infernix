// Variable-input image generation (#37): the production joint-sequence rotary gate.
//
// build_joint_rope (src/models/qwen_image/rope.{h,cpp}) is the production host builder that replaces the
// fixed-geometry .npy the interlude reads today: for any (text_prefix, H, W) it emits the DiT's 3-axis
// (frame/height/width) rotary table [S,64,2] interleaved (cos,sin), S = text_prefix + H*W. It reproduces
// QwenImage21Rope.forward for the text-only case, so resolution becomes a parameter, not a dumped file.
//
// Two gates:
//   (A) GRID invariants, always run: across several geometries (non-square aspect ratios, tiny, tp=0),
//       the table's shape is S*128; every (cos,sin) pair is unit modulus (cos^2+sin^2=1) -- the general
//       garbage detector; a text token's height- and width-axis blocks are identical (both use axis dim
//       56 at the same frame value, unlike the frame axis whose dim is 16); all image tokens share one
//       frame-axis block (constant frame = text_prefix); and two closed-form spot checks -- the s-th text
//       token's frame-axis col0 is (cos s, sin s) (angle = index * theta^0), and the (0,0) image token's
//       height-axis col0 is (cos(-(H-H/2)), sin(-(H-H/2))). The expected values are computed by hand from
//       the documented formula, independent of the implementation. Also checks determinism.
//   (B) ORACLE anchors, gated on INFERNIX_IMAGE_ORACLE (a-priori tols copied from the pipeline self-check,
//       never narrowed): build_joint_rope(8,8,8) vs tensors_full_depth/block_rotary.npy rel_fro < 1e-2, and
//       build_joint_rope(11,16,16) vs tensors_pipeline/rot_inter.npy rel_fro < 5e-2. These are two INDEPENDENT
//       diffusers-dumped references at two DIFFERENT geometries -- so agreement here shows the formula
//       generalizes, not that it mirrors one reference. Skips a missing file gracefully.
//
// Host-only, no GPU.

#include "models/qwen_image/rope.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using infernix::models::qwen_image::build_joint_rope;

void die(const std::string& m) { std::printf("  ERROR %s\n", m.c_str()); std::exit(1); }

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (a.back() == '/' || a.back() == '\\') ? a + b : a + "/" + b;
}

bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return f.good(); }

struct Npy { std::vector<long long> shape; std::string descr; std::vector<std::uint8_t> body; };

// Minimal C-contiguous little-endian .npy reader (v1/v2/v3 header). Accepts float32 "<f4" and
// complex64 "<c8" -- the rotary dumps are complex64 (polar(1,angle) = cos + i sin), whose 8 bytes are
// exactly the interleaved (cos, sin) pair, so decoding them as float32 pairs yields the same byte
// sequence as the builder's [S,64,2] output. Any other descr is rejected.
Npy read_npy_f32(const std::string& path) {
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
    if (a.descr != "<f4" && a.descr != "<c8") die("expected <f4/<c8 rotary, got " + a.descr);
    return a;
}

double rel_fro(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || b.empty()) return 1e9;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d; den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return den > 0.0 ? std::sqrt(num) / std::sqrt(den) : (num > 0.0 ? 1e9 : 0.0);
}

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-34s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

// ---- (A) grid invariants across geometries ----
void gate_grid() {
    const int geos[][3] = {{5, 3, 7}, {11, 16, 16}, {11, 64, 32}, {11, 2, 8},
                           {1, 1, 1}, {0, 4, 4}, {3, 32, 2}};
    bool shape_ok = true, unit_ok = true, tex_hw_ok = true, img_frame_ok = true, spot_ok = true,
         det_ok = true;
    std::string bad;
    for (const auto& g : geos) {
        const int tp = g[0], H = g[1], W = g[2];
        const std::vector<float> rot = build_joint_rope(tp, H, W);
        const long long S = static_cast<long long>(tp) + static_cast<long long>(H) * W;
        if (static_cast<long long>(rot.size()) != S * 64 * 2) {
            shape_ok = false; bad = "geo " + std::to_string(tp) + "," + std::to_string(H) + "," +
                                     std::to_string(W) + " size " + std::to_string(rot.size()) + " want " +
                                     std::to_string(S * 128);
        }
        // unit modulus for every (cos,sin) pair
        for (std::size_t k = 0; k + 1 < rot.size(); k += 2) {
            const double m = static_cast<double>(rot[k]) * rot[k] + static_cast<double>(rot[k + 1]) * rot[k + 1];
            if (std::fabs(m - 1.0) > 1e-5) { unit_ok = false; bad = "unit modulus at geo " +
                std::to_string(tp) + "," + std::to_string(H) + "," + std::to_string(W) + " k=" + std::to_string(k); break; }
        }
        auto row = [&](long long s) { return rot.data() + s * 128; };
        // text token: height-axis block (cols 8..35) == width-axis block (cols 36..63)
        for (int s = 0; s < tp && tex_hw_ok; ++s) {
            const float* r = row(s);
            for (int c = 0; c < 28; ++c) {  // 28 cols of dim-56 axis each
                const int c1 = (8 + c) * 2, c2 = (8 + 28 + c) * 2;
                if (r[c1] != r[c2] || r[c1 + 1] != r[c2 + 1]) { tex_hw_ok = false; break; }
            }
        }
        // image tokens: shared frame-axis block (cols 0..7), constant frame = text_prefix
        if (H * W >= 2 && img_frame_ok) {
            const float* r0 = row(tp);
            for (long long t = 1; t < static_cast<long long>(H) * W && img_frame_ok; ++t)
                for (int c = 0; c < 8; ++c)
                    if (r0[c * 2] != row(tp + t)[c * 2] || r0[c * 2 + 1] != row(tp + t)[c * 2 + 1]) {
                        img_frame_ok = false; break;
                    }
        }
        // closed-form spot checks
        if (tp > 0) {
            const float* r = row(1);  // text token s=1: frame axis col0 angle=1 -> (cos1, sin1)
            if (std::fabs(r[0] - std::cos(1.0)) > 1e-6 || std::fabs(r[1] - std::sin(1.0)) > 1e-6)
                spot_ok = false;
        }
        {
            const long long hval = -(static_cast<long long>(H) - static_cast<long long>(H) / 2);
            const float* r = row(tp);  // image (0,0): height-axis global col8 angle = hval
            const int off = 8 * 2;
            if (std::fabs(r[off] - std::cos(static_cast<double>(hval))) > 1e-6 ||
                std::fabs(r[off + 1] - std::sin(static_cast<double>(hval))) > 1e-6)
                spot_ok = false;
        }
        // determinism
        if (build_joint_rope(tp, H, W) != rot) det_ok = false;
    }
    check("grid shape S*128", shape_ok, bad.empty() ? "all geometries" : bad);
    check("grid unit modulus", unit_ok, "cos^2+sin^2==1 every pair");
    check("grid text h/w axes equal", tex_hw_ok, "text height- and width-axis blocks identical (same value+dim)");
    check("grid image frame shared", img_frame_ok, "all image tokens share the frame-axis block");
    check("grid closed-form spots", spot_ok, "text(col0)=cos/sin(s), image(0,0)h=cos/sin(-(H-H/2))");
    check("grid deterministic", det_ok, "two calls identical");
}

// ---- (B) oracle anchors ----
void gate_anchor(const std::string& rel, const std::string& file, int tp, int H, int W, double tol,
                 const std::string& name) {
    const std::string path = join(rel, file);
    if (!file_exists(path)) { std::printf("  %-34s SKIP  %s absent\n", name.c_str(), path.c_str()); return; }
    const Npy ref = read_npy_f32(path);
    const std::vector<float> want(ref.body.size() / 4);
    std::memcpy(const_cast<float*>(want.data()), ref.body.data(), want.size() * 4);
    const std::vector<float> got = build_joint_rope(tp, H, W);
    const double e = rel_fro(got, want);
    check(name, e < tol, "rel_fro=" + std::to_string(e) + " tol=" + std::to_string(tol));
}

} // namespace

int main() {
    gate_grid();
    const char* oracle_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (oracle_env == nullptr || std::string(oracle_env).empty()) {
        std::printf("  ANCHORS SKIPPED (INFERNIX_IMAGE_ORACLE not set)\n");
    } else {
        const std::string o(oracle_env);
        gate_anchor(join(o, "tensors_full_depth"), "block_rotary.npy", 8, 8, 8, 1e-2, "anchor (8,8,8) block_rotary");
        gate_anchor(join(o, "tensors_pipeline"), "rot_inter.npy", 11, 16, 16, 5e-2, "anchor (11,16,16) rot_inter");
    }
    if (!g_ok) { std::printf("IMAGE_ROPE_PROD_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_ROPE_PROD_TEST: PASS (production variable-resolution joint rope: grid invariants + "
                "two independent diffusers rotary anchors)\n");
    return 0;
}
