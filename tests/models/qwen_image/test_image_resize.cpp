// P1: the condition-image resize gated against Pillow itself.
//
// oracle_resize.json (regenerate with: python -I tests/models/qwen_image/gen_oracle_resize.py) holds
// synthetic RGBA fixtures, their PNG encoding, and what
//     PIL.Image.resize((w, h), resample=LANCZOS, reducing_gap=None)
// produced for them -- the call VaeImageProcessor.resize makes for this pipeline, whose processor is
// built with no resample override (docs/superpowers/specs/2026-10-31-image-edit-lanczos-resize-design.md).
//
// The gate is exact byte equality. Each input PNG is decoded with the SHIPPED PNG decoder, so the
// case exercises the same bytes the HTTP route would see; each expected plane is compared as raw
// RGBA, so nothing between the engine's answer and the oracle's answer is itself under test.
//
// One guard proves the gate is not trivially satisfiable: a 2x2 bilinear kernel written here must
// DISAGREE with the oracle on the high-frequency fixture. A Lanczos bug that quietly degenerates to
// bilinear therefore cannot pass.
//
// Host-only, no GPU: it runs while serve.exe owns the card.

#include "media/decode/decode.h"
#include "media/decode/png_decode.h"
#include "serve/reference_image.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
namespace md = infernix::media::decode;
namespace srv = infernix::serve;

void die(const std::string& message) { std::printf("  ERROR %s\n", message.c_str()); std::exit(1); }

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail = {}) {
    std::printf("  %-26s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { die("cannot open " + path); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> b64(const std::string& text) {
    try {
        return srv::base64_decode(text);
    } catch (const std::exception& e) {
        die(std::string("oracle base64: ") + e.what());
    }
    return {};
}

// A deliberately naive 2x2 bilinear, used ONLY to show the oracle is filter-sensitive.
std::vector<std::uint8_t> naive_bilinear(const std::uint8_t* src, int sw, int sh, int dw, int dh) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(dw) * dh * 4);
    const double fx_step = static_cast<double>(sw) / dw;
    const double fy_step = static_cast<double>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        double fy = (y + 0.5) * fy_step - 0.5;
        fy = std::min(std::max(fy, 0.0), static_cast<double>(sh - 1));
        const int y0 = static_cast<int>(std::floor(fy));
        const int y1 = std::min(y0 + 1, sh - 1);
        const double wy = fy - y0;
        for (int x = 0; x < dw; ++x) {
            double fx = (x + 0.5) * fx_step - 0.5;
            fx = std::min(std::max(fx, 0.0), static_cast<double>(sw - 1));
            const int x0 = static_cast<int>(std::floor(fx));
            const int x1 = std::min(x0 + 1, sw - 1);
            const double wx = fx - x0;
            for (int c = 0; c < 4; ++c) {
                const double a = src[(static_cast<std::size_t>(y0) * sw + x0) * 4 + c];
                const double b = src[(static_cast<std::size_t>(y0) * sw + x1) * 4 + c];
                const double d = src[(static_cast<std::size_t>(y1) * sw + x0) * 4 + c];
                const double e = src[(static_cast<std::size_t>(y1) * sw + x1) * 4 + c];
                const double v = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
                out[(static_cast<std::size_t>(y) * dw + x) * 4 + c] =
                    static_cast<std::uint8_t>(std::min(std::max(std::lround(v), 0L), 255L));
            }
        }
    }
    return out;
}

std::size_t first_mismatch(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want,
                           std::size_t& count) {
    count = 0;
    std::size_t first = std::string::npos;
    if (got.size() != want.size()) { return 0; }
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) {
            if (first == std::string::npos) { first = i; }
            ++count;
        }
    }
    return first;
}

} // namespace

#ifndef INFERNIX_SOURCE_DIR
#    define INFERNIX_SOURCE_DIR "."
#endif

int main() {
    const std::string path =
        std::string(INFERNIX_SOURCE_DIR) + "/tests/models/qwen_image/oracle_resize.json";
    const json o = json::parse(read_file(path));
    std::printf("Pillow %s, resample %s, reducing_gap %s\n",
                o.at("pil_version").get<std::string>().c_str(),
                o.at("resample").get<std::string>().c_str(),
                o.at("reducing_gap").is_null() ? "null" : "set");

    for (const auto& c : o.at("cases")) {
        const std::string name   = c.at("name").get<std::string>();
        const int sw             = c.at("source_w").get<int>();
        const int sh             = c.at("source_h").get<int>();
        const int dw             = c.at("dst_w").get<int>();
        const int dh             = c.at("dst_h").get<int>();
        const std::vector<std::uint8_t> png = b64(c.at("input_png_b64").get<std::string>());
        const std::vector<std::uint8_t> want = b64(c.at("expected_rgba_b64").get<std::string>());

        if (!md::is_png(png)) { check(name + ".input_is_png", false, "not a PNG"); continue; }
        md::RgbaImage src;
        try {
            src = md::decode_png_rgba(png, md::Policy{}.max_decoded_pixels);
        } catch (const std::exception& e) {
            check(name + ".decode", false, e.what());
            continue;
        }
        if (src.width != sw || src.height != sh) {
            check(name + ".decoded_size", false,
                  "got " + std::to_string(src.width) + "x" + std::to_string(src.height) + " want " +
                      std::to_string(sw) + "x" + std::to_string(sh));
            continue;
        }

        std::vector<std::uint8_t> got;
        try {
            srv::resize_rgba_pil_lanczos(src.rgba.data(), sw, sh, dw, dh, got);
        } catch (const std::exception& e) {
            check(name + ".resize", false, e.what());
            continue;
        }
        std::size_t bad = 0;
        const std::size_t first = first_mismatch(got, want, bad);
        std::string detail = std::to_string(got.size()) + " bytes";
        if (bad) {
            detail += " mismatches=" + std::to_string(bad) + " of " + std::to_string(want.size());
            if (first != std::string::npos && got.size() == want.size()) {
                detail += " first=[" + std::to_string(first / 4 / dw) + "," +
                          std::to_string(first / 4 % dw) + ",c" + std::to_string(first % 4) + "] got " +
                          std::to_string(got[first]) + " want " + std::to_string(want[first]);
            }
        }
        check(name, bad == 0 && got.size() == want.size(), detail);

        if (name == "checker_alias") {
            const std::vector<std::uint8_t> bi = naive_bilinear(src.rgba.data(), sw, sh, dw, dh);
            std::size_t bi_bad = 0;
            first_mismatch(bi, want, bi_bad);
            check("bilinear_is_rejected", bi_bad > 0,
                  "naive bilinear differs on " + std::to_string(bi_bad) + " bytes");
        }
    }

    std::printf("%s\n", g_ok ? "RESIZE_ORACLE_RESULT fail=0" : "RESIZE_ORACLE_RESULT FAILED");
    return g_ok ? 0 : 1;
}
