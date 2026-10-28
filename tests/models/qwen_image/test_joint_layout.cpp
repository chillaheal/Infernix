// Instruction-guided image edit (M5): the host gate for the joint index map.
//
// Gates build_joint_layout (src/models/qwen_image/joint_layout.{h,cpp}) -- the whole index arithmetic of
// transformer_qwenimage21.py's forward (repeats, x4 expansion, img_in scatter rows, target_token_mask,
// image block ids, rope walk):
//   (A) TEXT-ONLY EQUALITY: the mask [0]*tp + [1]*(H*W/4) with one img_shapes entry must produce the
//       rope table build_joint_rope(tp,H,W) BIT-FOR-BIT (rel Frobenius 0, exact ==) across the
//       geometries the certified rope test uses, plus the structural fields ([text][target], scatter
//       rows 0..H*W-1, target mask on the target only). The interleaved route must not move the
//       certified text-to-image assembly.
//   (B) PORTRAIT ROW: the M1 geometry, interleaved -- 30 text, a 4056-token reference block (1014
//       vision slots), 12 trailer text, a 4096-token target. Expanded length 30+4056+12+4096; block ids
//       -1/0/1; scatter rows text -1, ref 0..4055, target 4056..8151; the rope frame advances by
//       max(h,w) AFTER each block (ref frame 30, trailer text starts at 30+78=108, target frame 120) and
//       each block's own height/width grid is centred (read off the closed-form col-0 angles).
//   (C) TWO ADJACENT IMAGES: no text between them, and they stay TWO blocks with TWO frame advances.
//   (D) INVARIANTS: slots x4 != block tokens, empty img_shapes, frames != 1, and a mask with no target
//       slots all throw std::invalid_argument (the M7 ledger's host-side refusal).
//
// Host-only, no GPU, no env.

#include "models/qwen_image/joint_layout.h"
#include "models/qwen_image/rope.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infernix::models::qwen_image::build_joint_layout;
using infernix::models::qwen_image::build_joint_rope;
using infernix::models::qwen_image::ImageShape;
using infernix::models::qwen_image::JointLayout;

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-34s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

std::vector<std::uint8_t> mask_of(std::vector<long long> runs) {
    // runs: alternating (text, image) counts, starting with text, ending with the target slots.
    std::vector<std::uint8_t> m;
    bool image = false;
    for (const long long n : runs) {
        for (long long i = 0; i < n; ++i) { m.push_back(image ? 1 : 0); }
        image = !image;
    }
    return m;
}

// The rope emitter's closed form: axis col 0 is angle = value * theta^0 = value, so a row's axis value
// is verified by comparing its first (cos,sin) pair against (cos v, sin v). Global cols: frame 0..7,
// height 8..35, width 36..63.
bool axis_value_is(const std::vector<float>& rope, long long row, int axis_col, long long v) {
    const float* r = rope.data() + static_cast<std::size_t>(row) * 128 + axis_col * 2;
    const double a = static_cast<double>(v);
    return std::fabs(r[0] - std::cos(a)) < 1e-6 && std::fabs(r[1] - std::sin(a)) < 1e-6;
}

long long frame_of(const std::vector<float>& rope, long long row) {
    for (long long v = -4096; v <= 4096; ++v) {
        if (axis_value_is(rope, row, 0, v)) { return v; }
    }
    return -99999;
}

void gate_text_only() {
    const long long geos[][3] = {{5, 4, 4}, {11, 16, 16}, {11, 64, 32}, {1, 2, 2}, {0, 4, 4}, {3, 32, 2},
                                 {7, 10, 6}};
    bool rope_exact = true, struct_ok = true, fields_ok = true;
    std::string bad;
    for (const auto& g : geos) {
        const long long tp = g[0], H = g[1], W = g[2];
        const std::vector<std::uint8_t> m = mask_of({tp, H * W / 4});
        const std::vector<ImageShape> shapes{ImageShape{1, H, W}};
        const JointLayout L = build_joint_layout(m, shapes);
        const std::vector<float> want = build_joint_rope(static_cast<int>(tp), static_cast<int>(H),
                                                         static_cast<int>(W));
        if (L.rope != want) {
            rope_exact = false;
            bad = "rope differs at " + std::to_string(tp) + "," + std::to_string(H) + "," +
                  std::to_string(W);
        }
        const long long S = tp + H * W;
        if (L.expanded_len != S || L.text_tokens != tp || L.ref_tokens != 0 || L.target_tokens != H * W ||
            L.base_len != tp + H * W / 4 || L.rope.size() != static_cast<std::size_t>(S) * 128 ||
            static_cast<long long>(L.repeats.size()) != L.base_len ||
            static_cast<long long>(L.scatter_row.size()) != S) {
            fields_ok = false;
            bad = "fields at " + std::to_string(tp) + "," + std::to_string(H) + "," + std::to_string(W);
        }
        for (long long i = 0; i < S; ++i) {
            const bool text = i < tp;
            if (L.expanded_image_pad_mask[static_cast<std::size_t>(i)] != (text ? 0 : 1) ||
                L.target_token_mask[static_cast<std::size_t>(i)] != (text ? 0 : 1) ||
                L.image_block_ids[static_cast<std::size_t>(i)] != (text ? -1 : 0) ||
                L.scatter_row[static_cast<std::size_t>(i)] != (text ? -1 : i - tp)) {
                struct_ok = false;
                bad = "structure at " + std::to_string(tp) + "," + std::to_string(H) + "," +
                      std::to_string(W) + " pos " + std::to_string(i);
                break;
            }
        }
        for (long long i = 0; i < L.base_len; ++i) {
            if (L.repeats[static_cast<std::size_t>(i)] != (i < tp ? 1 : 4)) { struct_ok = false; }
        }
    }
    check("text-only rope bit-exact", rope_exact, bad.empty() ? "7 geometries == build_joint_rope (rel F 0)" : bad);
    check("text-only structure", struct_ok, "[text][target]: mask/block ids/scatter rows/repeats");
    check("text-only fields", fields_ok, "expanded/text/ref/target/base counts and table size");
}

void gate_portrait() {
    const long long prefix = 30, ref_vision = 1014, ref_tokens = 4056, trailer = 12, target = 4096;
    const std::vector<std::uint8_t> m = mask_of({prefix, ref_vision, trailer, target / 4});
    const ImageShape ref{1, 52, 78}, tgt{1, 64, 64};
    const std::vector<ImageShape> shapes{ref, tgt};
    const JointLayout L = build_joint_layout(m, shapes);

    const long long S = prefix + ref_tokens + trailer + target;
    check("portrait expanded len", L.expanded_len == S && L.base_len == prefix + ref_vision + trailer + target / 4,
          std::to_string(S) + " expanded, " + std::to_string(L.base_len) + " base (30+1014+12+1024)");
    check("portrait counts", L.text_tokens == prefix + trailer && L.ref_tokens == ref_tokens &&
                               L.target_tokens == target,
          "text 42 / ref 4056 (= 4 x 1014) / target 4096");

    bool ids_ok = true, target_ok = true, repeats_ok = true;
    std::string scatter_bad;
    for (long long i = 0; i < L.expanded_len; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        long long want_block = -1, want_row = -1;
        // Scatter rows are rows of the PACKED img_in input cat[ref | target], so they follow the latent
        // packing order, not the joint-sequence order: the target block continues after the reference
        // block's 4056 rows even though the trailer text sits between them in the joint sequence.
        if (i >= prefix && i < prefix + ref_tokens) { want_block = 0; want_row = i - prefix; }
        else if (i >= prefix + ref_tokens + trailer) {
            want_block = 1; want_row = ref_tokens + (i - prefix - ref_tokens - trailer);
        }
        if (L.image_block_ids[k] != want_block) { ids_ok = false; }
        if (L.scatter_row[k] != want_row && scatter_bad.empty()) {
            scatter_bad = "row " + std::to_string(i) + ": want " + std::to_string(want_row) + ", got " +
                          std::to_string(L.scatter_row[k]);
        }
        const bool is_img = want_block >= 0;
        if (L.expanded_image_pad_mask[k] != (is_img ? 1 : 0) ||
            L.target_token_mask[k] != (want_block == 1 ? 1 : 0)) { target_ok = false; }
    }
    for (std::size_t i = 0; i < m.size(); ++i) {
        if (L.repeats[i] != (m[i] ? 4 : 1)) { repeats_ok = false; }
    }
    check("portrait block ids", ids_ok, "-1 text, 0 ref, 1 target (img_shapes order, not mask runs)");
    check("portrait scatter rows", scatter_bad.empty(),
          scatter_bad.empty() ? "text -1; ref rows 0..4055, target rows 4056..8151 packed order" : scatter_bad);
    check("portrait target mask", target_ok, "1 on the LAST block only (modulation split, not prefix len)");
    check("portrait repeats", repeats_ok, "where(mask,4,1) per base position");

    // The rope walk: frame advances by max(h,w) AFTER each image block; grids centre on zero.
    bool frame_ok = frame_of(L.rope, 0) == 0 && frame_of(L.rope, prefix - 1) == prefix - 1 &&
                    frame_of(L.rope, prefix) == prefix &&
                    frame_of(L.rope, prefix + ref_tokens) == prefix + std::max(ref.height, ref.width) &&
                    frame_of(L.rope, L.expanded_len - 1) ==
                        prefix + std::max(ref.height, ref.width) + trailer;
    check("portrait rope frame advance", frame_ok,
          "ref frame " + std::to_string(prefix) + ", trailer text starts at " +
          std::to_string(prefix + std::max(ref.height, ref.width)) + " (30+max(52,78)=108), target frame " +
          std::to_string(prefix + std::max(ref.height, ref.width) + trailer));
    // Row indices: the ref block is [prefix, prefix+ref_tokens); the trailer text follows it; the target
    // block is [prefix+ref_tokens+trailer, expanded_len). Reading the wrong one of these rows is the
    // classic off-by-a-run error in this route, so the gate names the row it checks.
    const long long ref_h0 = -(ref.height - ref.height / 2), ref_w0 = -(ref.width - ref.width / 2);
    const long long tgt_h0 = -(tgt.height - tgt.height / 2), tgt_w0 = -(tgt.width - tgt.width / 2);
    const long long tgt0 = prefix + ref_tokens + trailer, tgt_last = L.expanded_len - 1;
    std::string grid_bad;
    auto want_axis = [&](long long row, int col, long long v, const char* what) {
        if (!axis_value_is(L.rope, row, col, v)) {
            grid_bad = std::string(what) + " row " + std::to_string(row) + " axis col " + std::to_string(col) +
                       " want value " + std::to_string(v);
            return false;
        }
        return true;
    };
    bool grid_ok = want_axis(prefix, 8, ref_h0, "ref (0,0) height") &&
                   want_axis(prefix, 36, ref_w0, "ref (0,0) width") &&
                   want_axis(prefix + ref_tokens - 1, 8, ref_h0 + ref.height - 1, "ref last-row height") &&
                   want_axis(prefix + ref_tokens - 1, 36, ref_w0 + ref.width - 1, "ref last-row width") &&
                   want_axis(tgt0, 8, tgt_h0, "target (0,0) height") &&
                   want_axis(tgt0, 36, tgt_w0, "target (0,0) width") &&
                   want_axis(tgt_last, 8, tgt_h0 + tgt.height - 1, "target last-row height") &&
                   want_axis(tgt_last, 36, tgt_w0 + tgt.width - 1, "target last-row width");
    check("portrait rope grids centred", grid_ok,
          grid_ok ? "ref (0,0) = (" + std::to_string(ref_h0) + "," + std::to_string(ref_w0) + "), target (0,0) = (" +
                        std::to_string(tgt_h0) + "," + std::to_string(tgt_w0) + ") -- each block its own grid"
                  : grid_bad);
    bool unit_ok = true;
    for (std::size_t k = 0; k + 1 < L.rope.size(); k += 2) {
        const double mm = static_cast<double>(L.rope[k]) * L.rope[k] + static_cast<double>(L.rope[k + 1]) * L.rope[k + 1];
        if (std::fabs(mm - 1.0) > 1e-5) { unit_ok = false; break; }
    }
    check("portrait rope unit modulus", unit_ok, "cos^2+sin^2==1 on every pair of the expanded table");
}

void gate_adjacent_images() {
    // 5 text, then 7 CONTIGUOUS image slots: ref1 (1x2x2)=4 tokens (1 slot), ref2 (1x4x2)=8 tokens
    // (2 slots) with NO text between them, then the target (1x4x4)=16 tokens (4 slots).
    const std::vector<std::uint8_t> m = mask_of({5, 7});
    const ImageShape a{1, 2, 2}, b{1, 4, 2}, t{1, 4, 4};
    const std::vector<ImageShape> shapes{a, b, t};
    const JointLayout L = build_joint_layout(m, shapes);
    check("adjacent expanded len", L.expanded_len == 5 + 4 + 8 + 16 && L.ref_tokens == 12 &&
                                       L.target_tokens == 16,
          "5+4+8+16 = 33 expanded, ref 12 = 4+8");
    bool ids_ok = true;
    for (long long i = 0; i < L.expanded_len; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        long long want = -1;
        if (i >= 5 && i < 9) { want = 0; }
        else if (i >= 9 && i < 17) { want = 1; }
        else if (i >= 17) { want = 2; }
        if (L.image_block_ids[k] != want) { ids_ok = false; }
    }
    check("adjacent stay two blocks", ids_ok, "a run of ones is not one block: ids 0,1,2 by img_shapes");
    // Frame: block a at 5, block b at 5+max(2,2)=7, target at 7+max(4,2)=11.
    check("adjacent frame advances twice",
          frame_of(L.rope, 5) == 5 && frame_of(L.rope, 9) == 7 && frame_of(L.rope, 17) == 11,
          "5 / 5+max(2,2)=7 / 7+max(4,2)=11 (one shared position, advanced per block)");
}

template <typename F>
void throws(const std::string& name, F f) {
    bool threw = false;
    std::string what;
    try {
        (void)f();
    } catch (const std::invalid_argument& e) {
        threw = true;
        what  = e.what();
    } catch (...) {
        what = "wrong exception type";
    }
    check(name, threw, threw ? what : "no invalid_argument thrown");
}

void gate_invariants() {
    // The M1 geometry gate as a runtime refusal: the mask carries 8 image slots (x4 = 32 tokens) while
    // the portrait grids want 4056+4096 -- the vision and VAE planes disagree, so the caller must 500
    // host-side instead of assembling a sequence whose slots cannot all be filled.
    throws("slots x4 mismatch throws", [] {
        const std::vector<ImageShape> shapes{ImageShape{1, 52, 78}, ImageShape{1, 64, 64}};
        return build_joint_layout(mask_of({30, 8}), shapes);
    });
    throws("empty img_shapes throws", [] {
        const std::vector<ImageShape> none;
        return build_joint_layout(mask_of({4, 4}), none);
    });
    throws("temporal extent throws", [] {
        const std::vector<ImageShape> shapes{ImageShape{2, 2, 2}};
        return build_joint_layout(mask_of({4, 4}), shapes);
    });
    throws("all-text mask throws", [] {
        const std::vector<ImageShape> shapes{ImageShape{1, 4, 4}};
        const std::vector<std::uint8_t> all_text(8, 0);
        return build_joint_layout(all_text, shapes);
    });
}

} // namespace

int main() {
    gate_text_only();
    gate_portrait();
    gate_adjacent_images();
    gate_invariants();
    if (!g_ok) { std::printf("IMAGE_JOINT_LAYOUT_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_JOINT_LAYOUT_TEST: PASS (joint index map: text-only rope bit-identical to "
                "build_joint_rope, interleaved portrait blocks/scatter/target mask/frame advance, "
                "adjacent images stay separate, x4 invariant throws)\n");
    return 0;
}
