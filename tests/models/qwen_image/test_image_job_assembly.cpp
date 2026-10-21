// Variable-input image generation (#37): the variable-resolution job-assembly INTEGRATION gate.
//
// Each production module is already individually verified against an oracle (scheduler bitwise + grid,
// build_joint_rope bitwise + grid, build_img_mask bitwise + synthetic, fill_normal moments). The one thing
// a SINGLE fixed-resolution oracle can never prove is that the modules AGREE with each other for an
// arbitrary (h, w): the rope joint length, the img_mask joint length, the scheduler's token count and the
// init-latent element count must all describe the SAME job. This test drives the real production API
// (ImageGeometry::derive / ImageJob::schedule / build_joint_rope / build_img_mask / fill_normal) across
// square, non-square, snapped-to-32, tall and tiny geometries with varying text_len and steps, and asserts
// the cross-module invariants that make one coherent job descriptor:
//   1. packed_h*packed_w == tokens/4                     (2x2 packing consistent)
//   2. build_joint_rope rows == text_len + packed_tokens (rope spans the whole joint sequence)
//   3. build_img_mask len == build_joint_rope rows       (rope and mask describe the SAME joint length)
//   4. img_mask == [0]*text_len + [1]*packed             (text slots 0, target slots 1)
//   5. geometry tokens == 4 * packed_tokens              (the scheduler's mu-arg is the UNPACKED count,
//                                                         exactly 4x the rope/latent packed grid)
//   6. schedule: sigmas == steps+1, timesteps == steps   (schedule length tracks the caller's steps)
//   7. init latent has packed_tokens*64 elements and is finite (right size for [packed,64], no NaN/inf)
//
// Host-only, no oracle, no GPU -- always runs. Guards the variable-resolution assembly the single oracle
// row cannot. (The end-to-end GPU wiring of these into the interlude is the remaining, blackout-gated step.)

#include "models/qwen_image/image_job.h"
#include "models/qwen_image/scheduler.h"
#include "models/qwen_image/rope.h"
#include "models/qwen_image/img_mask.h"
#include "models/qwen_image/noise.h"
#include "models/qwen_image/text_embeds.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace infernix::models::qwen_image;

bool g_ok = true;
void expect(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-40s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    if (!ok) g_ok = false;
}

void one_case(std::uint32_t h_px, std::uint32_t w_px, int text_len, std::uint32_t steps) {
    const std::string tag = std::to_string(h_px) + "x" + std::to_string(w_px) + " tp=" +
                            std::to_string(text_len) + " steps=" + std::to_string(steps);

    const ImageGeometry g = ImageGeometry::derive(h_px, w_px);
    const std::uint32_t packed_tokens = g.tokens / 4;
    const std::uint32_t packed_h = g.hl / 2, packed_w = g.wl / 2;
    expect("packing consistent " + tag,
          g.hl % 2 == 0 && g.wl % 2 == 0 && packed_h * packed_w == packed_tokens && g.tokens == 4 * packed_tokens,
          "hl=" + std::to_string(g.hl) + " wl=" + std::to_string(g.wl) + " tokens=" + std::to_string(g.tokens) +
              " packed=" + std::to_string(packed_tokens));

    const std::vector<float> rope = build_joint_rope(text_len, static_cast<int>(packed_h), static_cast<int>(packed_w));
    const std::size_t joint = static_cast<std::size_t>(text_len) + packed_tokens;
    expect("rope spans joint " + tag, rope.size() == joint * 128,
          "rows=" + std::to_string(rope.size() / 128) + " joint=" + std::to_string(joint));

    const std::vector<std::uint8_t> base(static_cast<std::size_t>(text_len), std::uint8_t{0});  // text-only: all-zero
    const std::vector<std::uint8_t> mask = build_img_mask(base.data(), 1, text_len, static_cast<int>(packed_tokens));
    expect("mask==rope joint " + tag, mask.size() == rope.size() / 128,
          "mask=" + std::to_string(mask.size()) + " rope_rows=" + std::to_string(rope.size() / 128));

    bool mask_layout = mask.size() == joint;
    for (std::size_t i = 0; mask_layout && i < joint; ++i) {
        const std::uint8_t want = (i < static_cast<std::size_t>(text_len)) ? 0 : 1;
        if (mask[i] != want) mask_layout = false;
    }
    expect("mask text0/target1 " + tag, mask_layout, "[0]*tp + [1]*packed");

    const SchedulerConfig cfg;  // model-fixed config
    ImageJob job; job.height = h_px; job.width = w_px; job.steps = steps;
    const Schedule sch = job.schedule(cfg);
    expect("schedule length tracks steps " + tag,
          sch.sigmas.size() == static_cast<std::size_t>(steps) + 1 && sch.timesteps.size() == steps,
          "sigmas=" + std::to_string(sch.sigmas.size()) + " timesteps=" + std::to_string(sch.timesteps.size()) +
              " steps=" + std::to_string(steps));

    const std::size_t latent = static_cast<std::size_t>(packed_tokens) * 64;  // [packed, 64] channels
    std::vector<float> lat(latent);
    fill_normal(1234ULL, std::span<float>(lat));
    bool lat_finite = lat.size() == latent;
    for (std::size_t i = 0; lat_finite && i < lat.size(); ++i)
        if (!std::isfinite(lat[i])) lat_finite = false;
    expect("init latent count + finite " + tag, lat_finite,
           "count=" + std::to_string(lat.size()) + " == packed*64");
}

// The real interlude does NOT hard-code text_len: it DERIVES it from the text tower via
// build_prompt_embeds -- the DiT joint sequence's text block is the KEPT (unpadded, post-drop) token count
// from encoder_attention_mask. The geometry cases above only ASSUMED a text_len; this gate closes the
// ORIGIN -> CONSUMPTION link: prompt_embeds emits text_len, and rope + img_mask consume the SAME text_len.
void gate_text_len() {
    const int dim = 4;

    // Single item (the oracle path): seq_len 13 all-valid, drop the 2-token system turn -> 11 kept, no
    // padding -> max_len 11, mask all-kept. That 11 IS the oracle text_len, now derived not assumed.
    {
        std::vector<int> am(13, 1);
        std::vector<float> h(13 * dim, 0.5f);
        const TextEmbeds te = build_prompt_embeds(h.data(), 1, 13, dim, am.data(), 2);
        expect("prompt_embeds single max_len==11", te.max_len == 11 && te.batch == 1,
               "max_len=" + std::to_string(te.max_len));
        int kept = 0;
        for (int l = 0; l < te.max_len; ++l) kept += te.encoder_attention_mask[l];
        expect("prompt_embeds single mask all-kept", kept == te.max_len,
               "kept=" + std::to_string(kept) + " of " + std::to_string(te.max_len));
        expect("prompt_embeds single embeds sized",
               te.prompt_embeds.size() == static_cast<std::size_t>(te.max_len) * dim,
               "embeds=" + std::to_string(te.prompt_embeds.size()));

        // CHAIN: the DERIVED text_len -> rope + img_mask at 256x256 (packed 64) -> joint 75.
        const int text_len = te.max_len;
        const std::size_t packed = 64;
        const std::vector<float> rope = build_joint_rope(text_len, 8, 8);  // hl/2, wl/2
        expect("derived text_len -> rope joint", rope.size() / 128 == static_cast<std::size_t>(text_len) + packed,
               "rows=" + std::to_string(rope.size() / 128) + " = text_len " + std::to_string(text_len) + " + 64");
        const std::vector<std::uint8_t> base(static_cast<std::size_t>(text_len), std::uint8_t{0});
        const std::vector<std::uint8_t> mask = build_img_mask(base.data(), 1, text_len, static_cast<int>(packed));
        bool layout = mask.size() == static_cast<std::size_t>(text_len) + packed;
        for (std::size_t i = 0; layout && i < mask.size(); ++i)
            if (mask[i] != (i < static_cast<std::size_t>(text_len) ? 0 : 1)) layout = false;
        expect("derived text_len -> img_mask [0]*tl+[1]*64", layout, "joint " + std::to_string(mask.size()));
    }

    // Batch: the DiT text_len is max_len = LONGEST kept across items (right-padded uniformly). item0 6 valid
    // drop 2 -> 4 kept; item1 8 valid drop 2 -> 6 kept -> max_len 6. Asserts text_len is the padded width.
    {
        std::vector<int> am = {1, 1, 1, 1, 1, 1, 0, 0,   // item0
                               1, 1, 1, 1, 1, 1, 1, 1};  // item1
        std::vector<float> h(2 * 8 * dim, 0.25f);
        const TextEmbeds te = build_prompt_embeds(h.data(), 2, 8, dim, am.data(), 2);
        expect("prompt_embeds batch max_len==longest kept", te.max_len == 6 && te.batch == 2,
               "max_len=" + std::to_string(te.max_len) + " (item0 4, item1 6)");
        int k0 = 0, k1 = 0;
        for (int l = 0; l < te.max_len; ++l) {
            k0 += te.encoder_attention_mask[0 * te.max_len + l];
            k1 += te.encoder_attention_mask[1 * te.max_len + l];
        }
        expect("prompt_embeds batch per-item kept", k0 == 4 && k1 == 6,
               "row sums " + std::to_string(k0) + ", " + std::to_string(k1));
    }
}

} // namespace

int main() {
    gate_text_len();
    const struct { std::uint32_t h, w; int tp; std::uint32_t steps; } cases[] = {
        {512, 512, 11, 40},    // oracle geometry -> joint 267 (11 + 256)
        {256, 256, 11, 40},    // joint 75 (11 + 64), matches the img_mask oracle
        {1024, 1024, 11, 40},  // large square
        {512, 256, 7, 8},      // non-square, small steps
        {768, 512, 20, 20},    // tall, long text, mid steps
        {32, 32, 3, 2},         // tiny: packed 1, joint 4
        {500, 300, 11, 40},    // snapped to 480x288 -> odd packed (135)
        {525, 199, 5, 12},     // snapped to 512x192
    };
    for (const auto& c : cases) one_case(c.h, c.w, c.tp, c.steps);
    if (!g_ok) { std::printf("IMAGE_JOB_ASSEMBLY_TEST: FAIL\n"); return 1; }
    std::printf("IMAGE_JOB_ASSEMBLY_TEST: PASS (variable-resolution job descriptor self-consistent across "
                "geometries: rope==img_mask joint length, tokens==4*packed, schedule length, latent size)\n");
    return 0;
}
