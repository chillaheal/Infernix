// CPU numeric oracle for the FULL Qwen-Image-2.1 DiT top level (num_layers=1 build).
//
// Reproduces, in naive float32 (double accumulators), everything on the outside of the single
// transformer block that the block oracle (test_image_block.cpp) already qualifies: the img_in /
// txt_in projections, the sinusoidal timestep embedding + TimestepEmbedding with causal_condition
// t-doubling, the shared global modulation, the 3-axis RoPE positional grid, build_token_metadata,
// the joint text+image repeat_interleave assembly, and the norm_out / proj_out tail. It then runs
// the block and compares the final velocity prediction against the upstream sample.npy, plus every
// intermediate the upstream forward captured. All 32 blocks are structurally identical to block 0,
// so this qualifies the whole forward graph except block-to-block chaining.
//
// Needs the oracle .npy dump (INFERNIX_IMAGE_ORACLE = the dir holding tensors/,
// tensors_toplevel_weights/, tensors_block_weights/). Skips (77) when absent.

#include "qwen_image_oracle_ref.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qo = qioracle;

int main() {
    const char* root_env = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (root_env == nullptr || root_env[0] == '\0') {
        std::printf("SKIP: INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    const std::string root(root_env);
    const std::string T = qo::join(root, "tensors");
    const std::string TW = qo::join(root, "tensors_toplevel_weights");
    const std::string BW = qo::join(root, "tensors_block_weights");

    auto F = [&](const std::string& dir, const std::string& name) { return qo::read_npy(qo::join(dir, name)).floats(); };

    // ---- constants (from tensors/manifest.json) ----
    const std::size_t D = 4096;
    const std::size_t heads = 32;
    const std::size_t head_dim = D / heads;    // 128
    const float EPS = 1e-6F;
    const std::size_t text_prefix = 8;
    const std::size_t H = 8, W = 8;
    const std::size_t img_tokens = H * W;      // 64
    const std::size_t slots = 16;              // target_tokens / 4
    const std::size_t vlm = text_prefix + slots;  // 24
    const std::size_t S = text_prefix + img_tokens;  // 72
    const std::size_t td = 256;
    const double time_factor = 1000.0;

    // ---- inputs ----
    const std::vector<float> in_hidden = F(T, "in_hidden_states.npy");       // [64,64]
    const std::vector<float> in_enc = F(T, "in_encoder_hidden_states.npy");  // [8,4096]
    const std::vector<float> in_t = F(T, "in_timestep.npy");                 // [1]
    const std::vector<std::uint8_t> in_mask = qo::read_npy(qo::join(T, "in_img_mask.npy")).bools();  // [24]

    // ---- toplevel weights ----
    const std::vector<float> w_img_in = F(TW, "toplevel_img_in_weight.npy");                    // [4096,64]
    const std::vector<float> w_tnorm = F(TW, "toplevel_txt_in_text_norm_weight.npy");           // [4096]
    const std::vector<float> w_in_layer = F(TW, "toplevel_txt_in_in_layer_weight.npy");         // [4096,4096]
    const std::vector<float> w_out_layer = F(TW, "toplevel_txt_in_out_layer_weight.npy");       // [4096,4096]
    const std::vector<float> w_tl1 = F(TW, "toplevel_time_text_embed_timestep_embedder_linear_1_weight.npy");  // [4096,256]
    const std::vector<float> w_tl2 = F(TW, "toplevel_time_text_embed_timestep_embedder_linear_2_weight.npy");  // [4096,4096]
    const std::vector<float> w_mod = F(TW, "toplevel_modulation_1_weight.npy");                 // [16384,4096]
    const std::vector<float> w_nolin = F(TW, "toplevel_norm_out_linear_weight.npy");            // [4096,4096]
    const std::vector<float> w_proj = F(TW, "toplevel_proj_out_weight.npy");                    // [64,4096]

    // ---- block-0 weights (owned so the BlockWeights pointers stay valid) ----
    qo::BlockWeights bw;
    static std::vector<std::vector<float>> owned;
    owned.reserve(9);
    owned.push_back(F(BW, "blk0_attn_to_q_weight.npy"));
    owned.push_back(F(BW, "blk0_attn_to_k_weight.npy"));
    owned.push_back(F(BW, "blk0_attn_to_v_weight.npy"));
    owned.push_back(F(BW, "blk0_attn_to_out_0_weight.npy"));
    owned.push_back(F(BW, "blk0_attn_norm_q_weight.npy"));
    owned.push_back(F(BW, "blk0_attn_norm_k_weight.npy"));
    owned.push_back(F(BW, "blk0_img_mlp_proj_weight.npy"));
    owned.push_back(F(BW, "blk0_img_mlp_gate_layer_weight.npy"));
    owned.push_back(F(BW, "blk0_img_mlp_out_weight.npy"));
    bw.to_q = &owned[0];
    bw.to_k = &owned[1];
    bw.to_v = &owned[2];
    bw.to_out = &owned[3];
    bw.norm_q = &owned[4];
    bw.norm_k = &owned[5];
    bw.mlp_proj = &owned[6];
    bw.mlp_gate = &owned[7];
    bw.mlp_out = &owned[8];

    int failures = 0;
    auto check = [&](const char* name, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
        const double r = qo::rel_fro(got, ref);
        const bool ok = (r < tol) && std::isfinite(r);
        std::printf("  %-16s rel_fro=%.3e  tol=%.1e  %s\n", name, r, tol, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    };

    // ---- img_in ----
    std::vector<float> hs_img;
    qo::matmul_xwt(in_hidden, w_img_in, hs_img, img_tokens, D, 64);
    check("img_in", hs_img, F(T, "img_in.npy"), 2e-3);

    // ---- txt_in (ZeroCenterRMSNorm -> in_layer -> gelu(tanh) -> out_layer) ----
    std::vector<float> tn, tmid;
    qo::zerocenter_rmsnorm(in_enc, tn, w_tnorm, EPS);
    qo::matmul_xwt(tn, w_in_layer, tmid, text_prefix, D, D);
    for (float& x : tmid) x = qo::gelu_tanh(x);
    std::vector<float> txt_out;
    qo::matmul_xwt(tmid, w_out_layer, txt_out, text_prefix, D, D);
    check("txt_in", txt_out, F(T, "txt_in.npy"), 2e-3);

    // ---- timestep embedding with causal_condition t-doubling: rows [t, 0] ----
    std::vector<float> tvec = {in_t[0], 0.0F};
    std::vector<float> tproj = qo::temporal_timesteps(tvec, td, time_factor);  // [2,256]
    std::vector<float> temb_a, temb;
    qo::matmul_xwt(tproj, w_tl1, temb_a, 2, D, td);
    for (float& x : temb_a) x = qo::silu(x);
    qo::matmul_xwt(temb_a, w_tl2, temb, 2, D, D);  // [2,4096]
    check("temb", temb, F(T, "temb.npy"), 2e-3);

    // ---- global modulation = Linear(SiLU(temb)), [2, 4D] ----
    std::vector<float> silu_temb(temb.size());
    for (std::size_t i = 0; i < temb.size(); ++i) silu_temb[i] = qo::silu(temb[i]);
    std::vector<float> modulation;
    qo::matmul_xwt(silu_temb, w_mod, modulation, 2, 4 * D, D);
    check("modulation", modulation, F(T, "modulation.npy"), 2e-3);

    // ---- token metadata + RoPE positional grid ----
    std::vector<std::uint8_t> tmask(S);
    for (std::size_t s = 0; s < S; ++s) tmask[s] = (s >= text_prefix) ? 1 : 0;
    const std::vector<std::uint8_t> ref_tmask = qo::read_npy(qo::join(T, "block_target_mask.npy")).bools();
    {
        int tm_bad = 0;
        for (std::size_t s = 0; s < S; ++s) if ((ref_tmask[s] ? 1 : 0) != tmask[s]) ++tm_bad;
        std::printf("  %-16s mismatches=%d  %s\n", "target_mask", tm_bad, tm_bad ? "FAIL" : "OK");
        if (tm_bad) ++failures;
    }
    const std::vector<float> rot = qo::build_rope(text_prefix, H, W);  // [72,64] re/im interleaved
    check("rotary", rot, F(T, "block_rotary.npy"), 5e-3);

    // ---- joint assembly: [txt_out(8), zeros(16)] repeat_interleave([1*8,4*16]) + place img_in ----
    std::vector<std::size_t> repeats(vlm);
    for (std::size_t i = 0; i < vlm; ++i) repeats[i] = in_mask[i] ? 4 : 1;
    std::vector<float> joint_rows(vlm * D, 0.0F);
    for (std::size_t r = 0; r < text_prefix; ++r)
        for (std::size_t d = 0; d < D; ++d) joint_rows[r * D + d] = txt_out[r * D + d];
    // remaining slots stay zero
    std::vector<float> block_in(S * D, 0.0F);
    std::vector<std::uint8_t> ipad(S, 0);
    {
        std::size_t pos = 0;
        for (std::size_t r = 0; r < vlm; ++r) {
            for (std::size_t k = 0; k < repeats[r]; ++k, ++pos) {
                ipad[pos] = in_mask[r] ? 1 : 0;
                for (std::size_t d = 0; d < D; ++d) block_in[pos * D + d] = joint_rows[r * D + d];
            }
        }
        qo::require(pos == S, "joint expansion length mismatch");
    }
    {
        std::size_t it = 0;
        for (std::size_t s = 0; s < S; ++s) {
            if (!ipad[s]) continue;
            for (std::size_t d = 0; d < D; ++d) block_in[s * D + d] = hs_img[it * D + d];
            ++it;
        }
        qo::require(it == img_tokens, "image placement count mismatch");
    }
    check("block_in", block_in, F(T, "block_in.npy"), 2e-3);

    // ---- the block (already independently qualified) ----
    std::vector<float> block_out;
    qo::block_forward(block_in, modulation, rot, tmask, bw, S, D, heads, text_prefix, EPS, block_out);
    check("block_out", block_out, F(T, "block_out.npy"), 2e-3);

    // ---- norm_out (scale-only AdaLayerNormContinuous) ----
    std::vector<float> nlin, scale_sel, normed, norm_outv;
    qo::matmul_xwt(silu_temb, w_nolin, nlin, 2, D, D);
    scale_sel.assign(S * D, 0.0F);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t row = tmask[s] ? 0 : 1;
        for (std::size_t d = 0; d < D; ++d) scale_sel[s * D + d] = nlin[row * D + d];
    }
    qo::layernorm(block_out, normed, S, D, EPS);
    norm_outv.assign(S * D, 0.0F);
    for (std::size_t i = 0; i < S * D; ++i) norm_outv[i] = normed[i] * (1.0F + scale_sel[i]);
    check("norm_out", norm_outv, F(T, "norm_out.npy"), 2e-3);

    // ---- proj_out -> sample ----
    std::vector<float> sample;
    qo::matmul_xwt(norm_outv, w_proj, sample, S, 64, D);
    check("sample", sample, F(T, "sample.npy"), 2e-3);

    if (failures) {
        std::printf("IMAGE_FORWARD_TEST: FAIL (%d stage(s))\n", failures);
        return 1;
    }
    std::printf("IMAGE_FORWARD_TEST: PASS (S=%zu D=%zu heads=%zu head_dim=%zu prefix=%zu)\n", S, D, heads, head_dim, text_prefix);
    return 0;
}
