// Step 8 (8D): the PRODUCTION raw-prompt tower route, end to end on device.
//
// Drives serve/image_text_tower.h's make_tower_prompt_embeds_provider -- the exact function
// POST /v1/images/generations runs for a prompt STRING -- through its real surfaces: the processor's
// compiled chat template + genuine VL tokenizer (8C host-qualified), the real FP8-v2 .infernix
// artifact's streamed text tower (bind_text_encoder), and the FP8 RowScale embed_tokens dequantized
// once into the BF16 host cache.
//
// NUMERICAL GATE (rewritten twice on evidence; every earlier finding preserved as a metric).
//   Round 1-2 gate: the FP8 route's rows vs the HF BF16 dump's pre_norm_hidden at 2e-2. WRONG oracle
//     -- the HF tensors are DIFFERENT weights from this artifact's FP8 planes, so it measured the
//     artifact's own weight-quantization delta on top of any route error (rel_fro=1.394e-01, kept
//     visible below). Per the doctrine -- "Packed inputs are independently decoded with their stored
//     scales. Qualify production routes directly against that oracle, not another kernel or plausible
//     model output" -- the oracle is rebuilt from THIS v2 artifact's FP8 RowScale planes.
//   Round-3 gate: device route vs that decoded-weight oracle, but the oracle ran f32 ACTIVATION math
//     (copied from the BF16 tower cert). It passed the 33-token marks item at 6.575e-02 but the
//     25-token "a black cat" hit 1.235e-01 -- marginally over the 1e-1 that basis was given. ROOT
//     CAUSE (not a defect): the qualified BF16 tower cert documents that "device-vs-f32 is dominated
//     by bf16-ACTIVATION rounding (device runs bf16 like the model, the host runs f32)... so the local
//     bound must clear the bf16 floor", and it therefore does NOT gate device-vs-f32 (1e-1 is a
//     per-layer LOCALIZATION bound there); it gates the bf16-vs-bf16 device-vs-pre_norm at 2e-2, which
//     is unavailable here because the FP8 weights differ from the HF BF16 dump. A bf16 device route
//     against an f32 oracle carries that bf16 floor, and it is input-dependent (6.6e-2 .. 1.24e-1).
//   Fix (no tolerance loosening): the oracle now stores activations in BF16 at every inter-op
//     boundary -- the SAME BF16-represented operands the device receives (the tower is BF16 activations
//     end to end; ops::linear/rmsnorm/residual_add/rope/attention all take and return BF16) -- while
//     keeping double accumulators WITHIN each reduction. With operand values matching on both sides
//     (bf16 activations, e4m3-decoded FP8 weights, embed rounded to BF16 exactly as the provider's
//     host cache does), device-vs-oracle now measures only the kernels' accumulation order, which the
//     FP8 A16 op cert bounds per matmul at relative-L2 3.9e-3 (one BF16 unit roundoff). A genuine
//     route defect still blows the gate: the known pre-GQA-fix K/V-layout error was 1.56e+00.
//   The pre_norm comparison is PRESERVED as two reported metrics -- decoded-oracle-vs-pre_norm (the
//     FP8 weight-quantization delta) and device-vs-pre_norm (the round-2 1.394e-01 disclosure) -- so
//     the whole criterion basis, including the adverse numbers, stays auditable.
//
// ids and the drop count come from the production renderer (make_tower_prompt_renderer), the same
// code the provider runs internally; its agreement with the oracle manifest's drop_idx and with the
// pre_norm row count is gated for the ascii item (that guards the pre_norm metric slice).
//
// The #17 ledger is gated at 0 image-owned device bytes before and after each call (tower Buffers
// fully released). Skips (77) without INFERNIX_TEST_ARTIFACT / INFERNIX_QWENVL_PROCESSOR /
// INFERNIX_IMAGE_ORACLE; returns 2 (runnable) with no CUDA device.

#include "serve/image_text_tower.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/weight.h"
#include "models/qwen_image/image_pipeline.h"
#include "models/qwen_image/image_weights.h"
#include "models/qwen_image/qwen_image_oracle_ref.h"
#include "ops/common/canonical_math.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace qo  = qioracle;
namespace qi  = infernix::models::qwen_image;
namespace art = infernix::artifact;
namespace sv  = infernix::serve;

namespace {

// device-vs-BF16-operand-FP64-accumulation oracle: pure kernel accumulation across 36 layers. A real
// route defect was 1.56e+00; the per-matmul accumulation bound is 3.9e-3, so 1e-1 clears the
// accumulation floor by a wide margin while still failing the known-defect regime by ~15x.
constexpr double TOL_ROUTE = 1e-1;

float from_bf16(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// Store a float vector in place as BF16 (round-to-nearest-even, then read back), matching the BF16
// activation tensor every production tower op writes between ops. The oracle keeps double
// accumulators WITHIN each reduction; only the stored operand representation is made to match the
// device, so device-vs-oracle reflects accumulation order, not the (accepted, non-defect) bf16-vs-f32
// activation gap the f32-activation oracle of rounds 1-3 put into the gate.
void round_bf16(std::vector<float>& v) {
    for (float& f : v) f = from_bf16(infernix::ops::canon::f32_to_bf16_rn(f));
}

bool g_ok = true;
void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("  %-44s %s  %s\n", name.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    std::fflush(stdout);
    if (!ok) g_ok = false;
}
void metric(const std::string& name, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rel_fro=%.3e (METRIC, not gate)", value);
    std::printf("  %-44s --  %s\n", name.c_str(), buf);
    std::fflush(stdout);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---- artifact decode (the independent oracle's weight source) ----------------------------------------
// Same plane semantics the production loader enforces (serve/image_text_tower.cpp dequantize):
// FP8 RowScale objects are [codes n*k e4m3][scales n BF16 at geometry.scale_offset], the value being
// e4m3(byte & 0x7F) * +/-bf16(scale[row]).

std::vector<float> decode_fp8_linear(const art::Reader& reader, const art::ParameterReference& ref,
                                     std::size_t n, std::size_t k) {
    qo::require(ref.binding.parts.size() == 1, "fp8 linear binds more than one object");
    const auto object         = ref.binding.parts.front().object;
    const infernix::WeightGeometry& g = reader.geometry(object);
    qo::require(g.format == infernix::QType::FP8_E4M3FN_ROW_BF16, "linear is not FP8_E4M3FN_ROW_BF16");
    qo::require(g.layout == infernix::QuantLayout::RowScale, "linear is not RowScale");
    qo::require(g.code_bytes == n * k, "linear code plane size mismatch");
    qo::require(g.scale_bytes == n * 2, "linear scale plane size mismatch");
    qo::require(g.scale_offset >= g.code_bytes, "linear scale plane overlaps the code plane");
    const std::vector<std::byte> obj = reader.read_object(object);
    qo::require(obj.size() >= g.scale_offset + g.scale_bytes, "linear object shorter than its geometry");
    const auto* codes  = reinterpret_cast<const std::uint8_t*>(obj.data());
    const auto* scales = reinterpret_cast<const std::uint16_t*>(obj.data() + g.scale_offset);
    std::vector<float> w(n * k);
    for (std::size_t row = 0; row < n; ++row) {
        const float scale = infernix::ops::canon::bf16_to_f32(scales[row]);
        for (std::size_t col = 0; col < k; ++col) {
            const std::uint8_t byte = codes[row * k + col];
            w[row * k + col] = infernix::ops::canon::e4m3_value(byte & 0x7FU) * ((byte & 0x80U) ? -scale : scale);
        }
    }
    return w;
}

std::vector<float> decode_bf16_vector(const art::Reader& reader, const art::ParameterReference& ref,
                                      std::size_t count) {
    qo::require(ref.binding.parts.size() == 1, "norm binds more than one object");
    const auto object = ref.binding.parts.front().object;
    qo::require(reader.geometry(object).format == infernix::QType::BF16, "norm vector is not BF16");
    const std::vector<std::byte> obj = reader.read_object(object);
    qo::require(obj.size() == count * 2, "norm vector byte count mismatch");
    const auto* bits = reinterpret_cast<const std::uint16_t*>(obj.data());
    std::vector<float> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = from_bf16(bits[i]);
    return v;
}

// One layer's roles, decoded from the artifact (the same binding the provider streams).
struct HostLayer {
    std::vector<float> q, k, v, o, gate, up, down, iln, pln, qn, kn;
};

HostLayer decode_layer(const art::Reader& reader, const qi::TextTowerWeights::Layer& L,
                       std::size_t D, std::size_t kv_width, std::size_t I, std::size_t HD) {
    HostLayer h;
    h.q    = decode_fp8_linear(reader, L.q, D, D);
    h.k    = decode_fp8_linear(reader, L.k, kv_width, D);
    h.v    = decode_fp8_linear(reader, L.v, kv_width, D);
    h.o    = decode_fp8_linear(reader, L.attention_output, D, D);
    h.gate = decode_fp8_linear(reader, L.mlp_gate, I, D);
    h.up   = decode_fp8_linear(reader, L.mlp_up, I, D);
    h.down = decode_fp8_linear(reader, L.mlp_down, D, I);
    h.iln  = decode_bf16_vector(reader, L.input_norm, D);
    h.pln  = decode_bf16_vector(reader, L.post_attention_norm, D);
    h.qn   = decode_bf16_vector(reader, L.query_norm, HD);
    h.kn   = decode_bf16_vector(reader, L.key_norm, HD);
    return h;
}

// ---- naive FP32 tower (double accumulators), the qualified pattern of the BF16 tower cert --------

void rmsnorm_rows(const std::vector<float>& x, std::vector<float>& y, std::size_t rows, std::size_t d,
                  const std::vector<float>& w, float eps) {
    y.assign(x.size(), 0.0F);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + r * d;
        float* yr       = y.data() + r * d;
        double ss = 0.0;
        for (std::size_t i = 0; i < d; ++i) ss += static_cast<double>(xr[i]) * xr[i];
        const float inv = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + eps);
        for (std::size_t i = 0; i < d; ++i) yr[i] = static_cast<float>(static_cast<double>(xr[i]) * inv * w[i]);
    }
}

// NeoX split-half RoPE over the head (full rotation), in place. inv_freq[i] = theta^(-2i/HD).
void rope_host(std::vector<float>& x, std::size_t heads, std::size_t S, const std::vector<int>& pos,
               const std::vector<float>& inv_freq, std::size_t HD) {
    const std::size_t half = HD / 2;
    for (std::size_t s = 0; s < S; ++s) {
        for (std::size_t h = 0; h < heads; ++h) {
            float* p = x.data() + (s * heads + h) * HD;
            for (std::size_t i = 0; i < half; ++i) {
                const double ang = static_cast<double>(pos[s]) * inv_freq[i];
                const float c    = std::cos(static_cast<float>(ang));
                const float sn   = std::sin(static_cast<float>(ang));
                const float x0   = p[i];
                const float x1   = p[i + half];
                p[i]             = x0 * c - x1 * sn;
                p[i + half]      = x1 * c + x0 * sn;
            }
        }
    }
}

// Fully-causal GQA attention: q [S,HQ,HD], k/v [S,HKV,HD] (kv head = h/GROUP), out [S,HQ,HD].
void attn_gqa(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
              std::vector<float>& o, std::size_t S, std::size_t HQ, std::size_t HKV, std::size_t HD,
              float scale, std::size_t GROUP) {
    o.assign(S * HQ * HD, 0.0F);
    std::vector<float> scores(S), pr(S);
    for (std::size_t s = 0; s < S; ++s) {
        for (std::size_t h = 0; h < HQ; ++h) {
            const std::size_t hk = h / GROUP;
            const float* qp      = q.data() + (s * HQ + h) * HD;
            for (std::size_t kk = 0; kk <= s; ++kk) {
                const float* kp = k.data() + (kk * HKV + hk) * HD;
                double dot = 0.0;
                for (std::size_t d = 0; d < HD; ++d) dot += static_cast<double>(qp[d]) * kp[d];
                scores[kk] = static_cast<float>(dot) * scale;
            }
            double mx  = -std::numeric_limits<double>::infinity();
            for (std::size_t kk = 0; kk <= s; ++kk) mx = std::max(mx, static_cast<double>(scores[kk]));
            double sum = 0.0;
            for (std::size_t kk = 0; kk <= s; ++kk) { pr[kk] = std::exp(scores[kk] - mx); sum += pr[kk]; }
            for (std::size_t kk = 0; kk <= s; ++kk) pr[kk] /= sum;
            float* op = o.data() + (s * HQ + h) * HD;
            for (std::size_t d = 0; d < HD; ++d) {
                double acc = 0.0;
                for (std::size_t kk = 0; kk <= s; ++kk) acc += pr[kk] * v.data()[(kk * HKV + hk) * HD + d];
                op[d] = static_cast<float>(acc);
            }
        }
    }
}

// One decoder layer in place on x [S,D] (identical math to the BF16 tower cert's hforward, over the
// artifact-decoded values).
void run_host_layer(const HostLayer& L, std::vector<float>& x, const std::vector<int>& pos,
                    const std::vector<float>& inv_freq, const qi::TextTowerDims& d) {
    const std::size_t D = d.hidden, HQ = d.heads, HKV = d.kv_heads, HD = d.head_dim, I = d.intermediate;
    const std::size_t GROUP = HQ / HKV;
    const std::size_t S     = x.size() / D;
    const float scale       = 1.0F / std::sqrt(static_cast<float>(HD));
    std::vector<float> h, q, k, v, attn, attn_o, h2, gate, up, hidden, down;

    // Each op output is stored BF16 to match the device's BF16 activation tensors; the residual x is
    // kept BF16 across layers exactly as ops::residual_add stores it. Accumulators stay double within
    // each matmul/softmax/rmsnorm reduction, so this is the same operand representation, not a mirror
    // of the kernels' arithmetic.
    rmsnorm_rows(x, h, S, D, L.iln, d.eps);       round_bf16(h);
    qo::matmul_xwt(h, L.q, q, S, D, D);            round_bf16(q);
    qo::matmul_xwt(h, L.k, k, S, HKV * HD, D);     round_bf16(k);
    qo::matmul_xwt(h, L.v, v, S, HKV * HD, D);     round_bf16(v);
    qo::rmsnorm_head(q, S * HQ, HD, L.qn, d.eps);  round_bf16(q);
    qo::rmsnorm_head(k, S * HKV, HD, L.kn, d.eps); round_bf16(k);
    rope_host(q, HQ, S, pos, inv_freq, HD);         round_bf16(q);
    rope_host(k, HKV, S, pos, inv_freq, HD);        round_bf16(k);
    attn_gqa(q, k, v, attn, S, HQ, HKV, HD, scale, GROUP); round_bf16(attn);
    qo::matmul_xwt(attn, L.o, attn_o, S, D, D);     round_bf16(attn_o);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn_o[i];
    round_bf16(x);
    rmsnorm_rows(x, h2, S, D, L.pln, d.eps);        round_bf16(h2);
    qo::matmul_xwt(h2, L.gate, gate, S, I, D);       round_bf16(gate);
    qo::matmul_xwt(h2, L.up, up, S, I, D);           round_bf16(up);
    hidden.resize(S * I);
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = qo::silu(gate[i]) * up[i];
    round_bf16(hidden);
    qo::matmul_xwt(hidden, L.down, down, S, D, I);   round_bf16(down);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += down[i];
    round_bf16(x);
}

// ---- items ---------------------------------------------------------------------------------------

struct Item {
    std::string label, dir, prompt;
    std::vector<int> ids;
    std::size_t drop   = 0;
    std::vector<std::uint16_t> bits; // the device route's rows [drop,S)
    std::vector<float> host_x;       // the decoded oracle's residual stream
    std::vector<float> pren;         // optional reference rows [drop,S) (metric slice only)
};

// Reads the embed object, checks the plane layout once, and gathers ids[t] into x with the SAME
// BF16 rounding the provider's host cache uses (so the residual streams start bit-identical).
void gather_embed_x0(const art::Reader& reader, const qi::TextTowerWeights& tw,
                     const std::vector<Item>& items, std::vector<std::byte>& scratch) {
    const art::ParameterReference& ref = tw.embed_tokens;
    qo::require(ref.binding.parts.size() == 1, "embed_tokens binds more than one object");
    const auto object                   = ref.binding.parts.front().object;
    const infernix::WeightGeometry& g   = reader.geometry(object);
    const std::size_t vocab             = tw.dims.vocab;
    const std::size_t hidden            = tw.dims.hidden;
    qo::require(g.format == infernix::QType::FP8_E4M3FN_ROW_BF16, "embed_tokens is not FP8_E4M3FN_ROW_BF16");
    qo::require(g.layout == infernix::QuantLayout::RowScale, "embed_tokens is not RowScale");
    qo::require(g.code_bytes == vocab * hidden, "embed_tokens code plane size mismatch");
    qo::require(g.scale_bytes == vocab * 2, "embed_tokens scale plane size mismatch");
    scratch = reader.read_object(object);
    qo::require(scratch.size() >= g.scale_offset + g.scale_bytes, "embed_tokens object shorter than geometry");
    const auto* codes  = reinterpret_cast<const std::uint8_t*>(scratch.data());
    const auto* scales = reinterpret_cast<const std::uint16_t*>(scratch.data() + g.scale_offset);
    for (const Item& item : items) {
        std::vector<float>& x = const_cast<Item&>(item).host_x;
        x.assign(item.ids.size() * hidden, 0.0F);
        for (std::size_t t = 0; t < item.ids.size(); ++t) {
            const long long id = item.ids[t];
            qo::require(id >= 0 && static_cast<std::size_t>(id) < vocab, "token id outside the embed table");
            const std::size_t row  = static_cast<std::size_t>(id);
            const float scale      = infernix::ops::canon::bf16_to_f32(scales[row]);
            float* xrow            = x.data() + t * hidden;
            for (std::size_t col = 0; col < hidden; ++col) {
                const std::uint8_t byte = codes[row * hidden + col];
                const float value       = infernix::ops::canon::e4m3_value(byte & 0x7FU) *
                                          ((byte & 0x80U) ? -scale : scale);
                xrow[col] = from_bf16(infernix::ops::canon::f32_to_bf16_rn(value));
            }
        }
    }
}

int run() {
    const char* art_env  = std::getenv("INFERNIX_TEST_ARTIFACT");
    const char* proc_env = std::getenv("INFERNIX_QWENVL_PROCESSOR");
    const char* ora_env  = std::getenv("INFERNIX_IMAGE_ORACLE");
    if (!art_env || !proc_env || !ora_env) {
        std::printf("SKIP: INFERNIX_TEST_ARTIFACT / INFERNIX_QWENVL_PROCESSOR / INFERNIX_IMAGE_ORACLE not set\n");
        return 77;
    }
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("NO_CUDA_DEVICE\n");
        std::fflush(stdout);
        return 2;
    }

    const std::string oracle(ora_env);
    std::printf("provider artifact=%s\n", art_env);
    const auto provider = sv::make_tower_prompt_embeds_provider(art_env, proc_env);
    const auto renderer = sv::make_tower_prompt_renderer(proc_env);

    art::Reader reader(art_env);
    art::Binder binder(reader);
    const qi::TextTowerWeights tw = qi::bind_text_encoder(binder);
    (void)std::move(binder).finish();
    const qi::TextTowerDims dims = tw.dims;
    qo::require(dims.heads % dims.kv_heads == 0, "heads not divisible by kv_heads");
    qo::require(tw.layers.size() == dims.layers, "bound layer count mismatch");
    std::printf("tower dims: layers=%u hidden=%u heads=%u kv=%u hd=%u inter=%u vocab=%u theta=%.0f eps=%.1e\n",
                dims.layers, dims.hidden, dims.heads, dims.kv_heads, dims.head_dim, dims.intermediate,
                dims.vocab, dims.rope_theta, dims.eps);

    std::vector<Item> items;
    for (const std::string& dir : {qo::join(oracle, "tensors_text_tower"),
                                   qo::join(oracle, "tensors_text_tower_marks")}) {
        const bool     is_ascii  = items.empty();
        const std::string manifest_path = qo::join(dir, "manifest.json");
        if (!file_exists(manifest_path)) {
            std::printf("  %-44s SKIP  no manifest.json (%s)\n",
                        is_ascii ? "ascii" : "marks", dir.c_str());
            qo::require(!is_ascii, "ascii oracle has no manifest.json");
            continue;
        }
        Item item;
        item.label        = is_ascii ? "ascii" : "marks";
        item.dir          = dir;
        const nlohmann::json manifest = nlohmann::json::parse(read_file(manifest_path));
        item.prompt       = manifest.at("prompt").get<std::string>();
        const long long manifest_drop = manifest.at("drop_idx").get<long long>();
        const sv::TowerPrompt rendered = renderer(item.prompt);
        item.ids          = rendered.ids;
        item.drop         = rendered.drop_rows;
        qo::require(item.drop < item.ids.size(), "drop_rows >= ids");
        std::printf("%s: prompt=\"%s\" S=%zu drop=%zu\n", item.label.c_str(), item.prompt.c_str(),
                    item.ids.size(), item.drop);

        const std::string pren_path = qo::join(dir, "pre_norm_hidden.npy");
        if (file_exists(pren_path)) {
            qo::NpyArray pren = qo::read_npy(pren_path);
            qo::require(pren.shape.size() == 3 && pren.shape[0] == 1 && pren.shape[2] == dims.hidden,
                        "pre_norm_hidden is not [1,S,hidden]");
            char buf[96];
            std::snprintf(buf, sizeof(buf), "renderer S=%zu pre_norm S=%zu", item.ids.size(), pren.shape[1]);
            check(item.label + " renderer S matches pre_norm", pren.shape[1] == item.ids.size(), buf);
            std::snprintf(buf, sizeof(buf), "renderer drop=%zu manifest drop_idx=%lld", item.drop, manifest_drop);
            check(item.label + " drop matches manifest", static_cast<long long>(item.drop) == manifest_drop, buf);
            if (g_ok && pren.shape[1] == item.ids.size()) {
                const std::vector<float> all = pren.floats();
                item.pren.assign(all.begin() + static_cast<std::ptrdiff_t>(item.drop * dims.hidden), all.end());
            }
        } else {
            std::printf("  %-44s --  no pre_norm_hidden.npy (metrics skipped)\n",
                        (item.label + " pre_norm availability").c_str());
        }

        const std::uint64_t before = qi::image_device_bytes_live();
        check(item.label + " ledger idle before", before == 0, "live=" + std::to_string(before));
        item.bits = provider(item.prompt);
        const std::uint64_t after = qi::image_device_bytes_live();
        check(item.label + " ledger released after", after == 0, "live=" + std::to_string(after));

        const std::size_t want = (item.ids.size() - item.drop) * dims.hidden;
        check(item.label + " output shape", item.bits.size() == want,
              "got " + std::to_string(item.bits.size()) + " want " + std::to_string(want));
        qo::require(item.bits.size() == want, "provider output shape mismatch");
        items.push_back(std::move(item));
    }

    // The host oracle pass over the artifact-decoded weights: each layer decodes once and advances
    // every item's residual stream (the tower weights are the same for both prompts).
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::byte> embed_scratch;
    gather_embed_x0(reader, tw, items, embed_scratch);
    std::vector<float> inv_freq(dims.head_dim / 2);
    for (std::size_t i = 0; i < inv_freq.size(); ++i)
        inv_freq[i] = std::pow(dims.rope_theta, -2.0F * static_cast<float>(i) / static_cast<float>(dims.head_dim));
    for (std::size_t l = 0; l < dims.layers; ++l) {
        const HostLayer L = decode_layer(reader, tw.layers[l], dims.hidden,
                                         static_cast<std::size_t>(dims.kv_heads) * dims.head_dim,
                                         dims.intermediate, dims.head_dim);
        for (Item& item : items) {
            std::vector<int> pos(item.ids.size());
            for (std::size_t t = 0; t < pos.size(); ++t) pos[t] = static_cast<int>(t);
            run_host_layer(L, item.host_x, pos, inv_freq, dims);
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  host oracle layer %2zu/%u done (%.0fs)\n", l + 1, dims.layers, el);
        std::fflush(stdout);
    }

    for (Item& item : items) {
        const std::size_t hidden = dims.hidden;
        std::vector<float> got(item.bits.size());
        for (std::size_t i = 0; i < item.bits.size(); ++i) got[i] = from_bf16(item.bits[i]);
        const std::vector<float> host_rows(item.host_x.begin() + static_cast<std::ptrdiff_t>(item.drop * hidden),
                                           item.host_x.end());

        const double e_route = qo::rel_fro(got, host_rows);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "rel_fro=%.3e (tol %.0e, bf16 operands, fp64 accum)", e_route, TOL_ROUTE);
        check(item.label + " FP8 route vs BF16-operand FP64 oracle", std::isfinite(e_route) && e_route < TOL_ROUTE, buf);

        if (!item.pren.empty()) {
            metric(item.label + " decoded oracle vs pre_norm (quant delta)", qo::rel_fro(host_rows, item.pren));
            metric(item.label + " FP8 route vs pre_norm (round-2 1.394e-01)", qo::rel_fro(got, item.pren));
        }
    }

    std::printf(g_ok ? "IMAGE_TOWER_PROVIDER_REAL_TEST: PASS\n" : "IMAGE_TOWER_PROVIDER_REAL_TEST: FAIL\n");
    std::fflush(stdout);
    return g_ok ? 0 : 1;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::printf("IMAGE_TOWER_PROVIDER_REAL_TEST EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
