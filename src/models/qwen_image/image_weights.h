#pragma once

// Host-side binding of the Qwen-Image-2.1 image job's weights from a production .infernix artifact.
// The logical parameter names, shapes and representations are the converter recipe's contract
// (tools/convert/qwen_image.py): the recipe's header makes those names the operand authority the
// loader binds to, and it derives every shape from the artifact's own component config. All
// parameters bind at Residency::Streamed -- the weights stay in the artifact's files and an image
// job reads each object through bounded staging (peak == one object) and uploads it into the VRAM
// its interlude yielded; nothing weight-sized is materialized up front.
//
// This binding IS the loader gate: a missing binding, a shape that differs from the Binding
// coverage, or a representation that differs from the recipe's (the 17 quantized roles are FP8 E4M3
// row-scaled, everything else direct BF16) throws artifact::ArtifactError when the plan is built --
// before any quiesce, yield or device work.

#include "artifact/binder.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace infernix::models::qwen_image {

// The DiT architecture dimensions, derived from the artifact's "dit" component config with the
// recipe's own expressions.
struct DiTDims {
    std::uint32_t layers  = 0;
    std::uint32_t heads   = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t hidden  = 0; // heads * head_dim
    std::uint32_t intermediate = 0; // hidden * mlp_ratio
    std::uint32_t image_in  = 0; // in_channels * patch_size^2 (dit/img_in columns)
    std::uint32_t image_out = 0; // out_channels * patch_size^2 (dit/proj_out rows)
    std::uint32_t context_in = 0;
    float eps = 0.0F;
};

struct TextTowerDims {
    std::uint32_t layers  = 0;
    std::uint32_t hidden  = 0;
    std::uint32_t heads   = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t intermediate = 0;
    std::uint32_t vocab   = 0;
    float rope_theta = 0.0F;
    float eps        = 0.0F; // rms_norm_eps
};

// One streamed reference per DiT-block role (the seven recipe-quantized linears plus the two
// BF16 QK norms the qualified block forward consumes).
struct DiTBlockWeights {
    artifact::ParameterReference q, k, v, attention_output, query_norm, key_norm;
    artifact::ParameterReference mlp_gate, mlp_up, mlp_down;
};

struct DiTWeights {
    DiTDims dims;
    artifact::ParameterReference img_in, proj_out, modulation, norm_out;
    artifact::ParameterReference txt_in_in, txt_in_out, txt_in_norm;
    artifact::ParameterReference time_linear_1, time_linear_2;
    std::vector<DiTBlockWeights> blocks;
    // Streaming bounds of the plan: the largest single object any demand addresses.
    std::uint64_t max_object_bytes       = 0;
    std::uint64_t max_block_object_bytes = 0; // one block's nine roles (one block resident at a time)
};

struct TextTowerWeights {
    TextTowerDims dims;
    artifact::ParameterReference embed_tokens, final_norm;
    struct Layer {
        artifact::ParameterReference input_norm, post_attention_norm;
        artifact::ParameterReference q, k, v, query_norm, key_norm, attention_output;
        artifact::ParameterReference mlp_gate, mlp_up, mlp_down;
    };
    std::vector<Layer> layers;
    std::uint64_t max_object_bytes = 0;
};

// The named VAE-decoder parameters the production decode route consumes, keyed by their artifact
// binding names ("vae/..." -- the recipe maps every VAE source tensor one-to-one as a direct BF16
// parameter, so the source key IS the contract). The decode walks a fixed chain: every operand it
// needs is a key here; a role the artifact no longer stores throws at bind time, not mid-decode.
struct VaeWeights {
    std::map<std::string, artifact::ParameterReference> by_name;
    std::uint64_t max_object_bytes = 0;
};

// The VAE per-channel latent denormalization pair (docs/maintainer/qwen-image-2-1-variables.md
// "Fixed model constants"): the artifact's "vae" component config carries the recipe's stored
// arrays; resolution-independent, read once and never a per-job input.
struct VaeDenorm {
    std::vector<float> mean; // [image_in]
    std::vector<float> std;  // [image_in]
};

// Read the denormalization pair from the artifact's "vae" component config. A missing key, a
// non-array value, a non-number entry or a length mismatch throws artifact::ArtifactError.
[[nodiscard]] VaeDenorm read_vae_denorm(artifact::Binder& binder);

// Resolve every logical DiT / text-tower / VAE (decoder + #75 encoder/quant_conv) parameter through
// the caller's Binder (so the caller's finish() sees one combined, streamed-only MaterializationPlan).
[[nodiscard]] DiTWeights bind_dit(artifact::Binder& binder);
[[nodiscard]] TextTowerWeights bind_text_encoder(artifact::Binder& binder);
[[nodiscard]] VaeWeights bind_image_vae(artifact::Binder& binder);

} // namespace infernix::models::qwen_image
