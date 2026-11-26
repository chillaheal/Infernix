#pragma once

// Qwen-Image-2.1 post-tower text conditioning: turn the Qwen3-VL text tower's pre-final-RMSNorm hidden
// state into the `prompt_embeds` / `encoder_attention_mask` the DiT consumes
// (docs/maintainer/qwen-image-2-1-variables.md, gap #1). Host-only, device-free: the CUDA execution
// runs the tower to `pre_norm_hidden`, then calls this on the host to apply the prompt-dependent
// drop + padding that is pure data movement, not arithmetic.
//
// This reproduces the reference pipeline's post-encode step line-for-line (image-spike/oracle/
// dump_text_tower.py): per batch item, gather the VALID (unmasked) token rows in order, drop the
// leading `drop_idx` of them (the system-role turn, computed host-side from the chat template -- see
// the tokenize slice), then right-pad every item to the longest dropped length and emit the matching
// attention mask. The drop is mask-aware (it drops the first `drop_idx` VALID rows, never raw index
// positions), which is what makes it correct under left padding and for the variable batch of
// N-condition-image / num_images_per_prompt jobs -- the general case the single-item oracle cannot
// exercise. The tower forward itself is qualified on-device (infernix_image_texttower_prod_cuda_test).

#include <cstddef>
#include <vector>

namespace infernix::models::qwen_image {

// The DiT text conditioning for one encode batch. prompt_embeds is [batch, max_len, dim] row-major
// (flat b*max_len*dim + l*dim + d), right-padded with zeros; encoder_attention_mask is [batch, max_len]
// with 1 on a real (kept) token and 0 on a pad slot. max_len is the longest kept length across the batch.
struct TextEmbeds {
    int batch   = 0;
    int max_len = 0;
    int dim     = 0;
    std::vector<float> prompt_embeds;           // [batch * max_len * dim]
    std::vector<int>   encoder_attention_mask;  // [batch * max_len]
};

// Build prompt_embeds from the tower's pre_norm_hidden [batch, seq_len, dim] and its padding
// attention_mask [batch, seq_len] (1 = valid token, 0 = pad), dropping the leading `drop_idx` valid
// rows per item. A negative `drop_idx` is clamped to 0; an item with fewer than `drop_idx` valid rows
// contributes an empty kept set. dim is the hidden width (4096 for the Qwen3-VL text tower).
[[nodiscard]] TextEmbeds build_prompt_embeds(const float* pre_norm_hidden, int batch, int seq_len,
                                             int dim, const int* attention_mask, int drop_idx);

} // namespace infernix::models::qwen_image
