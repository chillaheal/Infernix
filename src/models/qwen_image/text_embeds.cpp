#include "models/qwen_image/text_embeds.h"

#include <algorithm>

namespace infernix::models::qwen_image {

TextEmbeds build_prompt_embeds(const float* pre_norm_hidden, int batch, int seq_len, int dim,
                               const int* attention_mask, int drop_idx) {
    TextEmbeds out;
    out.batch = batch;
    out.dim   = dim;
    if (batch <= 0 || seq_len < 0 || dim <= 0) return out;
    drop_idx = std::max(0, drop_idx);

    // Pass 1: kept length per item = (#valid rows) - drop_idx, and the batch max. #valid counts only
    // attention_mask==1 rows, so left/right padding never counts toward the drop.
    const auto valid_count = [&](int b) {
        int c = 0;
        for (int l = 0; l < seq_len; ++l) c += attention_mask[b * seq_len + l] != 0 ? 1 : 0;
        return c;
    };
    int max_len = 0;
    for (int b = 0; b < batch; ++b)
        max_len = std::max(max_len, std::max(0, valid_count(b) - drop_idx));
    out.max_len = max_len;
    out.prompt_embeds.assign(static_cast<std::size_t>(batch) * max_len * dim, 0.0F);
    out.encoder_attention_mask.assign(static_cast<std::size_t>(batch) * max_len, 0);

    // Pass 2: copy the kept (valid, post-drop) rows into each item's slots in order, right-padded.
    for (int b = 0; b < batch; ++b) {
        int kept = 0, seen = 0;
        for (int l = 0; l < seq_len; ++l) {
            if (attention_mask[b * seq_len + l] == 0) continue;
            if (seen++ < drop_idx) continue;
            const float* src = pre_norm_hidden + (static_cast<std::size_t>(b) * seq_len + l) * dim;
            float* dst       = out.prompt_embeds.data() + (static_cast<std::size_t>(b) * max_len + kept) * dim;
            for (int d = 0; d < dim; ++d) dst[d] = src[d];
            out.encoder_attention_mask[b * max_len + kept] = 1;
            ++kept;
        }
    }
    return out;
}

} // namespace infernix::models::qwen_image
