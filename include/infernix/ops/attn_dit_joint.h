#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Qwen-Image-2.1 DiT joint block-causal softmax attention over the text-prefix / image-suffix
 * sequence.
 *
 * The DiT runs one attention over the joint sequence S = text_prefix + image_tokens. The mask is
 * block-causal: a text query token (s < text_prefix) attends only to text tokens up to and
 * including itself, so it sees strictly causal text and never any image token; an image query token
 * (s >= text_prefix) attends to the entire sequence (all text plus all image), which is the
 * bidirectional image suffix. Equivalently, a key token kk is visible to query s iff
 *
 *   allow(s, kk) = (s >= text_prefix) || (kk <= s)
 *
 * The score is (q . k) * 1/sqrt(head_dim), the row is softmaxed over the visible keys, and the
 * result is the probability-weighted sum of the value rows. No softcap and no attention dropout.
 * Every row has at least one visible key (its own diagonal kk = s), so no row is fully masked.
 *
 * All three operands and the output use the channel-first head layout [head_dim, heads, S, 1] with
 * head_dim (128) fastest, then heads, then the token. head_dim is fixed at 128; a caller dispatches
 * here only for a 128-wide head. heads is 1..40 and S is 1..16384. text_prefix is 0..S: 0 makes the
 * whole sequence one bidirectional image block, S makes it fully causal text. q, k, v, out are
 * BF16, contiguous and 4-byte aligned; out is a separate buffer that must not overlap q, k or v.
 * q, k and v may alias each other only if the caller has already written them.
 *
 * The independent oracle evaluates the masked softmax naively in FP64 from the BF16-represented
 * operands over the same channel-first index. The Op owns no workspace and no persistent state.
 */
void attn_dit_joint(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t text_prefix,
                    Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
