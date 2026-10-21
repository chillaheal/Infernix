#pragma once

// Qwen-Image-2.1 variable-prompt (#37) front half: turn a prompt's token ids into the Qwen3-VL text
// tower's INPUT embedding by gathering embed_tokens rows. The reference builds the tower input as
// inputs_embeds = embed_tokens[input_ids]: a pure row lookup of the BF16 embedding table, widened to
// FP32 (image-spike/verify_embed_gather.py proves this reproduces the dumped inputs_embeds.npy bit-exact).
// This is the paired bookend to build_prompt_embeds (which runs AFTER the tower): together they are the
// pure-data movement around the (on-device-qualified) 36-layer forward, so a raw prompt string can drive
// the whole chain. Host-only, device-free. The engine gathers from its materialized embed_tokens weight;
// here the table is a caller-supplied BF16 row-major [vocab, hidden] buffer (raw IEEE bits: BF16 is the
// top 16 bits of the FP32 value), so the gather is exact, not rounded.

#include <cstdint>
#include <vector>

namespace infernix::models::qwen_image {

// Gather one embedding row per id from the [vocab, hidden] BF16 table and widen each row to FP32 exactly
// (BF16 -> FP32 is the top-16-bit widening, so no precision is introduced or lost). Output is
// [n, hidden] row-major FP32, matching the layout the tower forward consumes. Throws
// std::invalid_argument if embed_bf16/ids is null, the table/shape is degenerate, or any id is outside
// [0, vocab). hidden is the model hidden width (4096 for the Qwen3-VL text tower).
[[nodiscard]] std::vector<float> build_inputs_embeds(const std::uint16_t* embed_bf16, long long vocab,
                                                     int hidden, const long long* ids, int n);

} // namespace infernix::models::qwen_image
