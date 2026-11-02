# P4 - two separate prompt bounds: a wire cap, and the model's joint sequence

Date: 2026-10-31. Branch: `image-edit`. Package P4 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_image_joint_layout_test`, `infernix_image_generation_service_test`,
`infernix_http_routes_test` (all host, no device).

## Problem

Two different limits had been collapsed into one constant, and it was the wrong one for both jobs.

`kMaximumImagePromptBytes` was 4096 (`docs/superpowers/specs/2026-10-31-image-edit-qwen21-parity-roadmap.md:21`).
That is a transport bound - what one POST may carry - but it was also acting as the model bound. It
rejected the kind of prompt this model is designed for: the shipped edit prompt enhancer answers a
long, structured instruction, and Qwen-Image-2.1's text-rendering use (lettering inside the picture)
routinely needs several thousand bytes of description. Rejecting those at 4 KiB made the feature
unreachable while telling the client nothing about why.

The limit that actually matters is the joint attention sequence: text rows + reference tokens + target
tokens must fit `ops::kAttnDitJointMaxTokens` = 20480, which is *not* a property of the prompt alone -
the same prompt is fine on 768^2 and over the ceiling on 2048^2. Before this package nothing checked
it on the host: an over-long request reached the joint attention kernel and failed there, after the
interlude had already yielded the LLM's VRAM, as the collapsed 500 that P6 fixes.

## Decision

Two independent bounds, neither of which truncates.

1. **Wire.** `inline constexpr std::size_t kMaximumImagePromptBytes = 16ULL * 1024;`
   (`src/serve/openai_images.h:79`), with a comment keeping it separate from the model bound. Over it:
   a 400 at parse time. Nothing is ever silently truncated - a truncated instruction is a different
   picture, which is the worst possible failure mode for an edit.
2. **Model.** A host admission before any VRAM is reserved, after both text lanes are rendered and
   before `image_need_bytes` (`src/serve/image_generation_service.cpp:725-745`), run over *both* lanes:

```cpp
    const qi::ImageGeometry canvas = qi::ImageGeometry::derive(effective.height, effective.width);
    const std::pair<const char*, std::size_t> lanes[] = {
        {"prompt", prompt.tokens()}, {"negative prompt", negative_prompt.tokens()}};
    for (const auto& [lane, rows] : lanes) {
        const long long joint = qi::joint_token_count(static_cast<long long>(rows),
            static_cast<long long>(vision_tokens), static_cast<long long>(canvas.tokens));
        if (joint > ops::kAttnDitJointMaxTokens) {
            throw ApiException{ApiError{400, "invalid_request_error",
                std::string("the ") + lane + " is too long for this canvas: it makes a joint sequence of " +
                    std::to_string(joint) + " tokens, and the model serves at most " +
                    std::to_string(ops::kAttnDitJointMaxTokens) + " (a smaller size or a shorter " +
                    std::string(lane) + " both work)", "prompt", "image_prompt_too_long"}};
        }
    }
```

The arithmetic is not restated here. `joint_token_count`
(`src/models/qwen_image/joint_layout.h:90`, defined `src/models/qwen_image/joint_layout.cpp:130`)
returns `prompt_rows + 3*vision_tokens + target_tokens` and is the same shape `build_joint_layout`
produces - `expanded = text_rows + 4*image_slots` with the mask holding `prompt_rows + target_slots`
ones and `vision_slots + target_slots` reference-side ones. `prompt_rows` includes the expanded
`<|image_pad|>` rows; the header says so because getting that wrong is an off-by-`4*vision` error that
would admit requests the kernel then rejects. `target_tokens` is a multiple of 4 (2x2 packs) and
negative or non-multiple-of-4 inputs throw.

The ceiling moved to the public header (`include/infernix/ops/attn_dit_joint.h:15`) so serve can admit
against the exact constant the kernel enforces (`src/ops/attn_dit_joint/attn_dit_joint.cpp:17` now
aliases it). One number, two users - that is the whole point.

The message names both numbers and both fixes, because the caller is usually a chat model deciding
what to do next. The bound is also advertised (`src/serve/http_server.cpp:669-675`):

```cpp
{"prompt", Json{{"max_bytes", kMaximumImagePromptBytes},
                {"truncated", false},
                {"too_long", "HTTP 400 image_prompt_too_long, naming the joint "
                             "sequence and its ceiling"}}},
```

The per-request half (does *this* prompt fit *this* canvas) cannot be advertised as a single number, so
the document says which error answers it.

## Errors

| condition | HTTP | code |
|---|---|---|
| prompt over the wire cap | 400 | parse-time `invalid_request_error` (unchanged route) |
| prompt or negative prompt over the joint ceiling | 400 | `image_prompt_too_long` |

Both are pre-start, so neither is ever reported as a draw failure.

## Tests

- `tests/models/qwen_image/test_joint_layout.cpp:389-424` (`gate_token_count`, called from `main`):
  three real layouts, `{37,1024}`/`{{1,64,64}}`, `{30,1014,12,1024}`/`{{1,78,52},{1,64,64}}`,
  `{5,7}`/`{{1,2,2},{1,2,4},{1,4,4}}`. The counts are not typed in - they are read back off the
  built mask (`target_slots = shapes.back().tokens()/4`, `vision = slots - target_slots`,
  `prompt_rows = mask_len - target_slots`) and compared with `L.expanded_len`, so the formula is pinned
  to the layout builder rather than to a hand-computed constant. Plus `joint_token_count(10, 0, 17)`
  throws.
- `tests/test_image_generation_service.cpp:860-892`: a 2048x2048 canvas (target 16384) with a renderer
  producing 4096 rows is served and reaches the probe (`probes == 1`); 4097 rows is a 400
  `image_prompt_too_long` with `probes == 0` and an empty store - the refusal never touches the device
  and leaves no job behind.
- `tests/test_http_routes.cpp`: the capability's `prompt` block matches the constant, says
  `truncated: false`, and names the error code.

## Out of scope

Raising the ceiling (P14, which also needs VAE tiling and a VRAM policy), token-level budget
negotiation, and any prompt rewriting (P10).
