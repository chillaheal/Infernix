# P5 - the text tower's M-RoPE axis split comes from the artifact, and fails closed

Date: 2026-10-31. Branch: `image-edit`. Package P5 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_image_dit_weights_test` (host, reads the real artifact).

## Problem

The condition text tower rotates its head with M-RoPE: each pair of dimensions in the 128-wide head
belongs to one of three axes (temporal, height, width), and the interleaved rule decides which. In
`src/models/qwen_image/text_tower.cu` that split was a compile-time `constexpr int kMropeSection[3]` -
the checkpoint's `[24, 20, 20]` typed into the source as if it were a property of the engine.

Nothing enforced that the two stayed together. A converted artifact whose `text_config.rope_scaling`
said something else - a different head width, a different split, `mrope_interleaved: false` - would
load, run, and produce *plausible* embeddings with the wrong rotation: the kind of defect that shows up
as "the pictures got a bit worse" and costs a day of bisection. The kernel also did not know whether
the checkpoint asked for the interleaved walk at all; it just always did it.

The pattern for doing this properly already exists in the repository: `src/models/qwen3_5/config.cpp:69-104`
parses and validates `mrope_section` when binding the chat model. The image tower had not been held to
it.

## Decision

Read the sections at bind time, require the interleaved flag, and validate against the axis ranges the
kernel actually implements. No fallback value anywhere.

`TextTowerDims` gains the fields (`src/models/qwen_image/image_weights.h:56-57`):

```cpp
    std::array<std::uint32_t, 3> mrope_section{};
    bool mrope_interleaved = false;
```

`bind_text_encoder` (`src/models/qwen_image/image_weights.cpp:206-241`) then:

- requires `text_config.rope_scaling` to be present, and `mrope_interleaved` to be a `true` boolean;
- requires `mrope_section` to be three unsigned numbers (`is_number_unsigned()` +
  `get<std::uint64_t>()`, the spelling this file already uses for config integers - the
  `get<std::int64_t>()` version MSVC rejects here), and their sum to equal `head_dim / 2`;
- walks the interleaved rule back - `pair % 3 == 1 && pair < 3*section[1]` -> H, `pair % 3 == 2 &&
  pair < 3*section[2]` -> W, otherwise T - counts the pairs each axis receives, and refuses if that
  count is not the declared section ("text M-RoPE sections exceed the interleaved axis ranges").

That last step is the one that earns the package. A section list can be self-consistent (`24/20/20`) and
still be unreachable by the interleaved walk for a given head width; requiring the sum is not enough,
and requiring the walk to reproduce the declared sections is what makes an inconsistent artifact
impossible to load.

The kernel now takes the spans as arguments (`src/models/qwen_image/text_tower.cu:85-99`): host-side
`h_span = 3*mrope_section[1]`, `w_span = 3*mrope_section[2]`, and `if (d % 3 == 1 && d < h_span) axis = 1;
if (d % 3 == 2 && d < w_span) axis = 2;`. The file's header comment says plainly that the split is not a
constant of this file and points at `text_config.rope_scaling.mrope_section` as its source.
`checked_dims` (`src/models/qwen_image/text_tower.cu:121-127`) independently refuses a dims whose
`mrope_interleaved` is false or whose sections do not sum to `head_dim / 2`, so a caller that built the
struct by hand instead of binding it cannot reach the kernel either.

This is fail-closed on purpose: the artifact on this machine
(`C:/Users/Micke/Desktop/Infernix FlashNext/models/Qwen-Image-2.1-FP8-v4.infernix`, its
`text_encoder/config.json`) says `mrope_section [24, 20, 20]`, `mrope_interleaved: true`,
`rope_theta 5000000`, and the converter materialises that. A checkpoint that says otherwise is a
different model and must be told about explicitly, not silently accommodated.

## Tests

`tests/models/qwen_image/test_image_dit_weights.cpp:137-141`, against the real artifact:

```cpp
        require(text.dims.mrope_interleaved, ...);
        require(text.dims.mrope_section[0] == 24 && text.dims.mrope_section[1] == 20 &&
                    text.dims.mrope_section[2] == 20,
                "text mrope_section differs from the checkpoint's 24/20/20");
```

so a conversion that drops or changes the field fails the suite rather than the pictures.

What is *not* tested: the rejection paths themselves (`mrope_interleaved` false, a non-unsigned or
mis-summing section, a section the interleaved walk cannot reproduce). Reaching them needs a doctored
`text_encoder/config.json`, which the current bind tests have no fixture for - both `checked_dims` and
the kernel live in a `.cu` translation unit. The requirement that this package can actually violate in
practice (an artifact that loses or changes the field) is the one the test above pins.

## Out of scope

Per-axis rope theta, the DiT's own positional scheme (not M-RoPE), and re-qualifying the numeric
output against the reference - the certification run's job, unchanged by this package.
