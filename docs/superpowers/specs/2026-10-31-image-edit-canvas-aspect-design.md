# P2 - an edit's canvas follows the reference aspect (wh_ratio / ratio_follow)

Date: 2026-10-31. Branch: `image-edit`. Package P2 of the Qwen-Image-2.1 parity roadmap.
Depends on: none, but it lands with P1 because P1's golden fixtures pin the geometry this package
changes.

## Problem

`OpenAIImagesRequest` hard-codes `width = 768`, `height = 768` (`src/serve/openai_images.h:32-33`)
and `parse_openai_images_request` only ever overrides them from an explicit `size`
(`src/serve/openai_images.cpp:110-113`, `parse_size` at `:41-69`). So a portrait reference edited
with no `size` is sampled on a square canvas, while `prepare_reference_planes` resized the condition
image to *its own* aspect at a 1024^2 area (`src/serve/reference_image.cpp:155-161`). The model is
conditioned on an aspect its target grid does not have.

The reference pipeline derives the canvas from the last reference image when height/width are
omitted (`image-edit/refs/q21_pipeline_qwenimage21.py:630-636`):

```python
        if height is None or width is None:
            width, height = calculate_dimensions(output_resolution * output_resolution,
                                                 image[-1].size[0] / image[-1].size[1])
        height = height or output_resolution
        width  = width  or output_resolution
```

`calculate_dimensions(area, ratio)` is `sqrt(area*ratio)` rounded to multiples of 32 on both sides -
which is exactly what `snap32` plus the two lines in `reference_image.cpp:155-158` already do for the
*reference*. The helper exists; it is just never applied to the canvas.

The checkpoint's own prompt enhancer expresses the same intent in its answer JSON: `wh_ratio`
("16:9", simplest integers) XOR `ratio_follow` (`"<image1>"`), mutually exclusive, never a size inside
`rewritten_prompt` (`image-edit/refs/q21_system_prompt_edit.txt:63-101, :175-188`). A single-image
edit with no size cue is `ratio_follow = "<image1>"`.

## Decision

Three explicit ways to ask for a canvas, exactly one of which may be given:

| request | canvas |
|---|---|
| `size: "WxH"` | that size, as today (32-multiples, `[32, 2048]`) |
| `wh_ratio: "W:H"` | that ratio, at the configured area preset |
| `ratio_follow: "<image1>"` | the reference's aspect, at the configured area preset |
| none, with a reference | **`ratio_follow` semantics**: the reference's aspect, at the preset |
| none, without a reference | the current 768x768 default (text-to-image is unchanged) |

The area stays a *deployment* policy, not a request semantic: `kCanvasTargetArea = 768 * 768`, the
value the existing comment at `openai_images.h:32-33` justifies for this card (1024^2 OOM'd the
interlude yield). Separating the two means a bigger card changes one constant and no wire behaviour,
and it keeps the joint sequence roughly token-neutral - `ratio_follow` never grows the canvas, it
rotates it. Raising the preset is P14 (2K regime), which also has to raise the attention `S` cap and
add VAE tiling; it is not free and is not in this package.

More than one of `size` / `wh_ratio` / `ratio_follow` is a 400, never a precedence. Silent precedence
would let a client that sends both believe its `size` was honored.

`ratio_follow` naming: `<image1>` is the tag the engine itself renders in `kTi2iImageBlock`
(`src/serve/image_text_tower.cpp:229`), so the wire and the template agree. With
`kMaximumImageReferences == 1` (`openai_images.h:63`), only `<image1>` is accepted; every other index
is a 400 rather than a guess. P9 (multi-reference) widens it.

## Changes

1. `src/serve/openai_images.h`
   - `OpenAIImagesRequest`: add `std::optional<std::string> wh_ratio;` and
     `std::optional<std::string> ratio_follow;`, plus `bool size_requested = false;`. Keep
     `width`/`height` non-optional - every consumer downstream wants a concrete grid.
   - `inline constexpr double kCanvasTargetArea = 768.0 * 768.0;` with the VRAM rationale and the
     pointer to P14.
2. `src/serve/openai_images.cpp`
   - `kAllowedKeys` (`:16-18`) gains `"wh_ratio"`, `"ratio_follow"`.
   - `wh_ratio` must be `"<int>:<int>"`, both > 0, ratio within `[1/64, 64]` (the range the reference
     geometry already survives); 400 `image_ratio_invalid`.
   - `ratio_follow` must be `"<image1>"`; 400 `image_ratio_follow_unsupported` otherwise.
   - XOR check across the three keys: 400 `image_size_conflict`.
   - New pure helper, the single place geometry is decided:
     `void resolve_canvas_geometry(const OpenAIImagesRequest&, std::uint32_t ref_w,
     std::uint32_t ref_h, std::uint32_t& width, std::uint32_t& height);`
     It snaps `sqrt(area*ratio)` / `sqrt(area/ratio)` to 32-multiples with the same banker's-rounding
     helper as the reference geometry, clamps each side into `[kMinimumImageDimension,
     kMaximumImageDimension]`, and leaves `width/height` alone when `size_requested`.
     `snap32` moves out of the anonymous namespace in `reference_image.cpp:23-35` into a shared
     internal header so both call sites use one implementation - two roundings that can drift is the
     bug this package is about.
3. `src/serve/reference_image.h` / `.cpp`
   - `ReferencePlanes` gains `std::uint32_t source_width_px, source_height_px;` (the decoded size
     *before* the snap), so the canvas can be resolved from the picture the request already carries
     without a second decode.
   - Fix the stale doc comment at `reference_image.h:63-64` ("PIL bilinear mapping (no antialias)")
     in the P1 commit; it is what let the wrong filter survive review.
4. `src/serve/image_generation_service.cpp`
   - The reference is prepared at `:619`, well before admission at `:693` and before `geom`/`inputs`
     at `:175-194`. Call `resolve_canvas_geometry` on the resolved planes there and use its result for
     `desc`, `schedule_cache_for`, `inputs.height_px/width_px` and `image_need_bytes`. One geometry
     decision, made before any VRAM is taken.
5. `src/serve/chat_image_tool.cpp`
   - The `generate_image` tool schema (`:19-31`) gains `wh_ratio` and `ratio_follow` string
     properties with the rules above in their descriptions, and the `size` description says plainly
     that omitting all three with a reference follows the reference's aspect. This is what makes the
     behaviour discoverable to a fresh session on any client: the tool schema is the only contract a
     client that never read a file has.

## Errors

| condition | HTTP | code |
|---|---|---|
| two of the three canvas keys | 400 | `image_size_conflict` |
| `wh_ratio` malformed / out of range | 400 | `image_ratio_invalid` |
| `ratio_follow` not `<image1>` | 400 | `image_ratio_follow_unsupported` |
| `size` unchanged | 400 | `image_size_invalid` |

All are parse-time, so none of them reach the catch-all `image_generation_failed` (that collapse is
P6).

## Tests

- `tests/test_openai_images_schema.cpp` (existing; `test_reference_image_field` at `:214`, runner at
  `:458`): the XOR matrix, `wh_ratio` shapes accepted/rejected, `ratio_follow` accepted/rejected,
  and that `size` still wins its own path byte-for-byte.
- Host geometry test for `resolve_canvas_geometry`: `3:4` reference -> portrait canvas whose sides are
  32-multiples and whose area is within one snap of the preset; `16:9`; `1:1`; extreme `1:64` clamped;
  text-to-image untouched at 768x768; and the property that `S` (latent + reference + text tokens)
  does not exceed what a 768^2 request costs.
- No GPU work is required by this package.

## Out of scope

Raising `kCanvasTargetArea` and anything it drags in (attention `S` cap, VAE tiling) - P14.
Multi-reference `ratio_follow` indices - P9. Server-side PE emitting these fields - P10.
