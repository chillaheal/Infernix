# P14 - The served operating point, and the four constants that hold it down

Date: 2026-10-31. Branch: `image-edit`. Package P14 of the Qwen-Image-2.1 parity roadmap.
**Status: designed, staged, not implemented.** Every number in the plan has to come from a measured
device peak, and the card is held by `serve.exe`; the reason this is not coded anyway is at the bottom.

## Problem

The reference pipeline's operating point is an **area**, not a size: `output_resolution: int = 1024`
(`image-edit/refs/q21_pipeline_qwenimage21.py:533`), used three ways --

- the canvas: `calculated_width, calculated_height = calculate_dimensions(output_resolution *
  output_resolution, aspect)` and `height = height or calculated_height` (:630-636),
- and the **condition images**: `input_width, input_height = calculate_dimensions(output_resolution *
  output_resolution, image_width / image_height)` (:665-666).

Our served default is `kCanvasTargetArea = 768.0 * 768.0` (`src/serve/openai_images.h:110`) -- 56% of the
model's own target area -- and the same snap is what the reference plane is resized through. So the edit
route feeds the model a smaller canvas *and* a smaller reference than the checkpoint's pipeline ever does.
That is a quality gap, not a preference: the reference's numbers are the ones the model was demonstrated
at.

And 1024² is not hypothetical on this machine: the `generate_image` tool's own default -- 1024x1024, 40
steps -- has been served end to end here (the bird edit, 166 s). The route already runs at the model's
area today; what is missing is the *default* and everything above it.

## Why it is not a constant change

The comment at `openai_images.h:108-109` already says raising it "also has to raise the joint-attention
token ceiling and add VAE tiling; it is not a constant change on its own". It is worse than that, in a way
worth naming precisely: **there are two ceilings and the admission check only sees one of them.**

| what | where | value |
|---|---|---|
| joint attention refusal | `src/ops/attn_dit_joint/attn_dit_joint.cpp:17,47,73` (`kMaxTokens = kAttnDitJointMaxTokens`) | 20480 |
| rope refusal | `src/ops/rope_c64/rope_c64.cpp:17` -- its own `constexpr std::int32_t kMaxTokens = 20480;` | 20480 |
| the serve's admission | `src/serve/image_generation_service.cpp:810` checks `joint > ops::kAttnDitJointMaxTokens` | one constant |

Raising only `ops::kAttnDitJointMaxTokens` therefore lets a request pass admission, yield the LLM's VRAM,
allocate the job, and *then* die inside `rope_c64` -- the exact class of failure
`src/models/qwen_image/joint_layout.h:87` warns about ("a job that dies on that throw inside" the pipeline
after the reservation). The two constants must move together, or better, rope_c64's private copy must
become the shared one so the situation cannot recur.

The arithmetic of the ceiling, with downscale 16 and a square canvas of side `s` and a same-size reference:

```
rows = 2 * (s / 16)^2 + text_rows            <= 20480   ->   s <= ~1619 px
```

So the current ceiling already admits roughly a 1600² edit with a same-size reference. A 2048² square edit
needs `2 * 128^2 = 32768` rows plus text -- it needs the ceiling at ~33k, which is what "the 2K regime"
actually means. `attn_dit_joint`'s workspace and the per-step `d_joint`/`d_hb` ping-pong
(`image_pipeline.cu:894-896`) both scale with `S`, and the ledger predicts them, so the ceiling raise is a
ledger event, not a `#define` edit.

The fourth constant nobody counts is the VAE. There is **no tiling anywhere**: the encoder gets the whole
reference plane and the decoder the whole latent, so the peak grows with pixel count in the decoder's
intermediate feature maps. Above ~1.5K² the DiT is no longer the thing that runs out.

## Design, staged so each step is a separate measured change

**P14a -- serve at the model's area (1024²).** Change `kCanvasTargetArea` to `1024.0 * 1024.0`, and make
the reference plane's resize fill the same area rather than only snapping to the 32-grid -- the two belong
together because the reference pipeline sizes the condition image with the *same* call it sizes the canvas
with (`:630-636` vs `:665-666`). Everything below 1600² already passes admission, so no ceiling moves here.
What must be re-established, not assumed: the per-geometry ledger cert (the peak rises roughly 1.8x in
target rows), the timings, and -- because this is a *quality* change -- the M9 behaviour gate with the new
fixture size, since the gate's own control numbers were measured at 512². Verification: the host geometry
tests are area-parameterised (`resolve_canvas_geometry` and the P2/P9 canvas cases keep asserting ratios
against the target area, so they follow the constant), and the device gates are re-run with the wall printed.

**P14b -- raise both ceilings together.** One shared constant for the joint-sequence bound, used by
`attn_dit_joint`, by `rope_c64` (deleting its private copy) and by the serve's admission check, so a
request can never be admitted by one and refused by another. Size it from the largest geometry the card
survives, printed by the run, and keep the admission error text honest -- it currently says
"a smaller size or a shorter prompt" (`image_generation_service.cpp:815`), which is only the truth while
the ceiling is the binding constraint and not the VRAM.

**P14c -- VAE tiling, both directions.** A tile + overlap-add path in the encoder and the decoder, with the
tile size a parameter the ledger can predict rather than a runtime guess. The constraints that make this
more than a loop: the VAE's own receptive field sets the minimum overlap (an underlapped tile shows as a
seam, and a seam is exactly the class of defect the project has already been burned by -- the documented
"green bar across the top edge" negative in `test_image_edit_behavior.cu` is a boundary artifact of this
family), and the tile count multiplies the encode calls for every reference, so the x4 vision/VAE invariant
and the per-plane admission still have to hold on the *whole* image, not per tile. Gate: a tiled decode of
the M9 fixture must be indistinguishable from an untiled one on the gate's own pixel statistics, and the
byte-zero ledger gate must stay green with the tiles allocated and freed inside the job.

## Why it is not implemented in this pass

The three stages each end in a number that has to be *measured on this card* -- a ledger cert that refuses
a row whose predicted total falls below its measured peak plus the floor (the mechanism is described in
`docs/maintainer/qwen-image-2-1-variables.md:227` around `kImageResidentBlocks`), a wall-clock ratio, and a
seam-free tile size. Writing those numbers from here would be inventing them, and an invented peak is how a
serve ends up OOM-killed in front of a user at 2 a.m. With `serve.exe` holding the GPU, the honest unit of
work for P14 is this plan plus the roadmap rows; the implementation is a card-free pass that starts with
P14a (a one-line default with a re-measured cert) and stops at the first number that does not print.

## Out of scope

Per-request resolution policy beyond the existing `size` / `wh_ratio` / `ratio_follow` (the ceiling is
global by design -- it is a memory statement, not a preference). Batched jobs. Anything about the
scheduler's `mu`, which already tracks resolution (`calculate_shift`).
