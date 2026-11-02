# P1 - condition-image resize must be Lanczos, not bilinear

Date: 2026-10-31. Branch: `image-edit`. Package P1 of the Qwen-Image-2.1 parity roadmap.

## Problem

`prepare_reference_planes` (`src/serve/reference_image.cpp:129-183`) resizes the reference to the
1024^2-area condition geometry with `resize_rgba_pil_bilinear` (`:82-112`), a 2x2 bilinear kernel
with no antialiasing prefilter. Its comment claims that is the PIL mapping the pipeline uses. It is
not.

The reference pipeline builds its processor without a resample override
(`image-edit/refs/q21_pipeline_qwenimage21.py:209-211`):

```python
self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,
                                        vae_latent_channels=self.latent_channels)
```

`VaeImageProcessor.__init__` defaults are `resample="lanczos"`, `reducing_gap=None`,
`do_convert_rgb=False`, `do_normalize=True`
(`image-edit/refs/diffusers_image_processor.py:113-118`). `preprocess` on PIL input calls
`self.resize(i, height, width, resize_mode="default")` (`:688-690`), and `resize` on a PIL image is
(`:495-501`):

```python
image = image.resize((width, height),
                     resample=PIL_INTERPOLATION[self.config.resample],
                     reducing_gap=self.config.reducing_gap)
```

So the reference is resampled with `PIL.Image.Resampling.LANCZOS` and `reducing_gap=None`, once:
the pipeline's own `image_processor.resize` at `:665-671` produces the vision copy and the later
`preprocess` resize is an identity (`self.size == size and box == (0,0)+self.size` -> `copy()`,
`image-edit/refs/pillow_Image.py:2401-2402`).

Every reference that is not already at the snapped size - almost every edit - therefore reaches the
VAE plane and the vision plane with the wrong high frequencies before the DiT runs. The M9 ship gate
cannot see it: `tests/models/qwen_image/test_image_edit_behavior.cu` encodes a 256^2 fixture and only
optionally cross-checks `prepare_reference_planes` (`:480`).

## Decision

Reproduce `PIL.Image.resize(size, resample=LANCZOS, reducing_gap=None)` bit-exactly for RGBA input,
and claim parity. Not "an antialiased filter".

Rejected:
- Reproducing all of `VaeImageProcessor.resize`: `fill`/`crop` modes, `get_default_height_width`
  snapping and `reducing_gap` are dead for this route - the pipeline passes explicit height/width,
  both already multiples of 32 (`snap32`, `reference_image.cpp:23-35`), and `reducing_gap` is None.
- Any correct-looking filter without the golden test: it cannot be defended later.

## Algorithm (ground truth: `image-edit/refs/pillow_Resample.c`, `pillow_Image.py`, `pillow_Convert_premult.c`)

All arithmetic below is copied from those files; nothing is invented.

`Image.resize` wrapper (`pillow_Image.py:2397-2439`), for `box = (0,0,W,H)`:

1. `self.size == size and box == (0,0)+self.size` -> copy. Keep the shortcut.
2. Mode `RGBA` -> `convert("RGBa")`, resize in `RGBa`, `convert("RGBA")` (`:2407-2410`).
   - premultiply `rgba2rgbA`'s inverse `rgbA2rgba` (`pillow_Convert_premult.c:420-431`):
     `out[c] = MULDIV255(in[c], a)`, `alpha` copied.
   - unpremultiply `rgba2rgbA` (`:435-452`): `a == 255 || a == 0` -> identity, else
     `out[c] = CLIP8((255 * in[c]) / a)` with truncating integer division.
   - `MULDIV255(a,b,t) = (t = a*b + 128, SHIFTFORDIV255(t))`, `SHIFTFORDIV255(a) = ((a>>8)+a)>>8`
     (`pillow_ImagingUtils.h`). `MULDIV255(v, 255) == v` for all `v` in `0..255`, so the opaque round
     trip is exact and one code path matches both PIL's RGB and RGBA paths.
3. `reducing_gap is None` -> no `reduce` prefilter. Assert the caller never sets one.
4. `self.size[1] > self.size[0] * 100 && size[1] < self.size[1]` -> two chained resizes with split
   boxes (`:2432-2436`). The inner routine takes a box, so reproduce it; it is unreachable for
   aspects the route admits (`target_w >= 32` at `:159-161`), but it is cheap and it keeps the
   wrapper honest.
5. Otherwise `self.im.resize(size, resample, box)` = `ImagingResample`.

`ImagingResampleInner` (`pillow_Resample.c:772-865`): horizontal pass first, then vertical; both
coefficient sets are computed from the *original* box; `need_horizontal = xsize != in->xsize ||
box[0] || box[2] != xsize` (same for vertical); the horizontal pass only visits rows
`[bounds_vert[0], bounds_vert[2*ysize-2] + bounds_vert[2*ysize-1])`.

`precompute_coeffs` (`:182-267`), per axis, `in0/in1` = box ends, `inSize` = source extent:

```
scale = filterscale = (in1 - in0) / outSize;      // double
if (filterscale < 1.0) filterscale = 1.0;          // upscale: kernel is NOT widened
support = 3.0 * filterscale;                        // LANCZOS support = 3.0 (:87)
ksize   = ceil(support) * 2 + 1;
center  = in0 + (xx + 0.5) * scale;                 // NOT (x+0.5)/scale - 0.5
xmin    = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
xmax    = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize;
xmax   -= xmin;
w[x]    = lanczos((x + xmin - center + 0.5) * (1.0 / filterscale));   // :248
k[x]   /= sum(w);   // only when sum != 0
```

`lanczos(x)` = `sinc(x) * sinc(x/3)` for `-3 <= x < 3`, else 0; `sinc(x) = x == 0 ? 1 : sin(pi*x)/(pi*x)`
(`:65-81`). Note the window is asymmetric: `x == -3` is *inside*, `x == 3` is outside.

Quantization and accumulation (`:270-284`, `:364-385`), `PRECISION_BITS = 32 - 8 - 2 = 22`:

```
kk[i] = (int)(k[i] < 0 ? -0.5 + k[i] * (1<<22) : 0.5 + k[i] * (1<<22))   // round-half-away
ss     = 1 << 21;                                                          // = +0.5
for x in [0,xmax): ss += (int)pixel * kk[x];                               // int32
out    = clip8(ss) = clip8_lookups[ss >> 22];                              // floor, then clamp 0..255
```

The `+0.5` seed plus the floor in `>> 22` *is* the rounding; do not add a second `lround`. All four
channels (including alpha) go through the same coefficient set, the `bands == 4` branch
(`:374-381`, `:472-483`).

Traps that silently differ from PIL, all of which the golden test is expected to catch:
- the `center` formula (`in0 + (x+0.5)*scale`, not the inverse-scale mapping in the current code);
- `(int)(v + 0.5)` truncation versus `lround`/`floor(v+0.5)` for negative `v`;
- widening the kernel on upscale (PIL clamps `filterscale` to 1 but keeps `support = 3*filterscale`
  *after* the clamp, so upscale uses `support = 3`);
- normalizing by `sum(w)` after the window, per output pixel;
- `clip8` on `ss >> 22` (arithmetic shift = floor) instead of clamping before the shift;
- resampling alpha linearly instead of in premultiplied space;
- doing the vertical pass on the *original* width (it runs on the horizontally resampled buffer).

## Test

Host-only golden, no GPU, so it runs while `serve.exe` owns the card (`Infernix/.agent/STATE.md`).

- Generator `tests/models/qwen_image/gen_oracle_resize.py`, run with the documented interpreter and
  `-I`, using the installed PIL 12.3.0 only (no torch/diffusers needed - the pipeline facts above are
  pinned by the citations in this file and re-checked by the edit behaviour gate).
- `tests/models/qwen_image/test_image_resize.cpp`, registered in
  `tests/models/qwen_image/tests.cmake` with `infernix_add_op_test(... NEEDS_SOURCE_DIR LIBRARIES
  infernix_serve infernix_core)`, next to `infernix_image_edit_template_test` (`:300-304`).
- Golden file `tests/models/qwen_image/oracle_resize.json`: per case the input PNG (base64), the
  requested `(w,h)`, and the expected RGBA bytes (base64, raw). Fixtures stay small
  (<= 256x256 in, outputs 16..320) so the file stays a few hundred kB.
- Tolerance: **exact byte equality**. A single LSB difference is a bug until proven to be a libm tie
  at a `kk` quantization boundary; if one is ever proven, record it here instead of widening the
  gate.
- Cases: identity; 2x down; 2x up; non-integer down (`200x150 -> 32x32`); aspect swap
  (`128x64 -> 64x128`); graded-alpha RGBA (premultiply path); `alpha == 0` pixels (the divide guard);
  opaque RGBA vs the same pixels as RGB (must agree, and both must match the golden);
  high-frequency checkerboard (this is what fails bilinear); `128x1` and `1x128` (edge clamp);
  the 32-multiple shapes the route actually produces.
- Structural assertion kept from today: `out.vision.rgb == composite_rgba_over_white(out.vae)` from
  the *same* resized buffer, i.e. the vision copy and the VAE copy stay consistent by construction.

## Out of scope here

`reducing_gap`/`reduce`, `fill`/`crop` resize modes, VAE tiling, canvas geometry (P2), output alpha
(P3).
