# P9 - Multi-reference end to end: every picture the client sent conditions the draw

Date: 2026-10-31. Branch: `image-edit`. Package P9 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_openai_images_schema_test`, `infernix_http_routes_test`,
`infernix_image_generation_service_test` (host, no device).

## Problem

The model side was already built for N references; the serve side carried one of each thing.

What was N-image before this change:

- `render_ti2i_template(prompt, image_count)` loops over the count and emits `<image1>`, `<image2>`, ...
  (`src/serve/image_text_tower.cpp:250-255`), and `expand_image_pads(ids, tokens_per_image)` expands
  one run per image.
- `build_joint_layout` scatters the reference blocks in order, and `joint_token_count` is
  `prompt_rows + 3 * vision_tokens + target_tokens` -- a sum over the whole picture set.
- `ConditionedVision` describes the whole set: `merger_rows` is `[total pad rows, hidden]`, each
  `deepstack_rows` level is `[total pad rows, hidden]`, and `grids` are in pad order
  (`src/models/qwen_image/text_tower.h:53-61`).
- The M7 ledger takes `vision_tokens` only to subtract pad rows from `text_tokens`, and reserves
  `all_img = ref_tokens + target` (`src/models/qwen_image/image_device_need.h`). N sequential ViT runs
  are therefore correctly sized by sums -- no ledger formula change was needed.

What was one: `kMaximumImageReferences = 1`, `OpenAIImagesRequest::image` was a single string,
`prepare_reference_planes(payload)` was called once, `reference_prompt_renderer_(text, {vision_tokens})`
passed one count, and `ImageJobExecutor` took `const ReferencePixels* reference` plus
`EditConditioning::planes` -- two pointers to the same picture, which is two things to keep in sync.

The consequence was not only missing capability but a silent one: `edit_body_from_multipart` kept the
LAST `image` part when a form carried two, so an SDK client posting a face and a body got a draw
conditioned on whichever part arrived second, with a 200 and no complaint.

## Decision

**The reference set is a vector the whole way down, and the sums are computed once, in one place.**

### Wire (`src/serve/openai_images.{h,cpp}`)

`std::string image` becomes `std::vector<std::string> images`, in request order. The JSON field accepts
the single string (unchanged, so no existing client special-cases us) or an array of up to
`kMaximumImageReferences`, which is now 4. That number is a route policy, not a model limit -- the
family allows more -- and what really bounds it is measured per request (each reference costs
`4 * its vision tokens` of joint sequence, admitted against `ops::kAttnDitJointMaxTokens`, plus its
latent tokens in the reservation). A payload past the cap is `400 image_count_invalid`, never a trim.

Every content rule that guarded the single string runs per reference: empty, URL, non-base64 data URI,
and the size pre-gate. Per reference, because each one is decoded on its own -- four half-size pictures
are not one big picture.

`ratio_follow` is now a *name*, `<imageI>`, and a name has to resolve. One parser,
`parse_image_reference_tag(tag, reference_count, index)`, accepts the form the engine itself renders in
the ti2i template (1-based, angle brackets) and refuses anything else. The shape is checked where the
key is read; whether it names a picture this request carries can only be answered after the references
are parsed, so that half runs at the end of the parse. Refusals, never fallbacks:

| request | result |
|---|---|
| two pictures, `<image2>` | accepted, the canvas follows the second |
| one picture, `<image2>` | 400 `image_ratio_follow_unsupported` |
| no picture, `<image1>` | 400 `image_ratio_follow_unsupported` |
| `image1` without brackets | 400 `image_ratio_follow_unsupported` |

The "no picture" row is a behaviour change from P2, which accepted the tag and ignored it. A parameter
that is silently ignored is the dropped-parameter failure this module's key whitelist exists to prevent,
and P9 is what made the difference observable.

### Multipart (`src/serve/openai_images_http.cpp`)

`edit_body_from_multipart` collects every `image` part -- text or file -- into a vector and emits one
reference as the plain string and more than one as the family's array. The `>1` count error it used to
raise is now the cap's error, with the cap in its message. The two halves of a multipart body are
separate containers in the wire library, so the order is text parts then file parts; what has to survive
is the set and each picture's own bytes, which is what the model reads.

### Service (`src/serve/image_generation_service.cpp`)

One loop over `request.images` calls `prepare_reference_planes` per payload; a bad payload keeps the
codecs' refusal codes and never echoes their words. Then three numbers are derived from the whole set,
once, in this order:

1. **Per plane, the x4 invariant.** `4 * vision_token_count(h, w) == (h/16) * (w/16)` or the request is
   a 500 `image_conditioning_inconsistent`. Checked per plane and not on the totals: one reference whose
   resize broke it must not be hidden by the other references' numbers.
2. **Sums for admission.** `ref_tokens` and `vision_tokens` accumulate, and `vision_per_image` keeps one
   count per reference -- which is what `reference_prompt_renderer_(text, vision_per_image)` needs, and
   what makes the rendered prompt's pad rows (and therefore the joint sequence admitted against
   `ops::kAttnDitJointMaxTokens`, and the `image_need_bytes` reservation) cover every picture.
3. **The followed reference for the canvas.** `ratio_follow` names one; with no tag it is the LAST
   reference, matching the reference pipeline deriving geometry from `image[-1]`
   (`image-edit/refs/q21_pipeline_qwenimage21.py:630-636`). A tag naming nothing was refused at parse,
   so only "named" and "none" reach here.

### Executor (`src/serve/image_generation_service.{h,cpp}`)

`EditConditioning::planes` is gone and the reference parameter is `std::span<const ReferencePlanes>
references`: both halves of the job -- the VAE encode and the vision tower -- read the SAME planes, from
one argument, in one order. The VAE loop appends each encode's latents and pushes its grid; the ViT runs
once per reference and concatenates `merger_rows`, every `deepstack_rows` level, and `grids`, taking the
merge constant and the level count from the first and throwing if a later reference disagrees about the
number of levels (the artifact, not the picture, decides that -- a disagreement is our bug, and a 500's
worth of it).

Three lists now agree because one loop drives them: the order the client sent, the order the ti2i
template numbered, and the order `build_joint_layout` scatters.

## Tests

`tests/test_openai_images_schema.cpp`

- `test_reference_image_field`: two references are both kept in order and each one's data URI prefix is
  stripped on its own; `<image2>` is accepted against a two-picture request; `<image3>` in a two-picture
  request and any tag in a zero-picture request are refused rather than clamped or ignored; the cap's
  count error still fires past four.
- `test_edit_route`: two file parts are two references (and both survive the shared parse), a text part
  plus a file part are two references, and a form past the cap is refused.
- `test_canvas_keys`: `ratio_follow` needs a picture to name.

`tests/test_image_generation_service.cpp` -- the two-reference block: two differently-shaped PNGs, no
size named. It asserts the renderer is handed `{1036, 1036}` (one run each, not one run for the pair),
that both planes reach the executor with the first still the wide one, that the canvas comes out
672x896 -- the LAST reference's aspect at `kCanvasTargetArea` -- that `ratio_follow = "<image1>"` moves
the canvas to 896x672 while both planes still arrive, and that the probe was asked for strictly more
bytes than the same request with one reference ("reserved for, not just conditioned on").

Host suite after the change: 100% passed out of 31
(`ctest -R "image|http_routes" -E "cuda|real|prod|behavior|pipeline_gate"`, `INFERNIX_TEST_ARTIFACT` =
`Qwen-Image-2.1-FP8-v4.infernix`).

## Not covered by a test binary

The executor's two loops are the production `ArtifactImageJobExecutor` and run on device; the service
test drives fakes, so the concatenation of `merger_rows`/`deepstack_rows`/`grids` and the VAE
`ref_latents` append are exercised by the real draw, not by this suite. The M9 behavior gate
(`tests/.../test_image_edit_behavior.cu`) is the check that the joint assembly actually consumes both
references, and it has not been re-run since the reference plane and canvas geometry changed.

## Out of scope

Per-reference *roles* (canvas vs material source) beyond the `ratio_follow` naming -- that is the
prompt enhancer's job (P10), which is the piece that decides which picture is which. A `mask` input
(does not exist in this pipeline). Changing `kMaximumImageReferences` past 4 (the ledger decides what
fits, and four 1024^2-area references already reach roughly half the joint ceiling).
