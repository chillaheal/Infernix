# Plan: P1 Lanczos condition resize

Spec: `docs/superpowers/specs/2026-10-31-image-edit-lanczos-resize-design.md`
Ground truth already on disk: `image-edit/refs/pillow_Resample.c`, `pillow_Image.py`,
`pillow_Convert_premult.c`, `pillow_ImagingUtils.h`, `diffusers_image_processor.py`,
`q21_pipeline_qwenimage21.py`.

1. **Generator** `tests/models/qwen_image/gen_oracle_resize.py`: build the fixtures listed in the
   spec with PIL + numpy only, resize each with
   `img.resize((w, h), resample=Image.Resampling.LANCZOS, reducing_gap=None)`, write
   `oracle_resize.json` (input PNG base64, size, expected raw RGBA base64). Run with the documented
   interpreter and `-I`.
   *Verify:* the JSON loads, every case's expected length equals `w*h*4`, and one case is eyeballed
   against a saved PNG.
2. **Golden test** `tests/models/qwen_image/test_image_resize.cpp` + `tests.cmake` registration
   (`infernix_add_op_test(... NEEDS_SOURCE_DIR LIBRARIES infernix_serve infernix_core)`, next to
   `infernix_image_edit_template_test` at `:300`). It calls the new resize entry point and demands
   exact bytes.
   *Verify:* it builds and FAILS against the current bilinear implementation on the checkerboard
   case. A test that passes before the fix is not this test.
3. **Implement** `resize_rgba_pil_lanczos` in `src/serve/reference_image.cpp`, replacing
   `resize_rgba_pil_bilinear` (`:82-112`): the `Image.resize` wrapper (premultiply/identity/unpremultiply,
   identity shortcut, extreme-aspect split) over `ImagingResampleInner` (horizontal then vertical)
   with `precompute_coeffs` + `normalize_coeffs_8bpc` + `clip8` arithmetic copied from the ref file.
   *Verify:* the golden test passes byte-exact; no fixture needs a tolerance.
4. **Header/doc truthfulness**: `reference_image.h:63-64` comment, and the `kReferenceTargetArea`
   comment at `reference_image.cpp:19-21`, state Lanczos and cite the pipeline lines.
   *Verify:* `grep -n bilinear src/serve/reference_image.*` returns nothing.
5. **Regression**: host suite run (`ctest -R image` for the non-GPU tests) plus the live edit smoke -
   one `prepare_reference_planes` of a real portrait through the route, checked with
   `image-edit/faceswap/_compare.py` / `verify_image.py` per the project CLAUDE.md.
   *Verify:* route still returns a picture; the identity check still holds.
6. `Infernix/.agent/STATE.md`: record the landed facts (file, test name, what the gate now pins).

Never run the GPU suite while `serve.exe` is alive (STATE); steps 1-4 are host-only.
