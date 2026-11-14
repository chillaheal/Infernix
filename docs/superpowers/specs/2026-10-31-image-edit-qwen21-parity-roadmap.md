# Qwen-Image-2.1 parity roadmap for the image-edit route

Date: 2026-10-31. Branch: `image-edit`. This is the queue; each package gets its own
`docs/superpowers/specs/<date>-<topic>-design.md` when it reaches the front, and the near batch has
them now.

Method, stated once so no finding has to be re-litigated: every row below was checked against the
shipped checkpoint (`image-spike/artifact/**`), the reference pipeline
(`image-edit/refs/q21_pipeline_qwenimage21.py`, which lives in huggingface/diffusers, not in the
QwenLM repo), `diffusers/image_processor.py`, and this engine's own source. Two external reviewers
(ChatGPT, Grok Build) reviewed the repository independently; where they disagreed, the primary
sources decided. "Not a gap" rows are listed too, precisely so nobody spends another session on them.

## Verified real, ordered by impact/effort

| # | Package | File and function | Status |
|---|---|---|---|
| P1 | Condition resize is bilinear, must be Lanczos | `src/serve/reference_image.cpp` `resize_rgba_pil_lanczos` | **landed** 2026-10-31, `infernix_image_resize_test` byte-exact vs Pillow 12.3.0 |
| P2 | Canvas does not follow the reference aspect | `src/serve/openai_images.{h,cpp}` `resolve_canvas_geometry`, `image_generation_service.cpp` `generate` | **landed** 2026-10-31, `test_canvas_keys` + `test_canvas_geometry` |
| P3 | The VAE's fourth channel is thrown away | `image_generation_service.cpp` `rgba_from_generated_channels` | **landed** 2026-10-31, `2026-10-31-image-edit-vae-alpha-channel-design.md`; four-channel plane -> RGBA, the rounding tie case pinned |
| P4 | 4096-byte prompt cap, no joint-sequence admission | `openai_images.h:79`, `image_generation_service.cpp:725-745`, `joint_layout.h:90` | **landed** 2026-10-31, `2026-10-31-image-edit-prompt-admission-design.md`; 16 KiB wire cap plus a host 400 `image_prompt_too_long`, never silent truncation |
| P5 | M-RoPE sections hard-coded | `image_weights.cpp:206-241`, `text_tower.cu:85-127` | **landed** 2026-10-31, `2026-10-31-image-edit-mrope-from-config-design.md`; bound from `text_config.rope_scaling`, interleaved required, fails closed |
| P6 | Post-start failures collapse into one 500 | `image_generation_service.cpp:886-903` | **landed** 2026-10-31, `2026-10-31-image-edit-failure-taxonomy-design.md`; `image_request_unservable` / `image_artifact_failed` / unchanged fallback |
| P7 | Engine seed and fixed-latent seam undocumented | `image_generation_service.cpp:763-776`, `image_pipeline.h:36-43` | **landed** 2026-10-31, `2026-10-31-image-edit-seed-and-latent-seam-design.md`; seed chosen at submit time and recorded, seam declared in the header |
| P8 | No `capabilities.image_edit` block, no `/v1/images/edits` | `openai_images_http.cpp` `handle_images`/`edit_body_from_multipart`, `http_server.cpp` `make_capabilities_json` | **landed** 2026-10-31, `2026-10-31-image-edit-edits-route-design.md`; JSON + multipart, one validator, 400 image_reference_required, image_edit capability block |
| P9 | Multi-reference end to end | `openai_images.{h,cpp}` `OpenAIImagesRequest::images` + `kMaximumImageReferences = 4` + `parse_image_reference_tag`, `openai_images_http.cpp` `edit_body_from_multipart`, `image_generation_service.{h,cpp}` plane vector, sums, `std::span<const ReferencePlanes> references` | **landed** 2026-10-31, `2026-10-31-image-edit-multi-reference-design.md`; per-plane x4 invariant, summed admission, canvas follows `ratio_follow` else the last reference |
| P10 | No prompt enhancer (PE-I2I) | `prompt_rewrite`/`rewritten_prompt` still in no source file; the serve now carries the contract | **contract landed** 2026-10-31, `2026-10-31-image-edit-prompt-enhancer-design.md`; `prompt_enhance` parses, `503 image_prompt_enhancer_unavailable` before any work, the shipped rules travel in `/v1/capabilities`. Model side open: needs the ~20 GB PE-I2I weights + a worker inside the existing VRAM interlude (user download) |
| P11 | CFG velocity re-rounded to BF16 every step when scale > 1 | `image_pipeline.cu:1040-1049` | **not a defect** 2026-10-31, verified against the scheduler source: `FlowMatchEulerDiscreteScheduler.step` upcasts only `sample`, never `model_output`, and closes with `prev_sample.to(model_output.dtype)` -- so the reference is BF16 at the step boundary too, and its three separately-rounded bf16 ops are *less* accurate than our fused double. No change; see STATE |
| P12 | The edit gate graded a prompt-drawable recolour, so a route that ignored the reference passed | `tests/models/qwen_image/test_image_edit_behavior.cu` | **widened** 2026-10-31, three new runs: a magenta disc the instruction never mentions (only the reference can put it there -- P3), the same run with the reference latents zeroed and the layout intact (the disc MUST be lost, or the assertion is named worthless in its own failure message), and a 544x608 canvas run (P2: geometry plus the subject under the independent x/y scale). P1 stays where it belongs, `test_image_resize.cpp` against a torch oracle. **Measured green 2026-10-31** on `Qwen-Image-2.1-FP8-v4.infernix` (`EDIT_BEHAVIOR_OK`, first device run of this gate since P1/P2/P3/P9 moved the reference-plane and canvas geometry): carried disc `r=223.8 g=27.3 b=148.2` kept while its field reads green 188.9, **blinded disc `r=38.4 g=185.4 b=58.2` lost to the field** -- the mutation control works, so the fourth channel is proven load-bearing, and the canvas run returned 544x608 with the subject 11.8 levels off. The first wording ("Keep the red square in the center") painted the field over the disc (r=53.4 g=187.1 b=52.3) and was changed, not widened: the checkpoint is not obliged to preserve what the sentence never claimed |
| P13 | No prefix KV cache | absent; `causal_condition: true` means the reference caches (`use_kv_cache` defaults **true**, refs `:534,590-592`) | **designed** 2026-10-31, `2026-10-31-image-edit-prefix-kv-cache-design.md`; valid here because our non-target rows already read the t = 0 modulation row (`image_pipeline.cu:931-940, 963-973`); ~64% of the rows through the block chain removed at the served point, ~512 KB/condition row of prefix to put in the ledger. Not coded: the cache is not bit-exact (the reference says so), so it must stay opt-in and be gated on M9 + a measured cached-vs-uncached bound + the printed ratio -- all device work |
| P14 | The served operating point is not the model's | `openai_images.h:110` 768^2; `attn_dit_joint/attn_dit_joint.cpp:17` 20480; **`rope_c64/rope_c64.cpp:17` a second, private 20480**; no VAE tiling | **designed, staged P14a/b/c** 2026-10-31, `2026-10-31-image-edit-operating-point-design.md`; the reference's point is an *area* (`output_resolution=1024`, refs `:533, 630-636` for the canvas and `:665-666` for the condition image, which we do not match); 1024^2 already runs here (166 s) so P14a is a default plus a re-measured cert; raising the ceiling means moving BOTH 20480s or a job is admitted then dies inside rope after the VRAM yield |
| P15 | No image LoRA | a case-insensitive `lora` over `src/` hits only `kColorableFamilies`/`calibrated`; no `safetensors` anywhere in `src/`; no image packer in `tools/flash_next/` | **designed** 2026-10-31, `2026-10-31-image-edit-lora-design.md`; the DiT is streamed from host block bytes (`kImageResidentBlocks = 2`), so the only design that costs nothing per step is merging `W + (alpha/r)*B@A` into the HOST bytes before `upload_block_async`; adapters as artifacts (the server never parses a downloaded file), bound per interlude, `base` digest refused at load, enumerated in `/v1/capabilities` + an `adapter`/`adapter_scale` field. Open: who converts community safetensors -- the user's call |
| P16 | Checkpoint `sample_sigmas` ignored | `src/models/qwen_image/scheduler.cpp` | deferred; the base checkpoint's field is null, so this only matters for Turbo |

## Checked and NOT gaps (do not spend a session on these)

`strength`, per-step CFG, `subseed` and a mask input are not parameters of `QwenImage21Pipeline`.
Guidance-distilled CFG is not used by 2.1. The scheduler matches this checkpoint's
`scheduler_config.json` exactly (`base_image_seq_len 256 / base_shift 0.5 / max_image_seq_len 8192 /
max_shift 0.9 / shift_terminal 0.02 / exponential`). `append_target_slots` IS implemented
(`image_pipeline.cu:480-481`). There is no norm-preserving CFG in 2.1 (the Edit pipelines have it,
2.1 does not). Vision `min_pixels`/`max_pixels` are null in this artifact's
`preprocessor_config.json`, so the processor's second resize is a no-op at a 32-snapped ~1024^2 area.
The vision/VAE alpha split matches the pipeline. Block-causal attention `(q >= kv) | same_image_block`
matches. The edit route uses the raw ti2i template with the system turn
"Comprehend and analyze the provided prompt."

## Sequencing

P1 and P2 landed together because P2 changes the geometry P1's golden fixtures pin. P3-P7 landed together on 2026-10-31 as the small host-verifiable batch (30/30 in the host image suite, no device work). Next in the queue are the two capability jumps, P8 (`capabilities.image_edit` + `/v1/images/edits`) and P9 (multi-reference), then P12 as the gate that makes P13-P15 safe, then P14.

## Verification policy for this roadmap

Host tensor oracle is authoritative for preprocessing (fast, no GPU, runs while `serve.exe` owns the
card). Fixed-latent GPU regression is authoritative for model impact. The 10-30 GB GPU suite is
release qualification, not a per-PR gate. Never run the GPU suite while `serve.exe` is alive.
