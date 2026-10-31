# Qwen-Image-2.1 variable inputs (engine spec)

The variable surface of one Qwen-Image-2.1 image job, as the engine must expose it. This is the input
contract that variable-input generation (size / prompt / seed / steps / N condition images) must make
tunable. Every entry is grounded in the diffusers reference
`diffusers.pipelines.qwenimage21.pipeline_qwenimage21.QwenImage21Pipeline.__call__` and the line-for-line
oracle dumper `image-spike/oracle/dump_pipeline.py`, which drove the pipeline's own sub-components
(`encode_prompt` / `prepare_latents` / `retrieve_timesteps` / `transformer` / `scheduler.step` /
`_unpack_latents` / `vae.decode`) in BF16.

Scope: this document enumerates the **inputs** and the input-dependent **derived math**. It does not
specify the endpoint (that is the `/v1/images/generations` contract, `docs/serving.md`) or the in-process
execution (the C4 interlude, `tests/models/qwen4_exp/test_image_interlude_dit_real.cu`). It states which
inputs the engine currently supports and which it does not yet.

## Caller-facing variable inputs

The `__call__` signature, with the value the 512×512 "Maine Coon in white smoke" oracle used and the
engine's support status. "Oracle" is that single fixed config, which the C++ qualification tests read from
fixed `.npy`/`.json` reference values and check the engine's own math against; the **production route takes
these as request inputs** (`docs/serving.md`), so a row's "Yes" means the engine accepts and derives that
input, not that it reproduces the oracle's bit pattern at every value.

| Input | Type / default | Meaning | Oracle value | Engine status |
|---|---|---|---|---|
| `prompt` | `str \| list[str]` | Text conditioning. Encoded by the text tower (Qwen text encoder) to `prompt_embeds [B, text_len, D_txt]`; `text_len` is **prompt-dependent**. | one string | **Yes** — the served `prompt` is a raw string and the Qwen-VL text tower runs in-process on the device; `text_len` follows the prompt. Batch/list form is not an input. |
| `image` | `PIL/np \| list[...] \| None` | N condition images. Read by the text encoder as vision context AND encoded by the VAE into latent tokens **prepended** to the noise. | `None` (text-only) | **Yes (single image)** — the served `image` field is decoded on the host, snapped to the reference geometry, read by the Qwen3-VL vision tower (its merger rows are scattered onto the image slots of the 36-layer text tower, including the three deepstack adds) and encoded by the VAE **encoder** into the tokens that overwrite those slots in the joint sequence. N=1, no mask, no inpainting region; the KV cache stays off on this route. The field is advertised because the behavioral gate (`infernix_image_edit_behavior_test`) grades the PICTURES a real edit produced -- not a copy of the reference, the subject held, the named change present. |
| `negative_prompt` | `str \| list[str] \| None` | CFG negative. Ignored unless `true_cfg_scale > 1`. | `None` | **Yes** as the endpoint's `negative_prompt` (served request field, encoded by the same text tower). Batch/list form is not an input. |
| `true_cfg_scale` | `float`, default `1.0` | CFG. Enabled only when `> 1` **and** a negative prompt is present. Doubles the per-step DiT forward (cond + uncond) → 2× VRAM and time. | `1.0` (no-CFG) | **Yes** as the endpoint's `true_cfg_scale` in `[1, 20]`; above 1 the step runs the DiT twice and combines `neg + scale * (cond - neg)` in BF16, and the endpoint requires the negative prompt. Step wall clock ~2×; device footprint grows only with the second pass's activations (the streamed block weights and the sequence-sized buffers are shared, sized by the longer of the two sequences). Numeric qualification against the reference's CFG branch is pending. |
| `height` | `int \| None` | Output height in px. Derived from condition-image aspect ratio (or `output_resolution`) if omitted; snapped down to a multiple of `vae_scale_factor * 2` (= 32). | `512` | **Yes** — the served `size` height, any multiple of 32 in `[32, 2048]`, snapped the same way. |
| `width` | `int \| None` | Output width in px; same snap. | `512` | **Yes** — the served `size` width, same range and snap. |
| `num_inference_steps` | `int`, default `40` | Denoising steps (== number of DiT forwards). | `40` | **Yes** — the served `steps` in `[2, 100]`; the schedule (sigmas, timesteps, resolution-dependent `mu`) is computed by the Engine for that step count and geometry. |
| `sigmas` | `list[float] \| None` | Custom denoise schedule. Default `linspace(1.0, 1/steps, steps)`. | default | **No** |
| `num_images_per_prompt` | `int`, default `1` | Batch multiplier for the latents. | `1` | **No** |
| `generator` | `torch.Generator \| list \| None` | Seed for the init-noise latents. | `seed = 1234` (mainecoon variant: its fixed seed) | **Yes** — the served `seed` drives the Engine's own init-noise fill (the same generator the oracle's `latents_init.npy` came from); omitted means the Engine picks one. |
| `latents` | `torch.Tensor \| None` | Pre-generated noisy latents; overrides `generator`. | `None` | **No** |
| `prompt_embeds` / `prompt_embeds_mask` | `torch.Tensor \| None` | Skip text encoding by supplying embeddings directly. | `None` | n/a (oracle uses exactly this path). |
| `negative_prompt_embeds` / `_mask` | `torch.Tensor \| None` | Skip negative encoding. | `None` | **No** |
| `output_type` | `str`, default `"pil"` | `"pil" \| "np" \| "pt" \| "latent"`. | `"pil"` (+ the `.npy`) | **No** |
| `attention_kwargs` | `dict \| None` | Passed to the attention processor. | `{}` | n/a |
| `callback_on_step_end` | `Callable \| None` | Per-step hook. | `None` | **No** |
| `output_resolution` | `int`, default `1024` | Target side length for aspect-ratio derivation and condition-image resize. | n/a (h/w explicit) | **No** |
| `use_kv_cache` | `bool`, default `True` | Cache the text + condition K/V after step 0 (valid because `causal_condition` makes those tokens step-independent). **Toggling this does not reproduce the same image bit-for-bit in BF16** — the prefill step and the cached decode steps tile the attention differently and land on different rounding; both agree with an fp32 reference to the same tolerance. Fix a sample by fixing this flag. | **`False`** | **Matches `False`** — the engine's DiT has no KV cache, so it reproduces the `use_kv_cache=False` oracle. Enabling it (the diffusers production default) requires a `QwenImage21KVCache` extract/cached mode in the engine DiT. |

`return_dict` (default `True`) only changes the Python return wrapper and is not an engine input.

## Input-dependent derived math

These are not caller parameters but change with the inputs above, so the engine must recompute them
per job (the oracle reads them from dumped `.npy`/`.json`):

- **Geometry.** `multiple_of = vae_scale_factor * 2` (32); `H, W` snapped down to it; `Hl = H/16`,
  `Wl = W/16`; target `tokens = Hl * Wl` (1024 at 512×512). With N condition images, `img_shapes` is
  `[B, [ (1, h_i, w_i) for each condition i, (1, Hl, Wl) ]]` and the condition images contribute
  `sum_i h_i * w_i` extra latent tokens prepended to the noise, so the DiT joint sequence is
  `text_len + cond_tokens + tokens`: the image contributes the **full** `Hl*Wl` latent grid —
  `patch_size = 1`, `in_channels = 64`, and `_pack_latents` is a plain spatial flatten (no 2×2 token
  packing) — so the same `tokens` is both the count that drives the shift and the target image-token
  count. The text-only oracle joint length is 11 (text) + 1024 (image) = 1035; the `tokens // 4`
  count (256 at 512×512) is the `img_mask` target-slot length (below), not a packed joint length.
  The init/final latent is a `[1, tokens, 64]` **token-major** tensor (flat element `t*64 + c`);
  the **init latent is raw
  N(0,1)** from the seed, and the VAE `* std + mean` denorm is applied only at decode time (never to
  the init latent).
- **Scheduler shift (`mu`).** `mu = calculate_shift(tokens, base_image_seq_len=256, max_image_seq_len=8192,
  base_shift=0.5, max_shift=0.9)` — **depends on `tokens`, i.e. on resolution** (1024 at 512×512 gives
  `mu = 0.5387096774193548`). The default sigmas are `linspace(1.0, 1/steps, steps)`; with
  `use_dynamic_shifting` they are rescaled elementwise by the exponential time shift
  `sigmas = exp(mu) / (exp(mu) + (1/sigma - 1))`, then stretched to the terminal (`shift_terminal = 0.02`).
  `timesteps = sigmas * num_train_timesteps` (`= 1000`), and a terminal `0.0` sigma is appended, so the
  stored sigmas have length `steps + 1` and the timesteps have length `steps`. The transformer is fed
  `timestep / 1000`; each step is a `FlowMatchEulerDiscrete` `scheduler.step`
  (`prev = (sample.float() + (sigma_next - sigma) * model_output).to(model_dtype)`).
- **`img_mask`.** `image_pad_mask` (text + condition slots) plus `target_tokens // 4` appended ones
  (one slot per 2×2 latent group). Length `= text_len + cond_slots + target_tokens // 4`.
- **VAE denorm.** `latents_denormed = latents * latents_std + latents_mean`, with `latents_mean` /
  `latents_std` the VAE's per-channel `[64]` constants (resolution-independent, dumped to
  `latents_mean.npy` / `latents_std.npy`).

## Fixed model constants (not variable)

Identical for every Qwen-Image-2.1 image; read once from the artifact, never a per-job input.

- DiT: `D = 4096`, 32 heads, `head_dim = 128`, time-dim 256, MLP 12288, 32 transformer blocks,
  `in_channels` (latent `z_dim`) 64.
- VAE: `vae_scale_factor = 16` (latent 16× downscale); `multiple_of = vae_scale_factor * 2` (32) is
  the reference's snap keeping the latent grid even so the `img_mask`/rope 2×2 **target-slot
  grouping** divides it — the latent itself is not 2×2-packed (64 channels throughout, token count
  `tokens`, `patch_size = 1`; the VAE's actual 2×2 spatial op is the AvgDown3D shortcut — a
  channel-group average over 2×2 with a temporal front zero pad, which never changes the latent
  grid), per-channel `latents_mean` / `latents_std` `[64]`.
- Scheduler config (`FlowMatchEulerDiscreteScheduler`, from `artifact/scheduler/scheduler_config.json`):
  `base_image_seq_len 256`, `max_image_seq_len 8192`, `base_shift 0.5`, `max_shift 0.9`,
  `num_train_timesteps 1000`, `shift 1.0`, `shift_terminal 0.02`, `use_dynamic_shifting true`,
  `time_shift_type exponential`, `use_karras/exponential/beta_sigmas false`, `invert_sigmas false`,
  `stochastic_sampling false`. (`shift 1.0` is the fixed-shift path and is not used because
  `use_dynamic_shifting` is on.)

## Engine gaps this spec drives (for variable-input generation)

The current interlude test reproduces exactly the oracle row: text-only, no-CFG, no condition images,
`use_kv_cache = False`, single seed, fixed h/w, fixed steps, default sigmas. To make inputs variable the
engine must additionally build:

1. **In-process text-tower forward** — variable `prompt` needs (a) the host tokenize slice (t2i template →
   tokenize → `_drop_idx` slice) and (b) the Qwen text encoder run in the engine (today `prompt_embeds` is
   precomputed). The tower forward is already **qualified** on-device (`test_image_texttower_prod_cuda.cu` vs
   `pre_norm_hidden.npy`), and the host tokenize slice is **verified** for both the ASCII oracle and a
   `\p{M}`-mark (Devanagari) oracle (`infernix_image_tower_tokenize_test` reproduces the Qwen3-VL processor's
   `input_ids` token-for-token from the genuine VL `tokenizer.json`). The engine's `validate_pipeline` now accepts
   the Qwen3-VL mark-exclusive `pre_tokenizer` alongside the mark-inclusive one; this is not a loosened gate — the
   two are behaviorally identical for this vocab (0 of 151669 merge tokens contain a combining mark: the corpus was
   NFC-normalized before BPE training, so a mark always falls to isolated byte-level tokens and both splits yield the
   same ids). The live Qwen3.5-LLM path (its `tokenizer.json` still declares the mark-inclusive form) is unchanged.
   The engine also computes `_drop_idx` itself at runtime (the system-only turn's token count = 14 for the t2i
   template), so the full host preprocessing `template -> tokenize -> _drop_idx -> [_drop_idx:]` needs no oracle.
   The **post-tower prep is now a production host module** — `qwen_image::build_prompt_embeds`
   (`src/models/qwen_image/text_embeds.{h,cpp}`) gathers each batch item's valid tower rows, drops the leading
   `_drop_idx` of them (mask-aware), right-pads to the longest kept length and emits `encoder_attention_mask` —
   the reference pipeline's post-encode step (`dump_text_tower.py`) verbatim. It is verified bitwise against the
   oracle `prompt_embeds.npy` / `encoder_attention_mask.npy` and by a synthetic multi-item batch with left- AND
   right-padding + distinct drop lengths (`infernix_image_text_embeds_test`), so the variable-N
   (multi-prompt / condition-image) padding path is covered, not just the single oracle row.
   Remaining GPU-bound work for the tower: load the text-tower weights + `embed_tokens` table from the artifact's
   `text` component, gather `inputs_embeds` from the sliced ids (now a landed module — `build_inputs_embeds`, see
   below), run the (qualified) 36-layer forward, then call `build_prompt_embeds` — i.e. the in-process
   orchestration, since the math of every stage is already qualified or verified. This needs the GPU (the server-held VRAM), not new math. **Partial step (build-verified):** the
   qualified on-device 36-layer forward output is now fed through the production `build_prompt_embeds` inside
   `infernix_image_texttower_prod_cuda_test` and gated (at the tower's own a-priori bf16 tolerance, not bit-exact
   — the device output is not the f32 dump) to reproduce the model's own DiT `prompt_embeds.npy` — proving the
   DiT conditioning can come from a forward, not only a dumped `prompt_embeds.npy`. The test's numerical gate is
   blackout-gated (the tower holds all 36 layers device-resident, ~16 GB, cannot run beside a live server). The
   The `embed_tokens` **gather** is now a production ENGINE module: `qwen_image::build_inputs_embeds`
   (`src/models/qwen_image/embed_gather.{h,cpp}`) computes `inputs_embeds[t] = widen(embed_tokens[ids[t]])` — a
   pure BF16 row lookup widened to FP32 (top-16-bit, exact), with an out-of-range id guard. Its host test
   (`infernix_image_embed_gather_test`) reproduces the real `inputs_embeds.npy` **bitwise** from the real
   `model.language_model.embed_tokens.weight` (BF16 `[151936,4096]`, read by safetensors header + slice seek) and
   `input_ids`, plus an always-runs synthetic exact gate — the C++ mirror of `image-spike/verify_embed_gather.py`.
   So the gather is qualified **in the engine's own language**, not only in Python, and the tower input is built
   from token ids with no torch-dumped `inputs_embeds`. Every front-half step is now individually verified
   (tokenize → `build_inputs_embeds` EXACT → 36-layer forward → `build_prompt_embeds`). What remains for a fully
   variable prompt is running the whole chain **end-to-end from a raw prompt string** (the on-device forward is
   blackout-gated) and having the in-process interlude **read the materialized `embed_tokens`** through the
   artifact Reader rather than raw safetensors — the converter already stores it as the object
   `text/embed_tokens` in `Qwen-Image-2.1-FP8.infernix` (`model.language_model.embed_tokens.weight`,
   `[0:622329856]` = 151936x4096 BF16), so only the reader (GPU-bound) is left, not the materialization.
2. **VAE encoder** — N condition images require the VAE *encoder* (the engine has the decoder only), plus
   the vision-encoder path that reads condition images as text-encoder context.
3. **True-CFG path** — a second (uncond) DiT forward per step and the `neg + scale*(cond-neg)` combine;
   the 8 GiB yielded budget must fit the cond and uncond weight sets (resident or streamed).
4. **Resolution-dependent scheduler + joint-sequence rope** — the resolution-dependent **scheduler**
   (`calculate_shift` mu from `tokens` + `build_schedule`) is a verified production host module
   (`infernix_qwen_image::scheduler`, gated bitwise vs the 512×512 pipeline oracle and across a resolution
   grid). The resolution-dependent **joint-sequence rope** for the text-only case is now the same:
   `qwen_image::build_joint_rope(text_len, Hl, Wl)` (`src/models/qwen_image/rope.{h,cpp}`) emits the DiT
   3-axis (frame/height/width) rotary `[S,64,2]` for `S = text_len + Hl*Wl` at any geometry, reproducing
   `QwenImage21Rope.forward` so resolution is a parameter rather than a per-geometry `.npy`. It is verified
   **bitwise (rel_fro 0.0)** against the diffusers `block_rotary.npy` at (8,8,8) and by grid invariants
   (unit modulus, text height/width-axis identity, shared image frame, two closed-form spots, determinism)
   across non-square, tall and `text_len=0` geometries (`infernix_image_rope_prod_test`). The joint
   **`img_mask`** is likewise now a verified production host module: `qwen_image::build_img_mask`
   (`src/models/qwen_image/img_mask.{h,cpp}`) reproduces `dump_pipeline.py:124` — `cat(image_pad_mask_base,
   ones(target_slots))` with `target_slots = (Hl·Wl)/4` the 2×2-packed target count; because the base is the
   text encoder's `image_pad` mask (all-zero, width `text_len`, for text-only; carrying the N condition
   images' slots otherwise), one rule covers text-only **and** N condition images. Verified bitwise against
   the diffusers `img_mask.npy` (256×256: base 11 + packed 64 = joint 75, matching `meta img_mask_len`) and by
   a synthetic multi-condition batch (`infernix_image_img_mask_test`). The scheduler and rope modules are now
   **wired into the assembled in-process GPU pipeline** (`infernix_image_pipeline_prod_cuda_test`): the denoise
   schedule is computed by `calculate_shift` + `build_schedule` and the DiT rotary by `build_joint_rope`, both
   driving the sampler/blocks instead of the fixed `timesteps.npy` / `rot_inter.npy`, each verified bit-exact
   against the values the fixed read supplied (schedule and rope `rel_fro = 0.0`, end-to-end latent/image gates
   unchanged). This runs beside the live server (no blackout) — streaming keeps device peak under the free-VRAM
   headroom. Note the engine path is **full-grid** (`img_tokens = Hl·Wl`, joint `= text_len + Hl·Wl`, no
   `img_mask`); the `img_mask` module's packed view (`target_slots = tokens/4`, joint `= text_len + packed`) is
   the diffusers/dump framing, self-consistent for the host modules but not the engine's job shape. A
   general-geometry run that proves **arbitrary resolution** in the assembled GPU pipeline is now in place
   (env `INFERNIX_IMAGE_SIZE=HxW`, seeded-only, no reference dump at that geometry): geometry drives tokens →
   `calculate_shift` mu → `build_schedule`, and `Hl/Wl` drive `build_joint_rope`, so resolution is a runtime
   parameter rather than a per-geometry `.npy`; verified at a non-oracle 128×128 (behaviour gate — finite,
   non-degenerate — since no dump exists there) with the 256px oracle gate rerun bit-identical. The remaining
   host-side wiring for variable inputs is the in-process text-tower (variable prompt, gap #1) and the
   encoder (gap #2). A cross-module **integration gate** (`infernix_image_job_assembly_test`) now
   drives all four host modules together — geometry/scheduler, `build_joint_rope`, `build_img_mask`, `fill_normal`
   — across arbitrary `(h, w)` (square, non-square, tall, tiny, snapped-to-32) and asserts they describe **one**
   job: rope rows == `img_mask` length == `text_len + packed`; `tokens == 4·packed` (the scheduler's μ-arg is the
   unpacked count, 4× the rope/latent packed grid); schedule length tracks `steps`; init latent == `packed·64` and
   finite. No single fixed-resolution oracle can prove this cross-module agreement. **Remaining under this gap:** only the condition-image multi-block **rope**
   position loop over `img_shapes` (per-block frame advance + the N prepended image blocks) — this has no
   oracle dump (the reference oracle is text-only) and depends on the VAE / vision encoder for the condition
   images (gap #2, GPU-bound), so it cannot be verified host-side until a condition-image reference dump exists.
5. **Seeding** — the init-noise latents are now DRAWN from a seed by `qwen_image::fill_normal`
   (`src/models/qwen_image/noise.{h,cpp}`, host module already qualified by its N(0,1) moments + seed
   contract) instead of read from `latents_init.npy`, and this is **wired into the assembled GPU pipeline**
   (`infernix_image_pipeline_prod_cuda_test`, env `INFERNIX_IMAGE_SEED`). A seeded init is intentionally NOT
   bit-comparable to the torch-seeded `latents_final.npy` (torch RNG ≠ `fill_normal`), so the seeded run gates
   on behaviour — it ran to completion, the final latent and decoded image are finite and non-degenerate — and
   the seed contract is checked at the pipeline level: the same seed reproduces a bit-identical image and a
   different seed produces a different one (verified beside the live server; the oracle-geometry gate stays
   green with `INFERNIX_IMAGE_SEED` unset). The explicit-`latents` override input remains unimplemented.
6. **`use_kv_cache = True`** — a `QwenImage21KVCache` extract/cached mode in the engine DiT, with the
   documented caveat that it produces a (valid but different) sample than `False`.

Resolution, seed and step count are now **runtime parameters** of the assembled GPU pipeline
(`infernix_image_pipeline_prod_cuda_test`, env `INFERNIX_IMAGE_SIZE` / `INFERNIX_IMAGE_SEED` /
`INFERNIX_IMAGE_STEPS`): at any geometry the schedule comes from `calculate_shift` + `build_schedule`, the
rotary from `build_joint_rope`, and the init latent from `fill_normal`, each verified at the oracle geometry
(bit-exact) and exercised end-to-end at a non-oracle geometry (behaviour gate — finite, non-degenerate). The
engine path is therefore no longer single-config. Still open are the **caller-facing inputs that need new
machinery**: the in-process text-tower forward for a variable `prompt` (gap #1), the VAE/vision encoder for N
condition images (gap #2), true-CFG (gap #3), `use_kv_cache=True` (gap #6), the condition-image multi-block
rope (gap #4 tail, oracle-less + encoder-bound) — and the caller-facing `/v1/images/generations` binding that
would expose these parameters as an HTTP endpoint (a separate task; the test harness sets them by env).

## Engine constants behind those inputs

The inputs above are the caller's. These are the engine's own fixed numbers: what a job is made of, how
much it may ask for, and what it may hold. Each row is the name, what it decides, where it is bound, and
what keeps it honest. This documents what exists; it does not propose a knob.

### Representation and residency

| Name | Decides | Bound at | Kept honest by |
|---|---|---|---|
| `kImageElementBytes` (2) | BF16 is the element type of every image activation, weight upload and intermediate | [`image_device_need.h:24`](../../src/models/qwen_image/image_device_need.h) | the ledger cert, which sizes its prediction with it and requires predicted ≥ measured ([`test_image_need_ledger_real.cu`](../../tests/models/qwen_image/test_image_need_ledger_real.cu)) |
| 17 quantized roles FP8 E4M3 row-scaled, every other role direct BF16 | which operands the artifact may store in FP8 | [`image_weights.h:11-14`](../../src/models/qwen_image/image_weights.h) | the binder itself: a representation that differs from the recipe throws `artifact::ArtifactError` when the plan is built, before any yield |
| `Residency::Streamed` for **all** image parameters | weights stay in the artifact's files and an object is staged only while a block needs it; peak is one object | [`image_weights.h:7-9`](../../src/models/qwen_image/image_weights.h) | the interlude certs, which run the production route on the VRAM the LLM yielded ([`test_image_interlude_prod_real.cu`](../../tests/models/qwen4_exp/test_image_interlude_prod_real.cu)) |
| `kImageResidentBlocks` (2) | how many DiT block slots the shipped route keeps resident at once — one symbol for the pipeline's double-buffered copy slots, the serving layer's `blocks_resident`, and the allowance's calibration, so they cannot drift apart | [`image_device_need.h:33`](../../src/models/qwen_image/image_device_need.h) | the same cert stamps the value it measured, and the ledger cert refuses a row whose predicted total falls below its measured peak plus the floor |
| `image_block_bytes(dims)` | one resident block's footprint (seven FP8 roles + two BF16 head norms), derived from the bound dims | [`image_device_need.h:62-65`](../../src/models/qwen_image/image_device_need.h) | a recipe change moves it with the dims; the resident term of the need is `blocks_resident × image_block_bytes` |

### Memory the admission promises

| Name | Decides | Bound at | Kept honest by |
|---|---|---|---|
| `kImageWeightResidentBytes` (3072 MiB) | the operands and op scratch the buffer inventory does not enumerate (DiT top-level projections, VAE weights, per-call workspace). Over-estimating refuses a job that could have run; under-estimating OOMs it. Calibrated to the residual left after the enumerated terms against a **live** job's per-step peak, not the cert's trivial inputs — currently ~1.6–2.5× the real peak | [`image_device_need.h:57`](../../src/models/qwen_image/image_device_need.h) | [`test_image_need_ledger_real.cu`](../../tests/models/qwen_image/test_image_need_ledger_real.cu) per row: predicted ≥ measured **and** ≥ measured + the floor; the stage bench supplies the live-job peak the cert alone would understate |
| `kInterludeMarginBytes` (512 MiB) | the free VRAM left below what a job takes: a yield that lands a hair short fails inside the interlude, where the restoration is the thing that breaks | [`image_generation_service.cpp:42`](../../src/serve/image_generation_service.cpp) | the failure cert's scenarios, which drive a job that does not fit and expect refusal, not a broken engine ([`test_image_interlude_failures_real.cu`](../../tests/models/qwen4_exp/test_image_interlude_failures_real.cu)) |
| the known over-count in the inventory | the structural terms are summed where the pipeline's buffers are partly simultaneous, which is why the 1024 row's structural terms alone already exceed its measured peak | [`image_device_need.h:47-56`](../../src/models/qwen_image/image_device_need.h) | deliberately left as conservatism: fixing it is a re-derivation of the inventory, not a constant. Recorded in the bundle B spec |

### Caller-facing limits (the endpoint's, restated here with their binding)

`docs/serving.md` is the authority on what a client may send; this table only points at where each limit is
enforced in code and what tests it.

| Name | Range / cap | Bound at | Kept honest by |
|---|---|---|---|
| `kMinimumImageDimension` / `kMaximumImageDimension` | every side a multiple of 32 in `[32, 2048]` | [`openai_images.h:60-61`](../../src/serve/openai_images.h) | `infernix_openai_images_schema_test` |
| `kMinimumImageSteps` / `kMaximumImageSteps` | `[2, 100]`; floor because the schedule divides by `steps - 1`, ceiling bounds how long one request holds the yielded VRAM | [`openai_images.h:67-68`](../../src/serve/openai_images.h) | same schema test |
| `kMinimumImageCFGScale` / `kMaximumImageCFGScale` | guidance in `[1, 20]`; above 1 doubles the per-step DiT forward | [`openai_images.h:50-51`](../../src/serve/openai_images.h) | same schema test |
| `kMaximumImagePromptBytes` | 4096 bytes per prompt | [`openai_images.h:57`](../../src/serve/openai_images.h) | same schema test |
| `kMaximumImageInputBytes` | a reference image's decoded payload at most 16 MiB | [`openai_images.h:56`](../../src/serve/openai_images.h) | the schema test's oversized-payload case (`infernix_openai_images_schema_test`) |
| `kImageArtifactRoutePrefix` | `/v1/images/`, the only route that serves a finished picture | [`openai_images.h:45`](../../src/serve/openai_images.h) | the artifact route's id-shape gate (`infernix_http_routes_test`, `infernix_image_generation_service_test`) |
| `kMaximumChatImageGenerations` | 4 pictures per chat request; each costs a full interlude | [`chat_image_tool.h:35`](../../src/serve/chat_image_tool.h) | `infernix_chat_image_tool_test` |
| `kMaximumChatImageRounds` | 8 internal rounds per chat request, so bad arguments end as an ordinary answer | [`chat_image_tool.h:85`](../../src/serve/chat_image_tool.h) | `infernix_chat_image_tool_test` |
| `image_retention_days` (14, `0` = off) | how long a generated picture stays on disk, which is how long its delivered URL answers before it is `404` | [`serve_options.h:90`](../../src/serve/serve_options.h), swept by `GeneratedImageRetention` ([`image_generation_service.h`](../../src/serve/image_generation_service.h)) | `infernix_image_generation_service_test`: kept inside retention, swept past it, never a picture an unfinished job owns, never a file that is not an artifact id, nothing when the retention is off |

The chat tool's `image` argument is not a new kind of input: it resolves in the chat layer to the route's
existing base64 `image` field, so everything above already covers it
([`chat_image_tool.cpp`](../../src/serve/chat_image_tool.cpp), `infernix_chat_image_tool_test`).
