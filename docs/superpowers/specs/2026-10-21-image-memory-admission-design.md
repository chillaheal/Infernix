# Image memory admission Design

**Goal:** An image request is admitted or refused from a computed device-memory need that includes the
reference-image tokens and the true-CFG sequence, decided before the LLM's VRAM is yielded, and a job
that cannot be served never allocates.

**Architecture:** One pure function in the model layer turns a job's geometry into its device-memory
need, reusing the same expressions the pipeline allocates with. The Engine exposes a probe that runs
the already-pure `plan_image_yield` against that need and answers before anything is moved, so a doomed
request never closes the submit door. The serving layer tokenizes before the yield (pure host work) so
the need is accurate rather than worst-case, refuses with a typed error when the grant is short, and
turns every image failure inside a chat round into a tool result instead of an HTTP failure. The
pipeline's existing device-memory ledger is the oracle that qualifies the prediction.

**Tech stack:** C++20, CUDA 13 (`sm_120a`), host tests through `infernix_add_test`, device tests for the
ledger qualification.

## Global Constraints

- The `.infernix` container stays the only C++ product artifact; the model layer must not learn serving
  types (the existing `ImageJobProgressView` seam is the pattern to follow).
- Advertised OpenAI/Anthropic behavior is an external contract: the schema tests and `docs/serving.md`
  change together with any route-visible behavior.
- Project-owned APIs keep no backward compatibility: a replaced formula, comment or error path is
  removed with its tests, not kept beside the new one.
- The exact route stays bit-exact: the admission work adds no arithmetic to the DiT/VAE math and must
  not change a generated picture's bytes.
- ASCII-only source, no new files in the image path that own a second device allocator: `Buf` stays the
  sole allocator and `image_device_bytes_live()` must stay 0 between jobs.

## Scope check

This is bundle A of three reviewed against an external code review (A safety/admission, B speed,
C features and documentation). B and C get their own specs and plans; nothing here moves weights,
kernels or the sampler.

## 1. The need is computed, not scaled

`image_yield_target()` (`src/serve/image_generation_service.cpp:42-54`) is removed: it scales a measured
512x512 peak by the output area, multiplies by 1.6, and clamps to `[7 GiB, 26 GiB]`. It cannot see a
reference image (`ref_tokens`, `src/models/qwen_image/image_pipeline.cu:320,328`) or true CFG widening
every sequence-sized buffer to the longer joint sequence (`image_pipeline.cu:332-342`), and its clamp
made the largest accepted sizes saturate at 26 GiB and die late.

New in `src/models/qwen_image/`:

```cpp
// One job's device-memory need: the bytes the pipeline will hold at its highest point, from the same
// expressions that size its allocations. `text_tokens` is the prompt's token count and
// `negative_tokens` the negative prompt's (0 without true CFG); `ref_tokens` is the sum of the
// encoded reference blocks' latent tokens.
struct ImageDeviceNeed {
    std::uint64_t dit_activations = 0; // the joint sequence at s_max: qkv, attn out, mlp, joint buffer
    std::uint64_t latent_and_velocity = 0;
    std::uint64_t resident_weights = 0; // the top-levels held across the loop plus one resident block
    std::uint64_t vae_peak = 0; // the largest simultaneous set of decode planes
    std::uint64_t rope_and_tables = 0;
    std::uint64_t total() const noexcept;
};

[[nodiscard]] ImageDeviceNeed image_device_need(const ImageGeometry& target, std::uint32_t text_tokens,
                                                std::uint32_t negative_tokens, std::uint32_t ref_tokens,
                                                const DiTShape&, const VaeShape&) noexcept;
```

`DiTShape` and `VaeShape` are two new plain structs holding the dimensions the binding already reads in
`image_weights.cpp` (hidden, mlp width, heads and head-dim, latent channels, the VAE's per-stage channel
counts) — no new model constants and no artifact read in the need path. The joint sequence is
`S = max(text + ref + img, text_neg + ref + img)` exactly as `s_max` computes it (`image_pipeline.cu:332-342`),
so CFG and references raise the need by construction.

`ref_tokens` is available before any device work: `src/serve/reference_image.cpp:21,133-142` snaps the
reference to the 1024^2 pixel budget and rounds to multiples of 32, and a reference contributes
`(height/16) * (width/16)` latent tokens — so the probe knows the count from the snapped geometry,
without the VAE encode that runs later inside the yield.

The serving layer keeps a small memo of `(need, measured peak)` per job for the log line, nothing more.

## 2. Admission in two steps, no blackout for a doomed job

**Before the yield.** `Engine` gains:

```cpp
struct ImageYieldProbe {
    bool fits = false;
    std::uint64_t target_bytes = 0;
    std::uint64_t freed_bytes_expected = 0; // what the plan can release (frames + KV, on their own grids)
    std::uint64_t device_free_bytes = 0;    // free before anything is unmapped
};
[[nodiscard]] ImageYieldProbe probe_image_yield(std::uint64_t target_bytes,
                                                std::uint64_t margin_bytes) const;
```

It takes `vram_mutex_`, reads the live pool snapshot and runs the existing `plan_image_yield`
(`src/models/qwen4_exp/memory_plan.h:201-240`, documented "Pure and single-threaded"): nothing is
quiesced, unmapped or resized. `fits` is true when `plan.total_freed + device_free_bytes` covers
`target + margin + display headroom`, where the headroom is the same quantity `display_headroom()`
already computes for `size_expert_frames` (`src/models/qwen4_exp/memory_plan.cpp:142-164`) — the probe
does not invent a second reserve. Throws `std::logic_error` when the loaded model supports no interlude,
like `run_image_interlude`.

Serving calls it before `runner_->run(...)`. A probe that does not fit answers
`503 image_vram_insufficient` with the need and the grant in the message and **never reaches
`begin_image_quiesce`**, so no other request stalls. `kInterludeMarginBytes` (`:41`) becomes
`512 MiB` and is passed to both the probe and the run.

**After the yield.** Inside the interlude lambda, before `job_executor_` is called, the service checks
`pre.freed_bytes_expected + pre.device_free_bytes >= need + margin + headroom`, where
`freed_bytes_expected` is the pre-yield probe's `plan.total_freed` and `device_free_bytes` what it
observed. A probe is not run again: the interlude holds the Program's VRAM lock, and the two numbers the
first probe returned plus the `context.freed_bytes` the lease reports are exactly what the run rests on.
A shortfall ends the job there with `image_vram_insufficient`, and the LLM restores through the path a
failed job already uses. No device-qualified pipeline signature changes, and `context.freed_bytes`
(`program.cpp:226`, `program_impl.h:5149`) — today read only by tests and the lease's own `active` flag —
becomes a value production acts on.

**Consequence for the yield's size.** The yield now asks for the job's need plus a fixed margin instead
of area x 1.6 clamped to 26 GiB. A 768x768 text-to-image job stops requesting ~22 GiB and evicting
almost the whole expert hot cache for bytes it will never touch, which makes every restore cheaper.

## 3. Text tokenization moves before the yield

An accurate need needs the prompt's token count, and tokenizing is pure host work: `PromptRenderer`
loads `tokenizer.json`/`tokenizer_config.json` and compiles the chat template
(`src/serve/image_text_tower.cpp:70-94`), and `render` returns ids (`:50-57`). Only the tower's 36-layer
forward wants VRAM, so `render` moves out of the interlude:

```cpp
// TowerTokenId is the id type PromptRenderer::render already returns; no new id representation.
// The prompt -> token ids half of the seam: pure host work, callable before the VRAM yield.
using PromptTokensProvider = std::function<std::vector<TowerTokenId>(const std::string& prompt)>;
// The ids -> prompt_embeds half: needs device memory, stays inside the interlude.
using PromptEmbedsProvider =
    std::function<std::vector<std::uint16_t>(const std::vector<TowerTokenId>&)>;
```

`make_tower_prompt_embeds_provider` gains the tokens half and `apps/serve/main.cpp:333-334` wires both;
the prompt string never re-enters the device path.

`ImageGenerationService::generate` resolves both prompt and negative prompt token ids before the probe,
so the number the probe checks is the number the job runs with. The tokenizer is lazy-bound exactly as
today; the first request's tokenizer load is host work that no longer holds the door closed.

Everything else that currently runs inside the interlude (the tower forward, PNG encode, base64, the
disk write, the first-use weight binding) stays there in this design and is bundle B's work.

## 4. Concurrency stays single-flight, honestly

- `429 image_interlude_in_flight` becomes a typed error: a `ImageInterludeInFlight` exception carrying
  the code, thrown by the runner adapter, mapped by type in `generate()` — replacing the substring
  match on the exception text (`src/serve/image_generation_service.cpp:427`).
- The comments that attribute a serving FIFO to the code are removed where they lie
  (`src/models/qwen4_exp/program/program.cpp:158-159`, `program.h:620-621`, `program_impl.h:5519`);
  they say the gate is single-flight and the caller is told, which is what the code does.
- `docs/serving.md` keeps its 429 row (`:1057`) and its "Only one image job runs at a time" (`:989`):
  both already match the code. Its claim that the yielded amount "scales with the requested size"
  (`:990`) is replaced by the computed-need wording.

## 5. A chat image failure becomes a tool result

`src/serve/chat_image_tool.cpp:124-126` rethrows anything that is not a 400, so one refused picture
kills the whole chat request. Every image failure inside a hidden tool round — refused for memory,
429, stalled, or a failed draw — becomes an error tool result whose text names the reason and, for a
refusal, the largest size that fits. The model can then retry smaller and the user still gets an answer.
The rounds' invisibility to the client is unchanged.

## 6. The restoration hole that can park the Program

`release_from_yield` discards `map_kv`'s false return (`program_impl.h:5200`) and then unconditionally
raises the logical KV limit (`:5201`), which `set_backed_pages` answers with
`std::logic_error("Paged KV: the pages past the new limit are in use")`
(`src/core/paged_kv_cache.cpp:253-254`) when the pages were never remapped — turning a plain job OOM
into `mark_interlude_degraded` through `program.cpp:204-207`.

Fix: when the remap cannot be completed, the pool's limit stays where the yield left it (a smaller KV
pool until `VramControl` grows it back) and the shortfall is logged as a warning with the byte counts.
The cache invariant is preserved and no Degraded state is entered for a soft shortage. A stale lease
still degrades as before: this covers only a mapping that physically cannot be satisfied.

Note on the review's framing: a CUDA error inside an image kernel does not reach the Degraded state at
all — `CUDA_CHECK` is `std::abort()` (`src/core/device.cu:68-72`) — so this fix addresses the C++ path,
which is the one the interlude can actually survive.

## 7. Verification

- Host test `infernix_image_device_need_test`: the need rises with reference tokens, with true CFG
  (through the longer of the two joint sequences) and with area; a 768 text-to-image need is below the
  same size plus one 1024-bounded reference, which is exactly the failure the review named.
- Host test for the probe over a fake snapshot: `Granted`, `Partial` below the need, `Denied` when the
  pools are frozen by an outstanding lease; the service refuses with `image_vram_insufficient` and the
  fake `job_executor_` is never called.
- Host test that token ids are resolved before `runner_->run` and that a refusal never calls the runner.
- Host test that a refused, in-flight and failed picture each reach the model as a tool result and that
  the chat response still completes.
- Host test for `release_from_yield`'s soft path: a remap that cannot be satisfied leaves the limit
  raised-not and the Engine still `is_available()`.
- Device test (the qualification of the model): for a matrix of sizes with and without a reference and
  with CFG, `image_device_bytes_peak()` stays at or below `image_device_need().total()` for the same
  inputs, and `image_device_bytes_live()` is 0 after. The ledger already exists and is the release gate
  for this module (`image_pipeline.h:93-99`).
- Existing host suite stays green, including the 11 serving/image tests already run for the heartbeat
  work.

## Explicitly not in scope

- Resident or double-buffered block weights, the attention kernel, the GPU Euler step, moving the tower
  forward/PNG/base64 out of the interlude, and the chat step default: bundle B.
- A `image` argument on the chat tool, generated-file garbage collection, and
  `docs/maintainer/qwen-image-2-1-variables.md`: bundle C.
- A real request queue, multi-image requests, vision conditioning into the text tower, and the
  artifact's missing scheduler config.
