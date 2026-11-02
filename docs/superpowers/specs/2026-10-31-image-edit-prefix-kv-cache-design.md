# P13 - Caching the step-independent half of the joint sequence

Date: 2026-10-31. Branch: `image-edit`. Package P13 of the Qwen-Image-2.1 parity roadmap.
**Status: designed, deliberately not implemented in this pass.** The reason is at the bottom, and it is a
correctness reason, not a scheduling one.

## Problem

`dit_velocity` (`src/models/qwen_image/image_pipeline.cu:834-976`) re-runs the whole 32-block chain over
all `S_p` joint rows on **every** denoising step, for both the conditional and (at CFG > 1) the
unconditional forward. Only one thing changes between steps: the target latents. The text rows come from
a stable `d_txt_p`, the reference latents are constant by construction (the edit route exists precisely
because the reference block is constant), and the modulation the non-target rows read is the **t = 0**
row, not the sampled one:

- the blocks take their modulation row from `ed.target_mask`, so "the non-target rows -- text AND the
  reference image -- read the t = 0 row of the [D,2] table while only the target block reads the sampled
  one" (`image_pipeline.cu:931-940`),
- `norm_out` does the same with `adaln_scale_masked(..., ed.target_mask, ...)`
  (`image_pipeline.cu:963-973`),
- and the attention pattern is fixed too: the non-target rows are strictly causal to the target while
  each image block stays internally bidirectional (`ed.image_ids`, same call).

So every K and V that a condition row produces in block *l* is bit-identical at step 0 and at step 39. The
reference ships exactly this: `QwenImage21KVCache`, `kv_cache_mode = "extract"` on step 0 and `"cached"`
after, enabled by `transformer.config.causal_condition`
(`image-edit/refs/q21_pipeline_qwenimage21.py:759-776, 792, 808`) -- and its own docstring gives the same
justification this file derives from the engine's code: "Cache the text and condition-image keys and
values after the first step. Valid because `causal_condition` modulates those tokens from `t = 0`, making
their activations step-independent" (`:590-592`), with `use_kv_cache: bool = True` (`:534`) -- the reference
has it on by default and we have it not at all. Our own maintainer notes already list
it as the open gap: "6. `use_kv_cache = True` -- a `QwenImage21KVCache` extract/cached mode in the engine
DiT" (`docs/maintainer/qwen-image-2-1-variables.md:200-201`).

## What it is worth

Numbers from `docs/maintainer/qwen-image-2-1-variables.md:84`: `D = 4096`, 32 heads of 128, MLP 12288, 32
blocks. Downsample is 16 (`ImageGeometry::downscale()`), so an `H x W` picture is `H/16 * W/16` rows.

At the served edit point -- canvas `kCanvasTargetArea = 768.0 * 768.0` (`src/serve/openai_images.h:110`), a
1024x1024 reference:

| rows | count |
|---|---|
| target | 48 * 48 = 2304 |
| reference | 64 * 64 = 4096 |
| text | ~30 (post-drop) |
| joint `S_p` | ~6430 |

Per block, the attention and MLP GEMMs cost linearly in rows, so caching the ~4126 condition rows removes
**about 64% of the per-block arithmetic** -- and at CFG > 1 it is paid twice per step. The rope for the
condition rows is likewise constant and can be built once instead of per step.

The ceiling is unaffected: `ops::kAttnDitJointMaxTokens = 20480` (`src/ops/attn_dit_joint.cpp:16`) bounds
the *sequence*, and the sequence does not shrink -- the cache changes what is recomputed, not the layout.
The admission ledger keeps predicting the same `S_p`.

## Design

**1. One buffer set, allocated the way every other device buffer in the pipeline is.**
A `DeviceArena` region of `2 * NB * condition_rows * D` BF16 = `512 KB per condition row` (32 blocks x
K and V x 4096 x 2 bytes). A 1024x1024 reference therefore asks for **~2.0 GB**. That number is the whole
risk of this feature and it must go through the existing mechanism, not beside it: the pipeline's peak is
predicted and certified by `src/models/qwen_image/image_device_need.h` (`kImageResidentBlocks` at :33 is
the same pattern -- one symbol for the pipeline, the serving layer and the allowance's calibration "so
they cannot drift apart"). The prefix cache becomes a **row in that ledger**, sized from `condition_rows`
and `NB`, or `image_device_bytes_live() == 0` gates and the cert stamping start disagreeing.

**2. `run_block` gains a cached range, not a new block.**
`run_block(..., ed.target_mask, ed.image_ids)` (`image_pipeline.cu:937-940`) grows one argument set:
`const PrefixCache* prefix` naming, per block, the device K/V of the condition rows and their count. With
it, the Q/K/V projections run over the target rows only; the attention op reads keys as `[cached
condition rows ; freshly computed rows]`, which the existing joint attention already expresses -- its
mask is already per-row (`ed.image_ids`) rather than one prefix boundary, which is what made the edit
route possible at all. Block 0's K/V on step 0 are written into the cache as they are computed ("extract");
later steps only read ("cached"). **The weights pipeline is untouched**: the 32-block host->device
lookahead (`issue_copy`, `lane.copy_done`, `lane.freed`) still streams every block, because the weights are
the reason the interlude exists; the saving is arithmetic, not bandwidth.

**3. The step-0 forward stays the certified one.**
This is the part that must not be improvised. The reference documents that its own cache "produces a
(valid but different) sample than `False`" (`qwen-image-2-1-variables.md:201`) -- reusing cached K/V is not
bit-identical to recomputing them, because the cached rows were rounded into BF16 storage instead of
flowing through the block again. So:

- the **uncached** path stays the default and stays what `infernix_image_edit_pipeline_prod_cuda_test`
  (M12) compares against the torch dump. A cache that quietly became the default would put every certified
  row comparison behind an approximation.
- the cached path is **opt-in per request** (a real field, not an env var, so a client can ask for it and
  a capability document can say whether it is available -- the project's standing rule for anything a
  fresh client must discover without being told).
- its gate is behavioural, in this order: (a) the M9 edit gate (`test_image_edit_behavior.cu`, including
  the P12 runs) passes with the cache on; (b) the seed contract holds *within* a mode -- same seed and same
  mode reproduce a bit-identical image, different seed a different one, which is how the seeding gap was
  qualified (`qwen-image-2-1-variables.md:194-199`); (c) the ledger holds zero bytes after the job with the
  cache allocated and freed inside it; (d) the wall-clock ratio is **printed**, because that ratio is the
  entire point of the package and a feature nobody measured is a feature nobody wanted.
- one further assertion worth having from the start: the cached and uncached runs at the same seed must
  stay inside a stated distance of each other -- the reference says "valid but different", and "different"
  without a bound is how a cache bug hides until someone prints a face.

## Why it is not implemented in this pass

Three of the four things this feature can get wrong are invisible to every test that can run right now:

- `serve.exe` holds the GPU, so nothing here can be executed, and the honest gate for a KV cache is a
  device run -- the semantics gate (M9) grades pixels, and a subtly wrong prefix would still produce a
  recoloured field with a kept disc.
- the ~2.0 GB prefix cannot be sized against a ledger cert that has to be stamped from a measured peak.
  Writing the number now would be inventing it, and the whole ledger design exists to stop exactly that.
- the cached-vs-uncached distance bound has to be *measured* before it can be asserted, for the same
  reason the M12 fp8 floor is described as "MEASURED, not retuned".

A half-wired cache -- buffer allocated, argument threaded, but never run -- is the worst state this could
be left in: it would look implemented and would corrupt every edit the server produces. So the package
lands as this design plus the roadmap row, and the implementation is one focused pass with the card free,
starting from the ledger row and the printed ratio.

## Out of scope

Caching across *jobs* (each job re-extracts: the condition depends on the prompt's pad layout). Caching
the CFG-negative half across steps is the same machinery with its own buffer and is included above.
Turbo/distillation schedulers, where 4 steps make the extraction cost dominate.
