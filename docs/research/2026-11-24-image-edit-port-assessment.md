# Image generation/edit lane: portability assessment for a port onto latest upstream Infernix

Date: 2026-11-24. Branch `image-generation`, HEAD `9bc42a7f`.
Upstream target: `origin` = https://github.com/Wallawalla47/Infernix.git, `origin/main` head `f9c2e42c`.
READ-ONLY analysis. No code, build, or `image-edit/` was touched.

REVISED 2026-11-24 against upstream head `f9c2e42c` (`f9c2e42cb3e987d6159ed0de7afd2205a806ffd6`,
2026-11-24 12:51:50 +0000, "Merge origin/main into fix/inkling2-defects"). The first draft was pinned
to the stale baseline `0a3f432b` (2026-11-23 12:59 +0000); `origin/main` has advanced by exactly 119
commits to `f9c2e42c` (`git rev-list --count 0a3f432b..f9c2e42c` = 119; upstream total now 2684, ours
427). That advance added a full second chat model family ("Inkling 2", `Architecture::Inkling2Loop`,
113 `inkling` commits) and `perf(qwen4_exp)` work that puts every expert in RAM. All anchors and churn
below were re-verified against the explicit SHA `f9c2e42c`, not the moving `origin/main` ref. See the
new section "Revised against f9c2e42c" near the end for what the 119 commits did and did not change; the
verdict and risk table have been updated in place.

Method note: `git merge-base HEAD origin/main` fails (unrelated roots; ours `fd2aa93c`, upstream
`410b9f6c`). So "since fork" is measured by content, not the commit graph. Files that are byte
identical between our HEAD and `origin/main` (`vram_monitor.cpp`, `qwen3_5/state/state_image.{h,cpp}`)
were last touched upstream on 2026-10-23/2026-10-25, which bounds the fork content baseline at about
2026-10-25. All "sinceFork" churn below is `git log origin/main --since=2026-10-25 -- <file>`.

## Verdict

This is a manual port, never a rebase or a merge (histories share no ancestor), and today it is a
medium-to-hard job, not a "drop the tree in and flip a flag" job. The feature itself is clean: all
149 lane-only files (qwen_image math, the four DiT/VAE/rope/adaln ops, the serve protocol/job-store/
reference/prompt-enhancer files, the converter, the tools, and ~90 tests) are ours-only with no
upstream counterpart, and upstream has zero image-generation code (`run_image_interlude`,
`openai_images`, `attn_dit_joint`, `/v1/images` all 0 hits in `origin/main`), so there is no feature
overlap to reconcile. The pain is entirely at the seams. The lane carries edits in about 19
upstream-owned files, and its VRAM-coexistence machinery is welded into the chat model's Program
private state inside the two highest-churn upstream files (`program_impl.h`: 114 upstream commits
since fork against `f9c2e42c`; `engine_core.h`: 39 commits). The latest 119-commit upstream advance
does NOT re-inflate those two seam files much (`program_impl.h` grew only +58/-6 across 3 commits,
`engine_core.h` unchanged in `0a3f432b..f9c2e42c`) - but it does harden upstream against the one
thing our interlude needs, by adding `perf(qwen4_exp)` "every expert in RAM" placement that keeps the
expert pool resident via `residency_->lendable_bytes()` (details in the revision section). Against
current upstream, the symbols the lane's yield/worker-ack code leans on are gone (`execution_mu_` removed;
`begin_quiesce`/`probe_image_yield`/`yield_for_image`/`release_from_yield`/`request_interlude_preempt`/
`wake_llm_worker`/`drain_llm_frontier`/`interlude_preempt_outstanding` never existed upstream), and
`engine.cpp` was refactored into a `std::variant` core architecture. The lane also has no build flag
today: it is wired unconditionally into upstream `serve`/`models`/`ops` targets, and its CLI flags sit
in a `serve_options.cpp` that upstream grew by ~51 flags. So today a port means re-anchoring ~19 files
by hand against 2684 upstream commits (was 2565 at `0a3f432b`). Plan 2 (`2026-11-14-image-family-plugin.md`) reduces that but,
as written, does not reach mechanical portability: its two hardest "surfaces" (move-the-lease-out,
worker-ack) assume chat-Program internals and worker hooks that upstream has since rewritten, and all
four surfaces still require editing upstream files. The realistic best outcome after plan 2 plus the
deltas named below is "sync a vendored subtree, re-apply a short named patch series over ~6-8 hunks,
rebuild behind `-DINFERNIX_IMAGE`" - which is achievable, but only if the lease is first turned into a
real narrow facade so ~150 chat-Program-internal references stop living in our diff.

## Hook-point risk table

"Ours-only" means the symbol/function is added by us (no upstream counterpart), so the port is pure
re-adding into a drifted file. "Re-anchor" means we edited around an upstream structure that may have
moved. "sinceFork" = upstream commits touching that file since ~2026-10-25.

| Rank | Upstream-owned file (hook) | Nature of our edit | Anchor status in origin/main | sinceFork churn / divergence | Port-breaking risk |
|---|---|---|---|---|---|
| 1 | `src/models/qwen4_exp/program/program_impl.h` | The whole image interlude/lease body (`probe_image_yield`, `yield_for_image`, `release_from_yield`, `rollback_failed_yield`, `begin_quiesce`, `begin_image_quiesce`); ~41 refs, body touches `residency_`(127), `plan_`(148), `pool_`(79), `kv_vmm_`(12), `execution_mu_`(0 - gone), `vram_mutex_`(5), `interlude_mu_`/`interlude_quiesced_epoch_` | ALL yield names 0 hits; `execution_mu_` 0 hits (renamed/removed); `vram_mutex_`/`residency_`/`kv_vmm_`/`plan_`/`pool_` survive (ref counts re-measured at `f9c2e42c`) | 114 commits vs `f9c2e42c` (only +58/-6 across 3 commits in the latest 119) | CRITICAL - highest-churn upstream file; extraction into a facade is not mechanical; NEW experts-in-RAM hardening below works against the freeze |
| 2 | `src/runtime/engine/engine_core.h` | Worker-ack seam: `request_interlude_preempt`, `wake_llm_worker`, `drain_llm_frontier`, `interlude_preempt_outstanding` | all 0 hits; generic `worker` exists but not these names | 39 commits vs `f9c2e42c`; UNCHANGED in `0a3f432b..f9c2e42c` | HIGH - plan assumes a hook that never existed upstream; must re-anchor to current worker code |
| 3 | `src/runtime/engine/engine.cpp` | Lease forwarding `impl_->qwen4->program` (hardcoded variant) | upstream refactored to `std::variant` `GenerationCore`/`ScoringCore`, `construct_qwen4_exp`, `qwen4_exp_instance.h`; `qwen4->program` still present but surrounding arch changed | 24 commits vs `f9c2e42c` (+126/-43 in the latest 119) | HIGH - forwarding must be rewritten against new variant cores (and now `Inkling2Loop` added a 4th core) |
| 4 | `src/serve/serve_options.cpp` / `.h` | 4 `--image-*` flags + parsing + help + validation | our flags collide with nothing (upstream still has 0 `--image-*` at `f9c2e42c`); file grew further with inkling2 launch options | 64 / 33 commits vs `f9c2e42c` | HIGH (textual) - large re-anchor region; re-add our 4-5 flags |
| 5 | `include/infernix/types.h` | `ImageYieldProbe` / `ImageInterludeContext` additions to protocol header | our two structs absent upstream; `GenerationRequest`/`GenerationOutcome`/`ChatTool` NO LONGER in this header (moved to `src/serve/request.h:266` and `src/serve/generation_service.h:33`) | 73 commits vs `f9c2e42c` (+31/-4 in the latest 119) | HIGH churn but LOWER port risk than drafted: the request types left the public header, shrinking what our diff shares with it |
| 6 | `src/serve/http_server.cpp` / `.h` | `/v1/images/*` routes, `handle_image_artifact`, capabilities image block (`http_server.cpp:659-788`), `/health` `image_jobs` (`:539`), `attach_images`/`images_` (`http_server.h:80,172`) | `HttpServer`, `register_routes`, `make_capabilities_json`, `/health`, `/v1/capabilities` all survive; upstream added NO `/v1/images` and NO image block in capabilities (only Anthropic multimodal-input "image" parsing) | 21 / 10 commits vs `f9c2e42c`; 0 inkling commits touched `http_server.*` | MEDIUM - big but structurally separable (plan 2 ImageModule); serve enumerates models from the registry, not hand-edits |
| 7 | `include/infernix/engine.h` | Public lease API `supports_image_interlude`/`run_image_interlude`/`probe_image_yield` (`:141-166`) | none exist upstream; additive into `Engine` | 8 commits vs `f9c2e42c` | MEDIUM - low churn, pure addition, but must fit new Engine shape (now 4 model cores) |
| 8 | `src/serve/openai_chat_http.cpp` / `openai_responses_http.cpp` / `anthropic_messages_http.cpp` | tool injection: `append_chat_image_tool` -> rounds -> `missing_image_url_suffix` | all our helpers 0 hits upstream; `ChatTool`/`tool_calls`/`GenerationRequest` machinery survives to re-anchor onto (now under `src/serve/`) | 11 / 11 / 12 commits vs `f9c2e42c` | MEDIUM - anchor survives, code is ours; re-inject at current handler structure |
| 9 | `apps/serve/main.cpp` | builder seam `:323-331` constructs ImageJobStore/Service, `attach_images` | `server.attach` survives | (re-verify count before port) | MEDIUM - small block, re-land |
| 10 | `src/models/qwen4_exp/program/program.{h,cpp}` | lease decls + image stream creation (`cudaSetDevice` fix) | file exists; `residency_` etc survive | 27 / 14 commits vs `f9c2e42c` | MEDIUM - tied to surface 1 |
| 11 | `src/{serve,models,ops}/CMakeLists.txt` | unconditional `include(.../sources.cmake)` for the 4 ops + qwen_image; serve links `infernix_qwen_image` | upstream owns these; no flag exists | 5 / 6 / 6 commits vs `f9c2e42c` | LOW-MEDIUM - mechanical once flag exists |
| 12 | `tests/cmake/ProductTests.cmake` | registers image test targets | upstream owns | 20 commits vs `f9c2e42c` | LOW - mechanical registration |

Non-hooks (upstream-owned but NOT edited by our lane, just matched "image"): `qwen3_5/state/state_image.*`,
`qwen4_exp/program/prefix/state_image.*`, their tests, `vram_monitor.cpp`. These are upstream chat
multimodal-INPUT/vision files; treat them as upstream, do not port our copies over them.

## Entanglement findings (where the port actually hurts)

The lane is cleanly separable for all the math/protocol code and painfully entangled for the
VRAM-coexistence machinery. The line-crossing calls:

1. Lease body reaches into chat Program private state. In `program_impl.h` the image interlude reads
   and mutates the chat model's residency bookkeeping directly: `residency_` (150 refs), `plan_` (133),
   `pool_` (106), `kv_vmm_` (33), `execution_mu_` (15), `vram_mutex_` (13), plus our own
   `interlude_mu_`/`interlude_quiesced_epoch_`. This is the qwen4_exp expert-offload / KV-residency /
   scheduler internals. `execution_mu_` does not exist in current upstream, so this is a concrete break,
   not just churn. A clean port needs these ~150 references to stay behind a narrow lease facade that the
   chat Program implements, instead of being our functions in upstream's file.

2. Serve layer depends on chat-model residency semantics, and upstream now actively resists the
   premise. `src/serve/image_generation_service.cpp` branches on `probe.expert_pool_frozen_by_lease`
   (`:260, :1100, :1370`) - the image VRAM admission decision is coupled to whether the chat lane froze
   its expert-offload pool for the lease. It also `#include`s `infernix/engine.h`
   (`image_generation_service.h:19`). That is the intended public seam, but the probe struct carries
   chat-internal residency state, so it belongs in `types.h` (rank 5) and must be redesigned as a narrow,
   stable contract, not a leak of residency internals. NEW REGRESSION (the 119 commits): the
   `perf(qwen4_exp)` "every expert in RAM" commits `a3f78ef8` and `18388f5b` (2026-11-24, both absent
   from `0a3f432b`) add `!tier_` (no SSD tier) branches to the gated-ring sizing in `program_impl.h`
   (~line 4399, 5299) that deliberately KEEP experts resident, taking fewer lendable frames from
   `residency_->lendable_bytes()` precisely so "the cache keeps those frames' experts". That is the exact
   pool our image interlude must freeze-and-release to yield VRAM. So `expert_pool_frozen_by_lease` no
   longer describes a neutral default the chat lane is happy to give up; freezing the pool is now a
   measured-regression decision upstream leans against. The facade in WP-B must expose an explicit
   "override the residency keep and evict the full expert set" primitive that the lease calls, rather than
   assuming the pool is freezable. See "Revised against f9c2e42c" for the commit evidence.

3. Image text tower reuses the chat model frontend. `src/serve/image_text_tower.cpp:4` includes
   `models/qwen3_5/frontend/chat_template.h` and uses `fi::CompiledChatTemplate::resolve` (`:131`). The
   generation route borrows the chat model's compiled chat-template machinery. Low upstream churn (9
   commits), so acceptable, but it is a real cross-lane dependency that must keep its public shape.

4. Additive dirs still depend on upstream Core ABI, and now on a hardened free-memory contract.
   `src/models/qwen_image/*` and the four ops include `core/tensor.h`, `core/weight.h`,
   `core/weight_view.h`, `core/device.h`, `core/vram_budget.h` (`image_pipeline.cu:4-7`,
   `dit_block.cpp:3`, `image_device_util.h:11-13`, each `ops/*/launch.h:3`). These are upstream-owned
   primitives (Core owns tensor/weight/device). A concrete consequence landed in
   `fix(core): never count other programs' VRAM as free` (`b821d961`, 2026-11-20): `VramSnapshot::device_free`
   is now capped at NVML's device-wide `physical_free` in `VramBudgetSource` (`src/core/vram_budget.cpp`),
   so startup sizing, the expert frames' growth, and the run-time pressure check all refuse to count any
   other holder's VRAM as free. Our `probe_image_yield` / image-admission path that estimates "how much the
   LLM can hand back and whether the picture fits" must be re-derived against that capped figure - a probe
   that assumed freed-then-reusable VRAM will now read a smaller free number than it expects. NOTE: `b821d961`
   was already in our stale baseline `0a3f432b` (it is NOT new in the latest 119), so this is a pre-existing
   entanglement the first draft under-analyzed, not a fresh regression. `vram_budget.{cpp,h}` did not change
   again in `0a3f432b..f9c2e42c`. This remains a normal "build against upstream headers" dependency (no hook
   to edit), but a Core API change can break the lane even with zero edits - track it.

5. No build isolation today. There is no `INFERNIX_IMAGE` option (root `CMakeLists.txt` has only
   `INFERNIX_BUILD_APPS`/`BUILD_TESTING`/`INFERNIX_BUILD_BENCHMARKS`). The lane is wired unconditionally:
   `src/models/CMakeLists.txt:22`, `src/ops/CMakeLists.txt:17,19,42,45`, `src/serve/CMakeLists.txt:20-27,47`.
   So every upstream build of `infernix_serve`/`infernix_models`/`infernix_ops` currently drags in the whole
   image lane. Verified good news: the four ops and `infernix_qwen_image` have NO consumers outside the image
   lane and tests, so they CAN move behind a flag with no external fallout.

## What upstream added that conflicts or makes part of our lane obsolete

- No generation-lane conflict: upstream has no DiT, no `/v1/images`, no interlude, no `openai_images`. Our
  generation feature is purely additive. Nothing is obsolete.
- Chat multimodal INPUT is upstream-owned (`qwen3_5/state/state_image.*`, `prefix/state_image.*`,
  `types.h` `VisionOffload`/`MediaKind::Image`/`ImageResizePolicy`/vision budgets, `examples/cli/messages/image_*.json`).
  This is the chat model reading pictures, NOT generating them. It does not duplicate our lane, but do not
  overwrite it when dropping our tree in. Our `image-spike/` and our `bench/fixtures/ttft/text/many_image_*` are ours;
  the chat-vision `state_image` files are upstream.
- No CLI flag collision on names (`--image-artifact`, `--image-prompt-enhancer-url`, `--image-retention-days`,
  `--image-text-processor` are ours; upstream has none), and no new upstream `--image-*`. RE-VERIFIED at
  `f9c2e42c`: `git grep -oE '\-\-image[a-z-]*'` over `src/serve` + `apps/` returns empty upstream, while our
  tree still has exactly those four. The inkling2 launch options (`--lora`, `--ablate`, `--a8-activations`,
  `--q8-weights`, `--spec dspark`, `--vision-offload`, token constraints, thinking budgets) added since the
  baseline do NOT overlap the four `--image-*` names. The only `serve_options` risk remains the growing region
  around our edits.
- No new overlap in the `/v1/capabilities` image block: upstream added no `/v1/images` route and no image
  entry to `/v1/capabilities` (0 inkling commits touched `http_server.*`; the only "image" hits are Anthropic
  multimodal INPUT parsing in `anthropic_messages_request.cpp`). Our capabilities image block does not collide.
- The model registry grew an `Architecture` enum (`src/models/registry.h:7` now
  `{ Qwen3_5, Qwen3_5Moe, Qwen4Exp, Inkling2Loop }`) and `engine.cpp` dispatches construction over it, but our
  image lane is NOT a registry `Architecture` - it is a separate `ImageGenerationService` attached via
  `attach_images()` with its own `/v1/images` routes (`http_server.h:80`). So admitting a chat family does not
  force us to edit the registry/enum; that pressure is upstream-internal.
- No artifact-container edits by our lane (`src/artifact/` shows 0 image commits, no image files). Our generation
  artifact is consumed through upstream's generic `artifact::Binder` via `qwen_image/image_weights.cpp`, and produced
  by our `tools/convert/qwen_image.py` (ours-only). So the container is a passive dependency, not a hook - but if
  upstream changes the `.infernix` container/binder format, `image_weights.cpp` binding breaks. Watch, do not fork.
  One passive-dependency watch item from the 119: inkling2 touched `src/artifact/formats.cpp` and
  `src/core/weight.{h,cpp}` / `src/core/weight_view.{h,cpp}` (new quant layouts: NVFP4 banks, lossless, a8);
  re-check `image_weights.cpp` binding against any container/weight-format change when the port runs.
- The inkling2 family also reused upstream `src/media/decode/*` (png_decode, resize) for its vision tower. If
  the image lane later wants a shared media/decode path rather than its own, that is now upstream-owned code;
  for now our lane is unaffected (it has its own pipeline).

## Pressure test of plan 2 against reality

Plan 2 claims four upstream surfaces so that a future drop only touches those. Checked one by one. (The
empirical family-admission measurement against upstream's own "Inkling 2" work, which puts the real number
at roughly six to eight seams for our narrower separate-service lane rather than four, is in the "Revised
against f9c2e42c" section below.)

1. Device-lease on Engine (move lease defs from `program_impl.h` to `src/runtime/expert_offload/device_lease.cpp`,
   keep public names). SEVERS? Partially. The move is the plan's linchpin but is presented as "free functions taking
   the locks they need as parameters"; the actual bodies touch ~150 chat-Program internals and `execution_mu_` (now gone).
   As written it stays unbuildable against upstream. MUST ADD: a real narrow facade (see deltas) so the chat Program
   keeps its own private state and the lease calls only the facade.
2. Worker-ack in `engine_core.h` where `interlude_preempt_outstanding` "already lives". SEVERS? No. That name never
   existed upstream; the whole block is ours grafted into a file that diverged +1169/-504. MUST CHANGE: re-anchor to
   current upstream worker code and pin the contract in `docs/maintainer/image-integration-points.md`.
3. Yield code in our own file called from `program_impl.h`. SEVERS? Partially; same coupling as (1), in the single
   highest-churn upstream file.
4. One serve-hook per protocol + `attach_images` by interface + capabilities contributed by module. SEVERS? Yes,
   this is the genuinely tractable one - all tool-injection helpers are ours, upstream's `ChatTool`/`tool_calls`
   anchor survives (and `GenerationRequest`/`GenerationOutcome` have moved out of `types.h` into
   `src/serve/request.h:266` / `src/serve/generation_service.h:33`, which is closer to our handlers), and the
   http_server image block is structurally separable. But it still edits `http_server.{h,cpp}`, the 3 protocol
   handlers, `serve_options`, `main.cpp`, CMake.

Net: even after plan 2, a port edits ~10 upstream files and re-anchors two files against rewritten chat-Program/engine
internals. It is better than today (no flag, ~19 edited files, unconditional wiring) but not "mechanical".

## Concrete deltas plan 2 must ADD or CHANGE

Work packages (with files and tests). Goal: a future port is "drop our subtree, flip `-DINFERNIX_IMAGE`, fix N named
seams" rather than a rewrite.

- WP-A Real build flag + conditional wiring (blocks everything). Root `CMakeLists.txt`: add `option(INFERNIX_IMAGE
  ...)`. Make unconditional includes conditional: `src/models/CMakeLists.txt:22`, `src/ops/CMakeLists.txt:17,19,42,45`,
  `src/serve/CMakeLists.txt:20-27,47`. Move the four ops to a standalone `infernix_image_ops` target and qwen_image
  behind the flag. Tests: build matrix with `-DINFERNIX_IMAGE=OFF` green (chat suite) and ON.
- WP-B Lease facade (the missing piece). Define a narrow interface (e.g. `ImageDeviceLease` in `include/infernix/image/`)
  that the chat Program implements; expose only: freeze/thaw residency pool, allocate/release in freed region, drain/wake
  worker, quiesced-epoch gate, probe-fit. The ~150 `residency_`/`plan_`/`pool_`/`kv_vmm_`/`vram_mutex_` refs stay inside
  upstream-owned `program_impl.h`; `device_lease.cpp` calls only the facade. Re-anchor `execution_mu_` to whatever
  replaced it upstream. Files: `program_impl.h`, `program.{h,cpp}`, new `device_lease.cpp`, `engine.cpp` (forward through
  new variant cores, not `impl_->qwen4->program`). NEW (from `f9c2e42c`): the facade must also expose an explicit
  "override the residency keep-policy and evict the full expert set for this lease" primitive, because the new
  experts-in-RAM branches (`a3f78ef8`/`18388f5b`) now default to keeping experts resident and will not hand the pool
  back on the old neutral path; and `probe-fit` must report free VRAM through the NVML-capped `VramSnapshot`
  (`b821d961`), not a budget-derived figure. Tests: the lease/coexistence suite in `tests/models/qwen4_exp/tests.cmake`
  plus a facade-contract host test asserting the lease reaches the Program only via the facade, and a case that forces
  the no-SSD-tier "every expert in RAM" configuration to prove the facade still gets the VRAM back.
- WP-C Worker-ack re-anchor. Replace plan's assumed `interlude_preempt_outstanding`/`request_interlude_preempt`/
  `wake_llm_worker`/`drain_llm_frontier` with the current upstream engine_core worker entry points; keep them named,
  narrow, image-agnostic. Files: `engine_core.h`.
- WP-D Probe/interlude types out of `types.h`. Move `ImageYieldProbe`/`ImageInterludeContext` into `include/infernix/image/`
  so the high-churn protocol header is not part of our diff; `expert_pool_frozen_by_lease` becomes a facade-returned
  capability (the WP-B eviction override), not a residency leak. Files: `types.h`, `engine.h`, `image_generation_service.cpp`.
  Favorable `f9c2e42c` change: upstream already moved `GenerationRequest`/`GenerationOutcome`/`ChatTool` out of
  `types.h` into `src/serve/request.h` / `src/serve/generation_service.h`, so the high-churn public header is no longer
  where our protocol edits would have sat - our diff shrinks and the handlers we anchor onto now live beside those types.
- WP-E Serve hook (plan 2 Task 4, keep) + re-anchor. `ImageModule` with one hook per protocol; change
  `attach_images(ImageGenerationService&)` to `attach_image_module(ImageModule&)` (`http_server.h:80,172`); move the
  capabilities image block (`http_server.cpp:659-788`) and `/health image_jobs` (`:539`) into `capabilities_block()`.
  Re-anchor the 3 protocol handler injection sites and the `main.cpp:323-331` builder against current upstream. Keep
  the byte-identical `/v1/capabilities` gate.
- WP-F Family seam (plan 2 Tasks 1-3) - fine, but note it does not by itself buy portability; it is orthogonal to the
  seam work above.
- WP-G Migration mechanics + docs. New `docs/maintainer/image-integration-points.md` enumerating every seam with current
  upstream anchor. Add the `--image-family` flag + docs/`--help` as an external-contract commit.

Migration mechanics options (recommendation):
- Vendored subtree for `plugins/image` (the 149 ours-only files): keeps one repo/one build, zero upstream edits for the
  feature, `git subtree`/`submodule` so a new upstream sync never touches them. Good.
- Separate library target `infernix_image` behind `-DINFERNIX_IMAGE`: necessary (WP-A) but not sufficient without the
  facade (WP-B).
- Patch series (`git format-patch`/`git am`) for exactly the seams (engine.h lease API, engine_core worker-ack, program
  facade hooks, serve options/routes/main, the 4 CMake conditionals). RECOMMENDED PRIMARY mechanism: with unrelated
  histories there is no merge/rebase, so a small, reviewable, re-appliable patch set over ~6-8 named hunks is the only
  realistic "port" primitive, and it matches the operator's own "rebase the patch" framing.
- Upstream contribution (best end state): land the generic seams (narrow lease facade, worker-ack primitive,
  attach-by-interface serve hook, `INFERNIX_IMAGE` flag) as upstream PRs so they become `origin/main` and we carry zero
  diff on those files; keep only the family/pipeline/ops/protocol code as our subtree.

One recommendation: hybrid - vendored subtree for the 149 ours-only feature files + a short upstream-contribution-first
patch series for the seams, collapsing to "sync subtree, `git am` the seam patches, fix the named anchors, build behind
`-DINFERNIX_IMAGE`". This is the only path that makes a future port mechanical, and it requires WP-B (the facade) as the
enabling change; without it the lane stays welded to the chat Program's rewritten internals.

## Revised against f9c2e42c (2026-11-24): what the 119 upstream commits did and did not change

New head verified: `git rev-parse f9c2e42c` = `f9c2e42cb3e987d6159ed0de7afd2205a806ffd6`, 2026-11-24
12:51:50 +0000, "Merge origin/main into fix/inkling2-defects"; `origin/main` == `f9c2e42c`; ours `HEAD`
= 427 commits, upstream = 2684; `git rev-list --count 0a3f432b..f9c2e42c` = 119. The stale baseline
`0a3f432b` was 2026-11-23 12:59 +0000, so the advance is one working day of upstream (the inkling2
family + `perf(qwen4_exp)`).

### Anchor re-verification against the explicit SHA (all confirmed)

Everything the risk table leans on was re-checked against `f9c2e42c` via `git show f9c2e42c:<path>`, not
the moving ref. Unchanged in status: `execution_mu_` 0 hits, all yield/interlude names 0, all worker-ack
names (`request_interlude_preempt`/`wake_llm_worker`/`drain_llm_frontier`/`interlude_preempt_outstanding`)
0 in `engine_core.h`, the lease API (`supports_image_interlude`/`run_image_interlude`/`probe_image_yield`)
0 in `engine.h`. Still-surviving upstream anchors our code grafts onto: `program_impl.h`
`vram_mutex_`=5, `residency_`=127, `kv_vmm_`=12, `plan_`=148, `pool_`=79; `engine.cpp` variant cores
(20 hits); `engine_core.h` generic `worker` present; all qwen4_exp program files and `engine_core.h`
still at identical paths. One anchor MOVED: `types.h` no longer holds `GenerationRequest`/
`GenerationOutcome`/`ChatTool` (they relocated to `src/serve/request.h:266` and
`src/serve/generation_service.h:33`) - this helps us (rank 5 downgraded).

### How much churn the 119 actually added on our seams

Less than the raw commit count suggests, because the entangled files had already diverged before the
baseline. Within `0a3f432b..f9c2e42c`: `program_impl.h` grew only +58/-6 across 3 commits (2 are the
experts-in-RAM perf, 1 a prefix-cache fix); `engine_core.h` changed 0 lines (unchanged); `engine.cpp`
+126/-43 (added `Inkling2Loop` as a 4th construction core); `types.h` +31/-4 (the relocation above). So
the two CRITICAL/HIGH seams did NOT get materially worse from this advance. The churn column in the table
above now reflects total `--since=2026-10-25` counts against `f9c2e42c`.

### Empirical test of plan 2's "four upstream surfaces"

The 119 commits contain upstream admitting a whole new chat family ("Inkling 2"), which is the direct,
same-repo measurement of how many shared surfaces a family actually costs. Method: `git log f9c2e42c
--grep=inkling -i` = 113 commits; `git show --stat` on the five foundational admission commits
(`35427922`, `8c39b02b`, `b48a7cb1`, `fbdb0b5f` "the Engine runs Inkling 2", `ecb8e913`); distinct files
touched across all 113 = 302, of which 192 are OUTSIDE the family's own `src/models/inkling2`/`tests/models/
inkling2` trees (shared seams), and 134 are code (excluding docs/model-cards/launcher-web/bench/fixtures/py).

That is emphatically NOT four surfaces. Even the MINIMAL chat wiring (not the later per-feature lossless/a8/
q8/media/grammar churn) required editing shared upstream files: `src/models/CMakeLists.txt`, `src/models/
registry.{h,cpp}` (the `Architecture` enum + name/parse maps + dispatch), `src/runtime/CMakeLists.txt`,
`src/runtime/engine/engine.cpp`, `engine_plan.{h,cpp}`, `model_instance.cpp` (+ a new `inkling2_instance.{h,cpp}`),
`src/serve/serve_options.{h,cpp}`, `apps/serve/main.cpp`, `apps/cli/{main,options}.{h,cpp}`, `include/infernix/
types.h` (new EngineOptions fields), `src/launcher/{launch_plan,model_defaults,option_support}.{h,cpp}` +
`launcher/CMakeLists.txt`, and the tests harness (`tests/CMakeLists.txt`, `tests/cmake/ProductTests.cmake`,
`tests/ops/tests.cmake`, `tests/ops/op_family.cpp`, `tests/test_engine_options.cpp`, `tests/test_serve_options.cpp`).
That is roughly 27 shared upstream code files for the minimal admission - and upstream admitted a family by
EDITING THE OTHER FAMILIES' SHARED FILES too (`qwen3_5/config.cpp`, `qwen3_5/frontend/*`, `qwen3_5/program/
planning/startup.cpp`, `qwen4_exp/load.cpp`, `qwen4_exp/model.h`). Families are NOT cleanly separated in this
codebase.

Reconciliation to our lane: our image lane is genuinely narrower than a chat family, so the 27 is not our
number - we are a separate `ImageGenerationService` attached by `attach_images()`, not a registry `Architecture`,
so we do NOT touch `registry.h`/`enum Architecture`/per-family construction. But plan 2's claim of exactly
"four upstream surfaces" is an undercount of the seams that actually carry VRAM semantics. The honest current
count of upstream-owned surfaces a port must re-anchor is about SIX to EIGHT: (1) `engine.h`/`engine.cpp` lease
API + forwarding across the now-4 variant cores, (2) `engine_core.h` worker-ack, (3) `program_impl.h` freeze/
resume (CRITICAL), (4) `src/core/vram_budget` probe coupling (contract hardened by `b821d961`), (5) the residency/
pool eviction override (hardened by `a3f78ef8`/`18388f5b`), (6) `serve_options` + protocol handlers +
`http_server` attach + capabilities block + `apps/serve/main.cpp`, (7) CMake conditionals (`INFERNIX_IMAGE`
option + the 3 CMakeLists), (8) `tests/cmake/ProductTests.cmake`. Two of those (4 and 5) are NEWLY hostile in
this advance. So plan 2's "four" should be restated as "four structural hooks plus two residency/VRAM contract
surfaces that must be owned behind the WP-B facade," and the plan must budget for them explicitly.

### The one NEW blocker the 119 commits introduced

`perf(qwen4_exp)` "every expert in RAM" (`a3f78ef8`, `18388f5b`, both 2026-11-24, absent from `0a3f432b`).
They add `!tier_` (no SSD tier) sizing branches in `program_impl.h` (~4399, ~5299) that keep the full expert
set resident via `residency_->lendable_bytes()`, with explicit measured reasoning that "the cache keeps those
frames' experts." Our image interlude's whole premise - freeze the expert pool and yield its VRAM to draw the
picture - now fights an upstream feature that was tuned to do the opposite. The port is no longer "re-anchor a
neutral hook"; it must override a policy upstream deliberately strengthened. This is the concrete new risk; the
rest of the advance (inkling2 serve/registry/launcher) does not touch our lane.

### What the advance made easier

- `GenerationRequest`/`GenerationOutcome`/`ChatTool` left `types.h` for the serve layer, shrinking our diff
  against the highest-churn public header and placing those types beside the protocol handlers we anchor onto.
- Serve model enumeration is registry-driven: 0 inkling commits touched `http_server.*`, confirming an added
  family does not force `http_server` edits, so our capabilities/routes stay separable.
- Still zero `--image-*` collisions and zero upstream `/v1/images` or capabilities image block.

## What I could not verify

- Exact fork commit (no merge-base). Fork baseline estimated at ~2026-10-25 from identical-file upstream commit dates;
  "sinceFork" counts are `--since=2026-10-25` approximations, not graph-exact. (The 119-commit advance itself IS
  graph-exact: `git rev-list --count 0a3f432b..f9c2e42c` = 119, `0a3f432b` still an ancestor of `f9c2e42c`.)
- Whether upstream's `program_impl.h`/`engine_core.h`/`engine.cpp` rewrites preserve any equivalent lock or worker
  primitive that our `execution_mu_`/worker-ack could map onto - I confirmed the named symbols are absent but did not
  trace upstream's replacement for each (would need a full read of upstream's 114-commit `program_impl.h` history).
- Runtime behavior of a real port (no build/run was performed, per the read-only constraint and the live :8888 engine).
- Whether upstream changes the `.infernix` binder/container format (a passive dependency that could break `image_weights.cpp`).
  The 119 DID touch `src/artifact/formats.cpp` and `src/core/weight.*`/`weight_view.*` (new quant layouts for inkling2),
  but I did not diff them line by line against our binder expectations - flag for the port, not verified here.
- Whether the new experts-in-RAM branches (`a3f78ef8`/`18388f5b`) are behind a launch flag or condition that a lease
  could disable to recover the old freezable behavior, or whether WP-B's eviction-override is strictly required. I read
  the diff hunks and the `!tier_ && wide_assist() && a4_bank()` gating but did not trace every runtime switch.
- The current upstream `serve_options` help/arity table in detail. It grew past the first draft's 343 with the inkling2
  launch options; I confirmed our four `--image-*` names do not collide but did not recount the total or diff the exact
  insertion points for our flags.
- Whether the four image ops moving to a standalone target would collide with any op-registration/CLI surface inkling2
  added under `src/ops/` (consumer check showed none outside the lane; upstream op-registration format not diffed).
- Whether `probe_image_yield`'s current free-VRAM math already routes through `VramBudgetSource` or bypasses it; I
  established the source now caps `device_free` at NVML `physical_free` but did not trace our probe's read path line by
  line (in our own tree, would need a focused read of `image_generation_service.cpp` / the probe producer).
