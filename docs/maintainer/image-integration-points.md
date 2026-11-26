# Image lane integration points (seam manifest)

**This file is the authority a future port reads first.** Before touching any upstream-owned
file after an upstream advance, run the checker over this manifest and fix every named anchor
that went missing. Do not start the port from a build error; start from this table.

    py -3.13 -I tools/check_image_seams.py --upstream <sha>     # exit 1 == a must-port anchor is gone

Companion detail, measurements and cost model (both are the basis for the numbers here):

- `docs/research/2026-11-24-image-edit-port-assessment.md` - per-seam risk table, what upstream
  added that is hostile to the lease, the plan-2 pressure test, the honest verdict that this is a
  manual port and never a rebase (unrelated histories, no merge-base).
- `docs/research/2026-11-24-image-port-cheap-wins.md` - the measured surface, why the build flag
  and the `plugins/image/` relocation are traps, and the **time-to-green**: **5-9 working days**
  at zero preparation, **3.5-6 working days** with the cheap preparation items landed (this
  manifest plus its checker, the arity pins, and the dry-run harness). The residual that no cheap
  prep removes is the `program_impl.h` re-derivation and the residency/VRAM contract fight, which
  needs the `ImageDeviceLease` facade.

Fixed reference points, restated so nobody re-derives them:

- Our tree and upstream share **no merge-base** (`git merge-base HEAD f9c2e42c` -> none). There is
  no rebase and no merge; the only port primitive is `git diff <ref> -- <file>` plus
  `git apply --3way` against blobs that both live in this one object DB.
- Upstream has **zero image-generation code**. `run_image_interlude`, `openai_images`,
  `attn_dit_joint`, `vae_decoder`, `/v1/images`, `/v1/capabilities`, `make_capabilities_json`,
  `handle_image_artifact`, `ImageJobLogRecord`, `infernix_media_encode` and `--image-artifact` all
  return **0 files** at `f9c2e42c`. Anything named `image` in upstream is chat multimodal INPUT
  (`qwen3_5/state/state_image.*`, `prefix/state_image.*`, Anthropic/PDF vision parsing) - upstream
  owns those, do not port our copies over them.
- `ChatTool` / `GenerationRequest` / `GenerationOutcome` live upstream in `src/serve/request.h:266`
  and `src/serve/generation_service.h:33`, **not** in `include/infernix/types.h`. Re-anchor the
  protocol handlers onto those, not onto the public header.
- `serve_option_arities()` exists **both** upstream (`src/serve/serve_options.h:246`,
  `src/serve/serve_options.cpp`) and in our tree, and our `tests/test_serve_options.cpp` mentions
  arity only once - upstream's help/arity consistency test is the pin we have not adopted.
- `cmake/FindFFMPEG.cmake` is **byte-identical** to upstream (blob `e86ffff16ec391929d0f83a914ee739561d2da17`)
  and contains **zero backslashes**. Do not blame a CMake syntax error on it. The real configure
  abort is `find_package(FFMPEG REQUIRED)` at `cmake/Dependencies.cmake:26` failing because
  `av*.lib`/headers are absent under the vcpkg triplet tree - historically an `x6-windows`
  (typo) triplet mix-up, not a module bug.
- The four lane `--image-*` flag names collide with nothing upstream, and upstream has no
  `/v1/images` route and no capabilities image block.

## How the hunk counts here were measured

Counts are **lane hunks in OUR tree**, not upstream churn. For each upstream-owned file we take the
lines that (a) contain a lane token from the documented vocabulary and (b) do **not** appear in
upstream's own version of the same file - that second condition is what removes chat-side false
positives. Contiguous clusters (gap of at most 6 non-lane lines) count as one hunk.

Total measured before the seams landed: **34 upstream-owned files, 147 lane hunks**.

That number is the *before* picture and is kept because the checker's must-port rows are still the
upstream side of the contract. The after picture is what the rows below say: in an upstream-owned
file the lane now appears only as **typed seam calls** (`category: seam`, checked against *our* tree by
`tools/check_image_seams.py`) plus the accepted residues listed under "Accepted residues" further
down. Everything else counted is an upstream-owned line that merely happens to contain one of the
words. Re-measure with the same two conditions before adding any row.

That is deliberately higher than the companion documents' "31 files, ~113 regions". The difference
is the second condition and a wider vocabulary, and it is not a disagreement to average away: the
earlier scan missed three real seams (`src/runtime/expert_offload/expert_residency.cpp`,
`tests/models/qwen4_exp/test_memory_plan.cpp`, `tests/test_request_log.cpp`) and undercounted
`http_server.cpp` (16 vs 19), `test_http_routes.cpp` (12 vs 17), `serve_options.cpp` (1 vs 3),
`apps/serve/main.cpp` (3 vs 4) and `program.h` (4 vs 5), while overcounting `program.cpp`
(7 vs 5). Where this manifest and the research documents differ, **this manifest is the one the
checker enforces** and the one to trust for a port.

Reproduce it with the vocabulary in the cheap-wins document (candidate C) before you trust any row.

## Category meanings

- `must-port` - image semantics cannot work without this hunk. The checker fails (`exit 1`) if a
  `must-port` anchor is missing.
- `port-if-missing` - a convenience we can lose: a comment, a reason bit, a test that pins a
  premise. The checker warns, does not fail.
- `ours-only` - no upstream counterpart at all. The checker skips these; they are pure re-adds and
  can only be checked by building.
- `seam` - an upstream-owned file the lane reaches only through its typed calls. Its `anchors:` are the
  lane's own calls and are checked against **our working tree** (a replay that lost a call fails), and
  its `residue:` tokens are counted in the same file against `lines:` (a replay that leaked a lane rule
  back in, or deleted a seam line, fails). A `must-port`/`port-if-missing` row may carry the same
  `residue:`/`lines:` pair to keep the lane gate on a file that still holds accepted residues.

An `anchors` list is a set of upstream **symbol/function/registration names** the hunk hangs off.
Never a line number - line numbers rot across upstream's 2684 commits. Every anchor listed here was
confirmed present in the named upstream file at `f9c2e42c` with
`git grep -c <anchor> f9c2e42c -- <path>`; the checker re-runs exactly that. For a `seam` row the
anchors are checked the other way round: they are the lane's own typed calls and the checker looks for
them in **our working tree**, because what a replay can silently lose is the call, not the symbol.

---

## The typed attach points

The lane reaches upstream-owned code through exactly five surfaces, each of them one or a few typed
calls into a file the lane owns. There is no registry, no name-driven dispatch, no `#ifdef`, no
configuration-driven hook table and no plugin discovery - the repository's own rules forbid those
(`AGENTS.md`: "Do not introduce generic model graphs, family base classes, plugin discovery,
string-driven execution..."), and a general dispatcher would be a worse port target than the five lines
below because it would have to be re-derived, not re-added.

| seam | upstream file it touches | lane file it reaches | the call the upstream file keeps |
|---|---|---|---|
| OPTIONS | `src/serve/serve_options.cpp` | `src/serve/image_serve_options.{h,cpp}` | `image_option_arity(name)`, `image_help_text()`, `parse_image_option(arg, options, require_value, parse_nonnegative_int)` |
| CAPABILITIES | `src/serve/http_server.cpp` | `src/serve/image_capabilities.{h,cpp}` | `add_image_capabilities(payload, options, image_model_id)` |
| MAIN_WIRING | `apps/serve/main.cpp` | `src/serve/image_startup.{h,cpp}` | `start_image_lane(server, service, options, logger)` |
| TOOL_INJECTION | the three protocol handlers and the three response renderers | `src/serve/image_chat_hook.{h,cpp}`, a facade over `src/serve/chat_image_tool.{h,cpp}` | `make_chat_image_hook` once per request, then `inject_chat_image_tool`, `chat_image_round_needed`, `chat_image_replay_needed`, `run_chat_image_rounds`, `chat_image_missing_url_suffix`, `chat_image_progress_note`, `add_chat_image_attempts`, `chat_image_tool_hidden` |
| BUILD | `src/serve/CMakeLists.txt`, `tests/cmake/ProductTests.cmake`, `tests/ops/tests.cmake`, `tests/models/qwen4_exp/tests.cmake` | `src/serve/image_lane.sources.cmake`, `tests/cmake/image_lane_tests.cmake`, `tests/ops/image_lane.tests.cmake`, `tests/models/qwen4_exp/image_lane_tests.cmake` | one `include(...)` per file, the same idiom upstream uses for its own `linear/tests.cmake` lists |

Three rules hold at every one of them, and a reviewer should refuse anything that breaks one:

- The lane owns its **text and its rules** (the tool's schema, the refusal codes, the progress prose,
  the field's name, the null rule, the shutdown order). What stays in the upstream file is what only
  that file can own: its request, its `service_`, its route label, its re-prepare/re-run callbacks.
- A call is typed and one-shot. Nothing in an upstream file may branch on a string, look a lane object
  up by name, or know what the lane does with what it hands over.
- A serve started without `--image-artifact` must be **byte-identical** to one before the lane existed,
  so every seam takes a null that means "no pipeline here" and the upstream file keeps its own branch
  only where it had one before.

The capabilities document is the one contract with a pinned artefact: `image-edit/_p42_capabilities_golden.json`
(20 185 bytes, sha256 `dfd49e23b81fc5c3355734cdc0fc020b56b74c39bb0660fc05aaa1d23bb969c4`), captured
live from `GET /v1/capabilities` before the seam moved it. It sits in the working folder's git-ignored
scratch directory, so it is a checkpoint and not a fixture: recapture it from the deployed serve before
deliberately changing any capabilities output, and diff against the old one.

To add a sixth attach point: write the lane file first with one typed entry point, call it from the
upstream file, give it a `category: seam` row whose `anchors:` name those calls and whose `residue:` and
`lines:` pin that file, and land it only when `tools/check_image_seams.py` is green.

---

## Accepted residues

These are the lane's fingerprints left in upstream-owned files on purpose. They are enumerated so that
nobody mistakes them for an unfinished seam and nobody removes one without a decision:

- `src/serve/http_server.h` - `attach_images(ImageGenerationService&)`, `run_image_job`,
  `image_model_id()`, the `handle_images*` / `handle_image_artifact` declarations, the
  `ImageGenerationService* images_` member and the `serve/openai_images.h` include. Deliberately not
  turned into a seam: adding a lane accessor or a `friend` to upstream's class would trade one upstream
  diff for another, so the handlers hand their own private state over to the lane as the typed
  `ChatImageDoors` instead. That struct carries `std::function<void*()> pipeline`, which is the whole
  price of not touching the header; the round trip is exact because `images_` is a plain
  `ImageGenerationService*` with no base class, and the handler fills it with `static_cast<void*>(images_)`.
- `src/serve/http_server.cpp` - the three `/v1/images/*` route registrations, `handle_image_artifact`,
  `image_model_id()`'s definition, the `/health` `image_jobs` figure, the image model object in
  `/v1/models`, and the two lane includes. These are the route surface, not lane rules.
- `src/serve/serve_options.h` - the four `--image-*` fields of `ServeOptions` and their comments.
- `src/serve/generation_service.h:74` - `std::optional<std::string> image_tool_attempts_json;`. A
  payload field carried from the tool's own writing to the renderers, not text the lane wants back.
- `src/serve/http_transport.{h,cpp}` - the `HeartbeatTicker` `watch` parameter typed
  `std::function<std::shared_ptr<ImageJobProgress>()>`, and `artifact_url_base` with
  `kImageArtifactRoutePrefix`, which the lane reads through the hook.
- `src/serve/request_log.h` - `ImageJobLogRecord`, `write_image_job`, `format_image_job_json`: the
  request-log record one draw leaves, whichever door it came in by.
- `src/serve/openai_common.cpp` - the comment and the model object that publish the image pipeline as a
  routeable resource on `/v1/models`.
- The lane cases inside upstream-owned **test** files (see "Lane tests inside upstream-owned test
  files"): they are replayed as patch hunks, not moved, because each one belongs to the protocol
  surface it tests.

---

## Ours-only lane files (pure re-adds)

Nothing here has an upstream counterpart; a replay copies them whole and builds. The checker skips
`ours-only` rows, so a build and the suites are the only evidence they are intact.

**The authority for what gets copied is [`image-lane.copy-set.json`](image-lane.copy-set.json)**, not
this page. It is generated from CMake's own generated projects by

```
py -3.13 -I tools/image_lane_copy_set.py --upstream <sha> --build-dir build-windows --write-json docs/maintainer/image-lane.copy-set.json
```

which reads the lane's own `.cmake` files, takes the target names out of them, reads each target's
`<target>.vcxproj` (the exact `ClCompile`/`ClInclude`/`CudaCompile`/`None` item set and
`AdditionalIncludeDirectories` CMake resolved) and closes the `#include` graph over those directories.
Everything the result reaches that the upstream ref lacks is copied - headers, `.cu` kernels, JSON
fixtures and the two Python tools the serve invokes. A target whose sources are all ours-only is taken
whole; a target upstream also builds (`infernix_serve`, `infernix_ops`, ...) contributes only the files a
lane `.cmake` file names, so the port carries the lane and not the fork's unrelated chat-model work.
Three reached files are deliberately excluded, each with its reason written in the JSON.

Regenerate the JSON whenever a lane file is added, deleted or newly reached.
`tools/check_image_seams.py` re-derives it and fails the tree when it goes stale, and
`tools/port_lane.ps1` refuses to start a port from a stale list - the first replay of this lane onto
upstream tip copied the 15 rows below and died at configure because the lane's `src/ops/*/sources.cmake`,
`src/models/qwen_image/sources.cmake` and every `.cu` kernel were missing.

The 15 rows below stay as the explained subset: what each of those files is for, and which seam reaches
it. `tools/port_lane.ps1` copies the union of these rows (plus each row's `companions`) and the JSON.

### src/serve/image_serve_options.cpp
    our path: src/serve/image_serve_options.cpp
    upstream path: src/serve/image_serve_options.cpp
    category: ours-only
    companions: src/serve/image_serve_options.h
    hunks: 0
    anchors: image_option_arity
    does: The four --image-* options, their arity, value checks, help text and diagnostics - reached from
      src/serve/serve_options.cpp by the OPTIONS seam.

### src/serve/image_capabilities.cpp
    our path: src/serve/image_capabilities.cpp
    upstream path: src/serve/image_capabilities.cpp
    category: ours-only
    companions: src/serve/image_capabilities.h
    hunks: 0
    anchors: add_image_capabilities
    does: Every image block of /v1/capabilities (model, endpoint, dimensions, steps, response formats,
      the edit route's enhancer rules, canvas refusals and the whole refusal vocabulary), reached from
      src/serve/http_server.cpp by the CAPABILITIES seam.

### src/serve/image_startup.cpp
    our path: src/serve/image_startup.cpp
    upstream path: src/serve/image_startup.cpp
    category: ours-only
    companions: src/serve/image_startup.h
    hunks: 0
    anchors: start_image_lane
    does: The lane's startup - the job store, the enhancer handle, the eight artifact-reading providers,
      server.attach_images, the retention sweep - and ImageLane's declaration order, which IS the
      shutdown contract. Reached from apps/serve/main.cpp by the MAIN_WIRING seam.

### apps/serve/main.cpp
    our path: apps/serve/main.cpp
    upstream path: apps/serve/main.cpp
    category: seam
    hunks: 2
    anchors: start_image_lane, serve/image_startup.h
    residue: image, Image, start_image_lane, ImageLane, ImageGenerationService, attach_images,
      ImageJobStore, image_artifact_path, image_text_processor_path, image_prompt_enhancer_url,
      image_retention_days, prompt_enhance, GeneratedImageRetention, make_engine_interlude_runner
    lines: 5
    does: The MAIN_WIRING seam, and the whole of what the serving main knows about images: one include of
      the lane's own header and the one typed call `start_image_lane(server, service, options, logger)`,
      kept alive until after `listen()` returns. It has no branch, no option name and no lane type of its
      own - everything about whether an image pipeline exists, and how it starts, belongs to
      `src/serve/image_startup.{h,cpp}`. Two hunks, and the 6a648048 port lost the second one: the hunk
      selector replayed the call and dropped `#include "serve/image_startup.h"`, which the build reported
      as `C3861: 'start_image_lane': identifier not found` - the same failure mode as
      `src/serve/http_transport.cpp`'s dropped include, and the reason both are recorded here.

### src/serve/image_chat_hook.cpp
    our path: src/serve/image_chat_hook.cpp
    upstream path: src/serve/image_chat_hook.cpp
    category: ours-only
    companions: src/serve/image_chat_hook.h
    hunks: 0
    anchors: make_chat_image_hook
    does: The typed facade the three handlers and three renderers call: the hook, the doors it is handed,
      the rounds, the progress note, the attempts field and the hidden-call rule. A thin layer over
      src/serve/chat_image_tool.{h,cpp}, which stays the tool's real owner.

### src/serve/chat_image_tool.cpp
    our path: src/serve/chat_image_tool.cpp
    upstream path: src/serve/chat_image_tool.cpp
    category: ours-only
    companions: src/serve/chat_image_tool.h
    hunks: 0
    anchors: kChatImageToolSchema
    does: The server-executed generate_image tool itself: its schema, the caps on images and rounds, the
      round loop and the attempts witness the clients read.

### src/serve/openai_images.cpp
    our path: src/serve/openai_images.cpp
    upstream path: src/serve/openai_images.cpp
    category: ours-only
    companions: src/serve/openai_images.h, src/serve/openai_images_http.cpp
    hunks: 0
    anchors: parse_openai_images_request
    does: The two image routes' request type and parser (sizes, seeds, the 1..4 references, the
      response_format vocabulary) and their handlers.

### src/serve/reference_image.cpp
    our path: src/serve/reference_image.cpp
    upstream path: src/serve/reference_image.cpp
    category: ours-only
    hunks: 0
    anchors: prepare_reference_planes
    does: The edit route's reference decode and the reference planes the VRAM ledger bills and the ViT
      lane forwards.

### src/serve/image_generation_service.cpp
    our path: src/serve/image_generation_service.cpp
    upstream path: src/serve/image_generation_service.cpp
    category: ours-only
    companions: src/serve/image_generation_service.h
    hunks: 0
    anchors: ImageGenerationService
    does: One picture job end to end: the probe and yield through the Engine's interlude, admission
      against the ledger, the Euler loop, the VAE decode, the enhancer call and the witnesses.

### src/serve/image_job_store.cpp
    our path: src/serve/image_job_store.cpp
    upstream path: src/serve/image_job_store.cpp
    category: ours-only
    companions: src/serve/image_job_store.h
    hunks: 0
    anchors: ImageJobStore
    does: The bounded store a picture's URL names, the retention sweep and the progress the heartbeat
      reads.

### src/serve/prompt_enhancer.cpp
    our path: src/serve/prompt_enhancer.cpp
    upstream path: src/serve/prompt_enhancer.cpp
    category: ours-only
    companions: src/serve/prompt_enhancer.h
    hunks: 0
    anchors: PromptEnhancer
    does: The loopback sidecar handle (load/rewrite/unload), its health and the answer contract that
      keeps a bad rewrite from reaching the pipeline as the client's instruction.

### src/serve/image_text_tower.cpp
    our path: src/serve/image_text_tower.cpp
    upstream path: src/serve/image_text_tower.cpp
    category: ours-only
    companions: src/serve/image_text_tower.h
    hunks: 0
    anchors: make_tower_prompt_renderer
    does: The RAW-prompt text tower: the family's chat templates, the ti2i blocks with their
      <imageN> tags, and the host halves of the prompt renderers.
      Note for the next port: the lane reaches upstream for two things here, and upstream moves them.
      `CompiledChatTemplate::resolve` still lives in `models/qwen3_5/frontend/chat_template.h`, but the
      tokenizer this file encodes its rendered prompt with (and that the two host suites
      `tests/models/qwen_image/test_image_tower_tokenize.cpp` and `test_image_edit_template.cpp` declare
      with `using Tokenizer = ...`) moved out of `models/qwen3_5/frontend/tokenizer.h` into
      `src/text/tokenizer.h` as `infernix::text::Tokenizer` / `TokenizerResources` before 6a648048, and
      that header no longer exists. A replayed copy of the fork's file therefore fails with
      `C2039: 'Tokenizer': is not a member of 'infernix::models::qwen3_5::frontend'`; the fix is the
      include and the two qualified names, not a re-plumbing. `TokenizerResources` grew a fourth
      `pipeline` member that defaults to `TokenizerPipeline::Qwen`, which is what the image processor's
      tokenizer.json declares, so the lane's three-positional construction stays correct.

### src/serve/image_lane.sources.cmake
    our path: src/serve/image_lane.sources.cmake
    upstream path: src/serve/image_lane.sources.cmake
    category: ours-only
    hunks: 0
    anchors: target_sources
    does: The BUILD seam's own half: the twelve lane translation units and the three libraries only the
      lane needs. Reached from src/serve/CMakeLists.txt by one include.

### tests/cmake/image_lane_tests.cmake
    our path: tests/cmake/image_lane_tests.cmake
    upstream path: tests/cmake/image_lane_tests.cmake
    category: ours-only
    hunks: 0
    anchors: infernix_chat_image_tool_test
    does: The seven lane suites (job store, the two route schemas, the service, the enhancer contract,
      the device-need ledger, the chat tool, the PNG writer). Reached from
      tests/cmake/ProductTests.cmake by one include.
      The file ends with a `foreach` over `infernix_chat_image_tool_test`, `infernix_http_routes_test`
      and `infernix_request_log_test` giving each `${CUDAToolkit_INCLUDE_DIRS}` (guarded by
      `if(TARGET ...)`). Those three host suites reach `src/core/device.h` - a lane header pulls
      `models/qwen_image/image_weights.h` into their translation unit, which takes
      `artifact/materializer.h`, which includes `core/device.h` for `DeviceFault` - and they link only
      libraries written in plain C++, so CMake puts no CUDA toolkit include path on their `cl.exe` line
      and `cuda_runtime.h` is not found. The fork never noticed because the Visual Studio generator
      imports the CUDA SDK's own `CUDA 13.4.props`, which adds the toolkit include dir to the whole
      project; a Ninja build does not. Re-add the block with the three names, not fewer: the two
      upstream-owned suites gained the same need the moment their lane hunks put a lane header in their
      TU, and they are named from here because their registration lives upstream.

### tests/ops/image_lane.tests.cmake
    our path: tests/ops/image_lane.tests.cmake
    upstream path: tests/ops/image_lane.tests.cmake
    category: ours-only
    hunks: 0
    anchors: infernix_dit_adaln_test
    does: The DiT oracle suites (complex RoPE, joint block-causal attention, AdaLN-Zero) and the
      joint-sequence ceiling. Reached from tests/ops/tests.cmake by one include.

### tests/models/qwen4_exp/image_lane_tests.cmake
    our path: tests/models/qwen4_exp/image_lane_tests.cmake
    upstream path: tests/models/qwen4_exp/image_lane_tests.cmake
    category: ours-only
    hunks: 0
    anchors: infernix_qwen4_exp_image_interlude_prod_real_test
    does: The seven LLM<->image coexistence suites on the qwen4_exp Program. Reached from
      tests/models/qwen4_exp/tests.cmake by one include.

---

## Lane-only patch selection

`git diff <upstream> HEAD -- <path>` is not the lane and must not be replayed as if it were. Measured on
this tree against upstream tip `6a648048aece`: the fork's diff adds **5947 lines, of which 680 (11.4%)
mention the lane at all**, across **2231 files that differ from upstream tip**. Replaying it whole
overwrites upstream's own refactors with the fork's older shape of the same file; that is exactly how the
first port failed, at `include/infernix/types.h` (the fork's `KvCacheStorage` and `ThinkingBudgetPolicy`
where upstream moved those into `include/infernix/kv_format.h`), in `src/core/device.cu`.

`tools/image_lane_patch_series.py` therefore keeps a hunk only when both hold:

1. at least one of its changed lines matches a lane token (the names the lane's own files and seams
   use, listed at the top of the tool), and
2. none of its deleted lines is unrelated upstream code — the lane may remove upstream lines it replaces,
   and only those.

Plus the reviewed per-file deny table `FORK_ONLY` (each entry read in the hunk that carries it, `--verbose`
prints every hunk with its header and verdict). On the tip above the result is 25 hunks over 19 rows, and
15 rows legitimately carry no lane change at all.

The rows that stay outside mechanical selection are the shared contracts this fork trails — `include/infernix/types.h`,
`include/infernix/engine.h`, `src/runtime/engine/engine_core.h`, `src/runtime/engine/engine.cpp`,
`src/models/qwen4_exp/program/program.{h,cpp}`, `src/models/qwen4_exp/program/program_impl.h`,
`src/runtime/expert_offload/expert_residency.cpp`, `src/serve/serve_options.{h,cpp}`. Those are
`tools/port_lane.ps1 -Manual <paths>` (or `-CopiesOnly` for the whole set): the target keeps upstream's
file and a person adds what the lane needs, because only the lane's own declarations are additive and the
fork's shape of everything else in those files is not the lane's business.

## Serve layer

### src/serve/http_server.cpp
    our path: src/serve/http_server.cpp
    upstream path: src/serve/http_server.cpp
    category: must-port
    hunks: 19
    anchors: register_routes, api_route_pattern, handle_unrendered_http_error
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 50
    does: Registers the three /v1/images/* routes through api_route_pattern (the upstream
      double-prefix rule), defines HttpServer::attach_images, carries the images_ dispatch and the
      /health image_jobs figure, and owns the /v1/capabilities route and its document. The CAPABILITIES
      seam has since moved every image block out of it: the whole `image_generation` / `image_edit` /
      `canvas` / refusal vocabulary is written by `src/serve/image_capabilities.cpp` and reached by the
      single typed call `add_image_capabilities(payload, options, image_model_id)` at the end of
      `make_capabilities_json`. What remains here is the enumerated residue of the route surface, not a
      lane rule: the route registrations, `handle_image_artifact`, `image_model_id()`, the `images_`
      reads on /health, and the two lane includes. If the counted 39 lines move, re-derive them from
      this row before trusting a build - the checker prints every one of them.

### src/serve/http_server.h
    our path: src/serve/http_server.h
    upstream path: src/serve/http_server.h
    category: must-port
    hunks: 5
    anchors: attach, register_routes, api_route_pattern
    does: Lane header include, ImageGenerationService forward declaration, the public
      attach_images(ImageGenerationService&) seam, the private /v1/images handler declarations,
      and the images_ member. Upstream's only public seam is attach(GenerationBackend&) and
      register_routes() is private, so there is no module list to join - these stay edits to
      upstream's class.

### src/serve/http_transport.cpp
    our path: src/serve/http_transport.cpp
    upstream path: src/serve/http_transport.cpp
    category: must-port
    hunks: 4
    anchors: parse_json_body
    does: THREE things, and none of them is a compile error if lost.
      1. The lane header include at the top of the file (`serve/openai_images.h`, for
         `kImageArtifactRoutePrefix` in `artifact_url_base`). One line, and nothing else in the
         file hints at it, so it is the easiest seam to lose silently in a port - and it is exactly what the
         hunk selector dropped at 6a648048 (upstream had grown two more system includes in the block the
         hunk's context needed), leaving an undeclared identifier that only the build reports.
      2. The sink writes are serialized. Upstream tip has no lock here because at its base only the
         handler thread ever touches `sink_`; the lane writes it from the ticker's thread while a picture
         round blocks that handler, so `write()`, `write(vector)` and `poll()` each take `write_mutex_`
         and delegate to the private `write_locked()` (the fork's own copy takes the lock around the body
         inline instead - same contract, no `heartbeat_event_` to keep out of the lock). `write_locked()`
         RETURNS whether the sink refused the item, and `poll()` returns that value: upstream's own
         `tests/test_http_transport.cpp` ("failed SSE heartbeat did not cancel its request") is what fails
         if a replay makes it `void` and ends `poll()` with `return false` - a refusable, silent loss of
         mid-stream disconnect detection, since nothing else in the file reads the result. Losing the lock
         is not a crash in a test: it is two threads interleaving bytes into one SSE stream and a torn
         `last_write_`, which shows up as a client's parse error at some later load.
      3. `beat()` and the `HeartbeatTicker` definitions - the transport's quiet-interval comment without
         `poll()`'s writability probe, and the ticker thread that calls it. The fork's
         `src/serve/http_transport.cpp:189-287` is the authority for both; on the 6a648048 port the
         ticker's watchdog gained one more line, the `armed` gate
         (`if (progress->armed.load(...))` around the elapsed check), because the fork armed nothing and
         a clock stamped at job construction would have flagged the prompt enhancer's sidecar call, the
         weight upload and the text tower as wedged draws. See the three handler rows for who passes
         `watch`, and `serve/image_job_progress.h` for what `armed` means.
      What upstream tip adds and this file must KEEP untouched beside them: `set_heartbeat` and
      `heartbeat_event_`, which serve the protocol's own quiet-stream event from `poll()` - the lane's
      ticker covers the span where `poll()` cannot run, not the span where it does.

### src/serve/http_transport.h
    our path: src/serve/http_transport.h
    upstream path: src/serve/http_transport.h
    category: must-port
    hunks: 5
    anchors: parse_json_body, client_disconnected
    does: Besides `artifact_url_base`, this header carries the whole of the lane's SSE liveness contract -
      the five hunks whose loss no compiler reports. `#include "serve/image_job_progress.h"`,
      `<condition_variable>` and `<thread>` (the fork's lines 3-4, 11, 17); the comment on `write()` that
      says why a lock is there at all; `beat()` and `is_cancelled()` beside `poll()`; the private
      `write_locked()` and `std::mutex write_mutex_`; and the `HeartbeatTicker` class itself
      (`src/serve/http_transport.h:88-121`). The fork's copy is the authority for their text. The failure
      mode of losing them is the same shape as losing the .cpp's: the exe builds, every chat answer and
      every non-streaming draw works, and a streaming draw goes silent for its whole 60-330 s with the
      stall watchdog permanently false. Keep upstream tip's `set_heartbeat`/`heartbeat_event_` and the
      `kHeartbeatComment` doc as they are - the lane adds a second, thread-owned liveness path beside
      them, it does not replace upstream's.
      The declaration of `artifact_url_base(const httplib::Request&, const ServeOptions&)` - the one
      place every route that hands out a generated-image URL gets its prefix (so a chat answer and an
      images response name the same GET), between upstream's `parse_json_body` and `client_disconnected`.
      Ours-only in shape, so it is re-added rather than replayed. Not in the manifest until the 6a648048
      build reported `C3861: 'artifact_url_base': identifier not found` at `src/serve/openai_images_http.cpp:281`
      while the .cpp already defined it: the definition without the declaration is exactly what the fork's
      diff looks like once the header hunk is dropped.

### src/serve/openai_chat_http.cpp
    our path: src/serve/openai_chat_http.cpp
    upstream path: src/serve/openai_chat_http.cpp
    category: seam
    hunks: 1
    anchors: make_chat_image_hook, inject_chat_image_tool, chat_image_round_needed, run_chat_image_rounds, chat_image_missing_url_suffix, chat_image_progress_note, chat_image_watch_progress
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 23
    does: The TOOL_INJECTION seam on the Chat Completions route. The file keeps only its own request,
      its own `service_`, its own re-prepare/re-run callbacks and the two reads of its private state
      that the lane cannot make (`images_`, `run_image_job`), handed over as the typed
      `ChatImageDoors` inside the one `make_chat_image_hook` call. Everything the lane decides - that
      the `generate_image` tool is offered, when a round is owed, what the drawn picture's URL suffix
      says, what the SSE silence carries while a draw runs, and what a chat draw is filed under in the
      request log - lives in `src/serve/image_chat_hook.cpp`. Re-anchor the single hook call inside
      upstream's `HttpServer::handle_chat_completions` before `RequestLogMetadata` is built, the two
      round blocks where upstream's own tool-round machinery ends (aggregate and streaming), and the
      progress note where the streaming heartbeat is constructed; `route_label` stays the handler's own
      word "openai_chat". Note that the counted `lines:` includes upstream's own
      `new_openai_chat_tool_call_id()` - the count is a fingerprint of the seam, not a claim that every
      counted line is the lane's.
      Inside each round block the handler also constructs the lane's `HeartbeatTicker` (declared in
      `src/serve/http_transport.{h,cpp}`, whose row says why it lives there) with TWO lane doors: the
      progress note and - added on the 6a648048 port, where upstream's `set_heartbeat` turned out to
      cover only what `poll()` runs - `chat_image_watch_progress(*hook)` as the watchdog's `watch`. A
      port that drops the second argument leaves the transport compiling, the stream beating and the
      fail-closed stall check permanently false: the job would hold the interlude's VRAM forever with
      nobody observing it. `serve/image_job_progress.h` and `src/models/qwen_image/image_pipeline.cu`
      are what make the second argument mean something.

### src/serve/openai_responses_http.cpp
    our path: src/serve/openai_responses_http.cpp
    upstream path: src/serve/openai_responses_http.cpp
    category: seam
    hunks: 1
    anchors: make_chat_image_hook, inject_chat_image_tool, chat_image_round_needed, chat_image_replay_needed, run_chat_image_rounds, chat_image_missing_url_suffix, chat_image_progress_note, chat_image_watch_progress
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 35
    does: The same TOOL_INJECTION seam on the Responses route, with the handler's own word
      "openai_responses" as `route_label`. This file additionally asks `chat_image_replay_needed(hook)`
      in the two places it decides whether the conversation has to stay whole for the rounds that would
      re-render it (the hosted-tool-search copy and the `messages.clear()`), which is the only reason the
      lane is named outside the round blocks. NOT the lane, and not to be touched: `redact_input_image_urls`,
      the `include=message.input_image.image_url` query and `PromptProgress` - upstream's chat-vision
      input handling, which is why this row counts more lines than its siblings.
      Like its siblings, each round block also constructs the lane's `HeartbeatTicker` with the progress
      note AND `chat_image_watch_progress(*hook)` as the stall watchdog's `watch` - see the Chat
      Completions row for what dropping the second argument silently costs, and
      `src/serve/http_transport.{h,cpp}` for why the ticker lives there.

### src/serve/anthropic_messages_http.cpp
    our path: src/serve/anthropic_messages_http.cpp
    upstream path: src/serve/anthropic_messages_http.cpp
    category: seam
    hunks: 1
    anchors: make_chat_image_hook, inject_chat_image_tool, chat_image_round_needed, run_chat_image_rounds, chat_image_missing_url_suffix, chat_image_progress_note, chat_image_watch_progress
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 24
    does: The same TOOL_INJECTION seam on the Messages route, with "anthropic_messages" as
      `route_label`. Where the old code said "the hidden image call of the round before is no longer
      part of this one", the handler keeps its own call bookkeeping and the reason now reads in
      `src/serve/image_chat_hook.cpp`; the handler's remaining comment says only what the handler owns.
      Like its siblings, each round block also constructs the lane's `HeartbeatTicker` with the progress
      note AND `chat_image_watch_progress(*hook)` as the stall watchdog's `watch` - see the Chat
      Completions row for what dropping the second argument silently costs, and
      `src/serve/http_transport.{h,cpp}` for why the ticker lives there.

### src/serve/openai_chat_response.cpp
    our path: src/serve/openai_chat_response.cpp
    upstream path: src/serve/openai_chat_response.cpp
    category: seam
    hunks: 2
    anchors: add_chat_image_attempts, chat_image_tool_hidden
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 12
    does: The renderer half of the TOOL_INJECTION seam on Chat Completions: the field's name, its
      null rule and the hidden-call rule all live in `src/serve/image_chat_hook.cpp`. Note the shape
      change a replay must keep: the file-local `chunk()` and `usage_chunk()` helpers take
      `const GenerationOutcome* outcome = nullptr` where they once took a pre-built
      `Json image_tool_attempts`, and exactly one of the two carriers is handed the pointer per stream -
      the same single-emission rule as before, now asked of the lane. Two of the counted lines are
      upstream's own `new_openai_chat_tool_call_id()`. The second hunk is the OTHER half of the
      hidden-call rule, in `OpenAIChatStream::finish()`: a hidden streamed call the terminal result does
      not carry is dropped before upstream's positional check, because the picture rounds replace the
      round that streamed it with the prose answer (see `src/serve/chat_image_tool.cpp`) and the check
      would otherwise throw "streamed tool calls do not match terminal output" on every streamed draw.
      A replay that keeps only the `tool_call_delta` half leaves the streaming picture route broken --
      host test `tests/test_chat_image_tool.cpp`, case
      `test_chat_renderer_hides_the_server_run_call`. In that same function the suppression is asked of
      the CALL a delta belongs to (`calls_[i].name`), never of `delta.name`: only the opening delta
      carries a name, so a continuation or a closing delta of a hidden call would be published with its
      argument bytes otherwise -- which is exactly how this hunk first failed in the ported tree, and what
      `check(!published_a_call, ...)` in that test case catches. The counted line is that hunk's comment naming
      `serve/chat_image_tool.cpp`; the code around it carries no lane token and reads `calls_[i].hidden`
      from the first hunk's own bookkeeping.

### src/serve/openai_responses_response.cpp
    our path: src/serve/openai_responses_response.cpp
    upstream path: src/serve/openai_responses_response.cpp
    category: seam
    hunks: 2
    anchors: add_chat_image_attempts, chat_image_tool_hidden
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 6
    does: The same renderer seam on the Responses protocol: `built.body` (a plain `nlohmann::json`)
      gets the field through the lane, and the streamed call's hidden flag comes from
      `chat_image_tool_hidden(delta.name)`. Upstream's `OpenAIResponse` shape is where it re-attaches.
      The second hunk is that flag's other half, in `Impl::emit_calls()`: a hidden streamed call the
      round's result does not carry is dropped before upstream's positional match, which otherwise
      throws on every streamed draw once the picture rounds replace that round's answer with the prose
      one (see `src/serve/chat_image_tool.cpp`) -- host test `tests/test_chat_image_tool.cpp`, case
      `test_responses_renderer_hides_the_server_run_call`. It reads `calls[i].hidden` from the first
      hunk, so its only counted line is the comment naming `serve/chat_image_tool.cpp`.

### src/serve/anthropic_messages_response.cpp
    our path: src/serve/anthropic_messages_response.cpp
    upstream path: src/serve/anthropic_messages_response.cpp
    category: seam
    hunks: 2
    anchors: add_chat_image_attempts, chat_image_tool_hidden
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 8
    does: The same renderer seam on the Messages route. This is the one protocol whose body and
      terminal event are built with the file-local `OrderedJson` (`nlohmann::ordered_json`), so it
      reaches the lane through the `ordered_json` overload of `add_chat_image_attempts` while the other
      two use the `nlohmann::json` one; both overloads read the attempts through the sorted document
      first, which is what keeps one written attempt's keys in the same order on all three protocols.
      Note for a replay: this protocol's positional check is NOT in `finish()` but in
      `AnthropicMessagesStream::call_blocks()` ("streamed Anthropic tool calls do not match terminal
      output"), which matches `calls_` against the segment's calls by position and id. So the Messages
      route needs the hidden-call drop exactly where the other two renderers need theirs: in
      `call_blocks()`, before the match, a hidden streamed call the segment does not carry is dropped
      (the fork reads its own `calls_[i].hidden`, the port asks the lane by name). Without it a streamed
      picture request on `/v1/messages` throws there too, in this fork as well as on a newer tip -- the
      host case `test_anthropic_renderer_hides_the_server_run_call` is what found it, in the fork.
      Counted with the other two renderers' rule: `tests/test_chat_image_tool.cpp` covers all three.
      The same function's silent branch in `tool_call_delta()` must still record what it streamed,
      including `call.complete` from the delta that carries it: the positional match demands the call's own
      completeness, and a hidden call that stays booked but never completes fails that match on a round
      that mixes a client tool in. The fork's own copy of that branch did exactly that until 2026-11-26
      (it returned before `call.complete` was set), and the mixed segment of
      `test_anthropic_renderer_hides_the_server_run_call` is the check that says so.
      The two overloads are both load-bearing: dropping the `ordered_json` one does not compile (a
      non-const `nlohmann::json&` cannot bind to the temporary a conversion to the other specialization
      would produce), so the shape of the call is enforced by the compiler rather than by review.

### src/serve/serve_options.cpp
    our path: src/serve/serve_options.cpp
    upstream path: src/serve/serve_options.cpp
    category: seam
    hunks: 3
    anchors: image_option_arity(name), image_help_text(), parse_image_option(arg, options, require_value, parse_nonnegative_int)
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 7
    does: The OPTIONS seam. The four --image-* flags, their arity, their value checks, their help text
      and their diagnostics all live in `src/serve/image_serve_options.{h,cpp}`; this file keeps three
      typed calls - the arity question inside upstream's `serve_option_arities()`/`ServeOptionArity`
      table, the help concatenation, and the parser hook before upstream's own accept-list - and nothing
      else. Re-anchor those three calls in upstream's current parser; upstream accepts nothing outside
      its own table, so a flag that reaches neither the table nor the lane hook makes the server refuse
      to start (build-green, test-green, and the failure looks like a bad command line). One of the
      counted lines is upstream's own `--vision-offload` help text. The four ServeOptions members
      themselves stay upstream-owned (see `src/serve/serve_options.h`).

### src/serve/serve_options.h
    our path: src/serve/serve_options.h
    upstream path: src/serve/serve_options.h
    category: must-port
    hunks: 2
    anchors: serve_option_arities
    does: The four ServeOptions fields the flags populate (image artifact, prompt-enhancer URL,
      retention days, text processor).

### src/serve/request_log.h
    our path: src/serve/request_log.h
    upstream path: src/serve/request_log.h
    category: must-port
    hunks: 3
    anchors: ServerLogEnvironment, format_request_done_json
    does: Adds our own `ImageJobLogRecord` (source, protocol, correlation, asked-vs-drawn canvas), the
      free `format_image_job_json` declaration beside upstream's other `format_*_json`, and
      `JsonlRequestLog::write_image_job` beside the other writers. The third hunk is the
      `#include <vector>` our copy of this header carries and upstream tip's does not:
      `ImageJobLogRecord` holds `std::vector<std::string> image_bytes`, so without it the record builds
      only while some unrelated header happens to pull `<vector>` in, and the first such change turns a
      lane port into a `C2039`/`C2065` in an upstream-owned file. Not the "observability only" row this
      manifest used to claim: the .cpp row's definitions and the two lane callers need these declarations,
      so losing them is a compile error (defining an undeclared member) or, if only the .cpp half is lost,
      `fatal error LNK1120: 1 unresolved externals` on `write_image_job` - which is exactly what the
      6a648048 port hit, because no row covered the .cpp at all. The struct is ours and lives in an
      upstream-owned file.

### src/serve/request_log.cpp
    our path: src/serve/request_log.cpp
    upstream path: src/serve/request_log.cpp
    category: must-port
    hunks: 2
    anchors: event_base, write_throughput
    does: The two definitions behind the header's image declarations: `format_image_job_json` (fork
      request_log.cpp:953, built on upstream's own `event_base`, its anchor for that hunk) and
      `JsonlRequestLog::write_image_job` (fork :1083, between upstream's `write_request_error` and
      `write_throughput`, which is its anchor). Both are pure re-adds of ours-only code into an
      upstream-owned file, and neither can be spotted by reading the file: the port that lost them still
      compiled every translation unit and died only at the link of `infernix-serve.exe`.

### src/serve/openai_common.cpp
    our path: src/serve/openai_common.cpp
    upstream path: src/serve/openai_common.cpp
    category: must-port
    hunks: 3
    anchors: make_models_list, make_model_object, resolve_openai_model
    does: The image model's OpenAI model object. `make_image_model_object_json` (with the comment
      explaining that the pipeline is a routeable resource on an OpenAI model object, not a claim that
      the chat model emits images), the `if (!image_model_id.empty())` entry appended to
      `make_models_list`'s array, and the public `make_image_model_object` wrapper. CORRECTION from the
      6a648048 port: this row used to say "comment only, loseable" and it is not - without the push_back
      `/v1/models` answers with only the chat model and `GET /v1/models/<image id>` has no answer at all,
      which the joint-limits and route tests both read. Upstream has since grown a second top-level
      `{"models", ...}` key beside `data` on the same array, so the lane's entry has to be added before
      BOTH keys are built (a naive re-add that copies the fork's `Json data` shape drops `models`).
      The third hunk is `validate_openai_model(std::string_view, std::initializer_list<std::string_view>)`
      (fork `openai_common.cpp:297`), the strict sibling that sits beside upstream's `resolve_openai_model`
      (its anchor - upstream has no such function of its own, so an anchor naming it would only ever
      report the lane's re-add): `POST /v1/images/generations` and `/v1/images/edits` call it with
      `{public_model_id_, image_model_id()}`, and without it neither public id is accepted. The 6a648048
      build found it as `C3861: 'validate_openai_model': identifier not found` at
      `src/serve/openai_images_http.cpp:240`.

### src/serve/openai_common.h
    our path: src/serve/openai_common.h
    upstream path: src/serve/openai_common.h
    category: must-port
    hunks: 2
    anchors: make_models_list, make_model_object, resolve_openai_model
    does: The declarations the .cpp row implements: `make_models_list`'s trailing
      `std::string_view image_model_id = {}` parameter (with the comment saying why the image entry is a
      bare model object), `make_image_model_object(std::string_view, std::int64_t)`, and
      `validate_openai_model(std::string_view, std::initializer_list<std::string_view>)` beside upstream's
      `resolve_openai_model` (its anchor; the include block also needs `<initializer_list>`). Missed by the
      6a648048 port's manifest, which had no row for this header at all: the lane's `http_server.cpp`
      call sites then fail to compile against upstream's five-parameter form.

### src/serve/generation_service.h
    our path: src/serve/generation_service.h
    upstream path: src/serve/generation_service.h
    category: must-port
    hunks: 2
    anchors: GenerationOutcome, prefix_miss_diagnostic_failed, is_available
    does: Two hunks. (1) `[[nodiscard]] infernix::Engine& engine() noexcept` on `GenerationService`, directly
      after upstream's `is_available()` (its anchor) - the only handle the serving startup has on the loaded
      Engine, which `image_startup.cpp` needs for `make_engine_interlude_runner(service.engine())` because
      the interlude is Engine-owned, not protocol-owned. Upstream's rewrite of this class keeps `engine_`
      private behind the `GenerationBackend` accessors, so this is ours-only in shape and the 6a648048 build
      found it as `C2039: 'engine': is not a member of 'infernix::serve::GenerationService'`.
      (2) `std::optional<std::string> image_tool_attempts_json` on `GenerationOutcome`, at the end of the
      struct beside `prefix_miss`. It is the only object that carries what the hidden chat tool did per
      picture to all three protocol writers in both their streaming and aggregate forms, so a client's
      `image_tool_attempts` (and the refused-vs-failed distinction the product contract promises) does not
      exist without it. Carried pre-serialized because this header owns no JSON type. A row this
      manifest lacked before the 6a648048 port, where the field had to be re-added by hand.

## Public API and protocol headers

### include/infernix/engine.h
    our path: include/infernix/engine.h
    upstream path: include/infernix/engine.h
    category: must-port
    hunks: 3
    anchors: class Engine, EngineOptions
    does: The public lease API on Engine: supports_image_interlude, run_image_interlude,
      probe_image_yield. Purely additive into upstream's class, but it must fit upstream's
      current shape, which now dispatches over four variant cores.

### include/infernix/types.h
    our path: include/infernix/types.h
    upstream path: include/infernix/types.h
    category: must-port
    hunks: 2
    anchors: EngineOptions
    does: Adds ImageYieldProbe and ImageInterludeContext, including
      expert_pool_frozen_by_lease and begin_image_quiesce. Today these leak chat residency
      state into the highest-churn public header; WP-D moves them to include/infernix/image/.
      Favorable upstream change already banked: GenerationRequest/GenerationOutcome/ChatTool
      left this header for src/serve/, so our diff here is smaller than the draft assumed.

## Engine runtime and the VRAM lease (the hard seams)

### src/runtime/engine/engine.cpp
    our path: src/runtime/engine/engine.cpp
    upstream path: src/runtime/engine/engine.cpp
    category: must-port
    hunks: 1
    anchors: GenerationCore, construct_qwen4_exp
    does: Forwards the lease API to the chat Program via impl_->qwen4->program. Upstream refactored
      this file into std::variant GenerationCore/ScoringCore construction and has since added
      Inkling2Loop as a fourth core, so the hardcoded variant member access has to be rewritten
      against the current dispatch, not re-applied.

### src/runtime/engine/engine_core.h
    our path: src/runtime/engine/engine_core.h
    upstream path: src/runtime/engine/engine_core.h
    category: must-port
    hunks: 3
    anchors: worker_loop, set_maintenance_waker
    does: The worker-ack seam: the image lane needs the LLM worker to quiesce, hand back
      in-flight work and be woken again. Our names request_interlude_preempt, wake_llm_worker,
      drain_llm_frontier and interlude_preempt_outstanding **never existed upstream** - plan 2's
      assumption that they "already live" here is false. Re-anchor onto upstream's real worker
      entry points (worker_loop, the maintenance waker, the preemption victim path) and keep the
      hook narrow and image-agnostic.

### src/models/qwen4_exp/program/program_impl.h
    our path: src/models/qwen4_exp/program/program_impl.h
    upstream path: src/models/qwen4_exp/program/program_impl.h
    category: must-port
    hunks: 23
    anchors: residency_, vram_mutex_, kv_vmm_
    does: The entire image interlude/lease body: probe_image_yield, yield_for_image,
      release_from_yield, rollback_failed_yield, begin_quiesce, begin_image_quiesce, the
      interlude_mu_ / interlude_quiesced_epoch_ track. It reads and mutates the chat Program's
      private residency bookkeeping directly. This is the single highest-churn upstream file and
      the reason a port is not mechanical. **Known-broken on re-derive, listed here so nobody is
      surprised:** our body also uses execution_mu_ (0 hits upstream - removed), and upstream's
      perf(qwen4_exp) "every expert in RAM" branches now deliberately keep the expert pool
      resident, so freezing it is a policy override rather than a neutral hand-back.

### src/models/qwen4_exp/program/program.h
    our path: src/models/qwen4_exp/program/program.h
    upstream path: src/models/qwen4_exp/program/program.h
    category: must-port
    hunks: 5
    anchors: class Program
    does: Declares the lease entry points and the quiesce/release hooks on Program, plus the
      image stream member. Tied to program_impl.h; moves when the lease facade does.

### src/models/qwen4_exp/program/program.cpp
    our path: src/models/qwen4_exp/program/program.cpp
    upstream path: src/models/qwen4_exp/program/program.cpp
    category: must-port
    hunks: 5
    anchors: set_maintenance_waker
    does: Defines the lease entry points, creates the image CUDA stream (including the
      cudaSetDevice fix that makes it land on the right device), and forwards quiesce/release.

### src/runtime/expert_offload/expert_residency.h
    our path: src/runtime/expert_offload/expert_residency.h
    upstream path: src/runtime/expert_offload/expert_residency.h
    category: must-port
    hunks: 1
    anchors: class ExpertResidency, lendable_bytes
    does: Declares `any_lease_out()`, which the .cpp row's scan implements and the lease's
      `interlude_yieldable()` asks. It is the only honest answer to "is any frame on loan right now":
      `lendable_bytes()` alone cannot say it, because a pool already at its never-lent quarter reads the
      same and is a legitimate answer. This row was missing when the 6a648048 port ran, so the header
      edit had to be derived by hand while the .cpp definition came across with the hunk - the pair fails
      to compile if only one half lands.

### src/models/qwen4_exp/program/elastic_kv.cpp
    our path: src/models/qwen4_exp/program/elastic_kv.cpp
    upstream path: src/models/qwen4_exp/program/elastic_kv.cpp
    category: port-if-missing
    hunks: 1
    anchors: map_kv
    does: Consumes the lane's injected KV remap fault inside `map_kv` (`take_kv_remap_fault() &&
      need.size() > 1 ? need.size() - 1 : need.size()`, returning `chunks == need.size()`), which is what
      makes the lease's "the remap fell short" rollback branch reachable from a test. In our tree `map_kv`
      sits inline in `program_impl.h`; upstream moved it here, so the fork's hunk never covers it and the
      fault stays declared with nothing consuming it. Production behaviour is unchanged either way - lose
      this and the injected-failure test asserts on a fault that cannot fire.

### src/runtime/expert_offload/expert_residency.cpp
    our path: src/runtime/expert_offload/expert_residency.cpp
    upstream path: src/runtime/expert_offload/expert_residency.cpp
    category: port-if-missing
    hunks: 1
    anchors: loaned_frames
    does: A comment in the loaned-frames scan explaining why a frozen pool is a reason and not a
      verdict, tied to ImageYieldProbe::expert_pool_frozen_by_lease. Readability only.

## Media decoding (the reference image's PNG reader)

### src/media/decode/png_decode.h
    our path: src/media/decode/png_decode.h
    upstream path: src/media/decode/png_decode.h
    category: must-port
    hunks: 1
    anchors: decode_png, is_png
    does: Declares the lane's RGBA reader: `struct RgbaImage` (width, height, 4 bytes per pixel) and
      `decode_png_rgba(span, max_decoded_pixels)`, which `src/serve/reference_image.cpp` calls for every
      PNG reference picture. Alpha is a real fourth channel here, never composited, because the
      image-VAE reference path consumes it directly - unlike upstream's `decode_png`, which drops it.
      The file is upstream-owned and had no row before the 6a648048 port, where both declarations had to
      be appended by hand below upstream's own `decode_png`.

### src/media/decode/png_decode.cpp
    our path: src/media/decode/png_decode.cpp
    upstream path: src/media/decode/png_decode.cpp
    category: must-port
    hunks: 1
    anchors: decode_png
    does: The RGBA8 implementation the header row declares, on the same native-PNG/zlib path as
      `decode_png` (WIN32 only, where FFmpeg's vcpkg build has no png decoder). PORT NOTE from 6a648048:
      upstream rewrote this file's chunk walker (it added a `Transparency` struct and a `keep_alpha`
      argument that returns the source's own alpha plane), so replaying the fork's copy of the decoder
      would have duplicated upstream's work. The lane's reader was instead landed as the ours-only file
      `src/media/decode/png_decode_rgba.cpp`, a thin adapter over upstream's
      `decode_png(bytes, budget, /*keep_alpha=*/true)` that interleaves the samples and fills 255 where
      the source declares no transparency. Re-derive it that way against any upstream that keeps
      `keep_alpha`; if upstream ever drops it, the fork's walker is the fallback.

## Tokenizer

### src/models/qwen3_5/frontend/tokenizer.cpp
    our path: src/models/qwen3_5/frontend/tokenizer.cpp
    upstream path: src/text/tokenizer.cpp
    category: must-port
    hunks: 2
    anchors: validate_pipeline, kInklingSplitPattern
    does: The lane's half of the Qwen tokenizer's pre-tokenizer gate. The image text tower encodes the
      rendered prompt with the Qwen3-VL processor's OWN `tokenizer.json` (the directory
      `--image-text-processor-path` points at, and `INFERNIX_QWENVL_PROCESSOR` for the host suites), and
      that descriptor declares the mark-EXCLUSIVE Split pattern - `[\p{L}\p{M}]+` replaced by `\p{L}+`
      and `[^\s\p{L}\p{M}\p{N}]+` replaced by `[^\s\p{L}\p{N}]+` - beside the mark-inclusive one this
      scanner implements. Two hunks: the `kQwenSplitPatternVL` constant with the comment that proves the
      two are byte-identical for this vocab (no merge token carries a combining mark: 0 of 151669
      tokens), and the `require` that accepts either for the Qwen pipeline. They are BEHAVIORALLY
      IDENTICAL, so nothing on the encode path changes - but refusing the descriptor is a
      `std::invalid_argument` out of the tokenizer constructor, which every image job hits before the
      DiT sees a byte, and `tests/models/qwen_image/test_image_tower_tokenize.cpp` asserts that
      construction succeeds as its own gate.
      UPSTREAM MOVED THIS FILE: at `6a648048` the tokenizer left `src/models/qwen3_5/frontend/` for
      `src/text/tokenizer.cpp` (`infernix::text::Tokenizer`/`TokenizerResources`/`TokenizerPipeline`),
      so re-derive the two hunks against the new file, keep the lane's gate off Inkling's pipeline
      (`qwen &&` in front of the VL alternative), and move the `#include` in
      `src/serve/image_text_tower.cpp` and in the two host suites with it - the fork reached the class
      through `models/qwen3_5/frontend/tokenizer.h` (via `chat_template.h`), which no longer exists.
      Forgetting the include is a `C2039`/`C3861` on `Tokenizer`/`TokenizerResources`; forgetting the
      hunks is not a compile error at all: the exe builds, every chat answer works, and the failure
      only surfaces where the lane first builds the tower (the providers `src/serve/image_startup.cpp:45-49`
      make lazily for a draw), where the route's own failure mapping reports it. The host suite is what
      catches it: `infernix_image_tower_tokenize_test` and `infernix_image_edit_template_test` both
      construct the real tokenizer from the processor directory and take a non-zero exit when it throws.

## Application entry

### apps/serve/main.cpp
    our path: apps/serve/main.cpp
    upstream path: apps/serve/main.cpp
    category: seam
    hunks: 1
    anchors: start_image_lane
    residue: image, Image, prompt_enhance, attach_images, run_image_job, generate_image, kChatImageToolName, chat_tool, ChatImage
    lines: 5
    does: The MAIN_WIRING seam. What used to be a contiguous builder block here (the job store, the
      enhancer handle, the generation service with its eight artifact-reading providers,
      server.attach_images and the retention sweep) is now `src/serve/image_startup.{h,cpp}`, so this
      file carries one include, one comment and one typed call:
      `auto image_lane = infernix::serve::start_image_lane(server, service, options, logger);`.
      Re-anchor it exactly where the old block sat - after upstream's `server.attach(service)` and its
      warmup, and before the `ServingScope` that arms the stop path - because the returned handle owns
      the shutdown contract (sweep stops, then the service, then the store) and must outlive
      `listen()`. Upstream's own `server.attach` stays the anchor for the position; the lane never
      touches it.

## Build files

### src/serve/CMakeLists.txt
    our path: src/serve/CMakeLists.txt
    upstream path: src/serve/CMakeLists.txt
    category: seam
    hunks: 1
    anchors: image_lane.sources.cmake
    residue: image_lane
    lines: 2
    does: The BUILD seam for the serve library. The twelve lane translation units and the three
      libraries only the lane needs (`infernix_qwen_image`, `infernix_qwen_image_runtime`,
      `infernix_media_encode`) moved to the ours-only `src/serve/image_lane.sources.cmake`, which this
      file reaches with one appended `include("${CMAKE_CURRENT_SOURCE_DIR}/image_lane.sources.cmake")`
      after its own `target_link_libraries` - the same idiom upstream already uses for its split
      `linear/tests.cmake` lists. Re-add just that line; the lane's own file is a pure re-add and no
      lane entry should ever appear inside `add_library(infernix_serve ...)` again.

### src/models/CMakeLists.txt
    our path: src/models/CMakeLists.txt
    upstream path: src/models/CMakeLists.txt
    category: must-port
    hunks: 1
    anchors: infernix_model_loading
    does: One include(qwen_image/sources.cmake) line for the qwen_image library. Upstream grew an
      inkling2/sources.cmake line at the same spot, so the insertion point moved.

### src/ops/CMakeLists.txt
    our path: src/ops/CMakeLists.txt
    upstream path: src/ops/CMakeLists.txt
    category: must-port
    hunks: 2
    anchors: infernix_ops
    does: Four include(<dir>/sources.cmake) lines for the ours-only ops attn_dit_joint, dit_adaln,
      rope_c64, vae_decoder. Upstream has none of those dirs and adds new op dirs at the same
      rate, so the block must be re-added against upstream's current include list.

### src/media/CMakeLists.txt
    our path: src/media/CMakeLists.txt
    upstream path: src/media/CMakeLists.txt
    category: must-port
    hunks: 1
    anchors: infernix_media_pdf
    does: Defines the ours-only infernix_media_encode target consumed only by the lane and its
      tests. Upstream put infernix_media_pdf at that exact spot, so a naive re-add collides.
      Target absent upstream (0 hits) - the definition is ours-only, the file is upstream-owned.
      On the 6a648048 port everything lane here is APPENDED below upstream's own targets (never edited
      into them): the `infernix_media_encode` block, and one WIN32-gated
      `target_sources(infernix_media_decode PRIVATE decode/png_decode_rgba.cpp)` that lands the lane's
      RGBA reader (see the `src/media/decode/png_decode.cpp` row) inside upstream's decoder target
      wherever that target carries the native PNG path.

### tests/CMakeLists.txt
    our path: tests/CMakeLists.txt
    upstream path: tests/CMakeLists.txt
    category: must-port
    hunks: 1
    anchors: infernix_stage_test_runtime_dlls, cmake/ProductTests.cmake
    does: Exactly one added line, `include(models/qwen_image/tests.cmake)`, between upstream's
      `include(models/qwen4_exp/tests.cmake)` and `include(ops/tests.cmake)` (confirmed by
      `git diff f9c2e42c -- tests/CMakeLists.txt`, whose whole ours-only side is that one line).
      Cheapest seam in the set. Upstream's file has no `infernix_add_test` call at all, so the
      anchors here are the include block and the trailing `infernix_stage_test_runtime_dlls`
      call - both stay, and upstream also grew `include(models/inkling2/tests.cmake)` at the same
      spot, which ours lacks (fork drift, not a lane seam).

### tests/cmake/ProductTests.cmake
    our path: tests/cmake/ProductTests.cmake
    upstream path: tests/cmake/ProductTests.cmake
    category: seam
    hunks: 1
    anchors: image_lane_tests.cmake
    residue: image_lane
    lines: 2
    does: The BUILD seam for the product suites. The seven lane test targets (the job store, the two
      route schemas, the service, the prompt enhancer's contract, the device-need ledger, the chat
      image tool and the PNG writer) moved verbatim to the ours-only
      `tests/cmake/image_lane_tests.cmake`, reached by one appended
      `include("${CMAKE_CURRENT_LIST_DIR}/image_lane_tests.cmake")`. Re-add that line only - the suite
      list itself never belongs in this file again, and `${CMAKE_CURRENT_LIST_DIR}` still resolves to
      `tests/cmake` because the lane file lives beside it.

### tests/models/qwen4_exp/tests.cmake
    our path: tests/models/qwen4_exp/tests.cmake
    upstream path: tests/models/qwen4_exp/tests.cmake
    category: seam
    hunks: 1
    anchors: image_lane_tests.cmake
    residue: image_lane
    lines: 2
    does: The BUILD seam for the LLM<->image coexistence suites. All seven (the yield, the interlude
      op, the DiT forward, the full pipeline, the production route, the failure machine and the
      admission-queue budget), with their `LIBRARIES infernix_qwen_image_runtime` links and their
      `SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600` properties, moved verbatim
      to the ours-only `tests/models/qwen4_exp/image_lane_tests.cmake`, reached by one
      `include("${CMAKE_CURRENT_LIST_DIR}/image_lane_tests.cmake")` at the place they used to sit.
      The remaining `image` line in this file is upstream's own
      `test_prefix_state_image.cpp` (chat-vision prefix cache) - upstream-owned, do not port our copy
      over it.

### tests/ops/tests.cmake
    our path: tests/ops/tests.cmake
    upstream path: tests/ops/tests.cmake
    category: seam
    hunks: 1
    anchors: image_lane.tests.cmake
    residue: image_lane
    lines: 2
    does: The BUILD seam for the DiT op suites. The four with the FP64/complex oracles -
      `infernix_rope_c64_test`, `infernix_attn_dit_joint_test`, `infernix_dit_adaln_test` (each via
      upstream's `infernix_add_op_test`, defined at tests/cmake/InfernixTests.cmake:32) and
      `infernix_image_joint_limits_test` (via `infernix_add_test`, linking `infernix_ops` and
      `infernix_serve`, host-only by design so it must never skip) - moved verbatim to the ours-only
      `tests/ops/image_lane.tests.cmake`, reached by one
      `include("${CMAKE_CURRENT_LIST_DIR}/image_lane.tests.cmake")` where they sat, before
      upstream's `infernix_finish_op_families()` has run for anything of ours. Re-add that line only.
      The remaining ours-only lines in this file (the INFERNIX_TMP_* entries and the attention oracle
      scheduling) are fork drift, not lane seams - do not port them back.

## Lane tests inside upstream-owned test files

### tests/test_http_routes.cpp
    our path: tests/test_http_routes.cpp
    upstream path: tests/test_http_routes.cpp
    category: must-port
    hunks: 17
    anchors: api_route_pattern
    does: The external-contract pins for the lane: the /v1/images/* routes registered through
      api_route_pattern including the /v1/v1/ double-prefix and bare-path 404 assertions, and the
      full /v1/capabilities JSON including every image_generation field. This is the only pin of
      the capabilities shape in either tree - upstream has no capabilities test at all, so nothing
      upstream fails if the block drifts. Keep it green.

### tests/test_request_log.cpp
    our path: tests/test_request_log.cpp
    upstream path: tests/test_request_log.cpp
    category: port-if-missing
    hunks: 2
    anchors: format_request_done_json, RequestLog
    does: Covers ImageJobLogRecord rendering (asked-vs-drawn canvas, source, protocol).

### tests/models/qwen4_exp/test_memory_plan.cpp
    our path: tests/models/qwen4_exp/test_memory_plan.cpp
    upstream path: tests/models/qwen4_exp/test_memory_plan.cpp
    category: port-if-missing
    hunks: 1
    anchors: test_expert_cap, HostMemoryDemand
    does: Pins the premise of expert_pool_frozen_by_lease: while the LLM holds a residency lease
      the frame pool is kept whole, so the plan's promise is elastic KV. It is the test that makes
      the freeze reason bit falsifiable; losing it lets the bit be set by a plan that freed
      nothing.

### tests/ops/linear/test_fp8_a16.cpp
    our path: tests/ops/linear/test_fp8_a16.cpp
    upstream path: tests/ops/linear/test_fp8_a16.cpp
    category: port-if-missing
    hunks: 2
    anchors: run_flash_next_fp8_a16
    does: Adds run_qwen_image_fp8_a16() - the lane's FP8 A16 route qualification against the FP64
      oracle at the templated-prompt T32/T64 boundary - and folds it into the runner sum.

---

## Known-absent upstream symbols (re-derive, do not treat as anchors)

Confirmed 0 hits at `f9c2e42c`. These are ours-only names our hunks call. They can never be
checker anchors; each is a re-derivation the port must do by hand:

| Name | Where our code uses it | Status upstream |
|---|---|---|
| `execution_mu_` | `program_impl.h` lease body | **removed** - must map onto upstream's current lock (only `vram_mutex_` remains at :7307) |
| `request_interlude_preempt`, `wake_llm_worker`, `drain_llm_frontier`, `interlude_preempt_outstanding` | `engine_core.h`, `program.{h,cpp}` | never existed - re-anchor to `worker_loop` / `set_maintenance_waker` / the preemption victim path |
| `supports_image_interlude`, `run_image_interlude`, `probe_image_yield`, `ImageYieldProbe`, `ImageInterludeContext`, `begin_image_quiesce`, `expert_pool_frozen_by_lease` | `engine.h`, `types.h`, the whole lease | ours-only |
| `make_capabilities_json`, `/v1/capabilities`, `/v1/images`, `handle_image_artifact`, `openai_images` | `http_server.*` | ours-only - upstream's only "capabilities" text is the unrelated `ChatTemplateCaps` |
| `infernix_media_encode`, `attn_dit_joint`, `dit_adaln`, `rope_c64`, `vae_decoder`, `infernix_qwen_image*` | all CMake seams | ours-only targets |
| `ImageJobLogRecord` | `request_log.h` | ours-only |
| `--image-artifact`, `--image-prompt-enhancer-url`, `--image-retention-days`, `--image-text-processor` | `serve_options.*` | ours-only, no upstream `--image-*` at all |

Hostile-by-design upstream changes a porter must plan against, not re-anchor around: the
experts-in-RAM keep policy (`a3f78ef8`, `18388f5b`) and the NVML-capped free-VRAM contract
(`b821d961`, `VramSnapshot::device_free` capped at `physical_free`). Both are in the assessment
document with commit evidence.

## What to do if a must-port anchor is missing after an upstream merge

Run the checker; it exits 1 and prints `<path> <anchor> MISSING` for each failure. Then, per row:

1. **Read upstream's file, do not guess a replacement name.** `git show <ref>:<path>` and find what
   the anchor became. `git log <ref> --oneline -- <path>` names the commit that moved it, and its
   message usually says why. A rename is a 5-minute fix; a removal is not.
2. **Classify the failure.** Renamed or moved -> update the `anchors:` value in this manifest to the
   real upstream name and re-run until it passes. Removed outright -> the hunk is no longer a
   re-anchor, it is a re-derivation: file it with the `program_impl.h` class and budget it as days,
   not minutes. Never leave a stale anchor in the table to make the checker go green.
3. **Never downgrade a `must-port` to `port-if-missing` to silence the checker.** That is the one
   edit that turns a real port break into a shipping bug. If a row genuinely is only a convenience,
   say so in the commit message and in this file, with the reason.
4. **Do not add markers to upstream-owned files** (`INFERNIX-IMAGE:BEGIN` and friends). Upstream has
   no such convention (confirmed: zero hits at `f9c2e42c`), the ~220 comment lines would themselves
   drift, and upstream would ask for them back in any PR. This manifest is the marker discipline.
5. **Prefer upstreaming the seam.** For each missing anchor, ask whether the hook could be generic
   and image-agnostic. Land it as an upstream PR and the row becomes `ours-only` with zero diff -
   that is the only end state that shrinks this table permanently.
6. **Then run the dry-run harness** (`tools/image_upstream_dryrun.ps1 -Ref <sha>`) to confirm the
   foreign tree still configures and the host targets still build before committing to the port, and
   check `docs/research/2026-11-24-image-edit-port-assessment.md` for whether the advance also moved
   the residency/VRAM contract underneath the lease.
