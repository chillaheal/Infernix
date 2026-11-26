# Image lane: cheap port preparation, evaluated against real code in both trees

Date: 2026-11-24. Our branch `image-generation`, HEAD `9bc42a7ff1351d315b542fe797cd1f170a80e07c`.
Upstream `origin/main` = `f9c2e42cb3e987d6159ed0de7afd2205a806ffd6` (2026-11-24 12:51:50 +0000), read only
through `git show f9c2e42c:<path>` / `git ls-tree -r --name-only f9c2e42c` / `git diff HEAD f9c2e42c`.
Unrelated histories confirmed this session: `git merge-base HEAD f9c2e42c` -> NO MERGE BASE.

READ-ONLY. No build, no configure, no edit of tracked code, `image-edit/` untouched, the live
`http://127.0.0.1:8888` server untouched. This file is the only artifact written and it is not committed.

Scope set by the operator: "what more can we do NOW so that when a new Infernix ships, porting image edit
is a small job ... No giant jobs now." Plan 2's Task 0 (WP-A flag + WP-B `ImageDeviceLease` facade +
WP-C worker-ack anchors + WP-D types out of `types.h`) is the big refactor and is out of scope here.
Companion background: `docs/research/2026-11-24-image-edit-port-assessment.md`.

## Measured surface (the numbers this audit stands on)

All measured this session, not carried over.

| Measurement | Command shape | Result |
|---|---|---|
| Files in our tree / upstream tree | `git ls-tree -r --name-only` | ours 3060, upstream 3385 |
| Ours-only files (no upstream counterpart) | `comm -23` of the two name lists | 377 total; by top dir: 179 `image-spike`, 80 `tests`, 79 `src`, 25 `docs`, 5 `include`, 4 `tools`, 2 `bench`, 1 `apps` |
| Files present in BOTH but differing | `git diff --diff-filter=M --name-only HEAD f9c2e42c` | 1143 (931 code under src/include/apps/tests/cmake) |
| Upstream-owned shared files that carry lane hunks | intersection of the above with `git grep -l <lane tokens> HEAD -- src include apps tests cmake` | **31 files** (listed in candidate C below) |
| Distinct contiguous lane regions inside those 31 files | line-cluster scan with the full lane vocabulary (`attach_images`, `qwen_image`, `interlude`, `quiesce`, `yield_for_image`, `images_`, `handle_images`, ...) | **~111 regions** |
| Plus lane-adjacent upstream-owned file missed by the token scan | `src/media/CMakeLists.txt` | 1 more file, ~2 regions (defines our ours-only `infernix_media_encode` target at `:24-33` where upstream added `infernix_media_pdf`) |
| Total upstream-owned files a port must touch | 31 + `src/media/CMakeLists.txt` | **32 files, ~113 hunks** |
| Total divergence hunks in those files (HEAD vs upstream, both directions) | `git diff HEAD f9c2e42c -- <f> \| grep -c '^@@'` | ~600, concentrated in `program_impl.h` 227, `types.h` 57, `serve_options.cpp` 47 |

Correction to the companion assessment's risk table, row 6, found while measuring: `/v1/capabilities` and
`make_capabilities_json` are NOT upstream anchors that "survive". They never existed upstream at all.
`git grep -n "make_capabilities_json\|/v1/capabilities" f9c2e42c -- src` returns nothing, and the same
grep at the stale baseline `0a3f432b` returns nothing. Ours is the only home: `src/serve/http_server.cpp:593`
(route) and `:1061` (`{"object", "infernix.capabilities"}`). Upstream's only "capabilities" text is the
unrelated `ChatTemplateCaps` at `src/serve/openai_common.h:37-41`. Consequence: the capabilities block is a
pure re-add (cheap), but nothing upstream pins it, so upstream may change or remove the surrounding
`register_routes()` and `/health` fields with no upstream test failing.

## Verdict table

| # | Candidate | Verdict | Files it touches | Diff size (honest) | Risk | Why |
|---|---|---|---|---|---|---|
| A | `option(INFERNIX_IMAGE)` + conditional wiring, no facade | **DROP as "cheap"; the CMake half survives as A-prime** | root `CMakeLists.txt`, `src/models/CMakeLists.txt:22`, `src/ops/CMakeLists.txt:17,19,42,45`, `src/serve/CMakeLists.txt:20-27,47-48`, `src/media/CMakeLists.txt:24-33`, `tests/CMakeLists.txt:11`, `tests/cmake/ProductTests.cmake:7-9,58-103`, `tests/models/qwen4_exp/tests.cmake:108-131`, `tests/ops/tests.cmake`, `tests/models/qwen_image/tests.cmake`, `tests/test_http_routes.cpp`, `tests/test_serve_options.cpp`, `apps/serve/main.cpp:7-9,345-391`, `src/serve/http_server.{h,cpp}`, `src/serve/http_transport.cpp:3`, `src/serve/{openai_chat_http,openai_responses_http,anthropic_messages_http}.cpp`, `src/serve/{openai_chat_response,openai_responses_response,anthropic_messages_response}.cpp:4` | ~24 files, ~70 guard sites, ~150-200 added lines | MEDIUM-HIGH | OFF is NOT buildable today (every blocker named below), and OFF is unreachable by CMake-only gating, so the flag necessarily costs `#ifdef` hunks inside the six protocol handlers -- files that today carry only 1-6 lane regions each. It buys CI/build isolation, not portability, and it makes the worst seams worse. |
| A' | Lane source manifests: move each lane source list into an ours-only `*.sources.cmake`, leave one `include()` line per upstream CMake file | **DO-NOW** | ours-only new: `src/serve/image_sources.cmake`, `src/ops/image_ops_sources.cmake`, `src/models/qwen_image/` already has `sources.cmake`, `src/media/image_encode_sources.cmake`, `tests/cmake/ImageTests.cmake`; one-line edits in the 5 files A already owns | 5 new ours-only files (~30 lines total), 5 one-line edits in files that are already seams | LOW | No behaviour change, no `#ifdef`, no new upstream file enters the seam set, and it converts the CMake half of a future port from a merge into a copy. It is also the artifact that makes A or D cheap later. |
| B | Relocate ours-only lane files under `plugins/image/` | **DROP** | `git mv` of ~157 ours-only files, `src/{serve,models,ops,media}/CMakeLists.txt`, and a NEW hunk in upstream-owned root `CMakeLists.txt` (`add_subdirectory(plugins)`) | 157 file moves + ~10 CMake edits | MEDIUM, and regret-prone | Include paths survive only by adding a second include root (`cmake/InfernixTargets.cmake:3-7` is upstream-owned; the per-target `target_include_directories` escape avoids it), but the move ADDS an upstream seam that does not exist today, leaves `src/serve/` holding chat files and `plugins/image/serve/` holding lane files with the same headers, and buys nothing that A'+C-manifest does not already buy. A 157-file `git mv` also destroys the blame a porter reads. |
| C | Marker discipline (`INFERNIX-IMAGE:BEGIN/END`) + seam-patch generator + `git am` re-apply | **DO-NOW without markers; DEFER markers; DROP the `git am` script for now** | one ours-only `docs/maintainer/image-integration-points.md` or `.agent/` manifest (file + anchor symbol + region class per hunk), plus a ~120-line checker script; zero upstream-file edits if markers are skipped | manifest ~200-350 lines of data; checker ~120 lines | LOW (manifest+checker) / MEDIUM (markers) | `git format-patch`/`git am` need a base commit; with no merge-base the only workable primitive is `git diff f9c2e42c -- <file>` + `git apply --3way`, which works because both blobs live in one object DB. Marker comments would add ~220 lines to the 32 upstream-owned files and upstream itself has ZERO such convention (`git grep -E "INFERNIX-(IMAGE\|BEGIN\|END)\|FORK\|LOCAL-" f9c2e42c -- src include apps tests cmake` -> no hits), so markers are diff we would carry forever and upstream would reject in a PR. |
| D | Dry-run drift harness (foreign checkout + seam apply + configure/build) | **DO-NOW, configure-and-audit only** | ours-only `tools/port/` or `.agent/` scripts; a throwaway checkout dir and a separate build dir (never `build-windows/`, which the live server's binaries come from) | ~150-200 lines (bash + python -I), no repo edits | LOW | This is the only candidate that finds drift before a port is on the clock. Honest limit: there is no host-only build -- every lane test links `infernix_serve`/`infernix_engine` and so drags the CUDA archives (`src/serve/CMakeLists.txt:44-48`, `tests/cmake/ProductTests.cmake:74,97`, `tests/models/qwen4_exp/tests.cmake:108-131`), and upstream now hard-gates the toolkit at CUDA >= 13.4 (`f9c2e42c:CMakeLists.txt:21-27`, absent in ours). Configure needs no GPU and no port; a build does not disturb the server either, but costs upstream's own measured 12.6 minutes (`f9c2e42c:build_native.bat:9-12`). |
| E | External-contract pins: arity/help consistency, `/v1/capabilities`, `/v1/images` route contract | **DO-NOW, cheapest real item** | `src/serve/serve_options.cpp` + `.h` (add `serve_option_arities()` accessor, already seam files), `tests/test_serve_options.cpp` (port upstream's self-contained `HelpOption`/`help_options`/`says_repeatable` block) | ~2 lines production + ~60 lines test | LOW | Upstream's newest code added exactly this pin (`f9c2e42c:tests/test_serve_options.cpp:139-168`) and our fork does not have it. Without it a build-green, test-green port still fails at startup: upstream's parse loop "accepts nothing outside this table" (`f9c2e42c:src/serve/serve_options.cpp:24-27`) and the launcher runs `infernix-serve --help` as a machine-read contract. We already pay for the other two pins: `tests/test_http_routes.cpp:110-120` (routes via `api_route_pattern`, incl. the `/v1/v1/` double prefix) and `:130-200` (the full capabilities JSON, incl. every `image_generation` field). |
| F | Anything upstream now offers that lets the lane attach with fewer upstream edits | **DO-NOW (select 4 real anchors); the "clean attach point" does not exist** | `src/serve/serve_options.h:246`, `src/serve/http_server.h:58` `api_route_pattern`, the `INFERNIX_HAVE_PDFIUM` optional-feature idiom, and the launcher `--help` contract | documentation + the E edits; no new upstream files | LOW | Verdict below. No route table, no module registry, no ops-registration macro, no plugin/discovery hook, no extension doc. Four genuine cheap anchors exist, one of which (`serve_option_arities`) is worth adopting today. |

## Evidence per candidate

### A. Would `INFERNIX_IMAGE=OFF` build today? No. Every blocker, named

There is no flag anywhere in the tree: our root `CMakeLists.txt:18-20` declares only
`INFERNIX_BUILD_APPS`, `BUILD_TESTING`, `INFERNIX_BUILD_BENCHMARKS`; upstream `f9c2e42c:CMakeLists.txt:30-47`
adds `INFERNIX_CUDA_LINEINFO` and `INFERNIX_PDL_SKEW` and still no feature flag. So an OFF build does not
exist to be broken -- it has to be created, and the lane has unconditional references in all of these:

Target level (link fails if the target disappears):
1. `src/serve/CMakeLists.txt:20-27` -- six lane translation units inside `add_library(infernix_serve ...)`
   (`image_job_store.cpp`, `openai_images.cpp`, `reference_image.cpp`, `openai_images_http.cpp`,
   `chat_image_tool.cpp`, `image_generation_service.cpp`, `prompt_enhancer.cpp`, `image_text_tower.cpp`),
   and `:47-48` links `infernix_qwen_image infernix_qwen_image_runtime infernix_media_encode`. Upstream's
   own version (`f9c2e42c:src/serve/CMakeLists.txt:1-44`) has none of these, so it is a text conflict, not an
   absent hook.
2. `src/models/CMakeLists.txt:22` `include(qwen_image/sources.cmake)`; upstream's version
   (`f9c2e42c:src/models/CMakeLists.txt:17-23`) has `include(inkling2/sources.cmake)` at :18 instead.
3. `src/ops/CMakeLists.txt:17,19,42,45` -- the four ours-only op dirs. Confirmed ours-only by directory
   listing: upstream `src/ops/` has `attn_input_proj` (shared, NOT ours-only), `screened_head`,
   `rmsnorm_rope`, `lossless`, `lora`, `direction_cut`, `w4a4_sparse_moe`, `inkling2` but has NO
   `attn_dit_joint`, `dit_adaln`, `rope_c64`, `vae_decoder`.
4. `src/media/CMakeLists.txt:24-33` defines ours-only `infernix_media_encode` (only consumers:
   `src/serve/image_generation_service.cpp`, `tests/test_media_encode.cpp`, `tests/test_image_*`; upstream
   put `infernix_media_pdf` at that exact spot -- `f9c2e42c:src/media/CMakeLists.txt:25-30`).

Compile level (a bare `#include` of a lane header that no longer resolves):
5. `apps/serve/main.cpp:7-9` includes `serve/image_generation_service.h`, `serve/image_job_store.h`,
   `serve/image_text_tower.h`; the builder block runs contiguously from about `:345` through `:391`
   (`ImageJobStore`, `PromptEnhancer`, `ImageGenerationService`, `server.attach_images(*images)` at `:383`,
   `GeneratedImageRetention`). Contiguous -- one guard.
6. `src/serve/http_server.h:12` `#include "serve/openai_images.h"`, `:30` fwd decl, `:80`
   `attach_images(ImageGenerationService&)`, `:172` `ImageGenerationService* images_ = nullptr;`.
7. `src/serve/http_server.cpp:3,7,9,11` lane header includes; `:1273` `HttpServer::attach_images`
   definition; `:576,583,589` the three `/v1/images/*` routes; `:593` `/v1/capabilities`;
   `:686,700,819,927,956,989-991,1059` the capabilities image block. 16 lane regions measured.
8. `src/serve/http_transport.cpp:3` `#include "serve/openai_images.h"` -- 1 line, easy to miss.
9. `src/serve/openai_chat_response.cpp:4`, `src/serve/openai_responses_response.cpp:4`,
   `src/serve/anthropic_messages_response.cpp:4` -- each `#include "serve/chat_image_tool.h"`, and the calls
   `chat_image_attempts_for_response(outcome)` at `openai_chat_response.cpp:164` and
   `openai_responses_response.cpp:386`.
10. `src/serve/openai_chat_http.cpp:59,139,354`, `src/serve/openai_responses_http.cpp:388,496,731`,
    `src/serve/anthropic_messages_http.cpp:203,298,501` -- `append_chat_image_tool(...)` and
    `missing_image_url_suffix(...)` call sites, 9 in total, inside upstream functions.

Test level (configure fails, not just build):
11. `tests/CMakeLists.txt:11` `include(models/qwen_image/tests.cmake)` -- one line; upstream's
    `tests/CMakeLists.txt` differs from ours by exactly 1 hunk, so this is the cheapest of all.
12. `tests/cmake/ProductTests.cmake:7-9` (`infernix_media_encode_test` links ours-only
    `infernix_media_encode`), `:58-103` (six lane test targets). Upstream rewrote this file
    (`git diff --stat HEAD f9c2e42c -- tests/cmake/ProductTests.cmake` = 172 lines), so these are real merges.
13. `tests/models/qwen4_exp/tests.cmake:108,121,131` link `infernix_qwen_image_runtime`;
    `tests/ops/tests.cmake` registers the four lane ops.

Not blockers (safe to leave unconditional, and should stay): `include/infernix/types.h` lane structs
(8 lines, 1 region, no lane dependency), `include/infernix/engine.h:141-166` lease API,
`src/runtime/engine/engine.cpp:419-438` forwarding (1 region), `program.{h,cpp}`, `program_impl.h`. Those are
chat-program-side and always built -- which is precisely why the flag does not reduce the hard part.

Verdict basis: OFF is ~24 files and ~70 guard sites, and 6 of them are upstream-owned protocol handlers whose
lane footprint today is only 1-6 regions. Guards in those files are NEW hunks in files a porter must
re-anchor anyway, i.e. the flag raises future port cost while buying only build isolation. That is why A is a
DROP as a "cheap item" and why A-prime (manifests, no guards) is the DO-NOW.

### B. Relocation cost measured

Our lane headers are included project-src-relative (`"serve/openai_images.h"`, `"models/qwen_image/..."`),
resolved by the two include roots in `cmake/InfernixTargets.cmake:3-7` (`${PROJECT_SOURCE_DIR}/include` and
`${PROJECT_SOURCE_DIR}/src`). A `plugins/image/` subtree that mirrors `serve/`, `models/qwen_image/`, `ops/...`
would keep ~157 files' own `#include` lines untouched if a third include root were added -- but
`infernix_internal_includes()` is upstream-owned, so the clean route is a per-target
`target_include_directories(infernix_image PUBLIC ${PROJECT_SOURCE_DIR}/plugins/image)` in our own
CMake file, which only covers lane consumers, not `src/serve/http_server.cpp`.
The decisive cost is that hanging `plugins/` off the build requires `add_subdirectory(plugins)` in
upstream-owned root `CMakeLists.txt`, a file that today carries ZERO lane hunks -- a new permanent seam --
while nothing in the seam inventory shrinks. Plus ~157 `git mv`s of ours-only files with no behaviour change
and no test that would catch a mistake other than a full build. DROP.

### C. Marker discipline and seam patches, judged honestly

Unrelated histories do NOT block patches (patches are text, and `git apply --3way` finds both blobs in this
one object DB because `f9c2e42c` is a fetched commit). What blocks a reliable generator is hunk
delimitation, and the measurement says the token heuristic gets 95 percent there but not all the way:
with the full lane vocabulary, the 31 seam files resolve into ~111 contiguous regions -- `apps/serve/main.cpp`
3, `engine.h` 1, `types.h` 1, `program.cpp` 7, `program.h` 4, `program_impl.h` 17, `http_server.cpp` 16,
`http_server.h` 4, `openai_chat_http.cpp` 5, `openai_responses_http.cpp` 6, `anthropic_messages_http.cpp` 6,
`serve_options.cpp` 1, `ProductTests.cmake` 3, `tests/ops/tests.cmake` 1, `test_http_routes.cpp` 12.
Two failure modes make markers tempting but not worth their own cost:
- Token-blind hunks: the `program_impl.h` lease body at `:5982` onward (`ImageYieldProbe
  probe_image_yield` `:6013`, `ImageYieldLease yield_for_image` `:6052`, `rollback_failed_yield` `:6171`,
  `release_from_yield` `:6184`, quiescence track `:6237`+) reads upstream-owned private state
  (`residency_`, `plan_`, `pool_`, `kv_vmm_`, `vram_mutex_`), so it is a re-derivation, not a re-anchor, and no
  delimiter makes it mechanical.
- Comment-only false positives, e.g. `src/serve/request_log.h:85` and `include/infernix/types.h` (2 real
  lane lines inside 57 divergence hunks).

Upstream has no marker convention whatsoever: `git grep -nE "INFERNIX-(IMAGE|BEGIN|END)|BEGIN/END|OUR-|FORK|LOCAL-|UPSTREAM" f9c2e42c -- src include apps tests cmake` returns nothing. Adopting markers would be us writing ~220 comment lines into 32 upstream-owned files that upstream would ask us to drop in any PR, and which would themselves drift.

What to build instead (cheap, ours-only, zero upstream-file cost): one manifest file enumerating, per seam
file, the anchor (function or CMake variable name the hunk hangs off), the lane region class
(re-add / re-anchor / re-derive), and the test that covers it; plus a checker that for each manifest entry
runs `git grep -c <anchor> f9c2e42c -- <file>` and prints MISSING. That converts the port's dominant cost
(discovering at build time that an anchor moved) into a weekly report, at ~200 data lines and ~120 script
lines, with no build required. `git apply --3way` rehearsal is a follow-on once the manifest is trusted;
a `git am` script is not achievable without first synthesizing a common base commit, which is the big job.

### D. Dry-run harness, smallest useful version, and the real configure prerequisites

Measured facts:
- `cmake/FindFFMPEG.cmake` is BYTE-IDENTICAL to upstream (`git diff --stat HEAD f9c2e42c -- cmake/FindFFMPEG.cmake`
  prints nothing) and `cmake/Dependencies.cmake:26` `find_package(FFMPEG REQUIRED)` is identical in both trees
  (`f9c2e42c:cmake/Dependencies.cmake:26`). So a foreign checkout hits the same gate we do -- but the failure
  is NOT a raw-backslash syntax error. `grep -n '\\\\' cmake/FindFFMPEG.cmake cmake/Dependencies.cmake` finds
  NO backslashes in either file; every path there is forward-slash. What actually aborts configure is the
  REQUIRED find: `cmake/FindFFMPEG.cmake:16-22` (no `VCPKG_ROOT`/`VCPKG_TARGET_TRIPLET` in the environment ->
  fail), `:28-33` (missing `avformat/avcodec/avutil/swscale .lib` in `<tree>/lib` -> fail),
  `:35-39` (missing `include/libavformat/avformat.h` -> fail), reported via
  `find_package_handle_standard_args` at `:45-48`. The workaround is exactly what `build_native.bat:12-13,20`
  already does: export `VCPKG_ROOT=C:\vcpkg` and `VCPKG_TARGET_TRIPLET=x64-windows`, or pass
  `-DCMAKE_PREFIX_PATH=C:\vcpkg\installed\x64-windows` directly.
- Triplet requirement is real and singular: this machine has only `C:\vcpkg\installed\x64-windows` (and a
  `vcpkg` metadata dir). No other triplet exists, so a foreign checkout must use `x64-windows`.
- New hard prerequisite upstream added and we do not have: CUDA >= 13.4, `f9c2e42c:CMakeLists.txt:21-27`
  (`if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.4) message(FATAL_ERROR ...)`) with the CUDA 13.2
  miscompile rationale. Our root `CMakeLists.txt` has no version gate at all. Both `v13.3` and `v13.4` are
  installed here, so a harness must pin `-DCMAKE_CUDA_COMPILER` at `...\v13.4\bin\nvcc.exe`, which is what
  `f9c2e42c:build_native.bat:18,44` does.
- `PDFIUM_ROOT` is optional and only warns (`f9c2e42c:cmake/Dependencies.cmake:47-68`), so a foreign tree
  configures without it; but if the harness reuses our `cmake/` files it must not assume they are current --
  upstream added `cmake/EmbedLauncherAssets.cmake` (ours lacks it) and grew `Dependencies.cmake` by 27 lines
  and `InfernixTargets.cmake` by 8.
- Generator conflict hazard: upstream's own script DELETES `build-windows` if it was configured with a
  different generator (`f9c2e42c:build_native.bat:38-42`, Visual Studio generator -> Ninja Multi-Config). Our
  tree is configured with the Visual Studio generator (`build_native.bat:17`). A harness reusing that path
  would delete the build tree the live `:8888` server's binaries come from. It must use a fresh dir
  (`build-drift/`).
- There is no host-only target set. `infernix_serve` links `infernix_engine` + `infernix_cuda_runtime`
  (`src/serve/CMakeLists.txt:44-48`), and every lane host test links `infernix_serve` or
  `infernix_qwen_image_runtime` (`tests/cmake/ProductTests.cmake:74,97`;
  `tests/models/qwen4_exp/tests.cmake:108,121,131`), so even a "host" test builds the CUDA archives.
  Configure needs no GPU at all; a build needs the toolkit, not the card. CMake still requires a CUDA
  compiler because the project is `project(infernix LANGUAGES C CXX CUDA)` at both
  `CMakeLists.txt:15` and `f9c2e42c:CMakeLists.txt:15`.
- Test-side gating that lets a harness run safely against the live server: upstream still grants
  `SKIP_RETURN_CODE 77` and the `gpu;real` label exactly as we lean on them
  (`f9c2e42c:tests/CMakeLists.txt:25`, `f9c2e42c:tests/cmake/InfernixTests.cmake:34,55`,
  `f9c2e42c:tests/cmake/ProductTests.cmake:144`), and `infernix_add_test`/`infernix_add_op_test`/
  `infernix_add_family_op_test` are unchanged in shape (`tests/cmake/InfernixTests.cmake:9-58`). No new
  convention appeared that we must adopt; upstream only added more `SKIP_RETURN_CODE 77` call sites
  (`tests/models/inkling2/tests.cmake:7-55`). So `ctest -E 'real|gpu'` (or excluding `LABELS "gpu;real"`)
  plus `RUN_SERIAL TRUE` on the gpu ones is still the correct contract.

Smallest useful version, therefore: fetch, `git worktree`/archive `f9c2e42c` into a scratch dir, apply the
seam set per the C manifest, run `cmake` configure with upstream's own argument line copied verbatim from
`f9c2e42c:build_native.bat:43-49` into a separate `build-drift/`, then print (a) every manifest anchor that
no longer exists, (b) every `include()`/link entry our manifests add that upstream has renamed, and (c) the
configure error log. Stop there. The build is a separate opt-in flag, and the doc should say plainly that
the harness does not prove compilability.

### E. Contract pins, and what they cost

Already owned (maintenance is already paid):
- `/v1/images` route contract, including upstream's `/v1/v1/` double-prefix rule:
  `tests/test_http_routes.cpp:82,89` register the lane routes through `api_route_pattern`, and `:110-112`
  asserts `post_status("/v1/images/edits") == 200 && post_status("/v1/v1/images/edits") == 200 &&
  post_status("/images/edits") == 404`. Our fork already carries `api_route_pattern`
  (`src/serve/http_server.h:64`, `http_server.cpp:204`), matching `f9c2e42c:src/serve/http_server.h:58` and
  `:218`, so our lane routes already obey the newest upstream URL contract. Nothing to change, and upstream
  has no `/v1/images` at all (`git grep -l "v1/images\|image_generation" f9c2e42c -- tests` -> empty), so no
  upstream change can silently collide.
- `/v1/capabilities` shape: `tests/test_http_routes.cpp:130-200` asserts `object == "infernix.capabilities"`,
  the null-without-options case at `:131-133`, and every `image_generation` field (model,
  `artifact_endpoint == "/v1/images/<id>"`, response formats, `chat_tool == "generate_image"`,
  `chat_max_images_per_request == 4`, `reference_image`, prompt/negative_prompt/canvas ceilings). This is the
  ONLY pin in either tree -- upstream has no capabilities test at all. Cost to keep: zero extra, we wrote it
  for our own contract.

Cheap addition worth doing now: upstream's help/arity consistency test, which our fork lacks.
`f9c2e42c:tests/test_serve_options.cpp:51-90` defines `HelpOption`, `help_options()` and `says_repeatable()`
inside the test file itself, and `:139-168` asserts that the help text lists exactly the options the parser
accepts, with matching arity and repeatability, in both directions ("serve help omits X",
"serve help gives X the wrong arity", "serve help lists X, which the parser refuses"). It depends on
`serve_option_arities()` (`f9c2e42c:src/serve/serve_options.h:246`), which ours does not export -- ours has
the table and the enforcement (`src/serve/serve_options.cpp:25,29,135-137`) but no accessor. Our lane flags are
already in the table at `src/serve/serve_options.cpp:103-106` and their help text at `:578-592`, and
`ConfigOption` defaults match upstream's (`src/product/config/config_file.h:17-12`), so our four flags would
pass upstream's test today. Adopting the accessor (2 lines) plus the test block (~60 lines) is the cheapest
real insurance in this whole audit, because the failure it prevents is a startup refusal, not a compile error.

Cost/benefit: no maintenance beyond keeping our four help lines' arity honest, which upstream's test then
enforces for us. Upstream grew its option set from our fork's 292 to 351 quoted flags
(`grep -c '"--[a-z0-9-]+"'` on the two `serve_options.cpp` files), so the table is the single hottest text in
the serve layer and the one place where a silent port break is otherwise invisible.

### F. What upstream offers that we do not use

There is no clean attach point. Evidence: upstream `HttpServer` exposes exactly one public seam,
`attach(GenerationBackend& service)` (`f9c2e42c:src/serve/http_server.h:75`), and `GenerationBackend` is the
chat-generation interface (`f9c2e42c:src/serve/generation_service.h:157`), not a module list. `register_routes()`
is private (`f9c2e42c:src/serve/http_server.h:120`) and every handler is a private member (`:125-138`). No
route table, no registration macro, no registry of modules, no ops-registration macro (ops are added by an
explicit `include(<dir>/sources.cmake)` line in upstream's file, `f9c2e42c:src/ops/CMakeLists.txt:22-53`),
no plugin/discovery hook -- consistent with the project rule in `AGENTS.md`: "Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution". No extension docs either: the docs
tree at `f9c2e42c` has no add-a-family or add-a-route guide (`docs/maintainer/` is 30 design/contract records;
the only grep hits for extension language are design records).

Four real anchors that do exist and that we should use:
1. `serve_option_arities()` (`f9c2e42c:src/serve/serve_options.h:246`) plus upstream's own help/arity test.
   This is the closest thing to a registry in the serve layer, and adopting it costs 2 production lines.
   This is the single highest-leverage cheap item.
2. Upstream's optional-feature idiom, which we should copy when the flag is ever built so a future PR reads
   native: `cmake/Dependencies.cmake:47-68` (`set(PDFIUM_ROOT ... CACHE PATH)` + `find_package(CONFIG ...
   NO_DEFAULT_PATH)` + `set(INFERNIX_HAVE_PDFIUM ON/OFF)` + cache-invalidation guard),
   `src/media/CMakeLists.txt:25-30` (target always exists, link and define only under `if(INFERNIX_HAVE_PDFIUM)`),
   `src/media/pdf/pdf.h:6` + `pdf.cpp:9,54` (`#ifdef INFERNIX_HAVE_PDFIUM`, and the interface refuses without
   it). Note upstream uses a `set(... OFF)`/`find_package` shape, not `option()`, because the feature is
   external-dependency driven -- for a lane that is always in-tree, `option(INFERNIX_IMAGE)` is the honest
   analogue, and the `if(TARGET)`/`target_compile_definitions` shape is what to imitate.
3. `api_route_pattern()` -- already adopted by our lane routes (see E). Keep it in the manifest so a porter
   re-adds the routes through it rather than hard-coding `/v1/`.
4. The launcher `--help` contract, which is a NEW surface nobody counted and it does not shrink our work but
   it does raise the bar on our help text: `f9c2e42c:src/launcher/serve_help.cpp:126-170` runs
   `infernix-serve --help`, parses it into option/argument/section/description, caches per executable mtime
   (`src/launcher/serve_help.h:42-48`), and errors if nothing parses; `apps/launcher/web/src/pages/configure/options.ts:2`
   states the intent "every option of `infernix-serve --help` has a control here";
   `src/launcher/launch_plan.cpp:248` holds a hardcoded list of options that need companions, and
   `src/launcher/args.cpp:94-100` a hardcoded switch list. Our fork has no `src/launcher/` at all. On a port,
   our four `--image-*` flags become launcher-visible and must keep upstream's exact help grammar -- which is
   another reason to adopt E now.

## Answer 1: honest time-to-green if a new Infernix ships tomorrow

From the measured surface only (32 upstream-owned files, ~113 lane hunks, ~600 divergence hunks to navigate),
not from vibes. Figures are skilled-developer wall-clock ranges for a porter who already knows the lane
(= us), assuming the port is done on a machine with the artifacts and CUDA 13.4 ready.

Zero prep today:
- Mechanical re-adds where the file barely moved (the 6 CMake/test-registration files, `types.h` 2 real lane
  lines, `engine.h` 1 region, `engine.cpp` 1 region, `main.cpp` 1 contiguous block, the 3 response-handler
  includes): about 1 day.
- Re-anchoring the protocol handlers and `http_server.{h,cpp}` (`http_server.cpp` 16 regions, incl. the
  ours-only capabilities block which is a re-add not a re-anchor, plus 9 injection call sites across 3
  handlers, and upstream's `register_routes()`/`/health` shape having moved under a block nothing upstream
  pins): 1.5 to 2.5 days, mostly reading upstream to find the new call points.
- The two hard re-derivations, unchanged from the companion assessment: `program_impl.h` (17 lane regions
  reading `residency_`/`plan_`/`pool_`/`kv_vmm_`, against `execution_mu_` which no longer exists upstream) and
  the residency keep-policy fight introduced by `a3f78ef8`/`18388f5b`, plus the NVML-capped free-VRAM contract
  from `b821d961`: 2 to 4 days, and this is where a naive port ships green-but-wrong.
- Serve options + arity table (59 upstream options more than our fork) + the launcher help contract: 0.5 day
  IF you know the arity table is mandatory, otherwise a day of "the server refuses to start".
- Rebuilding and getting host tests green (full build is upstream's own measured 12.6 min per cycle): 0.5 day.
Total: roughly 5 to 9 working days, dominated by `program_impl.h` and the serve layer. The often-quoted
"6-8 seams" is a count of conceptual surfaces from the companion assessment; mechanically it is 32 files and
~113 hunks, and a plan that budgets by surfaces will under-schedule the port by roughly the difference.

With each DO-NOW item done (items are additive to the same baseline; they shorten the tail, they do not
touch the 2-4 day `program_impl.h` core):
- A' manifests only: 4.5 to 8.5 days (the CMake/test-registration half becomes copy instead of merge; saves
  about half a day, and removes the class of "a lane source silently missing from a target").
- C manifest + anchor checker only: 4 to 7 days, and the shape of the work changes -- the 1.5-2.5 day
  re-anchoring block collapses toward 0.5-1 day because every broken anchor is named before the port starts.
  This is the largest single reduction available cheaply.
- D harness only: no reduction in the port itself, but drift is caught while it is 1 day old instead of 119
  commits old; on a weekly cadence it historically converts most of the re-anchoring block into small
  commits. Standing cost ~150 script lines, no build time.
- E pins only: 4.5 to 8.5 days (kills the "builds green, server refuses to start" failure mode and the
  launcher help-grammar failure mode; saves ~0.5 day and one very demoralizing class of bug).
- All four DO-NOW items: honest range 3.5 to 6 days, with the residual being exactly the `program_impl.h`
  re-derivation and the residency/VRAM contract fight -- the part that genuinely needs WP-B and cannot be
  bought down by cheap preparation. Everything outside that residual becomes a checklist.

## Answer 2: the best cheap item, and the trap

Best cheap item: the C seam manifest plus an anchor checker (and D is the same file's runtime). Concretely:
one ours-only manifest listing, for each of the 32 upstream-owned files, the anchor symbol, the region class,
and the covering test; plus ~120 lines that run `git grep -c <anchor> <upstream-sha>` and print MISSING, with
the optional configure-only rehearsal on top. It is the only candidate here that attacks the port's dominant
cost (finding out at build time that an anchor moved) instead of a peripheral one, it touches zero
upstream-owned files, it needs no build, and it is the artifact that makes A, B and every later decision
cheap. If only one thing gets done, it should be the manifest; E's `serve_option_arities()` accessor plus
upstream's help/arity test is the 30-minute companion that prevents the ugliest failure mode, and both are
cheap enough to do together.

The trap that looks cheap and is not: candidate A, the build flag. It is described as "one `option()` plus
about six conditional wiring points", and six of those points are real -- but they are the whole story only
for the CMake half. Because `infernix_serve` compiles eight lane translation units and links
`infernix_qwen_image`/`infernix_qwen_image_runtime`/`infernix_media_encode` (`src/serve/CMakeLists.txt:20-27,47-48`),
an OFF configure cannot drop those targets while the serve layer still includes the lane headers, so the flag
forces `#ifdef` guards into ~24 files including nine call sites inside three upstream-owned protocol handlers
and a lane header include in `src/serve/http_transport.cpp:3` that nothing else hints at. It adds hunks to the
exact files whose hunk count is the port's cost, it does not touch the hard part (`program_impl.h`), and OFF
must then be kept green forever, which nobody will. It is the first third of Plan 2's WP-A wearing a cheap
label. Candidate B (moving to `plugins/image/`) is the second trap and is disqualified by a single fact: it
requires a new `add_subdirectory()` in upstream-owned root `CMakeLists.txt`, so it adds a seam while
churning 157 files.

## What upstream changed that moves our "6-8 seams" number

- `/v1/capabilities` / `make_capabilities_json` are ours-only, absent upstream at `f9c2e42c` AND at the
  `0a3f432b` baseline. The companion assessment's claim that the anchor "survives" is wrong. Net effect on
  the count: `http_server.cpp`'s 16 regions split into fewer re-anchors and more pure re-adds (cheaper), but
  the block loses its only protection against upstream drift, because nothing upstream pins it (raises D's
  value).
- Upstream's serve-option arity table is now an enforced whitelist with an enumerable accessor
  (`f9c2e42c:src/serve/serve_options.h:246`, `serve_options.cpp:24-29,152`) and a help/arity consistency test
  (`tests/test_serve_options.cpp:139-168`). Our four `--image-*` flags must exist in the table, the parser AND
  the help text, with matching arity, or the server refuses to start. That is a third mandatory anchor per
  flag, i.e. +4 hunks in one file we already count, but with a silent-startup failure mode -- so it is
  operationally a new seam even though it is textually inside an existing one.
- Upstream grew a launcher subsystem that runs `infernix-serve --help` as a machine-read contract and expects
  every listed option to have a UI control (`f9c2e42c:src/launcher/serve_help.cpp:126-170`,
  `src/launcher/serve_help.h:42-48`, `apps/launcher/web/src/pages/configure/options.ts:2`,
  `src/launcher/launch_plan.cpp:248`). Our fork has no launcher. This is a genuinely new surface: +1 to the
  surface count (the help-text grammar), not +1 file to our seam set.
- Upstream's hard CUDA >= 13.4 gate (`f9c2e42c:CMakeLists.txt:21-27`) and its switch to Ninja Multi-Config
  with a build-dir-wipe-on-generator-mismatch (`f9c2e42c:build_native.bat:38-49`) are new configure-time
  prerequisites a fresh checkout must satisfy, and a hazard for any harness that reuses `build-windows/`.
- Good news that does not change the count but does change the port's risk: upstream still grants
  `SKIP_RETURN_CODE 77` and the `gpu;real` label exactly as before (`f9c2e42c:tests/CMakeLists.txt:25`,
  `tests/cmake/InfernixTests.cmake:34,55`, `tests/cmake/ProductTests.cmake:144`), added no `/v1/images`, no
  `--image-*` flag, no capabilities entry, and our lane routes already use upstream's `api_route_pattern`
  contract. No new test-label convention appeared that we must adopt.

## What I could not verify

- Configure and build times for a foreign checkout. Nothing was configured or built (read-only constraint), so
  the 12.6 minute figure is upstream's own published number from `f9c2e42c:build_native.bat:9-12`, not a
  measurement by me, and whether a foreign `f9c2e42c` checkout actually configures cleanly on this machine
  (ffmpeg libs present in `C:\vcpkg\installed\x64-windows\lib`, CUDA 13.4 nvcc discovery, the launcher's
  `EmbedLauncherAssets.cmake` asset step, which our tree lacks entirely) is untested.
- Whether upstream's `program_impl.h` retained any lock or worker primitive equivalent to our `execution_mu_`
  and the worker-ack names. Confirmed absent by name; upstream's replacement was not traced (needs the full
  114-commit history of that file).
- The per-file split of the ~113 lane regions into "pure re-add" versus "true re-anchor" versus
  "re-derivation". I classified by token presence and by whether the region reads upstream-private state, but
  did not read all 32 files' upstream counterparts line by line. The 5-9 / 3.5-6 day ranges therefore carry
  maybe plus or minus 30 percent.
- Whether our four `--image-*` help lines pass upstream's `says_repeatable()` and `help_options()` grammar
  byte for byte. The arity defaults and the `--flag VALUE` shape look correct against
  `src/serve/serve_options.cpp:578-592`, but the parser lives in upstream's test file and was not executed
  against our help text (no build permitted).
- Whether `probe_image_yield`'s free-VRAM math already routes through the NVML-capped `VramBudgetSource`
  (`b821d961`), and whether the `!tier_` experts-in-RAM branches (`a3f78ef8`/`18388f5b`) are behind a switch a
  lease could disable. Both carried forward from the companion assessment as open; neither is cheap enough to
  close in this pass, and both gate only the WP-B work that is out of scope today.
- `src/artifact/formats.cpp` and `src/core/weight.*`/`weight_view.*` were touched by upstream's 119-commit
  advance (new quant layouts) but not diffed line by line against what `src/models/qwen_image/image_weights.cpp`
  binds. A passive-dependency watch item, flagged, not verified.
