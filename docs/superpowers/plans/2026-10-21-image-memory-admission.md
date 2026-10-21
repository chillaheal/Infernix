# Image Memory Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refuse an image job from a computed device-memory need before the LLM yields any VRAM, and never allocate on a grant that cannot cover the job.

**Architecture:** A pure `image_device_need()` in the model layer computes a job's device footprint from the pipeline's own buffer inventory (including reference tokens and the true-CFG sequence). `Engine::probe_image_yield()` runs the already-pure `plan_image_yield()` against that need and answers before anything is quiesced. The serving layer tokenizes on the host before the probe so the need is exact, refuses with a typed error, and turns every chat image failure into a tool result. The pipeline's device ledger (`image_device_bytes_peak()`) qualifies the model.

**Tech Stack:** C++20, CUDA 13 / `sm_120a`, CMake + ctest (host tests are plain `main()` + `check()` helpers registered through `infernix_add_test`).

## Global Constraints

- Build tree is `build` (VS 18 2026), configured by `../build-infernix.bat`; build with `cmake --build build --config Release --target <t> -j 8`, test with `ctest --test-dir build -C Release -R <pattern>`. There is no `build-windows` on this machine.
- Commits use the inline identity only: `git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit ...`. Never edit git config.
- ASCII-only source. No non-ASCII punctuation in code, comments or tests.
- The model layer must not know serving types; the existing `ImageJobProgressView` is the pattern for a models-facing observer seam.
- `Buf` (`src/models/qwen_image/image_device_util.h`) stays the only device allocator in the image path, and `image_device_bytes_live()` must be 0 between jobs.
- The generated picture's bytes must not change: nothing in this plan adds arithmetic to the DiT/VAE math.
- `docs/serving.md` and the schema tests change in the same commit as any route-visible behavior.
- Do not relink or restart the deployed `infernix-serve.exe`; this plan builds libraries and test executables only.

---

### Task 1: `image_device_need()` — the computed footprint

**Files:**
- Create: `src/models/qwen_image/image_device_need.h`
- Modify: `src/models/qwen_image/CMakeLists.txt` (the host `infernix_qwen_image` lib gains the header; confirm the lib lists headers before adding anything)
- Test: `tests/test_image_device_need.cpp`
- Modify: `tests/cmake/ProductTests.cmake`

**Interfaces:**
- Consumes: `ImageGeometry` (`src/models/qwen_image/image_job.h:15-34`), `DiTDims` (`src/models/qwen_image/image_weights.h:27-37`).
- Produces: `infernix::models::qwen_image::ImageDeviceNeed` and `image_device_need(const ImageDeviceNeedRequest&) -> ImageDeviceNeed`, used by the serving layer in Task 4.

- [ ] **Step 1: Read the inventory this must match**

Read `src/models/qwen_image/image_pipeline.cu:307-342` (the `S`/`s_max` derivation), `:410-456` (the 13 + 3 workspace buffers and every per-step buffer), `:423-446`, `:366-397` (constants and both rope tables), `:399-408` (the resident top-level weights), `:620-665` (the VAE decode's plane sequence). Also read `VaeGraph::dup_up`, `VaeGraph::att`, `VaeGraph::dresnet` and `VaeGraph::drms` (in `src/models/qwen_image/image_pipeline.cu`, the `struct VaeGraph` above line 300) and write down, per call, how many planes it has live at once. Those counts become `kVaePlanesPerStage` in Step 3 — they are copied from that code, not guessed.

- [ ] **Step 2: Write the failing host test**

Create `tests/test_image_device_need.cpp` in the repo's host-test style (`main()` plus a local `check(bool, const char*)` like `tests/test_image_job_store.cpp`):

```cpp
#include "models/qwen_image/image_device_need.h"

#include <cstdio>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what); }
}

qi::DiTDims test_dims() {
    qi::DiTDims d;
    d.layers = 32; d.heads = 32; d.head_dim = 128; d.hidden = 4096;
    d.intermediate = 12288; d.image_in = 64; d.image_out = 64; d.context_in = 4096;
    return d;
}

std::uint64_t need_of(std::uint32_t side, std::uint32_t text, std::uint32_t neg, std::uint32_t ref) {
    qi::ImageDeviceNeedRequest req;
    req.geometry   = qi::ImageGeometry::derive(side, side);
    req.dims       = test_dims();
    req.text_tokens = text;
    req.negative_tokens = neg;
    req.ref_tokens = ref;
    req.blocks_resident = 1;
    return qi::image_device_need(req).total();
}

} // namespace

int main() {
    // The review's failure: the same target size with a reference must cost more than without.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 0, 4096), "a reference raises the need");
    // True CFG sizes every sequence buffer by the longer joint sequence.
    check(need_of(768, 200, 0, 0) < need_of(768, 200, 512, 0), "true CFG raises the need");
    // A longer negative pass drives s_max, so CFG with a long negative is the wider of the two.
    check(need_of(512, 64, 4096, 0) == need_of(512, 4096, 64, 0), "s_max is the longer joint sequence");
    // Area dominates: 2048^2 is far past 768^2 and far past the same size plus one reference.
    check(need_of(768, 200, 0, 0) < need_of(2048, 200, 0, 0), "area raises the need");
    check(need_of(1024, 200, 0, 4096) < need_of(2048, 200, 0, 0), "area beats one reference");
    // The 512x512 certified route peaked at 6183 MiB; the model must not predict wildly below it.
    check(need_of(512, 200, 0, 0) >= 6183ULL * 1024ULL * 1024ULL, "covers the certified 512 peak");
    // Every term is non-negative and the total is the sum of the parts.
    qi::ImageDeviceNeedRequest req;
    req.geometry = qi::ImageGeometry::derive(768, 768);
    req.dims     = test_dims();
    req.text_tokens = 200;
    req.blocks_resident = 1;
    const auto need = qi::image_device_need(req);
    check(need.total() == need.dit_activations + need.latent_and_velocity + need.resident_weights +
                            need.vae_peak + need.rope_and_tables,
          "total is the sum of the terms");
    if (failures != 0) { std::printf("%d check(s) failed\n", failures); return 1; }
    std::printf("image_device_need: ok\n");
    return 0;
}
```

Register it in `tests/cmake/ProductTests.cmake` beside the other serve/model host tests (copy the shape of an existing `infernix_add_test(infernix_image_job_store_test ...)` entry, linking `infernix_qwen_image`):

```cmake
infernix_add_test(infernix_image_device_need_test test_image_device_need.cpp)
target_link_libraries(infernix_image_device_need_test PRIVATE infernix_qwen_image)
```

Adjust the target's library name to whatever `src/models/qwen_image/CMakeLists.txt` actually names the host lib (read it first; the `.cu` pipeline is a separate runtime lib and the test must not link CUDA).

- [ ] **Step 3: Run it and confirm it fails to compile**

Run: `cmake --build build --config Release --target infernix_image_device_need_test`
Expected: `image_device_need.h: No such file or directory`.

- [ ] **Step 4: Write the header**

Create `src/models/qwen_image/image_device_need.h`, header-only like `image_job.h`:

```cpp
#pragma once

// One image job's device-memory need, computed from the buffer inventory the production pipeline
// actually allocates (src/models/qwen_image/image_pipeline.cu), not from a scaled measurement. The
// serving layer asks for it before the LLM yields any VRAM, so a job that cannot be served is refused
// with nothing moved. The image ledger (image_device_bytes_peak()) is the oracle: the qualification
// test asserts the predicted total is never below the measured peak for the same inputs.

#include "models/qwen_image/image_job.h"
#include "models/qwen_image/image_weights.h"

#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen_image {

// The bytes the resident weights hold across the loop: the DiT's top-level projections, one resident
// block, and the VAE's operands. Calibrated once against the certified 512x512 route's measured peak
// (6183 MiB, interlude_s8dprod_test.out.log) minus this file's computed activation terms at that
// geometry, and re-checked by tests/models/qwen_image/test_image_need_ledger_real.cu.
inline constexpr std::uint64_t kImageWeightResidentBytes = 1024ULL * 1024ULL * 1024ULL;

// BF16 element size everywhere in the image pipeline.
inline constexpr std::uint64_t kImageElementBytes = 2;

// The VAE decode's widest stage: how many full planes a stage holds live at once, per stage kind,
// copied from VaeGraph::dup_up/att/dresnet/drms in image_pipeline.cu.
struct VaeStage {
    std::uint32_t channels_out;
    std::uint32_t planes; // simultaneous full [C,H,W] planes at that stage's resolution
};

// The decoder's fixed five up-stages plus the mid block, as image_pipeline.cu:620-665 runs them.
inline constexpr VaeStage kVaeMidStage{1152, 4};
inline constexpr VaeStage kVaeUpStages[5] = {
    {1152, 4}, {1152, 4}, {576, 4}, {288, 4}, {144, 4},
};

struct ImageDeviceNeedRequest {
    ImageGeometry geometry;      // the target image's snapped geometry
    DiTDims dims;                // the bound DiT's dimensions
    std::uint32_t text_tokens = 0;    // prompt rows the joint sequence carries
    std::uint32_t negative_tokens = 0; // negative-prompt rows (0 without true CFG)
    std::uint32_t ref_tokens = 0;      // reference-image latent tokens ahead of the target
    std::uint32_t blocks_resident = 1; // DiT blocks resident at once (the streamed route: one)
};

struct ImageDeviceNeed {
    std::uint64_t dit_activations = 0;   // block workspace + the joint/ping-pong/norm buffers at s_max
    std::uint64_t latent_and_velocity = 0; // [ref+target] latent, hs_img, the [s_max,Cin] velocity
    std::uint64_t resident_weights = 0;  // top-levels, the resident block(s), the VAE operands
    std::uint64_t vae_peak = 0;          // the widest simultaneous set of decode planes
    std::uint64_t rope_and_tables = 0;   // both rope tables and the shared constants

    [[nodiscard]] std::uint64_t total() const noexcept {
        return dit_activations + latent_and_velocity + resident_weights + vae_peak + rope_and_tables;
    }
};

// The need, exactly as the pipeline sizes itself: every sequence-shaped buffer follows
// `s_max = max(text + ref + target, negative + ref + target)` (image_pipeline.cu:332-342), so a
// reference image and true CFG both raise it by construction.
[[nodiscard]] constexpr ImageDeviceNeed image_device_need(const ImageDeviceNeedRequest& req) noexcept {
    const std::uint64_t target = req.geometry.tokens; // hl * wl
    const std::uint64_t all_img = static_cast<std::uint64_t>(req.ref_tokens) + target;
    const std::uint64_t d      = req.dims.hidden;
    const std::uint64_t mlp    = req.dims.intermediate;
    const std::uint64_t cin    = req.dims.image_in;
    const std::uint64_t s_pos  = static_cast<std::uint64_t>(req.text_tokens) + all_img;
    const std::uint64_t s_neg  = static_cast<std::uint64_t>(req.negative_tokens) + all_img;
    const std::uint64_t s_max  = s_neg > s_pos ? s_neg : s_pos;
    const std::uint64_t text_max = req.negative_tokens > req.text_tokens
                                     ? static_cast<std::uint64_t>(req.negative_tokens)
                                     : static_cast<std::uint64_t>(req.text_tokens);

    ImageDeviceNeed need;
    // 13 workspace buffers at sD and 3 at sMlp (image_pipeline.cu:412-420), plus the joint buffer and
    // its ping-pong partner, the two norm stages and the [s_max,Cin] velocity output (:434-446).
    need.dit_activations = (13ULL + 3ULL) * s_max * d * kImageElementBytes + 3ULL * s_max * mlp * kImageElementBytes +
                           4ULL * s_max * d * kImageElementBytes + s_max * cin * kImageElementBytes;
    // The evolving [ref+target] latent, its projected image hidden, and both text embed stages.
    need.latent_and_velocity = all_img * cin * kImageElementBytes + all_img * d * kImageElementBytes +
                               2ULL * text_max * d * kImageElementBytes;
    // Interleaved rope tables (S * 64 * 2 floats, BF16 on device) for both passes and the shared
    // ones/zeros/tnorm constants (2 * D each).
    need.rope_and_tables = (s_pos + s_neg) * 64ULL * 2ULL * kImageElementBytes + 5ULL * d * kImageElementBytes;
    // The widest decoder stage: planes * channels * pixels, at that stage's resolution. Up-stages
    // double the side, so the last stages are the expensive ones (image_pipeline.cu:620-665).
    std::uint64_t side = req.geometry.hl > req.geometry.wl ? req.geometry.hl : req.geometry.wl;
    std::uint64_t hl   = req.geometry.hl, wl = req.geometry.wl;
    need.vae_peak = kVaeMidStage.planes * static_cast<std::uint64_t>(kVaeMidStage.channels_out) * hl * wl *
                    kImageElementBytes;
    for (const auto& stage : kVaeUpStages) {
        const std::uint64_t bytes = stage.planes * static_cast<std::uint64_t>(stage.channels_out) * hl * wl *
                                    kImageElementBytes;
        if (bytes > need.vae_peak) { need.vae_peak = bytes; }
        hl *= 2;
        wl *= 2;
    }
    (void)side;
    // The final 4-channel plane at full pixel resolution, plus the operands held across the loop.
    const std::uint64_t final_plane = 4ULL * static_cast<std::uint64_t>(req.geometry.height) *
                                      static_cast<std::uint64_t>(req.geometry.width) * kImageElementBytes;
    if (final_plane > need.vae_peak) { need.vae_peak = final_plane; }
    need.resident_weights = kImageWeightResidentBytes * (1ULL + req.blocks_resident) / 2ULL;
    return need;
}

} // namespace infernix::models::qwen_image
```

Replace `kImageWeightResidentBytes` and the `kVaeMidStage`/`kVaeUpStages` plane counts with the values you wrote down in Step 1 (the literal `1 GiB` and the `4`s above are the starting guesses the ledger test in Task 2 corrects; the header comment says where they come from). `resident_weights` is one allowance for the top-levels plus the resident block allowance — write it as the two named terms you derived in Step 1 if the code lets you separate them; the test only pins `total()`.

- [ ] **Step 5: Build and run the test**

Run: `cmake --build build --config Release --target infernix_image_device_need_test -j 8` then `ctest --test-dir build -C Release -R infernix_image_device_need_test --output-on-failure`
Expected: `100% tests passed`. If "covers the certified 512 peak" fails, the weight allowance is too small — raise `kImageWeightResidentBytes` to the value Step 1's inventory implies rather than deleting the check.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen_image/image_device_need.h src/models/qwen_image/CMakeLists.txt tests/test_image_device_need.cpp tests/cmake/ProductTests.cmake
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "feat(image): compute a job's device-memory need from its buffer inventory"
```

---

### Task 2: Qualify the model against the ledger

**Files:**
- Test: `tests/models/qwen_image/test_image_need_ledger_real.cu` (new, artifact-gated like its siblings)
- Modify: `src/models/qwen_image/image_device_need.h` (the calibrated constants, if the ledger says otherwise)

**Interfaces:**
- Consumes: `image_device_need()` (Task 1), `image_device_bytes_peak()` / `image_device_bytes_live()` / `reset_image_device_peak()` (`src/models/qwen_image/image_pipeline.h:93-99`), `run_image_job`.
- Produces: the calibrated `kImageWeightResidentBytes` and `kVaeStage::planes` values that Task 4's refusal threshold rests on.

- [ ] **Step 1: Read the nearest existing device test**

Read `tests/models/qwen_image/test_image_pipeline_gate.cu` (and one `*_real.cu` sibling under `tests/models/qwen4_exp/`) and copy exactly how it is registered, how it locates the artifact (`INFERNIX_IMAGE_ARTIFACT` or the CMake variable the siblings use) and how it skips without one. Do not invent a new mechanism.

- [ ] **Step 2: Write the qualification test**

For each case in the table, run the job and compare:

```cpp
struct Case { std::uint32_t side; std::uint32_t text_tokens; std::uint32_t ref_tokens; float cfg; };
constexpr Case kCases[] = {
    {512,  200, 0,    1.0F}, // the certified route this plan must not regress
    {768,  200, 0,    1.0F},
    {768,  200, 4096, 1.0F}, // the review's case: same target, plus one reference
    {768,  200, 0,    3.5F}, // true CFG
    {1024, 200, 0,    1.0F},
};
```

Assert, per case: `image_device_bytes_peak() <= predicted_total`, and after the job `image_device_bytes_live() == 0`. Call `reset_image_device_peak()` before each case. Print predicted and measured per case so a calibration change is readable in the log.

- [ ] **Step 3: Run it where the artifact exists**

Build the registered target, then run it with the artifact path the existing gate tests use. If no artifact is reachable from this session, say so in the report and leave the constants at their starting values — the prediction is then verified only by the host model in Task 1, and Task 4's threshold inherits that uncertainty (do not claim the ledger qualified it).

- [ ] **Step 4: Calibrate, then commit**

Set `kImageWeightResidentBytes` and the `planes` fields to the smallest values that satisfy every case with headroom, and update the header comment to name the measurement it came from.

```bash
git add src/models/qwen_image/image_device_need.h tests/models/qwen_image/test_image_need_ledger_real.cu tests/cmake/ProductTests.cmake
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "test(image): qualify the device-memory need against the image ledger"
```

---

### Task 3: `Engine::probe_image_yield()` — answer before anything moves

**Files:**
- Modify: `include/infernix/types.h` (next to `ImageInterludeContext`, `:1565-1568`)
- Modify: `include/infernix/engine.h` (`:114-130`)
- Modify: `src/models/qwen4_exp/program/program.h` (`:619-640`), `program.cpp`, `program_impl.h` (`:5062-5160`)
- Modify: `src/runtime/engine/engine.cpp` (`:386-398`)
- Test: `tests/models/qwen4_exp/test_image_interlude_prod_real.cu`

**Interfaces:**
- Consumes: `plan_image_yield()` (`src/models/qwen4_exp/memory_plan.h:240`), the live pool snapshot the sizing path already builds, `DeviceContext`'s memory info.
- Produces: `infernix::ImageYieldProbe` and `Engine::probe_image_yield(std::uint64_t target_bytes, std::uint64_t margin_bytes) const -> ImageYieldProbe`, plus `Qwen4ExpInstance::program->probe_image_yield(...)`.

- [ ] **Step 1: Read the yield's own accounting**

Read `src/models/qwen4_exp/program/program_impl.h:5062-5160` (`yield_for_image`: the request it builds, the `before` measurement, the rollback) and grep for where `VramSnapshot` is constructed (`Grep "VramSnapshot" src/models/qwen4_exp`) so the probe reads the same snapshot the sizing path reads rather than a second view of the world.

- [ ] **Step 2: Add the public type and the Engine method**

In `include/infernix/types.h` beside `ImageInterludeContext`:

```cpp
// The answer to "would an image job of this size fit?" computed before anything is quiesced or
// unmapped: the pure yield plan over the live pools plus the device's own free bytes. `fits` holds
// when freed_bytes_expected + device_free_bytes covers target_bytes + margin_bytes + the display
// headroom the sizing path already reserves.
struct ImageYieldProbe {
    bool fits = false;
    std::uint64_t target_bytes = 0;
    std::uint64_t margin_bytes = 0;
    std::uint64_t freed_bytes_expected = 0; // plan_image_yield's total_freed, on the pools' own grids
    std::uint64_t device_free_bytes = 0;    // free before anything is unmapped
    std::uint64_t headroom_bytes = 0;       // the reserved display headroom
};
```

In `include/infernix/engine.h`, after `run_image_interlude` (`:128-130`), and extend the interlude comment block:

```cpp
    // Whether an image job wanting `target_bytes` (plus `margin_bytes` kept below the headroom) would
    // fit right now. Runs the same pure plan the interlude applies, against the live pools, without
    // quiescing, unmapping or resizing anything: a caller that gets `fits == false` refuses the job
    // before the submit door ever closes. Throws std::logic_error when the Engine is moved from or
    // its model supports no interlude.
    [[nodiscard]] ImageYieldProbe probe_image_yield(std::uint64_t target_bytes,
                                                   std::uint64_t margin_bytes) const;
```

In `src/runtime/engine/engine.cpp`, mirror `run_image_interlude` (`:390-398`) exactly — same moved-from check, same "supports no image interlude" message:

```cpp
infernix::ImageYieldProbe Engine::probe_image_yield(std::uint64_t target_bytes,
                                                   std::uint64_t margin_bytes) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->qwen4 == nullptr || impl_->qwen4->program == nullptr) {
        throw std::logic_error("Engine: this model's Program supports no image interlude");
    }
    return impl_->qwen4->program->probe_image_yield(target_bytes, margin_bytes);
}
```

- [ ] **Step 3: Implement it in the Program**

Declare in `program.h` next to `run_image_interlude`, and implement in `program.cpp`/`program_impl.h` as: take `vram_mutex_`, build the same `ImageYieldRequest` `yield_for_image` builds (`pool_bytes`, `frames`, strides/chunks, `frames_floor`, the KV mapping and floor, `lease_held`), call `plan_image_yield`, and fill the probe from `plan.total_freed` and the live free-bytes snapshot and the same headroom `display_headroom()` computes. It must move nothing: no `residency_->resize`, no KV shrink, no state transition, and it must not touch `interlude_mu_` (a probe during an in-flight interlude answers from the current pools). Do not call it from inside an interlude: the interlude already holds `vram_mutex_`.

- [ ] **Step 4: Extend the device test**

In `tests/models/qwen4_exp/test_image_interlude_prod_real.cu`, add a case that (a) probes a small need and expects `fits`, (b) probes an absurd need (say `1 << 40`) and expects `!fits`, and (c) asserts the probe left the pools where they were: the same `freed_bytes` a later real interlude reports, and a normal generation round after the probe still succeeds.

- [ ] **Step 5: Build and run what can run here**

`cmake --build build --config Release --target infernix_engine -j 8`, then build the interlude test target you extended and run `ctest --test-dir build -C Release -R image_interlude --output-on-failure`. Device cases report Skipped without an artifact — say so rather than calling them passed.

- [ ] **Step 6: Commit**

```bash
git add include/infernix/types.h include/infernix/engine.h src/models/qwen4_exp/program/ src/runtime/engine/engine.cpp tests/models/qwen4_exp/test_image_interlude_prod_real.cu
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "feat(engine): probe an image yield's fit before the LLM quiesces"
```

---

### Task 4: Serving decides before it blocks

**Files:**
- Modify: `src/serve/image_generation_service.h`, `src/serve/image_generation_service.cpp`
- Modify: `src/serve/image_text_tower.h`, `src/serve/image_text_tower.cpp`
- Modify: `apps/serve/main.cpp` (`:330-343`)
- Test: `tests/test_image_generation_service.cpp`, `tests/test_http_routes.cpp`

**Interfaces:**
- Consumes: `make_tower_prompt_renderer` (`src/serve/image_text_tower.h:32`), `TowerPrompt` (`:24-28`), `image_device_need` (Task 1), `Engine::probe_image_yield` (Task 3), `ImageGeometry::derive`.
- Produces: `PromptEmbedsProvider` taking `TowerPrompt`, `ImageInterludeRunner::probe(...)`, `ApiError` codes `image_vram_insufficient` (503) and the typed in-flight error.

- [ ] **Step 1: Write the failing service tests**

In `tests/test_image_generation_service.cpp`, following the file's existing fake-runner pattern:

1. The fake runner records `probe` calls. Assert a `generate()` for a size the fake says does not fit throws `ApiException` with `code == "image_vram_insufficient"`, status 503, **and that `run` was never called** (no yield, no blackout).
2. Assert the prompt and negative prompt are rendered (the fake renderer's counter rises) before the runner's `probe`, so the token counts the need uses are the counts the job runs with.
3. Assert the yielded target equals the computed need plus the new margin: the fake records `target_bytes`/`margin_bytes`, and `margin_bytes == 512 MiB`.
4. Fake a runner whose `run` reports a context with `freed_bytes` far below the need: the job must fail with `image_vram_insufficient` and the fake `job_executor_` must not have been called.
5. Keep every existing assertion green, including the progress 24/97/100 and stall cases (the fake executor signature changes with the provider, so update the fakes in the same commit).

- [ ] **Step 2: Run it and watch it fail**

`cmake --build build --config Release --target infernix_image_generation_service_test -j 8`
Expected: compile errors on the new seam (the plan's intended failure).

- [ ] **Step 3: Split the tower seam**

In `src/serve/image_text_tower.h`, change the provider's input and add nothing else — the renderer already exists:

```cpp
// The ids -> prompt_embeds half: needs device memory, so it runs inside the interlude. The caller
// renders (host work) first, which is what lets the service know the token count before the yield.
using PromptEmbedsProvider =
    std::function<std::vector<std::uint16_t>(const TowerPrompt&)>;
```

(`PromptEmbedsProvider` is declared in `image_generation_service.h`; move the typedef there and include `image_text_tower.h`'s `TowerPrompt` by declaring the struct in a small shared header or by moving `TowerPrompt` into `image_generation_service.h` — pick whichever the existing include graph already allows with no cycle, and delete the now-false sentence in `image_text_tower.h:1-10` about the provider being the whole route.)

`TowerProvider::operator()` (`src/serve/image_text_tower.cpp:108-113`) takes `const TowerPrompt&` and stops calling the renderer itself:

```cpp
    std::vector<std::uint16_t> operator()(const TowerPrompt& rendered) {
        std::lock_guard lock(mutex_);
        ensure_locked();
        return qi::run_text_tower(*reader_, tower_, embed_table_, rendered.ids, rendered.drop_rows);
    }
```

`make_tower_prompt_embeds_provider` keeps its name and returns the ids-taking provider; `apps/serve/main.cpp:330-343` builds `prompt_embeds` from it and `make_tower_prompt_renderer(processor_dir)` and hands both to the service.

- [ ] **Step 4: Add the seam the service needs**

In `src/serve/image_generation_service.h`:

```cpp
// Whether an image job of this need would fit without quiescing the LLM. The production runner asks
// the Engine; a fake answers from a field.
struct ImageYieldFit {
    bool fits = false;
    std::uint64_t freed_bytes_expected = 0;
    std::uint64_t device_free_bytes = 0;
    std::uint64_t headroom_bytes = 0;
};

class ImageInterludeRunner {
public:
    ...
    [[nodiscard]] virtual ImageYieldFit probe(std::uint64_t target_bytes, std::uint64_t margin_bytes) = 0;
};

// The renderer half of the tower seam: pure host work, safe before the yield.
using PromptRendererProvider = std::function<TowerPrompt(const std::string&)>;
```

`generate()`'s signature stays; the constructor takes the renderer as a fifth argument.

- [ ] **Step 5: Compute, probe, then decide**

Replace `image_yield_target` and `kInterludeMarginBytes` (`src/serve/image_generation_service.cpp:35-54`) with the margin, the need helper and a file-scope `constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;` (the old one was function-local and disappears with the formula):

```cpp
// The free VRAM the job keeps below what it may take: enough that a slightly mis-estimated job
// fails in its own allocation rather than poisoning the LLM's context.
constexpr std::uint64_t kInterludeMarginBytes = 512ULL * 1024ULL * 1024ULL;

std::uint64_t image_need_bytes(const qi::ImageGeometry& geometry, std::size_t text_tokens,
                              std::size_t negative_tokens, std::size_t ref_tokens,
                              const qi::DiTDims& dims) {
    qi::ImageDeviceNeedRequest req;
    req.geometry        = geometry;
    req.dims            = dims;
    req.text_tokens     = static_cast<std::uint32_t>(text_tokens);
    req.negative_tokens = static_cast<std::uint32_t>(negative_tokens);
    req.ref_tokens      = static_cast<std::uint32_t>(ref_tokens);
    req.blocks_resident = 1; // the streamed route holds one block at a time
    return qi::image_device_need(req).total();
}
```

and in `generate()`, after the reference is prepared host-side (`:352-365`) and before `store_.put` (`:367`):

```cpp
    // Rendering is host work: the token counts the need and the job must agree on exist before any
    // VRAM is yielded, so a request that cannot fit is refused without closing the submit door.
    const TowerPrompt prompt     = prompt_renderer_(request.prompt);
    const TowerPrompt neg_prompt = request.true_cfg_scale > kMinimumImageCFGScale
                                     ? prompt_renderer_(request.negative_prompt)
                                     : TowerPrompt{};
    // ReferencePixels carries the snapped pixel plane (src/serve/reference_image.h:17-21), so its
    // latent token count is known before the VAE encode that happens inside the yield.
    const std::uint32_t ref_tokens =
        reference.has_value()
            ? static_cast<std::uint32_t>((reference->height_px / qi::ImageGeometry::downscale()) *
                                         (reference->width_px / qi::ImageGeometry::downscale()))
            : 0U;
    const std::uint64_t need = image_need_bytes(qi::ImageGeometry::derive(request.height, request.width),
                                                prompt.ids.size() - prompt.drop_rows,
                                                neg_prompt.ids.size() - neg_prompt.drop_rows, ref_tokens,
                                                dit_dims_);
    const auto fit = runner_->probe(need, kInterludeMarginBytes);
    if (!fit.fits) { throw_vram_insufficient(need, fit); }
```

with one file-scope thrower both refusal sites use:

```cpp
[[noreturn]] void throw_vram_insufficient(std::uint64_t need, const ImageYieldFit& fit) {
    throw ApiException{ApiError{503, "server_error",
                                "image job needs " + std::to_string(need / kMiB) +
                                    " MiB; the model can yield " +
                                    std::to_string(fit.freed_bytes_expected / kMiB) + " MiB with " +
                                    std::to_string(fit.device_free_bytes / kMiB) + " MiB free",
                                "size", "image_vram_insufficient"}};
}
```

`dit_dims_` is the DiT's dimensions as seen without the artifact: the service takes them from the executor's own lazy binding if a cheap host-side accessor exists, and otherwise from a constant `qi::DiTDims` the image module already publishes for the shipped recipe. Read `src/models/qwen_image/image_weights.cpp:99-115` and pick whichever needs no artifact read on the request path; say which you picked.

Then pass `need` to the runner and guard inside the lambda, before `job_executor_`:

```cpp
        runner_->run(need, kInterludeMarginBytes, [&](const infernix::ImageInterludeContext& context) -> bool {
            // What the lease actually unmapped, against the same number the probe was asked about.
            if (fit.freed_bytes_expected + fit.device_free_bytes <
                need + kInterludeMarginBytes + fit.headroom_bytes) { ...same 503... }
```

A refusal after the yield must not mark the job `Failed` for a reason the client can retry as a 429 — it takes the `job_started == false` path already in `:449-452`; extend that block so `image_vram_insufficient` keeps its own code instead of being renamed `image_interlude_failed`.

- [ ] **Step 6: Typed in-flight error**

In the runner adapter (`:70-87`), catch the Program's "another interlude is already in flight" and rethrow `infernix::serve::ImageInterludeInFlight{}` (a new `struct ImageInterludeInFlight : std::runtime_error` in `image_generation_service.h`). In `generate()`'s handler, replace `message.find("already in flight")` (`:427`) with `dynamic_cast<const ImageInterludeInFlight*>(&error) != nullptr`. Keep the 429 wire shape identical.

- [ ] **Step 7: Build, run, and keep the neighbours green**

`cmake --build build --config Release --target infernix_image_generation_service_test infernix_chat_image_tool_test infernix_http_routes_test infernix_openai_images_schema_test -j 8`
`ctest --test-dir build -C Release -R "image|http_routes|serve_options" --output-on-failure`
Expected: all pass. (Run each pattern separately if the shell rejects a quoted `|`.)

- [ ] **Step 8: Commit**

```bash
git add src/serve/ apps/serve/main.cpp tests/test_image_generation_service.cpp tests/test_http_routes.cpp
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "feat(serve): admit image jobs from a probed yield instead of a scaled guess"
```

---

### Task 5: A refused picture is a tool result, not a dead chat

**Files:**
- Modify: `src/serve/chat_image_tool.cpp` (`:110-140`), `chat_image_tool.h`
- Test: `tests/test_chat_image_tool.cpp`

**Interfaces:**
- Consumes: `ApiError.code` values `image_vram_insufficient`, `image_interlude_in_flight`, `image_generation_failed`, `image_engine_degraded`.
- Produces: an error tool result the model sees, with the reason and, for a memory refusal, the largest side that fits.

- [ ] **Step 1: Write the failing test**

In `tests/test_chat_image_tool.cpp`: a service fake that throws `ApiException{ApiError{503,...,"image_vram_insufficient"}}` must produce (a) a completed chat response, not a thrown HTTP error, (b) a tool result whose content contains `image_vram_insufficient`, and (c) no image part in the assistant turn. Add a second case for `image_interlude_in_flight` and one for `image_generation_failed`.

- [ ] **Step 2: Run it, watch it fail** on the current `chat_image_tool.cpp:124-126` rethrow.

- [ ] **Step 3: Convert the failures**

Turn every non-400 image failure inside a hidden round into the tool-result error path the file already uses for a bad argument (read `:80-140` for that shape first, do not invent a new one):

```cpp
        } catch (const ApiException& error) {
            // A picture that could not be drawn is the model's problem to solve, not the user's
            // request to lose: the reason (and the size that would have fit) goes back as the tool
            // result and the rest of the answer still streams.
            return error_tool_result(call_id, error.error.code + ": " + error.error.message);
        }
```

`error_tool_result` is the name of the existing helper if one exists; otherwise add it beside the existing success-result builder in the same style. Keep 400s (`image_invalid`) on their current path.

- [ ] **Step 4: Run the tests**

`ctest --test-dir build -C Release -R chat_image_tool --output-on-failure`
Expected: pass, including the existing invisibility assertions.

- [ ] **Step 5: Commit**

```bash
git add src/serve/chat_image_tool.cpp src/serve/chat_image_tool.h tests/test_chat_image_tool.cpp
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "fix(serve): report a refused image as a tool result instead of failing the request"
```

---

### Task 6: A soft KV remap failure must not park the Program

**Files:**
- Modify: `src/models/qwen4_exp/program/program_impl.h:5197-5211`
- Test: `tests/models/qwen4_exp/test_image_interlude_failures_real.cu`

- [ ] **Step 1: Write the failing device test**: inject a restore whose `map_kv` cannot satisfy the full remap (the file already has an injection seam for a restore fault — read `:150-200` and reuse it), then assert the interlude's job failure propagates, the Engine is still `is_available()`, and a later generation round succeeds. Today this path reaches `set_backed_pages`' `std::logic_error` and ends in Degraded.

- [ ] **Step 2: Run it and confirm the current Degraded outcome.**

- [ ] **Step 3: Keep the limit where the yield left it**

```cpp
            // A remap the driver cannot satisfy leaves the KV pool at its yielded size rather than
            // claiming pages that are not backed: set_backed_pages would throw for exactly that, and
            // the cost is a smaller elastic pool until the sizing path grows it back.
            if (map_kv(lease.kv_pages_after, lease.kv_pages_before)) {
                if (pool_->backed_pages() < lease.kv_pages_before) {
                    pool_->set_backed_pages(lease.kv_pages_before);
                }
            } else {
                restored_full = false; // the caller logs the byte shortfall; the door reopens anyway
            }
```

Thread `restored_full` out to the interlude's diagnostic line (`program_impl.h:5151` pattern) so the shortfall is visible in the log, and confirm nothing downstream assumes `backed_pages == lease.kv_pages_before` (`Grep "kv_pages_before" src/models/qwen4_exp`).

- [ ] **Step 4: Run the failures test and the prod test**

`ctest --test-dir build -C Release -R image_interlude --output-on-failure`
Expected: pass where runnable, Skipped without an artifact — report which.

- [ ] **Step 5: Commit**

```bash
git add src/models/qwen4_exp/program/program_impl.h tests/models/qwen4_exp/test_image_interlude_failures_real.cu
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "fix(engine): keep the KV limit where a failed restore remap left it"
```

---

### Task 7: Contracts and comments that match the code

**Files:**
- Modify: `docs/serving.md` (`:985-990`, `:1049-1061`, the capabilities rows near `:1099-1100`)
- Modify: `src/models/qwen4_exp/program/program.cpp:158-159`, `program.h:619-622`, `program_impl.h:5519`
- Test: `tests/test_serve_options.cpp` or the capabilities test that already asserts the error/code inventory

- [ ] **Step 1: Replace the sizing sentence** in `docs/serving.md:987-990`: the yielded amount is computed from the request's geometry, its reference image and true CFG, and a request that cannot fit is refused with `image_vram_insufficient` **before the model yields anything**.

- [ ] **Step 2: Add the 503 row** to the error table beside `:1057`: `| 503 | image_vram_insufficient | the requested size, reference or CFG pass needs more VRAM than the model can yield now; retry smaller |`.

- [ ] **Step 3: Delete the FIFO claims** in the three comments found in Task 3/Step 1, leaving each comment stating the single-flight gate and the concurrent caller's answer. Keep `docs/serving.md`'s existing 429 row and "Only one image job runs at a time" — they are already true.

- [ ] **Step 4: State the chat behavior** in the chat-auto-image section (`:1122-1146`): a refused or failed picture returns the reason to the model as a tool result and the answer continues.

- [ ] **Step 5: Check the documentation contract**: `git diff --check` plus whatever capabilities/schema test asserts the code inventory (grep the tests for `image_interlude_in_flight` and extend it with `image_vram_insufficient` in the same commit).

- [ ] **Step 6: Commit**

```bash
git add docs/serving.md src/models/qwen4_exp/program/ tests/
git -c user.name="Ian Ranson" -c user.email="ian.ranson@users.noreply.github.com" commit -m "docs(serve): document the computed image yield and the preflight refusal"
```

---

### Task 8: Whole-tree verification

- [ ] Full build: `cmake --build build --config Release -j 8` (this does **not** relink the deployed `infernix-serve.exe` copy on the Desktop; report if the build tries to).
- [ ] Host suite: `ctest --test-dir build -C Release -R infernix_ --output-on-failure`, reporting passed/skipped counts and naming every skip that hides a claim this plan made (the ledger qualification in particular).
- [ ] Confirm the generated picture is unchanged: run the certified 512x512 route's existing comparison (the test named in `src/models/qwen_image/image_pipeline.h:5-10`) and report its verdict.
- [ ] Report: the need at 512/768/768+ref/1024, what the probe said for each, and — if the artifact was unreachable — that the ledger never qualified the model and the refusal threshold therefore rests on the computed inventory alone.
