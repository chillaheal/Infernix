# Image Interlude Heartbeat, Progress & Console Ownership — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep streaming chat requests alive through a server-executed image round (heartbeat every 5 s), report real generation progress (`: image 45%` SSE comments + `progress_percent` on the job route), watch dog-stalled jobs, and stop the image job's raw stderr lines from landing inside the stats panel.

**Architecture:** A ticker thread pumps a thread-safe `SseTransport` while `run_chat_image_rounds` blocks; the DiT job's natural step loop feeds an `ImageJobProgress` shared state that rides on the stored job record; the same state is the watchdog's witness and the route's status source.

**Tech Stack:** C++20, cpp-httplib (`httplib::DataSink`), spdlog (application logger), CUDA (qwen_image pipeline), GoogleTest-style host tests in `tests/`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-21-image-heartbeat-progress-design.md` (branch `image-heartbeat-progress`).
- The certified image route must be bit-identical when the new optional hooks are `nullptr`.
- Progress reaches the wire ONLY as SSE comment lines (`: ...\n\n`) — never unregistered SSE events.
- No new server options; intervals fixed: heartbeat 5 s (`SseTransport::kHeartbeatInterval`), stall 60 s.
- Build: `cmake --build build-windows -j` from repo root. Host tests: `ctest --test-dir build-windows -C Release -R <pattern>`.
- Commits: Conventional Commit subjects; identity inline:
  `git -c user.name='Ian Ranson' -c user.email='ian.ranson@users.noreply.github.com' commit -m "..."`
  (never edit git config). ASCII-only source files.

---

### Task 1: Thread-safe `SseTransport` + `HeartbeatTicker`

**Files:**
- Modify: `src/serve/http_transport.h` (class `SseTransport` lines 38-64; add ticker class after it)
- Modify: `src/serve/http_transport.cpp` (`SseTransport` impl lines 166-201)
- Test: `tests/test_http_transport.cpp` (extend)

**Interfaces:**
- Produces: `void SseTransport::beat(Clock::time_point now = Clock::now())`;
  `class HeartbeatTicker { HeartbeatTicker(SseTransport&, std::function<std::string()> next_comment); ~HeartbeatTicker(); }` — destructor joins.
- Consumes (test-side): existing `SseTransport` ctor `(httplib::DataSink&, std::atomic<bool>&, Clock::duration, Clock::time_point)`. Host tests register via `infernix_add_test` in `tests/cmake/ProductTests.cmake` (executables `infernix_http_transport_test`, `infernix_image_generation_service_test`, ...).

- [ ] **Step 1: Failing tests** in `tests/test_http_transport.cpp`:

```cpp
TEST(SseTransportBeat, WritesCommentOncePerInterval) {
    std::string out;
    httplib::DataSink sink;
    sink.write = [&out](const char* d, size_t n) { out.append(d, n); return true; };
    sink.is_writable = [] { return true; };
    std::atomic<bool> cancelled{false};
    SseTransport::Clock::time_point t{};
    SseTransport transport(sink, cancelled, std::chrono::seconds(5), t);
    EXPECT_FALSE(transport.poll(t));                       // first beat: now-last_write_==0 -> silent
    transport.beat(t + std::chrono::seconds(5));           // exactly one comment
    EXPECT_EQ(out, ": keep-alive\n\n");
}

TEST(HeartbeatTicker, BeatsWhileBlockedAndStopsOnDestruction) {
    // Fake sink whose write counts comments; run rounds blocked on a promise; ticker at 20 ms;
    // after >=3 counted beats set the promise, join, expect beats > 0 and exactly one join (no crash).
}
```

- [ ] **Step 2: Run, expect FAIL** (`HeartbeatTicker`/`beat` do not exist):
  `cmake --build build-windows -j --target infernix_http_transport_test`.

- [ ] **Step 3: Implement.** In `SseTransport`: add `#include <mutex> <condition_variable> <functional> <thread>`; member `std::mutex write_mu_;`; take it in `write(...)`, `poll(...)` and the new:

```cpp
void SseTransport::beat(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) return;
    const std::lock_guard<std::mutex> lock(write_mu_);
    if (now - last_write_ < heartbeat_interval_) return;
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) mark_cancelled();
    else last_write_ = now;
}
```

(`write` and `poll` lock the same mutex; keep their current bodies inside the lock.)
New class in `http_transport.h`:

```cpp
class HeartbeatTicker final {
public:
    HeartbeatTicker(SseTransport& transport, std::function<std::string()> next_comment);
    ~HeartbeatTicker();
    HeartbeatTicker(const HeartbeatTicker&) = delete;
    HeartbeatTicker& operator=(const HeartbeatTicker&) = delete;
private:
    void run();
    SseTransport& transport_;
    std::function<std::string()> next_comment_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::thread thread_;
};
```

`run()`: wait `mu_`+`cv_` for `SseTransport::kHeartbeatInterval` or `stopping_`; call
`next_comment_()` — if it returns a non-empty string, `transport_.write(comment)` in a try/catch
(`ClientDisconnected` -> return); else `transport_.beat(now)`. Dtor: set `stopping_`, notify, join.

- [ ] **Step 4: Run, expect PASS.** Also run the whole existing pattern:
  `ctest --test-dir build-windows -C Release -R http_transport` -> all green.
- [ ] **Step 5: Commit** `feat(serve): thread-safe SSE transport with a heartbeat ticker for image rounds`

---

### Task 2: Wrap `run_chat_image_rounds` with the ticker in the three streaming handlers

**Files:**
- Modify: `src/serve/anthropic_messages_http.cpp:246-256` (streaming image-rounds block)
- Modify: `src/serve/openai_chat_http.cpp` (same block; locate with `grep -n run_chat_image_rounds`)
- Modify: `src/serve/openai_responses_http.cpp` (same)
- Test: `tests/test_image_generation_service.cpp` stays green; behavior is covered live (Task 7).

**Interfaces:**
- Consumes: `HeartbeatTicker` (Task 1); `ImageGenerationService::last_job_progress()` (Task 4 — until then pass a null accessor).
- Produces: during any chat-executed image round, the request's SSE stream carries a comment at least every 5 s.

- [ ] **Step 1:** In each streaming handler, replace

```cpp
ChatImageRounds rounds = run_chat_image_rounds(...);
```

with (identical shape in all three):

```cpp
std::shared_ptr<ImageJobProgress> progress;  // serve/image_job_progress.h (Task 4)
auto progress_accessor = images_configured ? last_job_progress : std::function<std::shared_ptr<ImageJobProgress>()>{};
HeartbeatTicker ticker(transport, [&] -> std::string {
    if (!progress_accessor) return {};
    progress = progress_accessor();
    if (!progress) return {};
    const int pct = progress->percent.load(std::memory_order_relaxed);
    if (pct < 0) return {};
    return ": image " + std::to_string(pct) + "%\n\n";
});
ChatImageRounds rounds = run_chat_image_rounds(...);  // unchanged call
```

`last_job_progress` comes from the handler's `generation_service_` (new accessor, Task 5); until
Task 5 lands, wire the accessor as `nullptr` in a first commit so this task compiles standalone.
- [ ] **Step 2:** Build `cmake --build build-windows -j` — green.
- [ ] **Step 3:** `ctest --test-dir build-windows -C Release -R "chat_image_tool|image_generation_service|http_routes|anthropic_schema|openai_schema"` — green.
- [ ] **Step 4: Commit** `feat(serve): heartbeat the client through server-executed image rounds`

---

### Task 3: `ImageJobProgress` shared state + `run_image_job` hooks (steps + logger)

**Files:**
- Create: `src/serve/image_job_progress.h`
- Modify: `src/models/qwen_image/image_pipeline.h:105` (`run_image_job` signature)
- Modify: `src/models/qwen_image/image_pipeline.cu` (`log_mem` 281-285; loop 524-575; "DiT loop done" 575; VAE decode ~601)
- Test: `tests/test_image_job_store.cpp` (state tests) — no CUDA needed for the struct.

**Interfaces:**
- Produces:

```cpp
// src/serve/image_job_progress.h
namespace infernix::serve {
struct ImageJobProgress {
    std::atomic<int> percent{-1};                    // -1 unknown, else 0-100
    std::atomic<std::chrono::steady_clock::rep> last_progress{
        std::chrono::steady_clock::now().time_since_epoch().count()};
    std::atomic<bool> stalled{false};
    bool advance(int pct) {                          // false when stalled (caller throws)
        percent.store(pct, std::memory_order_relaxed);
        last_progress.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                            std::memory_order_relaxed);
        return !stalled.load(std::memory_order_relaxed);
    }
};
} // namespace infernix::serve
```

- Produces (models): `run_image_job(reader, dit, vae, inputs, stream,
  const ImageJobProgressView* progress = nullptr, std::function<void(const std::string&)> log_line = {})`
  where `struct ImageJobProgressView { virtual bool step(std::size_t i, std::size_t n) = 0; };`
  lives in `image_pipeline.h` (models must not include serve headers); the serve executor adapts it to `ImageJobProgress::advance`.

- [ ] **Step 1:** Failing test in `tests/test_image_job_store.cpp` (or a new small test file reusing an existing registration): `advance` stores percent/last_progress and returns false exactly when `stalled` is set.
- [ ] **Step 2:** Build target, expect FAIL; implement the header; expect PASS.
- [ ] **Step 3:** In `image_pipeline.cu`: replace the `log_mem` lambda body with `if (log_line) log_line(...)` keeping the exact message text; after each step in the loop at 525: `if (progress && !progress->step(i + 1, n_steps)) throw std::runtime_error("image job: stalled by watchdog");`; after the VAE decode block (before return, near 575+): `if (progress && !progress->step(n_steps + 1, n_steps + 2)) throw ...;`. The serve layer adds the final `+2` unit after PNG encode. `log_mem("pre-loop resident")` -> `if (log_line) log_mem2(...)` (same string).
- [ ] **Step 4:** Build; run `-R "image_job_store|image_generation_service"` green (existing `run_image_job` callers/tests pass defaults unchanged).
- [ ] **Step 5: Commit** `feat(image): optional progress and logging hooks on the certified image job`

---

### Task 4: Carry progress on the stored job; executor plumbing

**Files:**
- Modify: `src/serve/image_job_store.h` (`StoredImageJob` + `#include "serve/image_job_progress.h"`, add `std::shared_ptr<ImageJobProgress> progress;`)
- Modify: `src/serve/image_generation_service.h` (`ImageJobExecutor` alias gains `const std::shared_ptr<ImageJobProgress>& progress` parameter; new `[[nodiscard]] std::shared_ptr<ImageJobProgress> last_job_progress() const;` with `std::shared_ptr<ImageJobProgress> last_progress_;` + mutex)
- Modify: `src/serve/image_generation_service.cpp` (`generate`: create progress, put into record, set `last_progress_`, pass to executor; `ArtifactImageJobExecutor::operator()` forwards an adapter view + logger `[](const std::string& s){ INFERRIX_LOG_INFO("{}", s); }` — use the logger accessor already available in this TU (grep `logger` in `src/serve`); after PNG encode: `progress->advance(100);`)
- Modify: `src/serve/openai_images_http.cpp` / any test executor for the widened `ImageJobExecutor` signature (`tests/test_image_generation_service.cpp`, `tests/test_chat_image_tool.cpp` fake lambdas); `apps/serve/main.cpp` unchanged here (the logger lands on the factory in Task 6).

**Interfaces:**
- Consumes: `ImageJobProgress` (Task 3).
- Produces: `ImageGenerationService::last_job_progress()` -> `std::shared_ptr<ImageJobProgress>` (the most recent accepted job's state; shared_ptr copy under mutex).

- [ ] **Step 1:** Extend `tests/test_image_generation_service.cpp`: fake executor advances `progress->percent` (25,50,75,100); assert `last_job_progress()->percent == 100` after a successful `generate`, and the stored record's `progress->percent` matches during a fake stall (stalled set -> job fails with the watchdog message).
- [ ] **Step 2:** Build, expect FAIL (missing param/accessor); implement; expect PASS.
- [ ] **Step 3:** `ctest --test-dir build-windows -C Release -R "image_generation_service|image_job_store|openai_images_schema"` green.
- [ ] **Step 4: Commit** `feat(serve): carry image job progress on the stored job record`

---

### Task 5: Job status on `GET /v1/images/<id>` + watchdog in the ticker

**Files:**
- Modify: `src/serve/http_server.cpp` (`handle_image_artifact` ~605-640: for a job in `Running`/`WaitingForFrontier`/`Restoring` state, return `200` JSON `{"id":..., "status":"running", "progress_percent":N}` instead of the artifact)
- Modify: `src/serve/http_transport.h/.cpp` (`HeartbeatTicker` gains stall check: every beat, if a provided `std::shared_ptr<ImageJobProgress>` has `now - last_progress > 60 s`, set `stalled`; accessor lambda passed by handlers reuses Task 2's `progress_accessor`)
- Modify: three streaming handlers (pass progress into ticker's stall check — same accessor already wired)
- Test: `tests/test_image_job_store.cpp` + `tests/test_http_transport.cpp` (fake progress frozen -> ticker sets `stalled` at the fake-clock bound).

**Interfaces:**
- Consumes: `StoredImageJob::progress`, `HeartbeatTicker` (Tasks 1,4).
- Produces: `HeartbeatTicker(SseTransport&, std::function<std::string()> next_comment, std::function<std::shared_ptr<ImageJobProgress>()> watch = {})`; route status object for non-terminal jobs.

- [ ] **Step 1:** Failing tests: ticker sets `stalled` when `last_progress` is older than the stall bound (inject bound+clock as ctor params defaulting to 60 s/real clock for testability); store/route test for the running-status JSON shape.
- [ ] **Step 2:** Implement; both green.
- [ ] **Step 3:** `ctest -R "http_routes|image_job_store|http_transport"` green.
- [ ] **Step 4: Commit** `feat(serve): image job status route and stall watchdog`

---

### Task 6: Console ownership — route pipeline logs through the application logger

**Files:**
- Modify: `src/serve/image_generation_service.cpp` (`ArtifactImageJobExecutor` passes `log_line` into `run_image_job` — app logger, e.g. `generation_service`'s spdlog logger already reachable at serve wiring; grep `logger()` in `src/serve/http_server.cpp` for the owner)
- Test: manual/terminal (below) + existing build.

- [ ] **Step 1:** Replace the `fprintf(stderr, ...)` call sites' data path: in `ArtifactImageJobExecutor::operator()`, pass

```cpp
[log = logger_](const std::string& line) { if (log) log->info("{}", line); }
```

(`logger_` threaded into the executor factory `make_artifact_image_job_executor(path, logger)`;
update its single production call site in `apps/serve/main.cpp` — the application logger there is
the `std::shared_ptr<spdlog::logger>` also passed to `HttpServer`'s ctor at
`src/serve/http_server.cpp:206`.)
- [ ] **Step 2:** Build green; `ctest -R image_generation_service` green.
- [ ] **Step 3:** Live terminal check (visible window, restart via `.agent/_restart_serve.ps1` pattern): generate a 512x512 picture; confirm `image job: ...` lines appear as own lines above the `--log-stats-panel` rows, never concatenated onto `...ng rnds` line.
- [ ] **Step 4: Commit** `fix(image): log image job stages through the application logger, not raw stderr`

---

### Task 7: Docs, live A/B, final gate

**Files:**
- Modify: `docs/serving.md` (OpenAI Images section: `/v1/images/<id>` running-status object; note heartbeat comments `: image N%` during chat image rounds)
- Modify: `docs/README.md`/`docs/cli.md` only if they enumerate route response shapes for images (check `grep -n "images/" docs`).

- [ ] **Step 1:** Docs edits matching implemented shapes.
- [ ] **Step 2:** Full host-test gate:
  `ctest --test-dir build-windows -C Release -R "openai_schema|http_routes|serve_options|chat_image_tool|image_generation_service|image_job_store|openai_images_schema|media_encode|linear_noalias|anthropic_schema|http_transport"` — all pass.
- [ ] **Step 3:** Deploy to Desktop (`_swap_merge_serve.ps1` + `_restart_serve.ps1` pattern, visible windows; user's session pauses).
- [ ] **Step 4:** Live A/B: from a streaming client, request a 1024x1024 map-style prompt (>60 s); assert the HTTP 200 completes with a URL and no client timeout, and `requests.jsonl` shows the round. Screenshot/record terminal for the log-line fix.
- [ ] **Step 5:** Commit `docs(serve): document image job status and heartbeat behavior` and push branch: `git push fork image-heartbeat-progress`.
