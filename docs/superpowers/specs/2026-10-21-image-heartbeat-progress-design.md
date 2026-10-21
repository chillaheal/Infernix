# Image interlude heartbeat, progress, and console output ownership

Date: 2026-10-21
Status: approved by user in chat (ping first, progress next, console fix included)

## Problem

1. A chat request that triggers a server-executed image round (`#83` chat auto-image) goes silent
   on the wire for the whole interlude (30-90+ s for large pictures). The SSE heartbeat
   (`SseTransport::poll`, `: keep-alive\n\n` every 5 s) is only pumped from the generation loop's
   `is_cancelled` checks; while `make_picture` runs, no caller exists. Claude Code (and any
   read-timeout client) therefore drops the request even though generation is healthy.
   The design note in `chat_image_tool.h` ("the SSE heartbeat covers the silence while a picture
   is drawn") is not true of the implementation.
2. The client cannot see that the picture is progressing (no percentage anywhere), so long jobs
   are indistinguishable from hangs.
3. The image job's per-stage memory log writes raw `fprintf(stderr, "image job: ...")` from
   `run_image_job` while the `--log-stats-panel` console panel (`TerminalProgress`, which owns
   the terminal's bottom rows) is mid-repaint without a trailing newline: the message lands at
   the end of the panel's last row ("...ng rnds ...image job: pre-loop resident ..."). Raw
   stderr writes must not bypass the terminal owner.

## Design

### 1. Heartbeat pump during image rounds (phase 1)

- `SseTransport` gains internal thread safety: a `std::mutex` serializing `sink_.write` and
  `last_write_` across the engine-loop thread and the ticker thread; `beat(now)` = the locked
  heartbeat write; `poll()` keeps its public contract (cancel check + interval).
- New RAII `HeartbeatTicker` (in `http_transport.h/cpp`): holds `SseTransport&`; one worker
  thread waking every `kHeartbeatInterval` (or until destruction, via condvar) calling
  `transport.beat(Clock::now())`. Construction/destruction only around the image rounds in the
  three streaming handlers (`openai_chat_http.cpp`, `openai_responses_http.cpp`,
  `anthropic_messages_http.cpp`), i.e. wrapping `run_chat_image_rounds` while the picture is
  drawn.
- Cancellation semantics: the ticker's heartbeat write fails on a dead peer ->
  `cancelled_` (already atomic) -> `transport.poll()` used as `is_cancelled`/prepare predicate
  ends the rounds; the ticker sees `cancelled_` and exits. No new cancellation paths.
- Non-streaming aggregate requests are unchanged (they never had SSE; TCP keepalive already set).

### 2. Real progress (phase 2)

- `run_image_job(..., const ImageJobProgress* on_progress = nullptr)` — optional callback hook;
  the certified route passes nothing and is bit-identical. Callback events: `step(i, n)` after
  every denoise step, `stage("vae")`, `stage("png")`; total units = `steps + 2`.
- `ArtifactImageJobExecutor` (serve) forwards the callback into an `std::atomic<int> percent`
  owned by the in-flight image round; `ImageJobStore` job entries gain `progress_percent`
  (-1 before the hook exists, 0-100 while running), updated on the job thread.
- Wire text: the ticker emits `: image <pct>%\n\n` (falling back to `: keep-alive\n\n` when no
  percent is known) — SSE comments are guaranteed inert for every SDK parser; unregistered SSE
  *events* are not, so they are not used.
- `GET /v1/images/<id>` (existing artifact route) additionally reports `status: "running"` +
  `progress_percent` for in-flight jobs (the job is already stored at accept time).
- Server-side liveness: the ticker is the watchdog — if neither `percent` nor
  `last_progress_time` (both on the job's shared state) has advanced for
  `kImageJobStallSeconds` (60 s), it sets the job's `stalled` flag; the job's progress callback
  checks the flag at its next step and throws, so the interlude unwinds and fails the job
  through the existing `image_interlude_failed` path instead of a wedged pipeline holding the
  yield forever.

### 3. Console output ownership (bundled fix)

- Replace the `fprintf(stderr, ...)` `log_mem` lambda in `image_pipeline.cu` with logging
  through the application spdlog logger, obtained via an optional logger parameter on
  `run_image_job` (same default-nullptr shape as the progress hook; the serve route passes the
  operational logger it already owns). `TerminalProgress`/startup-log already serialize spdlog
  writes against panel repaints, so the message lands on its own line above the panel.
- No other raw stderr writers are in scope; `--log-stats-panel` behavior otherwise unchanged.

## Error handling

- Ticker thread must never outlive the handler's `SseTransport`: RAII, joined in the destructor
  on every path (normal, `ClientDisconnected`, `ApiException`).
- A heartbeat write racing token writes is ordered by the transport mutex; a failed write marks
  cancelled exactly as `poll()` does today.
- Progress callback failures: none are possible (it stores an integer); the stall watchdog is
  the only new failure it can trigger.

## Verification

- Host tests (`tests/`): fake slow image job behind the existing `ImageInterludeRunner` seam —
  assert heartbeats appear at the interval during the job, percent advances monotonically, a
  stalled fake fails the job at the watchdog bound, and the ticker stops on cancel.
- Schema/route test: `/v1/images/<id>` shape with a running job (status + progress_percent).
- Terminal: run the live server with the stats panel on, generate a picture, confirm the
  "image job" lines print as own lines above the panel.
- Live A/B: a map prompt (1024x1024, >60 s) from a streaming client that previously timed out
  must now complete.

## Explicitly not in scope

- Non-streaming request timeouts (no SSE to pump; client-side concern).
- Progress for the text-tower prompt-embed phase (short, pre-yield; stays silent).
- Any change to the images-route OpenAI contract beyond the additive status fields.
