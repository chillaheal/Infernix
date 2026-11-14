# Image admission refuses on a probe that cannot see the yield it would take

Date: 2026-11-05. Component: `src/serve/image_generation_service.cpp`, `src/models/qwen4_exp/program/`,
`src/runtime/expert_offload/`. Status: approved by the user for phase 1.

## The measured failure

The art pass of a chat session asked for pictures and every one was refused in 1-32 ms with
`503 image_vram_insufficient`. From `requests.jsonl` of instance `...66735038` (`image-edit/_p37_scan_vram.py`,
`_p37_timeline.py`, `_p37_net_check.py`): 18 refusals, 26 successes, all refusals

```
the image job needs 3701-3710 MiB; the model can yield 4224 MiB with 363 MiB already free and
512 MiB reserved below the headroom
```

so the promise plus free (4587 MiB) fell 138 MiB short of the gate 3701 + 512 + 512 = 4725. The refusals
coincide with chat load (`scheduler.running` 2, then 1) and no draw in flight; the same 768x768 at 40 steps
drew in 44 s with the chat quiet, and the same probe line promised 20992 MiB on 2026-11-03. Correlation 86
burned five refusals in 15 s while the model shrank its own canvas.

Two things are wrong, and they are independent.

**The probe is blind exactly when it is needed.** `probe_image_yield` (`program_impl.h`) builds the same
`ImageYieldRequest` the interlude applies, with `req.lease_held = residency_->lendable_bytes() == 0 &&
residency_->frames() != 0`. While the chat holds a residency lease, `plan_image_yield`
(`memory_plan.cpp:178`) sets `frames_movable = false` and the whole frame pool is kept -- so
`freed_bytes_expected` is KV-only, and the gate is compared against a yield the job would never ask for.
The mechanism that takes the ring back, `begin_image_quiesce` (`program_impl.h:6373-6428`), is only reached
inside `run`, which the refusal prevents. The serve refuses jobs the interlude would have drawn.

**The refusal advises the one thing that cannot help.** Below roughly 1024 the admission need is dominated
by a canvas-independent resident term (`kImageWeightResidentBytes` and the resident block slots): 768x768 and
640x640 both need 3701 MiB. `http_server.cpp:796-800` nonetheless tells every client that
`image_vram_insufficient` means "a smaller size is the fix", which is what the model did -- five times.

## What is already right (do not re-build it)

- The interlude is already single-flight per device: `Program::run_image_interlude` takes
  `interlude_mu_.try_lock()` and answers "another interlude is already in flight", which the serve
  translates to the type `ImageInterludeInFlight` (`image_generation_service.h:86`) and the HTTP layer to
  `429 image_interlude_in_flight`. A stampede of image jobs cannot quiesce side by side.
- The net inside the interlude already decides on delivered bytes: `available = context.freed_bytes +
  context.device_free_bytes` (`image_generation_service.cpp:1266`), re-walking the canvas and refusing only
  when the floor is unpayable, naming the delivered numbers.
- **Checked, not assumed: the two terms do not overlap.** `device_free_bytes` is read after the pools move
  (`program_impl.h:6131-6135`) and is `cudaMemGetInfo` free or budget-less-usage (`core/vram_budget.cpp:162-195`),
  so an open question was whether the delivered yield is counted twice. The logged post-yield refusals say
  no: a 768x768 refused with `yield=7552 free=0`, and the 2016x2016 case the code quotes carried
  `yield=14400 free=479`. Free never rises with the delivered yield, and those canvases do draw at those
  numbers. The net's ceiling stands; this change does not touch it.

## The change

Four edits, one seam field, no numeric change to any certified number.

### A. The probe says why it said no

`ImageYieldProbe` (`include/infernix/types.h:1903`) gains

```cpp
bool expert_pool_frozen_by_lease = false;
```

set in `probe_image_yield` as `req.lease_held && plan.frames_freed == 0 && residency_->any_lease_out()`.
`any_lease_out()` is new on `ExpertResidency` and answers `pool.controller->loaned_frames() != 0` for any
pool -- the fact the existing `lease_held` expression conflates with the quarter reserve floor
(`pool_lendable` is zero both when a lease is out and when only the never-lent quarter is left). The three
conditions together mean: the planner kept the frame pool *because the LLM is holding it*, and therefore the
promise this probe reports is not the frontier this job would get.

`fits` and every byte field stay bit-identical: this is an explanation, not a verdict.

`ImageInterludeRunner::Fit` (`image_generation_service.h:157`) gains the same field and
`EngineInterludeRunner::probe` copies it, so the host-seam test can drive it without an Engine.

### B. A blind no does not shrink the canvas

In the walk (`image_generation_service.cpp:1192-1219`), a probe that answers `!fits` **and**
`expert_pool_frozen_by_lease` ends the walk at the asked canvas: submit the job, let `run` quiesce, let the
delivered-bytes net decide. Shrinking is meaningless here -- the deficit was measured against a yield the
job never asked for. A `!fits` without the bit stays the honest no it is today: walk to the floor, keep the
largest canvas that cleared the gate, otherwise `throw_vram_insufficient`.

The net already refuses a canvas the card cannot pay for, and it refuses *before the first allocation*, so
the LLM restores through the ordinary failed-job path. Handing it the decision moves no risk into a kernel.

### C. A busy card is not a broken server, and the advice stops lying

- `Program::run_image_interlude` throws "no yieldable LLM frontier within the deadline" when
  `begin_image_quiesce` cleanly unwinds (`program.cpp:176-179`). Only "already in flight" is translated today
  (`image_generation_service.cpp:229-232`), so this one lands as a generic `500` and the client is told the
  draw *failed* when nothing objected. Add `ImageInterludeBusy` beside `ImageInterludeInFlight`, translate it
  at the same place, and map it to `429 image_interlude_busy: retry the same request` next to the existing
  `another_draw_holds_the_card` entry in the capability document.
- `throw_vram_insufficient` learns whether shrinking could have helped: the caller passes
  `canvas_is_the_lever`, computed once as `need_at(kVramCanvasFloorPx, kVramCanvasFloorPx) < need_at(ask)`.
  When it is false the message says so plainly -- the need does not fall with the canvas -- instead of
  sending the client to shrink. The standing sentence in `make_capabilities_json` is corrected the same way:
  a smaller size is the fix *when the need falls with the canvas*; the refusal says whether it does.

### D. Do not walk a need that cannot shrink

`need_at(floor) == need_at(ask)` means every step of the walk costs a probe and finds the same number. The
walk runs once when the need is canvas-independent: probe at the asked canvas, then the A/B decision. The
floor refusal and the delivered-bytes net are unchanged, so nothing that could have been drawn at a smaller
size is lost -- the need says there is no such size.

## Non-goals

- **No quiesce inside the probe.** The probe stays observational; a measurement that preempts a prefill
  changes the workload by being asked, and `begin_image_quiesce` already owns that transition.
- **No retry loop as the fix.** Repeating a probe whose input has not changed buys latency and contention,
  and can pass by luck while the blindness stands.
- **No interlude budget, cooldown or interruption quota.** That is the failure the user ruled out: the chat
  must keep serving and the picture must still be drawn. It also cannot starve anything today, because one
  draw holds the card. Revisit only with a measured replay cost.
- **No change to the certified memory ledger.** `plan_image_yield`, `pool_lendable`, the quarter floor, the
  gate arithmetic and `test_image_need_ledger_real.cu`'s numbers are untouched -- the reason bit is derived
  from what they already report.

## Tests

`tests/test_image_generation_service.cpp` (host seam, `FakeRunner`):

1. Probe `!fits` with `expert_pool_frozen_by_lease` set -> the job reaches `run` at the **asked** canvas and
   is drawn from the fake context, with no 503.
2. Probe `!fits` with the bit clear -> today's behaviour: refused with `image_vram_insufficient` at the floor.
3. `!fits` + bit set + a net whose delivered bytes cannot pay even the floor -> refused, and the message
   names the delivered bytes (the net still decides).
4. A canvas whose need equals the floor's -> the refusal says a smaller size will not help, and the walk
   probes once.
5. `run` throwing the busy words -> `429 image_interlude_busy`, not `500`.

`tests/test_memory_plan*.cpp`: `plan_image_yield` keeps `frames_freed == 0` for a lease-held request and
`any_lease_out()` distinguishes it from the quarter-floor case.
`tests/test_http_routes` / capabilities: `image_interlude_busy` is listed with its status.
`test_image_need_ledger_real.cu`: run unchanged, numbers unchanged.

## Verification after the build

Build narrow targets only (`cmake --build build --config Release --target infernix-serve`, then the touched
test targets), never a full `build`. Then through the running server, not a script: attach the picture
request as a chat round while the chat is busy, and require the record in `requests.jsonl` to show the draw
instead of a 503 -- the same probe line that produced 4224/363 today. Re-run
`image-edit/_p37_net_check.py` over the new instance and expect zero `image_vram_insufficient` rows for
768x768 while the chat runs. The serve restart is the user's.
