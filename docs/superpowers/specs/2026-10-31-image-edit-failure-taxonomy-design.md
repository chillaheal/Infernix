# P6 - a post-start failure keeps its family, so the caller can tell retry from reproduce

Date: 2026-10-31. Branch: `image-edit`. Package P6 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_image_generation_service_test` (host, no device).

## Problem

Once the image job had started, every exception from it became the same answer: HTTP 500 with code
`image_generation_failed`. That code is a fine name for "the draw did not produce a picture", and a bad
name for three genuinely different things that all funnelled into it:

- an **Op refusing the geometry it was handed** (`std::invalid_argument` from an ops shape/range check,
  e.g. `attn_dit_joint: S must be 1..20480`). The request is unservable as written; retrying reproduces
  it exactly.
- a **weight that could not be bound or streamed** (`artifact::ArtifactError`). Nothing about the
  request is wrong; the artifact lane failed.
- an actual device/CUDA fault (`std::runtime_error` from `cuda_ok`, `include/infernix/ops/…` /
  `image_device_util.h`), which is the case the code is named for.

The consumer is usually not a human. `src/serve/chat_image_tool.cpp:276-281` builds the tool's error
result out of exactly these two fields - `json_error_result(code, message)` - and hands it back to the
chat model, whose next action is chosen from them. Told `image_generation_failed` for a kernel that
refused the geometry, the model's sensible move is to call the tool again, which costs a full interlude
yield and returns the identical failure.

## Decision

Classify in the final catch of `generate`, and only when the job had actually started
(`src/serve/image_generation_service.cpp:886-903`). The message is never rewritten - it already carries
the underlying text; the code and parameter are what was being lost:

| family | how recognised | HTTP | param | code |
|---|---|---|---|---|
| an Op refused the geometry | `std::invalid_argument` | 500 | `image` | `image_request_unservable` |
| artifact bind/stream | `artifact::ArtifactError` | 500 | `image_artifact` | `image_artifact_failed` |
| anything else | - | 500 | - | `image_generation_failed` (unchanged) |

`dynamic_cast<const T*>(&error)` on the stored `std::exception_ptr`'s target, in the order above:
`ArtifactError` derives from `std::runtime_error`, not from `invalid_argument`, so the two families are
disjoint and the order is not load-bearing - it is written that way so a future type that joins a family
lands somewhere deliberate.

Why `std::invalid_argument` means "ours to fix" rather than "yours to fix": every device failure in the
image lane is a `std::runtime_error` (`cuda_ok`), so an `invalid_argument` from inside a running job can
only have come from an Op's own shape or range check. That is a case the host admission should have
refused before any VRAM was taken and did not - a bug in this process, and a deterministic one. The
comment at the site says so, because the distinction is invisible from the type alone.

Pre-start failures keep the codes they already had (400 decode/schema/size, 403/404, the 429 and 503
interlude refusals); this package touches only the post-start collapse, and the two new codes join the
existing ones rather than replacing them.

## Tests

`tests/test_image_generation_service.cpp:316-333`: two `ImageJobExecutor` lambdas, one throwing
`std::invalid_argument("attn_dit_joint: S must be 1..20480")` and one
`infernix::artifact::ArtifactError("text/embed_tokens is not bound")`, each asserted to produce
`image_request_unservable` and `image_artifact_failed` respectively; the existing
`image_generation_failed` case (a plain throw) still asserts its code, so the fallback is pinned too.

One trap worth recording, because it cost a debugging round: the first version built these executors
from a factory returning `std::exception_ptr` and did `throw make();`. `make()` is a *prvalue*, so that
throws the pointer object, which is not `std::exception`-derived; it escaped the `ApiException`-only
`attempt` helper and took the process down with `0xc0000409`. The test throws directly instead. The
`check()` helper also gained a trailing `std::flush` in this package, which is what made the abort
locatable at all.

## Out of scope

Retrying, circuit-breaking, or any policy change on the caller's side; a finer CUDA-error taxonomy
(all device faults stay one family, which is what the reference behaviour is); and the M9 GPU gate.
