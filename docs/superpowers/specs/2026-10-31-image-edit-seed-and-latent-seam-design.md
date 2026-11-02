# P7 - the seed is chosen at submit time, and `init_latent` is declared to be the fixed-latent seam

Date: 2026-10-31. Branch: `image-edit`. Package P7 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_image_generation_service_test` (host, no device).

## Problem

Two facts about how a picture comes out of this process were true but written nowhere a reader would
look, and one of them was actually wrong in the record.

**The seed was drawn inside the job.** `seed` is the public contract (`std::optional<std::uint64_t> seed`
in `OpenAIImagesRequest`, `src/serve/openai_images.h:43`, "product extension; absent => engine-chosen"),
but with no seed the number was minted after the interlude had been taken and the record's request JSON
carried no seed at all. So a stored job could not say which pixels it drew: the noise was a local
variable of the job body, and the one artifact clients and the log actually keep (`record.request`)
omitted it. Two runs of the same record were not comparable, and a user asking "why did that picture
look like that" had nothing to re-send.

**The fixed-latent seam was undocumented.** `ImageJobInputs::init_latent`
(`src/models/qwen_image/image_pipeline.h:36-43`) is the one input that makes a model regression test
possible: two builds that agree on `init_latent`, `prompt_embeds`, the schedule and the reference
latents differ *only* in the model. But nothing said that `run_image_job` takes it verbatim, nor that
the served route reaches it through `seed`, so the two usages were easy to confuse - and confusing them
produces a green test that says nothing about any client's request.

## Decision

**Choose the seed at submit time and record it.** Before the interlude and before the job body,
`src/serve/image_generation_service.cpp` takes the request's seed or draws a 64-bit one, and puts it in
the record's request JSON (`src/serve/image_generation_service.cpp:772`):

```cpp
    std::uint64_t seed = 0;
    if (request.seed.has_value()) {
        seed = *request.seed;
    } else {
        std::random_device random;
        seed = (static_cast<std::uint64_t>(random()) << 32U) | random();
    }
    // What the job actually ran, not what was asked: a stored canvas that disagreed with the picture
    // the client got back would make the record useless for a later diagnosis.
    record.request  = nlohmann::json{{"prompt", effective.prompt},
                                     {"width", effective.width},
                                     {"height", effective.height},
                                     {"steps", effective.steps},
                                     {"seed", seed}};
```

The comment above it states why the number is recorded at all: "the only way it can ever ask for that
picture again is if the server remembers the number", and it stays a *diagnosis* field -
`job_status_json` reports id, status and progress, so no response field changed.

The job body reads that value; the in-job mint is gone. One place still turns a seed into pixels -
`qi::fill_normal(seed, noise)` at `src/serve/image_generation_service.cpp:195`, now marked as exactly
that: this engine's own reproducible N(0,1) stream (`models/qwen_image/noise.h`), snapped to BF16 like
every other activation, and "a fixed-latent GPU regression does not go through this line". A reader
looking for "where does the noise come from" finds one line and its contract.

**Declare the seam in the header it belongs to.** `image_pipeline.h` now says: the arrays are taken
verbatim, `run_image_job` validates the size, uploads the bits and never draws noise itself - which is
what makes it the seam a model regression pins - and *the served route does not use it that way*: serve
turns `seed` into the buffer with `fill_normal` before calling. So a pinned-latent result is not a claim
about what any client's seed produces, and a seed is not portable to another implementation's generator
either. That last clause matters for a project whose reference is diffusers: a seed is only comparable
within this process.

No behaviour change for a client that passes a seed; the observable changes are the record and the
documentation. Nothing here adds an injection path for tests - `run_image_job` already accepted the
buffer, and the point of this package is that nobody has to guess which layer owns the noise.

## Tests

`tests/test_image_generation_service.cpp`, two assertions in the success region:

- a request with no seed still records one: `record->request.contains("seed")` and the value is an
  unsigned number - the regression this package fixes, since the field did not exist before;
- a request with `seed = 123456789` produces `record->request.at("seed") == 123456789` in a fresh
  service's own store, so the value travels submit -> record unchanged.

`infernix_image_generation_service_test` passes; the whole host image suite is 30/30
(`ctest -R image -E "cuda|real|prod|behavior|pipeline_gate"` with `INFERNIX_TEST_ARTIFACT` pointing at
`Qwen-Image-2.1-FP8-v4.infernix`).

## Out of scope

Sub-seeds / variation seeds and `strength` (not parameters of this pipeline - the roadmap's
"checked and NOT gaps"), making a seed reproducible across engine versions or in diffusers, and a
`seed` field in the HTTP response body (the record is where it lives; the response is the OpenAI shape).
