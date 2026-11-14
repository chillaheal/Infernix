# P8 - `/v1/images/edits` and `capabilities.image_edit`: the contract says how to change a picture

Date: 2026-10-31. Branch: `image-edit`. Package P8 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_openai_images_schema_test`, `infernix_http_routes_test` (host, no device).

## Problem

Everything an edit needs already worked, and nothing said so.

`POST /v1/images/generations` takes `image` (base64, one reference) and the whole edit path is behind
it - the vision tower, the reference latents, the ti2i template, `ratio_follow`. The `generate_image`
chat tool takes an `image` argument in three forms (a job id this server drew, that picture's own URL,
or `attachment:N`) and draws the edit inside the same chat request. So both doors were open.

What a client could not discover was the *edit* itself:

- `/v1/capabilities` advertised `image_generation.reference_image: true` - a boolean about the
  conditioning path - and nothing about how to ask for an edit, what a missing reference does, or that
  a mask is not a thing here. `make_capabilities_json` is the only contract a client that has read no
  file has (`image-edit/CLAUDE.md` makes that the project's test).
- The route a client looks for by name - the OpenAI images **edits** endpoint - did not exist. A client
  written against that API posts `multipart/form-data` with a file part named `image`; there was nothing
  to post to, and our JSON route would not have accepted that shape.
- Most importantly: a client that forgot its reference on the "edit" path would have gotten a
  text-to-image picture. `image` is optional on generations, so a mis-attached file answers an edit
  request with an unrelated drawing and a 200. A silent wrong answer is worse than a refusal.

## Decision

**A route of its own, not a flag on generations.** `POST /v1/images/edits` takes every field the
generations route takes (`model`, `prompt`, `n`, `size`/`wh_ratio`/`ratio_follow`, `steps`, `seed`,
`negative_prompt`, `true_cfg_scale`, `response_format`, `user`) and adds one rule: a reference is
required. Forgetting it is `400 image_reference_required`. The name is what makes the missing picture a
client error instead of a mode switch.

Both wire shapes an edit arrives in are accepted, and they are validated by **one** code path:
`handle_images(req, res, require_reference)` (`src/serve/openai_images_http.cpp`) is the single body
behind `handle_images_generations` and `handle_images_edits`. Two handlers that each parse their own
way is how a rule ends up enforced on one shape and missing on the other; one handler with one flag
cannot drift.

The multipart body is translated, not separately validated:

```cpp
// src/serve/openai_images.h
struct MultipartEditParts {
    std::vector<std::pair<std::string, std::string>> fields;  // text parts: name, value
    std::vector<std::pair<std::string, std::string>> files;   // file parts: name, bytes
};
[[nodiscard]] RequestJson edit_body_from_multipart(const MultipartEditParts& parts);
```

It produces exactly the JSON document `parse_openai_images_request` validates, so the key whitelist,
the canvas XOR, the step bounds and the base64 rules are the same rules for both shapes. Its own rules,
all of them refusals rather than interpretations:

| situation | result |
|---|---|
| a file part named `image` | base64-encoded here into the `image` field |
| a **text** part named `image` holding the base64 | also accepted (some clients send it that way) |
| more than `kMaximumImageReferences` references, in either mix | 400 `image_count_invalid` |
| a file part named `mask` | 400 `image_mask_not_supported` |
| any other file part name | 400 `image_unsupported_file_field` naming it |
| `n` / `seed` / `steps` whose text is all digits, `true_cfg_scale` whose text is a number | become JSON numbers, so an out-of-range value gets the ordinary range error rather than a type error |
| a repeated text part | the first wins (a duplicate is the client's ambiguity) |
| no reference at all | left absent - `parse_openai_images_edit_request` refuses it with its own code |

`mask` gets a code of its own on purpose. This pipeline has no mask input (the roadmap's "checked and NOT
gaps" list), and quietly dropping a mask an SDK client believes it sent would draw an unmasked picture -
the exact dropped-parameter failure the key whitelist at `src/serve/openai_images.cpp:16-20` exists to
prevent. Same reason for `image_unsupported_file_field`: an ignored part is a silent no.

URL references stay refused on this route (`image_url_not_supported`, unchanged from generations): the
route takes bytes. The chat tool is where the three forms (job id, picture URL, `attachment:N`) are
resolved, and the capability says that, because those are the two ways in and a client needs to know
which is which.

**The capability block.** `make_capabilities_json` (`src/serve/http_server.cpp`) gains a top-level
`image_edit` object, null when no image artifact is configured, alongside `image_generation`:

```cpp
{"model", image_model_id},
{"endpoint", "/v1/images/edits"},
{"fields", "every field /v1/images/generations takes"},
{"content_types", Json::array({"application/json", "multipart/form-data"})},
{"reference_required", true},
{"reference_max", kMaximumImageReferences},
{"reference_formats", Json::array({...base64 in the image field, a multipart file part named image,
                                    a multipart text field named image holding the base64...})},
{"accepts_url_reference", false},
{"chat_tool_image_argument", Json::array({"a job id img-<created>-<sequence> this server drew",
                                          "that picture's own URL",
                                          "attachment:N of the chat request"})},
{"missing_reference", "HTTP 400 image_reference_required -- this route never draws a picture nobody attached"},
{"mask", "not supported: HTTP 400 image_mask_not_supported"},
{"canvas_default", "the reference's own aspect (see image_generation.canvas)"}
```

It deliberately points at `image_generation` for the field list and the canvas rules instead of
restating the bounds: two copies of the same numbers is two things to keep in sync, and a capability
document that contradicts itself is worse than one that is quiet. `endpoints.images_edits` joins the
endpoint list so a client that only reads that map finds the route too.

## Tests

`tests/test_openai_images_schema.cpp` - `test_edit_route()`:

- no reference is 400 `image_reference_required` (param `image`); an *empty* `image` stays the
  generations `image_count_invalid`, because that rule runs first;
- the reference and the instruction survive the edit parse; the family's array-of-one form edits too;
- a bad `size` on an edit is still `image_size_invalid` - a reference does not make other mistakes
  legal;
- multipart: the file part lands as the base64 reference and the whole form parses through the shared
  path (steps, seed, true_cfg_scale included); `n`/`seed`/`steps`/`true_cfg_scale` become numbers, and
  `steps=abc` becomes the ordinary value error, not a type error; two file references, or a text plus a
  file reference, are `image_count_invalid`; `mask` is `image_mask_not_supported`; an unknown file part
  is `image_unsupported_file_field`; a base64 text part named `image` is accepted; a repeated text part
  keeps the first; a form with no reference reaches the edit route's own refusal.

`tests/test_http_routes.cpp`:

- the pattern answers under `/v1` and one doubled `/v1`, and nowhere else;
- `image_edit` is null with no image model, and with one it names the endpoint, `reference_required`,
  `accepts_url_reference: false`, both content types, both refusal codes and the chat tool's three
  image forms; `endpoints.images_edits` is the route.

Host suite after the change: 100% passed out of 31
(`ctest -R "image|http_routes" -E "cuda|real|prod|behavior|pipeline_gate"`, `INFERNIX_TEST_ARTIFACT`
= `Qwen-Image-2.1-FP8-v4.infernix`).

Not covered by a test binary: the real `HttpServer` end-to-end for this route (the test harness builds
its own `httplib::Server` for patterns and calls the parser directly for semantics). The shared tail -
model gate, 503 `image_generation_not_configured` before `attach_images`, the response object - is the
generations code, reached through the same function, and the live serve exercises it once restarted.

## Out of scope

`mask` input (does not exist in this pipeline), URL references on the JSON route, more than one
reference (P9 - `kMaximumImageReferences` and the M7 ledger decide that, not the parse), and any
server-side prompt rewriting (P10).
