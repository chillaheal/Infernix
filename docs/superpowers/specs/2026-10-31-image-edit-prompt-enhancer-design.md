# P10 - The prompt enhancer: what this build says when it cannot rewrite, and what lands when PE-I2I arrives

Date: 2026-10-31. Branch: `image-edit`. Package P10 of the Qwen-Image-2.1 parity roadmap.
**Status: the contract landed; the enhancer itself did not.** The weights are a 20 GB BF16 download this
machine does not have, and installing them is the user's call, not a background step. What is landed is
the half that can be true today: the route accepts the parameter, refuses it honestly when asked, and
the capability document tells a client that reads no file how to write an instruction that does not need
one.

Landed and host-verified: `infernix_openai_images_schema_test`, `infernix_http_routes_test`,
`infernix_image_generation_service_test` (100% of 31 host tests).

## Problem

Qwen-Image-2.1 does not expect the user's words to reach the DiT raw. The release ships two prompt
enhancers on Hugging Face, `Qwen/Qwen-Image-2.1-PE-T2I` and `Qwen/Qwen-Image-2.1-PE-I2I`
(`image-edit/refs/q21_README.md:244-246`): fine-tuned Qwen3.5-VL 9B models, one per task. The `edit` one
takes text **plus the 1..N source images** and answers a JSON object with exactly three fields --
`rewritten_prompt`, `wh_ratio`, `ratio_follow` (`refs/q21_system_prompt_edit.txt:175-183`) -- sampled at
temperature 1.0, top_p 0.95, top_k 20, `presence_penalty` 0, max_new_tokens 24000, thinking on
(`refs/q21_prompt_rewrite_README.md:175-197`).

Nothing in Infernix referenced `prompt_rewrite`, `PE-I2I` or `rewritten_prompt`. A client that sent
"make it an owl" got exactly that string into the joint sequence -- and the two things that made the
kingfisher edit weak in this project's own sessions were the missing push ("to a strong and unmistakable
degree") and the missing preservation discipline, both of which are the enhancer's job
(`refs/q21_system_prompt_edit.txt:26-51`).

## What the two external consultations said (recorded as read)

Asked per the standing instruction, in order:

- **Grok (Build mode)**: reached "Free tier limit reached" before answering. Nothing was read from it;
  the dump is `image-edit/_grok_p10.txt` and it contains the question and the limit notice only.
- **ChatGPT (Think mode, deep research)**: answered in full (`image-edit/_cg_p10f.txt`). Its
  recommendation was **(d), an orchestrated optional PE worker -- implemented first as (b), a separate
  process using the official Transformers/vLLM implementation, run inside the existing image-job VRAM
  interlude**, with (a) (a second Infernix artifact) as the eventual in-engine form once Qwen3.5-VL is
  natively supported, and explicitly **not (c) as the primary path**. Facts it cited from the primary
  sources: PE-I2I is "a separately fine-tuned Qwen3.5-VL 9B checkpoint", not a system prompt on the
  ordinary chat model; the card documents ~20 GB BF16 weights plus KV/runtime, comfortable on a 40 GB
  GPU and needing `max model length <= 12000` or reduced memory utilisation on 24 GB; a permanently
  resident PE server "would therefore be a poor default on a single GPU"; a separate process is an
  *architectural* isolation boundary and not a VRAM one, since both still compete for the same card.
  Its execution sequence was: chat -> acquire the image-job GPU lease -> PE inference -> release PE ->
  image job -> restore chat, so the GPU never holds LLM, PE and DiT at once, with the orchestrator
  keeping the worker warm across a batch of image jobs. On quality: "Qwen-Image-2.1 does not require
  PE-I2I to edit images ... PE-I2I is an additional quality layer, not a prerequisite", with the gain
  small for a targeted recolour and high for vague instructions ("make this look more cinematic"),
  "remove the distracting objects but preserve everything else", cross-image composites and multi-image
  coordination. On the existing tower: **no** -- and it flagged that the official architecture docs name
  the image model's own encoder as **Qwen3-VL 8B**, a version distinction worth checking against our
  artifact (this project's `CLAUDE.md` says Qwen2.5-VL; that claim was not re-verified here).
  It also said plainly that it could not read the private `image-edit` branch, so its judgement about
  how much of Qwen3.5-VL Infernix already supports is conditional.

That answer is why this package is split rather than closed: (b)/(d) both need the checkpoint, and a
20 GB download plus a second runtime on the user's card is not a background action.

## Decision (what landed today)

**The parameter is real, the refusal is honest, and the rules travel with the server.**

### The field (`src/serve/openai_images.{h,cpp}`)

`prompt_enhance` is a boolean on `OpenAIImagesRequest`, in the accepted-key whitelist and typed at
parse (a non-boolean is the ordinary 400 naming the field). It is whitelisted rather than left unknown
on purpose: the whitelist exists so that a dropped parameter is never mistaken for an honored one, and
"unsupported key" about a parameter that exists upstream is a less true answer than "this build has no
rewriter installed".

### The refusal (`src/serve/image_generation_service.cpp`)

`generate()` refuses `prompt_enhance: true` with `503 image_prompt_enhancer_unavailable`, naming the
missing model and pointing at the capability, **before** any reference is decoded, any probe is asked or
any job is stored. A client must not get a drawing made from words it believed had been rewritten --
that is the same class of failure as a silently ignored mask, one layer up.

The comment at the refusal states why the artifact's own vision-language tower cannot stand in: it is a
different checkpoint, never trained against the enhancer's system prompt, so running it would return the
picture's embeddings, not a better instruction.

### The contract (`src/serve/http_server.cpp`, `make_capabilities_json`)

`image_edit` gains three keys, so a fresh session of any client -- the project's standing test -- gets
the enhancer's discipline from the server it is talking to instead of from a file it was told to read:

- `prompt_enhancer`: `available: false`, `model: "Qwen/Qwen-Image-2.1-PE-I2I"` (one
  `kImagePromptEnhancerModel` constant, so the refusal and the document cannot disagree), `field`,
  `when_true` naming the 503 code, and `answers` naming the three fields -- including that `wh_ratio`
  and `ratio_follow` are *the same two the route already takes*, which is what makes the eventual
  server-side path a rewrite of the request rather than a new mechanism.
- `short_instruction`: says outright that a one-phrase instruction is not sufficient in this build,
  because nothing rewrites it.
- `instruction`: the shipped edit system prompt's rules, condensed and attributed to it -- lead with the
  operation; push the named attribute to a strong and unmistakable degree; hold everything else at input
  fidelity and say what stays by type, position and role, never by appearance; one blanket preservation
  clause beats a checklist; point identity at the reference instead of describing features; address
  multiple references as `<image1>`, `<image2>`, ... and name the canvas vs the material sources; keep
  every resolution out of the instruction; commit to exact characters for rendered text in the image's
  own dominant language; state preservation affirmatively; describe only what is visibly there; ask for
  nothing unrequested.

The last two are the reason this is not merely a stub: with them, the absence of a rewriter becomes
actionable information for the client, which is the difference between "the model is bad at edits" and
"the instruction was never an edit instruction".

## Tests

- `test_openai_images_schema.cpp::test_prompt_enhance_field`: absent means off; `true` is captured (the
  parse does not decide capability); `false` is captured; a non-boolean is 400 on `prompt_enhance`.
- `test_image_generation_service.cpp`: `prompt_enhance = true` -> 503
  `image_prompt_enhancer_unavailable` whose `param` is `prompt_enhance`, with `probes == 0`, `runs == 0`
  and `store.size() == 0`; the same request with the switch off is served. (`Attempt` gained a `param`
  member so a test can pin which field a refusal blames.)
- `test_http_routes.cpp`: `image_edit.prompt_enhancer` says unavailable, names the field and the model
  and carries the refusal code; `short_instruction` says a one-phrase instruction is not enough;
  `instruction` is an array of at least ten rules and mentions both `<image1>` and `ratio_follow`.

## What is NOT done, and the shape of the remaining work

Roadmap P10 stays open as **partial**. What remains, in the order the consultation put it:

1. **The weights.** `Qwen/Qwen-Image-2.1-PE-I2I` (~20 GB BF16) into its own directory, by the user.
   Nothing here downloads it.
2. **A worker that answers the three-field JSON.** First as a separate process on the official
   implementation (Transformers/vLLM), not a re-implementation, and not as a permanently resident server.
3. **The lease sequence.** PE inference *inside* the existing image-job interlude, between the LLM going
   quiet and the DiT coming up: acquire lease -> PE -> release PE -> image job -> restore chat. The card
   never holds LLM, PE and DiT together. Keep the worker warm across queued image jobs so the load cost
   is paid per batch, not per request.
4. **The validator.** When the worker exists: a strict parse of the answer (exactly three fields;
   `wh_ratio` XOR `ratio_follow`, one of them `""`; a single paragraph with no newlines; no resolution
   or ratio inside `rewritten_prompt` -- `refs/q21_system_prompt_edit.txt:185-192`), then the answer
   feeds the SAME parser this route already runs, so `wh_ratio`/`ratio_follow` go through
   `resolve_canvas_geometry` and an answer that breaks the rules is a server-side 500, not a client 400.
5. **The third value of `prompt_enhance`.** Once it works, `auto` (enhance when the instruction is
   vague or multi-image, skip it for a targeted recolour) is what the consultation's quality table
   argues for; today's boolean is the honest subset.

## Out of scope

Re-implementing Qwen3.5-VL inside Infernix (that is step (a), and it needs native support first).
Client-side enhancement libraries. Any change to how the DiT is conditioned -- PE changes the
*instruction*, not the pipeline.
