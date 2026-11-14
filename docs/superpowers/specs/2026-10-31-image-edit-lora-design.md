# P15 - Image LoRA: where an adapter can enter a pipeline that streams its weights from host memory

Date: 2026-10-31. Branch: `image-edit`. Package P15 of the Qwen-Image-2.1 parity roadmap.
**Status: designed.** No LoRA exists anywhere in the engine -- a case-insensitive search for `lora` under
`src/` returns only `kColorableFamilies` and `calibrated` substrings -- and none of this is implementable
without one external decision from the user (below), so the package lands as this design.

## Problem

The Qwen-Image ecosystem's distribution mechanism for styles and subjects is LoRA adapters, and the engine
has no way to take one. The consequence for the edit route is concrete: the tone problem this project
measured (a preserved face that will not re-grade from text alone) is exactly the kind of thing an adapter
carries -- the roadmap's own note is that "the reference must already carry the tone", and an adapter is how
you put it there without re-training anything.

## The one fact that decides the design

The DiT's weights are **not resident**. `dit_velocity` keeps host-side block bytes (`cache[l]`, whose nine
roles are `cc.q.h ... cc.key_norm.h`) and streams two blocks at a time into device slots while the SM
computes (`issue_copy`, `lane.copy_done`, `lane.freed`, `image_pipeline.cu:898-925`), because the whole DiT
does not fit beside everything else in the interlude -- `kImageResidentBlocks = 2`
(`src/models/qwen_image/image_device_need.h:33`).

That kills the obvious designs and points at the right one:

- **A runtime low-rank branch** (`x@A^T@B^T` added after each linear) needs a new op per role, 9 roles x 32
  blocks, and it puts extra work on the critical path of a step that is already the reason the interlude
  exists. Rejected: it pays forever for something that can be paid once.
- **A merged device copy of the DiT** cannot exist -- the merged weights are exactly the thing that does not
  fit.
- **Merge into the host-side block bytes** costs one pass over the host copies before the first step and
  changes nothing per step: `upload_block_async` streams the *adjusted* bytes, every op, ledger row and
  overlap event stays as it is. This is the design. `W' = round(W + (alpha / r) * B @ A)`, computed once per
  adapter per block, on the host side where the blocks already live.

## Design

**1. Adapters are artifacts, never downloaded files.** There is no safetensors reader in the engine (a
search for `safetensors` under `src/` returns nothing) and `tools/flash_next/` -- listed: `reference.py`,
`block_check.py`, `verify_artifact.py`, `verify_gguf_artifact.py`, `inspect_checkpoint.py`, the compare and
stripe helpers -- has no image packer in it. So the rule is: **the server parses only what `Reader`/`Binder`
already read**, and adapter conversion happens offline, out of process, producing an artifact that carries
its own digest. That is both the untrusted-data rule this project runs by (a third-party adapter is
downloaded data: its paths, sizes and contents are checked, never trusted, never executed) and the reason
the serve keeps its single code path for weights. First task of the implementation is therefore *not* C++:
it is where the packer lives and what the adapter artifact's section names are.

**2. Binding is per interlude, not per request.** Merging 32 blocks' worth of `B @ A` products is not
something to do inside a job's VRAM lease -- the merge is host work but the *result* is a full second set of
host block bytes, and a per-request adapter would churn that set between every request. So: adapters are
loaded and enumerated at startup (or on an explicit load), named, and a request selects one by name. The
`ImageGenerationService` keeps at most a bounded number of bound adapters, sized from the DiT's host block
bytes -- the same "one symbol so the pipeline, the serving layer and the allowance cannot drift apart"
pattern `kImageResidentBlocks` uses.

**3. It is discoverable, because that is the project's standing test.** `GET /v1/capabilities` grows
`image_generation.adapters` (and the edit route inherits it): each entry `name`, `rank`, `base` (the artifact
digest it was built for), and `strength` range. A fresh session of any client, reading no file, has to be
able to find out that `--style ink` exists. The request gains one field, `adapter`, plus `adapter_scale`;
both go through the same whitelist-and-type discipline as `prompt_enhance` in P10, and an unknown name is a
400 that lists what is bound -- a silently ignored adapter is the failure mode to avoid, since the picture
looks plausible either way.

**4. `base` is a load-time refusal, not a warning.** An adapter is a diff against specific weights; applied
to another build it produces a plausible-looking corruption. The bound is refused unless the artifact digest
in the adapter equals the digest of the artifact the service was started with -- the same digest the
pipeline's trace and cert records already carry.

## Gates

- **Zero-regression:** with no `adapter` in the request, the bytes uploaded per block are bit-identical to
  today's. This is assertable host-side against `Reader`/`Binder` output and is the gate that lets the
  feature land at all -- the certified route must not move.
- **Effect:** the M9 behaviour gate gains an adapter case only once an adapter exists to test with; until
  then the gate stays what it is. The right first adapter is a *tone* one, because that is the effect the
  project measured as unreachable from text (`CLAUDE.md`: the tanning instruction moved the face-minus-chest
  mean under 4 levels per channel) and it is graded by pixels with `faceswap/_tone.py`, not by opinion.
- **Ledger:** the merge and the merged host set live outside the device ledger (host bytes), but the
  byte-zero device gate must stay green across an adapter switch mid-queue, which is the case where a stale
  device copy of an old block would show up.

## The open decision

Which adapter format to accept, and who converts it. The engine has no safetensors path and no image packer
in-repo, so P15 needs either (a) a conversion tool added beside `tools/flash_next/` that reads a community
adapter and writes the container format `Reader` consumes, or (b) a policy of engine-native adapters only,
produced by the project's own training path. (a) is what makes the feature worth having -- community
adapters are the whole ecosystem -- and it means the server consumes a *derived* artifact rather than a
downloaded file. That is the user's call, and it is why this row is a design and not a diff.

## Out of scope

Adapter merging *chains* (start with one), LyCORIS/LoHa/LoKr variants, DreamBooth full fine-tunes, training
anything, and the PE-I2I path (P10) -- an adapter changes the weights, the enhancer changes the instruction,
and conflating them would make neither measurable.
