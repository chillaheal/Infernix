"""PE-I2I prompt-enhancer sidecar for the Infernix image-edit route.

One process, one checkpoint, loopback HTTP. The checkpoint is the shipped
Qwen/Qwen-Image-2.1-PE-I2I: a fine-tuned Qwen3.5-VL 9B (model_type qwen3_5,
Qwen3_5ForConditionalGeneration) that reads the instruction AND the source
pictures and answers the JSON object {rewritten_prompt, wh_ratio, ratio_follow}.
It is a different checkpoint from the chat model and from the image artifact's
own vision tower, so it is a separate process here rather than a template on a
tower that was never trained against its system prompt.

The model is loaded lazily and released on request: the RTX 5090 in this
deployment never holds the resident LLM, the enhancer and the DiT at once. The
serve side calls /load inside its image-job interlude and /unload when the batch
is done. Nothing is cached on disk; a request carries its pictures as base64.

Endpoints (all JSON, all 127.0.0.1 only):
  GET  /health   {"state","model_dir","loaded","device","free_mib"}
  POST /load     {"ok":true,"weight_bytes":...}      idempotent
  POST /unload   {"ok":true,"freed_mib":...}
  POST /rewrite  {"prompt":...,"images":[b64,...],"seed":42}
              -> {"ok","rewritten_prompt","wh_ratio","ratio_follow",
                  "thinking","parse_ok","seconds","error"}

Sampling is the release's own production setting for the edit task
(image-edit/refs/q21_prompt_rewrite_README.md:175-197): temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence_penalty 0, max_new_tokens 24000,
thinking ON (required -- both checkpoints were trained with a think block).
presence_penalty 0 is the edit task's value; t2i's 1.5 would need a
LogitsProcessor here because transformers has no native presence penalty.

Usage:
  python pe_worker.py --model-dir <dir with config.json + system_prompt.txt>
                      [--port 8899] [--parent-pid N] [--idle-unload-sec 180]
"""

from __future__ import annotations

import argparse
import base64
import binascii
import http.server
import io
import json
import logging
import os
import sys
import threading
import time

# What this process costs the card is the number serve is asked to promise before it loads it, so the
# allocator must not be what inflates it: a generate() that grows reserves a fresh block per new tensor
# size and strands the gaps between them. Expandable segments hand those gaps back to the driver.
# Set before torch is imported -- this module imports torch lazily inside the handlers, so a
# module-level default always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MAX_IMAGES = 8            # the route caps references too; this is the worker's own ceiling
MAX_IMAGE_BYTES = 32 << 20
MAX_PROMPT_CHARS = 8000
# The shipped processor has NO upper bound: processor_config.json asks for
# size.longest_edge = 16777216 pixels, so a 12 MP phone photo is fed to the vision tower at 12 MP and
# becomes ~12k merged tokens. The cost of that lands on the card at the exact moment serve has already
# promised the rest of it to the DiT, so the picture's own resolution would decide whether an edit
# request fits -- an answer nobody can predict from the API. The enhancer only has to SEE the picture
# well enough to name what changes: 1.25 MP is about 1280x1000, above what the DiT itself is ever
# asked to draw at, and keeps the vision cost inside the reservation. 0 turns the cap off.
MAX_IMAGE_PIXELS = 1_250_000
EDIT_FIELDS = ("rewritten_prompt", "wh_ratio", "ratio_follow")

# The edit task's production sampling, per the release README. Kept as one dict
# so a run's log line and the generate() call cannot disagree about what was used.
EDIT_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 24000,
}

log = logging.getLogger("pe_worker")


def _single_position_logits(model) -> dict:
    """Hand only the last position to the language head, and record how many it was handed.

    The hypothesis this was written for: transformers narrows the logits to the final row only from
    the second step on, so a prefill that ran the whole prompt through lm_head would materialise a
    few thousand rows of a 248320-wide vocabulary -- about 1.7 GiB of bf16, doubled again by the
    float32 softmax in sampling. Measured, it is not what happens: one real 5118-token edit gave
    max_positions = 1 and 23654 MiB against 23659 MiB with the slice removed (_p10_fix.json), so
    prefill logits cost nothing here and the memory above the weights was the attention backend.
    The slice stays as the guard -- if a future transformers stops honouring logits_to_keep in
    prepare_inputs_for_generation, this is what keeps the prefill off the head -- and
    max_positions is what proves it next time. Nothing else in this process reads the head, so the
    narrower answer is the same answer.
    """
    stats = {"wrapped": False, "max_positions": 0, "head_features": 0}
    head = getattr(model, "lm_head", None)
    if head is None or not callable(getattr(head, "forward", None)):
        log.warning("no lm_head on %s: the prefill pays full logits", type(model).__name__)
        return stats
    original = head.forward

    def forward(hidden_states, *args, **kwargs):
        if hidden_states.dim() == 3:
            stats["max_positions"] = max(stats["max_positions"], int(hidden_states.shape[1]))
            # The head's input width (hidden_size), not the vocabulary: this is what the head is
            # handed, and it is what tells a later reader whether the slice was reached at all.
            stats["head_features"] = int(hidden_states.shape[-1])
            if hidden_states.shape[1] > 1:
                hidden_states = hidden_states[:, -1:, :]
        return original(hidden_states, *args, **kwargs)

    head.forward = forward
    stats["wrapped"] = True
    return stats


def _fused_full_attention(torch):
    """Keep the full-attention layers out of the math backend, which owns the prefill peak.

    Measured, not assumed (_p10_owner.json, _p10_sdpa2.json, _p10_lever.json): the language model's
    attention reaches scaled_dot_product_attention as query [1,16,5118,256] against 4-head key and
    value with enable_gqa set -- transformers/integrations/sdpa_attention.py:57-62 hands it over
    that way whenever the mask is None (use_gqa_in_sdpa at :28-37). This torch build refuses GQA in
    both fused kernels ("both fused kernels require query, key and value to have the same
    num_heads", "Torch was not compiled with flash attention", "head_dim should be no more than
    128" for cuDNN), so every call falls through to math and materialises the score matrix in fp32:
    5118*5118*16*4 = 1600 MiB per block, several live at once, which is the ~4.4 GiB above the
    weights that made an edit request refuse with image_vram_insufficient. Forcing the backends
    without repeating the heads only turns the math pass into "No available kernel". The shape the
    fused kernels do accept is heads repeated out and no enable_gqa, and that same shape cost 384
    MiB in 0.02 s in _p10_sdpa2.json.

    So wrap the function for the duration of one generate(): repeat the key/value heads when they
    differ from the query's, drop enable_gqa, and ask for cuDNN then the memory-efficient kernel.
    If no fused kernel will take the call, run that call under torch's own policy instead of
    refusing the rewrite -- a slow rewrite is worth more than a failed one. Returns the undo.
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    real = F.scaled_dot_product_attention
    backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None, **kwargs):
        wanted = bool(kwargs.pop("enable_gqa", False))
        heads = int(query.shape[1])
        kv_heads = int(key.shape[1])
        if wanted and kv_heads and heads != kv_heads and heads % kv_heads == 0:
            groups = heads // kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        elif wanted:
            kwargs["enable_gqa"] = True           # nothing clean to repeat: leave it to torch

        def call():
            return real(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        scale=scale, is_causal=is_causal, **kwargs)

        try:
            with sdpa_kernel(backends, set_priority=True):
                return call()
        except RuntimeError as exc:
            if "No available kernel" not in str(exc):
                raise
            log.info("no fused kernel for %s x %s heads, falling back to torch's policy",
                     list(query.shape), [heads, kv_heads])
            return call()

    F.scaled_dot_product_attention = wrapped

    def undo():
        F.scaled_dot_product_attention = real

    return undo


class Enhancer:
    """Holds the checkpoint in GPU memory, or holds nothing at all."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._system_prompt = None
        self._device = None
        self.loaded_at = 0.0
        # How many rewrites are between "the card is about to be touched" and "generate() returned".
        # generate() runs OUTSIDE self._lock, so the lock is not what keeps unload() away from it -- this
        # count is. A lease measured from activity alone still expires mid-forward on a thinking run
        # longer than the lease, and empty_cache() landing there strands the running job's blocks.
        self._active = 0
        self.weight_bytes = 0
        self._logits_stats = {"wrapped": False, "max_positions": 0, "head_features": 0}
        self._load_stats = {"allocated_mib": -1, "reserved_mib": -1, "param_mib": -1,
                            "buffer_mib": -1, "top_buffers": []}

    # -- state ------------------------------------------------------------

    def state(self) -> str:
        return "ready" if self._model is not None else "unloaded"

    def free_mib(self) -> int:
        # Only while torch is actually in this process: importing it builds a CUDA context of a few
        # hundred MiB, and a health check asked from the outside must not be the thing that costs the
        # card that. -1 means "this process holds no device state to report".
        torch = sys.modules.get("torch")
        if torch is None:
            return -1
        try:
            free, _total = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        except Exception:                                    # noqa: BLE001 - diagnostics only
            return -1

    def peak_mib(self) -> int:
        # The largest reservation torch has held since the weights came up -- this rewrite's real cost,
        # which is what kEnhancerReserveBytes must be built from. nvidia-smi sees the same from outside
        # plus the CUDA context and cuBLAS workspaces, which belong to the cost too. -1 = not loaded.
        torch = sys.modules.get("torch")
        if torch is None:
            return -1
        try:
            return int(torch.cuda.max_memory_reserved() // (1024 * 1024))
        except Exception:                                    # noqa: BLE001 - diagnostics only
            return -1

    # -- lifecycle --------------------------------------------------------

    def load(self) -> dict:
        with self._lock:
            if self._model is not None:
                return {"ok": True, "loaded": True, "weight_bytes": self.weight_bytes,
                        "resident": self._load_stats, "already": True}
            config = os.path.join(self.model_dir, "config.json")
            if not os.path.isfile(config):
                raise FileNotFoundError("no config.json in " + self.model_dir)
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            started = time.time()
            log.info("loading %s", self.model_dir)
            load_kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
            # Everything in this process that is not the weights measured 5.7 GiB of peak
            # (_p10_squeeze.json: 23659 MiB on a 666 MiB idle card, 17964 MiB of it the safetensors).
            # That turned out to be the eight full-attention layers falling through to the math
            # backend -- fp32 score blocks of 5118*5118*16*4 = 1600 MiB each -- which is what
            # _fused_full_attention around generate() exists to prevent; the same job now costs
            # 19564 MiB (_p10_resident.json). Which kernel from_pretrained settles on is not visible
            # from the outside, so the environment still names it when asked to.
            attn = os.environ.get("PE_ATTN_IMPLEMENTATION", "").strip()
            if attn:
                load_kwargs["attn_implementation"] = attn
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_dir, **load_kwargs)
            except TypeError:                                 # transformers < 5 used torch_dtype
                load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_dir, **load_kwargs)
            model.eval()
            self._logits_stats = _single_position_logits(model)
            processor = AutoProcessor.from_pretrained(self.model_dir)
            weight_bytes = sum(
                p.numel() * p.element_size() for p in model.parameters())
            self._model, self._processor = model, processor
            self._device = str(next(model.parameters()).device)
            self.weight_bytes = weight_bytes
            self.loaded_at = time.time()
            # What is actually resident before anything runs, so that a gap above the weights can be
            # attributed. It answered the question it was written for: buffer_mib came back 0, which
            # ruled out a lazily built position or rope table (a BUFFER, invisible to a sum over
            # parameters) and left an activation peak -- the attention kernel -- as the only owner of
            # the 4.5 GiB. Kept because it is the measurement that separates "chase the kernel" from
            # "shrink the loaded model", and it has to happen before the counters are reset.
            buffers = [(name, int(tensor.numel()) * int(tensor.element_size()))
                       for name, tensor in model.named_buffers()]
            buffers.sort(key=lambda item: -item[1])
            self._load_stats = {
                "allocated_mib": int(torch.cuda.memory_allocated() // (1024 * 1024)),
                "reserved_mib": int(torch.cuda.memory_reserved() // (1024 * 1024)),
                "param_mib": int(weight_bytes // (1024 * 1024)),
                "buffer_mib": int(sum(size for _, size in buffers) // (1024 * 1024)),
                "top_buffers": [[name, size // (1024 * 1024)] for name, size in buffers[:6]],
            }
            # The load's own scratch (the safetensors staging, the first dispatch) is not something a
            # rewrite needs; hand it back and start the peak count here, so the peak /health reports is
            # the reservation of ONE real rewrite on top of the weights.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # The text stack and the vision stack choose separately; both are reported so a
            # measurement run can tell which kernel the peak was actually produced under.
            kernels = [getattr(getattr(model, "config", None), "_attn_implementation", None),
                       getattr(getattr(model.config, "vision_config", None),
                               "_attn_implementation", None)]
            log.info("loaded in %.1fs, %.2f GiB on %s, attn %s, logits %s, resident %s, "
                     "free now %d MiB",
                     time.time() - started, weight_bytes / 2**30, self._device,
                     kernels, self._logits_stats, self._load_stats, self.free_mib())
            return {"ok": True, "loaded": True, "weight_bytes": weight_bytes,
                    "seconds": time.time() - started, "device": self._device,
                    "attn": kernels, "logits": self._logits_stats,
                    "resident": self._load_stats,
                    "reserved_mib": self.peak_mib()}

    def unload(self) -> dict:
        with self._lock:
            if self._active:
                # A rewrite is inside generate() right now. Releasing the cache there does not kill the
                # running tensors -- it strands them and makes the next allocation pay for it, on a job
                # the client is still waiting for. So the release waits for the work instead: the idle
                # timer asks again in five seconds, and serve's own /unload at the end of the batch gets
                # there as soon as nothing is in flight. ok stays true because nothing went wrong.
                return {"ok": True, "loaded": True, "freed_mib": 0,
                        "deferred": "%d rewrite(s) in flight" % self._active}
            if self._model is None:
                return {"ok": True, "loaded": False, "freed_mib": 0}
            before = self.free_mib()
            import torch

            model, processor = self._model, self._processor
            self._model, self._processor, self._system_prompt = None, None, None
            del model, processor
            torch.cuda.empty_cache()
            return {"ok": True, "loaded": False, "freed_mib": self.free_mib() - before}

    # -- the one thing it is for -----------------------------------------

    def system_prompt(self) -> str:
        if self._system_prompt is None:
            path = os.path.join(self.model_dir, "system_prompt.txt")
            with open(path, "r", encoding="utf-8") as handle:
                self._system_prompt = handle.read().strip()
            if not self._system_prompt:
                raise RuntimeError("system_prompt.txt is empty in " + self.model_dir)
        return self._system_prompt

    def peak_allocated_mib(self) -> int:
        # Reserved is what the card lost; allocated is what was really needed. The gap is stranded
        # blocks, and only a gap worth closing is worth an allocator flag -- so the two are reported
        # side by side rather than one of them being trusted. -1 = not loaded.
        torch = sys.modules.get("torch")
        if torch is None:
            return -1
        try:
            return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
        except Exception:                                    # noqa: BLE001 - diagnostics only
            return -1

    def rewrite(self, prompt: str, images: list, seed: int,
                max_pixels: int = MAX_IMAGE_PIXELS,
                logits_to_keep_last: bool = True) -> dict:
        # The request's own shape is checked first, whether or not the weights are up: a client that
        # sent no pictures or an empty instruction is wrong in the same way at any time, and answering
        # it with "not loaded" would make a 400 depend on the lease.
        if not prompt.strip():
            raise ValueError("prompt is empty")
        if not images:
            raise ValueError("the edit task needs at least one source image")
        if len(images) > MAX_IMAGES:
            raise ValueError("too many images (the edit task addresses <image1>..)")
        pictures = [_decode_image(item, max_pixels) for item in images]

        import torch

        # Where the 4498 MiB above the weights comes from (_p10_resident.json: allocated at load is
        # exactly the parameter sum, every buffer is 0 MiB, so the rewrite itself creates the peak --
        # and 1830 MiB of it is what the interlude gate is short by). The allocator's history names
        # the Python line behind every block, which is the only way to tell a chunked prefill from a
        # KV cache from an over-eager vision tower without guessing. Gated by an environment
        # variable so a measurement run costs the production path nothing.
        snapshot = (os.environ.get("PE_MEM_SNAPSHOT") or "").strip()
        if snapshot:
            try:
                torch.cuda.memory._record_memory_history(
                    max_entries=int(os.environ.get("PE_MEM_HISTORY_ENTRIES", "60000")))
            except Exception as exc:                            # noqa: BLE001 - diagnostics only
                log.info("memory history unavailable: %s", exc)
                snapshot = ""

        with self._lock:
            if self._model is None:
                raise RuntimeError("the enhancer is not loaded; POST /load first")
            model, processor = self._model, self._processor

        # The idle lease measures from here, not from /load. This is a 9B model thinking with
        # max_new_tokens 24000, and a second rewrite inside the same batch would otherwise run under a
        # lease that started counting at load time and could expire while generate() still holds the card.
        self.loaded_at = time.time()

        # Images first, in the order given, because the shipped system prompt tells the model to
        # address them as <image1>, <image2>, ... in that order; reordering silently re-points
        # every reference in the rewrite (refs/q21_prompt_rewrite_README.md:66-73).
        content = [{"type": "image", "image": picture} for picture in pictures]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": content},
        ]

        started = time.time()
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = processor(images=pictures, text=[text], return_tensors="pt").to(
            model.device, torch.bfloat16)
        torch.manual_seed(seed)
        kwargs = dict(EDIT_SAMPLING)
        kwargs["eos_token_id"] = [processor.tokenizer.eos_token_id]
        if logits_to_keep_last:
            # The whole prompt goes through lm_head otherwise: this vocabulary is 248320 rows wide,
            # so a two-thousand-token prefill materialises about 1.2 GiB of logits that generate()
            # only ever reads the last row of. Keeping one logit is the same answer without that
            # tensor -- and this process shares the card with the DiT, so a gigabyte is a real term
            # in whether an edit request fits at all.
            kwargs["logits_to_keep"] = 1
        undo_attention = _fused_full_attention(torch)
        # The window unload() must not land in starts here, at the touch of the card, and ends when
        # generate() returns -- which for a thinking run can be longer than the idle lease.
        with self._lock:
            self._active += 1
        try:
            with torch.inference_mode():
                try:
                    out = model.generate(**inputs, **kwargs)
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    # A transformers old enough not to know the argument: it is an optimisation, not
                    # the task, so pay the logits rather than refuse the edit.
                    log.info("this transformers has no logits_to_keep; generating with full logits")
                    kwargs.pop("logits_to_keep", None)
                    out = model.generate(**inputs, **kwargs)
        finally:
            undo_attention()
            with self._lock:
                self._active -= 1
            # Quiet time counts from when the work ENDED, so an idle worker gives the weights back
            # idle_unload_sec after its last rewrite instead of after its first.
            self.loaded_at = time.time()
        generated = out[0][inputs["input_ids"].shape[1]:]
        answer = processor.tokenizer.decode(generated, skip_special_tokens=False)
        thinking, body = _split_thinking(answer)
        parsed = _parse_answer(body)
        if snapshot:
            # Written after the decode so the snapshot covers the whole generate() and nothing of the
            # answer handling. Recording stays off unless PE_MEM_SNAPSHOT named a path, so this is
            # invisible to a normal request.
            try:
                torch.cuda.memory._dump_snapshot(snapshot)
                log.info("memory history written to %s", snapshot)
            except Exception as exc:                            # noqa: BLE001 - diagnostics only
                log.info("memory snapshot failed: %s", exc)
            finally:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:                               # noqa: BLE001 - diagnostics only
                    pass
        # head_features, not vocab: _single_position_logits records the width the head is handed
        # (hidden_size), which is the fact that proves whether the slice was reached. Reading a key
        # it no longer returns raised KeyError after generate() and threw the finished rewrite away.
        log.info("rewrote in %.1fs (%d new tokens, parse_ok=%s), head saw %d positions of %d features",
                 time.time() - started, generated.shape[0], parsed["parse_ok"],
                 self._logits_stats["max_positions"], self._logits_stats["head_features"])
        return {
            "ok": True,
            "loaded": True,
            "logits": dict(self._logits_stats),
            "rewritten_prompt": parsed["rewritten_prompt"],
            "wh_ratio": parsed["wh_ratio"],
            "ratio_follow": parsed["ratio_follow"],
            "thinking": thinking,
            "parse_ok": parsed["parse_ok"],
            "seconds": time.time() - started,
            "new_tokens": int(generated.shape[0]),
            "image_sizes": [[p.width, p.height] for p in pictures],
            "sampling": EDIT_SAMPLING,
        }


def _decode_image(payload: str, max_pixels: int = MAX_IMAGE_PIXELS):
    from PIL import Image

    if isinstance(payload, str) and payload.startswith("data:"):
        payload = payload.split(",", 1)[-1]
    if len(payload) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 8:
        raise ValueError("an image exceeds the accepted budget")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("an image is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("an image exceeds the accepted budget")
    picture = Image.open(io.BytesIO(raw))
    if max_pixels and picture.width * picture.height > max_pixels:
        # Downscale to the budget, then to a whole number of 32-pixel patches (patch 16 x merge 2),
        # so the processor's own smart-resize has nothing left to round and cannot push the token
        # count back up. LANCZOS because this picture is the only thing the rewriter sees of the
        # user's edit subject.
        scale = (float(max_pixels) / float(picture.width * picture.height)) ** 0.5
        width = max(32, int(picture.width * scale) // 32 * 32)
        height = max(32, int(picture.height * scale) // 32 * 32)
        picture = picture.resize((width, height), Image.LANCZOS)
    picture.load()
    if picture.mode != "RGB":
        picture = picture.convert("RGB")
    return picture


def _split_thinking(answer: str) -> tuple:
    """The checkpoints emit a think block before the answer JSON (thinking is required).

    Thinking is required by the checkpoint, so a think block is expected before the JSON.
    A missing close marker costs only the thinking field -- the whole text is then treated
    as the answer and the parser still gets its chance, which is what the release's own
    runners do (parse_ok=false with the raw text kept, never a silent drop).
    """
    if "</think>" in answer:
        head, _, tail = answer.partition("</think>")
        head = head.split("<think>", 1)[-1]
        return head.strip(), tail.strip()
    return "", answer.strip()


def _parse_answer(body: str) -> dict:
    """Strict on the contract, tolerant about the wrapping.

    The answer must be the three shipped fields (refs/q21_system_prompt_edit.txt:175-183).
    Anything that does not parse is reported as parse_ok=false with the raw text kept,
    exactly as the release's own runners do -- a rejected answer is visible, never silent.
    """
    empty = {"rewritten_prompt": "", "wh_ratio": "", "ratio_follow": "", "parse_ok": False}
    text = body.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, finish = text.find("{"), text.rfind("}")
    if start < 0 or finish <= start:
        return empty
    try:
        record = json.loads(text[start:finish + 1])
    except json.JSONDecodeError:
        return empty
    if not isinstance(record, dict) or set(record) != set(EDIT_FIELDS):
        return empty
    out = {}
    for field in EDIT_FIELDS:
        value = record[field]
        if not isinstance(value, str):
            return empty
        out[field] = value.strip()
    out["parse_ok"] = bool(out["rewritten_prompt"])
    if not out["parse_ok"]:
        return empty
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "InfernixPE/1"
    enhancer: Enhancer  # set on the class by serve()
    timeout_guard = None

    def log_message(self, fmt, *args):  # route the access log through logging
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > (MAX_IMAGE_BYTES * (MAX_IMAGES + 1) * 4) // 3 + 4096:
            raise ValueError("the request body is too large")
        try:
            record = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("the request body is not JSON") from exc
        if not isinstance(record, dict):
            raise ValueError("the request body is not a JSON object")
        return record

    def do_GET(self) -> None:                                  # noqa: N802 - httplib spelling
        if self.path != "/health":
            self._send(404, {"ok": False, "error": "unknown path"})
            return
        self._send(200, {"ok": True, "state": self.enhancer.state(),
                         "model_dir": self.enhancer.model_dir,
                         "loaded": self.enhancer.state() == "ready",
                         "weight_bytes": self.enhancer.weight_bytes,
                         "device": self.enhancer._device,
                         "free_mib": self.enhancer.free_mib(),
                         "peak_reserved_mib": self.enhancer.peak_mib(),
                         "peak_allocated_mib": self.enhancer.peak_allocated_mib(),
                         "max_image_pixels": MAX_IMAGE_PIXELS,
                         "sampling": EDIT_SAMPLING})

    def do_POST(self) -> None:                                 # noqa: N802 - httplib spelling
        try:
            body = self._body()
            if self.path == "/load":
                self._send(200, self.enhancer.load())
            elif self.path == "/unload":
                self._send(200, self.enhancer.unload())
            elif self.path == "/rewrite":
                prompt = body.get("prompt", "")
                images = body.get("images", [])
                if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
                    raise ValueError("prompt must be a string under %d characters"
                                     % MAX_PROMPT_CHARS)
                if not isinstance(images, list) or not all(isinstance(i, str) for i in images):
                    raise ValueError("images must be a list of base64 strings")
                seed = body.get("seed", 42)
                if not isinstance(seed, int):
                    raise ValueError("seed must be an integer")
                # Both are measurement and escape hatches, not client API: serve never sends them, and
                # each is checked so a stray value cannot become an unbounded one.
                max_pixels = body.get("max_image_pixels", MAX_IMAGE_PIXELS)
                if not isinstance(max_pixels, int) or max_pixels < 0:
                    raise ValueError("max_image_pixels must be a non-negative integer")
                keep_last = body.get("logits_to_keep_last", True)
                if not isinstance(keep_last, bool):
                    raise ValueError("logits_to_keep_last must be a boolean")
                self._send(200, self.enhancer.rewrite(prompt, images, seed,
                                                     max_pixels=max_pixels,
                                                     logits_to_keep_last=keep_last))
            else:
                self._send(404, {"ok": False, "error": "unknown path"})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc), "code": "pe_bad_request"})
        except FileNotFoundError as exc:
            self._send(500, {"ok": False, "error": str(exc), "code": "pe_model_missing"})
        except Exception as exc:                                # noqa: BLE001 - the wire must not die
            log.exception("request failed")
            self._send(500, {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                             "code": "pe_internal"})


def idle_due(loaded_at: float, idle_unload_sec: int, now: float) -> bool:
    """The idle-lease rule, on its own, so a probe can hold it to a number without a model or a card.

    It measures from the LAST ACTIVITY, which is what Enhancer.rewrite stamps, not from the load: a
    lease measured from /load alone expires whatever the worker is doing, and the timer thread would
    then run unload() -- including torch.cuda.empty_cache() -- while another thread sits inside
    generate(). The launcher runs this worker with --idle-unload-sec 180 (STATE 2026-11-01), and a
    whole enhanced request measured 190.9 s (STATE:4522), so that window is not hypothetical.
    """
    return idle_unload_sec > 0 and now - loaded_at > idle_unload_sec


def watch_parent(parent_pid: int, idle_unload_sec: int, enhancer: Enhancer) -> None:
    """Exit when serve dies, and release the weights when a batch goes quiet.

    The idle timer is the second half of the lease: serve says /unload when it ends a
    batch, and if it never gets the chance (a client vanished mid-job) the weights go
    back on their own so the resident LLM is not starved by an orphaned worker. "Quiet"
    means no rewrite for idle_unload_sec -- not "the load is that old", which would pull
    the weights out from under a rewrite in flight.
    """
    while True:
        time.sleep(5.0)
        if parent_pid > 0:
            try:
                os.kill(parent_pid, 0)
            except OSError:
                log.info("parent pid %d is gone; exiting", parent_pid)
                os._exit(0)
        if enhancer.state() == "ready" and idle_due(enhancer.loaded_at, idle_unload_sec,
                                                     time.time()):
            log.info("idle past %ds; unloading", idle_unload_sec)
            enhancer.unload()


def _answers_as_worker(port: int) -> bool:
    """True when something on this port already answers like this program does."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as reply:
            record = json.loads(reply.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001 - a probe, not a request
        return False
    return bool(record.get("ok")) and "state" in record and "weight_bytes" in record


def serve(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True,
                        help="the PE-I2I checkpoint directory (config.json, system_prompt.txt)")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="exit when this pid is gone (serve passes its own)")
    parser.add_argument("--idle-unload-sec", type=int, default=180)
    parser.add_argument("--once", action="store_true",
                        help="load, rewrite one request from --input, print it, exit (smoke test)")
    parser.add_argument("--input", default="", help="one JSONL line for --once")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s PE %(levelname)s %(message)s")
    enhancer = Enhancer(args.model_dir)

    if args.once:
        line = args.input or sys.stdin.readline()
        record = json.loads(line)
        enhancer.load()
        print(json.dumps(enhancer.rewrite(record.get("prompt", ""),
                                          record.get("input_images", []) and
                                          [_read_file(p) for p in record["input_images"]],
                                          int(record.get("seed", 42))), ensure_ascii=False))
        return 0

    Handler.enhancer = enhancer
    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        # The ordinary case after a redeploy: the worker the last launcher started is still holding the
        # port. When the thing answering there is a worker of this kind, leave it the port and say so --
        # a traceback in the new window teaches nothing, and two processes cannot share 20 GB anyway.
        if not _answers_as_worker(args.port):
            log.error("cannot listen on 127.0.0.1:%d: %s", args.port, exc)
            return 1
        log.info("another prompt enhancer already answers on port %d; leaving it the port",
                 args.port)
        return 0
    threading.Thread(target=watch_parent, args=(args.parent_pid, args.idle_unload_sec, enhancer),
                     daemon=True).start()
    log.info("listening on 127.0.0.1:%d for %s (state %s)", args.port, enhancer.model_dir,
             enhancer.state())
    httpd.serve_forever()
    return 0


def _read_file(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


if __name__ == "__main__":
    sys.exit(serve(sys.argv[1:]))
