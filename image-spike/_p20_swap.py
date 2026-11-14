"""Harmonize a composite through the live engine, with the reference count and the switch as arguments.

One POST to /v1/images/edits. It prints the answer's lean fields -- prompt_enhance_outcome above all,
because after the reserve fix of 22:14 that field is the only thing that says whether the rewriter ran --
the seconds, the sha256 and the bytes it wrote, so a sweep can be judged from its log alone.

  python -I _p20_swap.py --out OUT.png --prompt "..." --ref A.png [--ref B.png]
                         [--size 512x576] [--steps 20] [--seed 7] [--enhance true|auto|off]
"""
import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8888/v1/images/edits"
MODEL = "Qwen-Image-2.1-FP8-v4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--size", default="512x576")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--enhance", default="true")
    a = ap.parse_args()

    refs = []
    for path in a.ref:
        with open(path, "rb") as handle:
            refs.append(base64.b64encode(handle.read()).decode("ascii"))
    body = {"model": MODEL, "prompt": a.prompt, "image": refs if len(refs) > 1 else refs[0],
            "n": 1, "size": a.size, "steps": a.steps, "seed": a.seed,
            "response_format": "b64_json"}
    if a.enhance in ("true", "false", "auto"):
        body["prompt_enhance"] = True if a.enhance == "true" else (
            False if a.enhance == "false" else "auto")
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    tag = a.out.split("\\")[-1]
    print("%s POST %s refs=%d json=%d size=%s steps=%d seed=%d enhance=%s"
          % (time.strftime("%H:%M:%S"), tag, len(refs), len(data), a.size, a.steps, a.seed, a.enhance),
          flush=True)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as answer:
            raw, status = answer.read(), answer.status
    except urllib.error.HTTPError as error:
        print("%s HTTP_ERROR %s %s in %.1fs body=%s"
              % (time.strftime("%H:%M:%S"), tag, error.code, time.time() - started,
                 error.read().decode("utf-8", "replace")[:400]), flush=True)
        return 1
    except Exception as exc:                                          # noqa: BLE001 - one request
        print("%s FAILED %s %s in %.1fs" % (time.strftime("%H:%M:%S"), tag, exc,
                                            time.time() - started), flush=True)
        return 1
    secs = time.time() - started
    doc = json.loads(raw)
    lean = {k: v for k, v in doc.items() if k != "data"}
    if not doc.get("data"):
        print("%s NO_IMAGE %s answer=%s" % (time.strftime("%H:%M:%S"), tag,
                                            json.dumps(lean)[:300]), flush=True)
        return 1
    image = base64.b64decode(doc["data"][0]["b64_json"])
    with open(a.out, "wb") as handle:
        handle.write(image)
    print("%s DONE %s HTTP %s in %.1fs outcome=%s sha256=%s bytes=%d"
          % (time.strftime("%H:%M:%S"), tag, status, secs,
             lean.get("prompt_enhance_outcome"), hashlib.sha256(image).hexdigest()[:16], len(image)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
