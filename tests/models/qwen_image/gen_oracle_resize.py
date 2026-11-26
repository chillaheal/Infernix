#!/usr/bin/env python3
"""Generate oracle_resize.json: the reference condition-image resize gated against PIL itself.

What the reference pipeline actually does to a condition image, transcribed:
  QwenImage21Pipeline builds VaeImageProcessor(...) with NO resample override, so
  VaeImageProcessor.__init__'s defaults apply: resample="lanczos", reducing_gap=None,
  do_convert_rgb=False. VaeImageProcessor.resize on PIL input is exactly
      image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"], reducing_gap=None)
  and Image.resize routes mode "RGBA" through the premultiplied "RGBa" mode. So PIL 12.3.0 here is
  the whole oracle: no torch, no diffusers.

Run with the isolated interpreter, from anywhere:
  <python> -I tests/models/qwen_image/gen_oracle_resize.py [output.json]

The C++ gate (test_image_resize.cpp) decodes input_png_b64 with the production PNG decoder and must
reproduce expected_rgba_b64 byte for byte. Fixtures are synthetic and deterministic, so this file can
be regenerated and reviewed at any time; inputs are stored as PNG (small, inspectable) and expected
outputs as raw RGBA (compared without any decoder in the way).
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

RESAMPLE = Image.Resampling.LANCZOS


def pattern(kind: str, w: int, h: int) -> np.ndarray:
    """Deterministic RGB source content. Sharp edges and high frequencies are the point: they are
    what a 2x2 bilinear kernel destroys and a Lanczos kernel does not."""
    x, y = np.meshgrid(np.arange(w, dtype=np.int64), np.arange(h, dtype=np.int64))
    if kind == "gradient":
        return np.dstack([
            (x * 3) % 256,
            (y * 5) % 256,
            (x + y) % 256,
        ]).astype(np.uint8)
    if kind == "checker8":
        c = (((x // 8) + (y // 8)) % 2) * 255
        return np.dstack([c, (c + (x % 17)) % 256, (255 - c) % 256]).astype(np.uint8)
    if kind == "hardrect":
        rgb = np.dstack([(x * 2) % 256, (y * 4) % 256, ((x * y) % 2) * 255]).astype(np.uint8)
        rgb[max(0, h // 5):max(1, 3 * h // 5), max(0, w // 6):max(1, 4 * w // 6)] = (255, 255, 255)
        rgb[max(0, 2 * h // 5):h, max(0, 3 * w // 6):w] = (0, 0, 0)
        rgb[0, :] = (7, 139, 140)          # exact top-row edges: clamping lives here
        rgb[:, 0] = (250, 128, 6)
        rgb[-1, -1] = (255, 0, 255)
        return rgb
    if kind == "noise":
        return np.random.RandomState(0xC0FFEE ^ (w * 7919 + h)).randint(
            0, 256, size=(h, w, 3), dtype=np.uint8)
    raise AssertionError("unknown pattern " + kind)


def alpha(kind: str, w: int, h: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(w, dtype=np.int64), np.arange(h, dtype=np.int64))
    if kind == "opaque":
        return np.full((h, w), 255, dtype=np.uint8)
    if kind == "graded":
        return (((x * 2 + y * 3) % 256)).astype(np.uint8)
    if kind == "zero_band":       # exercises the unpremultiply divide-by-zero guard
        a = np.full((h, w), 255, dtype=np.int64)
        a[h // 3: max(h // 3 + 1, 2 * h // 3), :] = 0
        return a.astype(np.uint8)
    raise AssertionError("unknown alpha " + kind)


CASES = [
    # name,               src,       dst,       pattern,     alpha
    ("identity_64",       (64, 64),  (64, 64),  "gradient",  "opaque"),
    ("down2_128",         (128, 128), (64, 64), "checker8",  "opaque"),
    ("down_160x120_32",   (160, 120), (32, 32), "hardrect",  "opaque"),
    ("up3_32",            (32, 32),  (96, 96),  "gradient",  "opaque"),
    ("aspect_swap",       (128, 64), (64, 128), "hardrect",  "graded"),
    ("alpha_graded",      (96, 96),  (64, 64),  "noise",     "graded"),
    ("alpha_zero_band",   (64, 64),  (32, 48),  "gradient",  "zero_band"),
    ("alpha_opaque",      (96, 96),  (64, 64),  "checker8",  "opaque"),
    ("checker_alias",     (128, 128), (40, 40), "checker8",  "opaque"),
    ("row_128x1",         (128, 1),  (32, 1),   "hardrect",  "opaque"),
    ("col_1x128",         (1, 128),  (1, 32),   "hardrect",  "opaque"),
    ("prod_192x128",      (192, 128), (96, 64), "noise",     "opaque"),
    ("odd_100x100",       (100, 100), (33, 67), "noise",     "graded"),
    ("odd_up_17x19",      (17, 19),  (64, 64),  "checker8",  "opaque"),
]


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("oracle_resize.json")
    cases = []
    for name, (sw, sh), (dw, dh), kind, akind in CASES:
        rgb = pattern(kind, sw, sh)
        a = alpha(akind, sw, sh)
        src = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
        got = src.resize((dw, dh), resample=RESAMPLE, reducing_gap=None)
        assert got.size == (dw, dh) and got.mode == "RGBA", name

        # PIL's RGB path must agree with its premultiplied RGBA path for an opaque source; that is
        # what lets the engine use one code path for both JPEG and PNG references.
        if akind == "opaque":
            as_rgb = Image.fromarray(rgb, "RGB").resize((dw, dh), resample=RESAMPLE,
                                                        reducing_gap=None)
            rgb_arr = np.asarray(as_rgb).reshape(-1, 3)
            rgba_arr = np.asarray(got)[:, :, :3].reshape(-1, 3)
            assert np.array_equal(rgb_arr, rgba_arr), (
                name + ": opaque PIL RGBA path diverges from the PIL RGB path")

        png = io.BytesIO()
        src.save(png, format="PNG")
        cases.append({
            "name": name,
            "source_w": sw, "source_h": sh,
            "dst_w": dw, "dst_h": dh,
            "input_png_b64": base64.b64encode(png.getvalue()).decode("ascii"),
            "expected_rgba_b64": base64.b64encode(np.asarray(got).tobytes()).decode("ascii"),
        })
        print("%-18s %4dx%-4d -> %4dx%-4d  alpha=%-9s %s" %
              (name, sw, sh, dw, dh, akind, "ok"))

    doc = {
        "generated_by": "tests/models/qwen_image/gen_oracle_resize.py",
        "pil_version": PIL.__version__,
        "resample": "LANCZOS",
        "reducing_gap": None,
        "cases": cases,
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":")), encoding="ascii")
    print("wrote %s (%d bytes, %d cases, Pillow %s)" %
          (out_path, out_path.stat().st_size, len(cases), PIL.__version__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
