"""Measure face vs chest skin tone in a swap output: is the head still paler than the body?

Usage: python -I _tone.py <swap.png> [...]
Regions are the face patch and the chest patch of the 544x608 route, scaled from the composite
geometry (man face box 180,68,108,110 and the pecs around 250,320 in the 525x581 source).
"""
import sys

import numpy as np
from PIL import Image


def mean(img, box):
    x, y, w, h = box
    return np.asarray(img.crop((x, y, x + w, y + h)), dtype=np.float64).reshape(-1, 3).mean(axis=0)


def main() -> int:
    for path in sys.argv[1:]:
        img = Image.open(path).convert("RGB")
        sx = img.width / 544.0
        sy = img.height / 608.0
        face = (int(205 * sx), int(110 * sy), int(75 * sx), int(55 * sy))
        chest = (int(250 * sx), int(320 * sy), int(70 * sx), int(40 * sy))
        f = mean(img, face)
        c = mean(img, chest)
        print("%-28s face RGB %6.1f %6.1f %6.1f | chest %6.1f %6.1f %6.1f | face-chest %5.1f %5.1f %5.1f"
              % (path.split("\\")[-1], f[0], f[1], f[2], c[0], c[1], c[2],
                 f[0] - c[0], f[1] - c[1], f[2] - c[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
