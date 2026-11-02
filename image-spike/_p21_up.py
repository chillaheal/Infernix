"""Upscale a composite to the size the model was trained to receive.

Why this exists: the man's photo is only 525x581, so the composite is 525x581 and every harmonize run so
far drew the head into a 512x576 canvas. The reference pipeline this engine mirrors resizes its condition
image to output_resolution, which defaults to 1024 on the long side, so a 525-wide reference is half the
pixel budget the DiT expects and the hair has no room to separate into strands. Lanczos carries no new
information, but it puts the reference on the grid the model was trained on.

  python -I _p21_up.py IN.png OUT.png WIDTHxHEIGHT
"""
import sys

from PIL import Image


def main():
    src, dst, size = sys.argv[1], sys.argv[2], sys.argv[3]
    width, height = (int(part) for part in size.lower().split("x"))
    image = Image.open(src).convert("RGB")
    print("src %s %s" % (src, image.size))
    image.resize((width, height), Image.LANCZOS).save(dst)
    print("wrote %s %dx%d" % (dst, width, height))
    return 0


if __name__ == "__main__":
    sys.exit(main())
