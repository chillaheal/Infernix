"""Stage the source photo as a PNG the edit route can consume.

Usage: python -I _owl_prepare.py <src.webp|src.png> <dst.png>
"""
import sys

from PIL import Image


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    img = Image.open(src).convert("RGB")
    print("source", src, img.size)
    img.save(dst, "PNG")
    print("wrote", dst, img.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
