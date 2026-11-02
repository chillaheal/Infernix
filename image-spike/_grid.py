"""Overlay a labelled pixel grid so head boxes can be read off by eye.

OpenCV 5 dropped CascadeClassifier, so face geometry comes from a visual read instead.
Usage: python -I _grid.py <in.jpg> <out.png> [step]
"""
import sys

import cv2


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        print("READ_FAIL", src)
        return 2
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    cv2.imwrite(dst, img)
    print("wrote", dst, w, "x", h, "step", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
