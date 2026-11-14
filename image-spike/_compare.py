"""Side-by-side montage for the identity check: original woman | swap result.

Usage: python -I _compare.py <woman.jpg> <swap.png> <out.png>
"""
import sys

import cv2
import numpy as np


def main() -> int:
    woman_p, swap_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    a = cv2.imread(woman_p, cv2.IMREAD_COLOR)
    b = cv2.imread(swap_p, cv2.IMREAD_COLOR)
    if a is None or b is None:
        print("READ_FAIL", a is None, b is None)
        return 2
    target_h = 512
    a = cv2.resize(a, (int(a.shape[1] * target_h / a.shape[0]), target_h),
                   interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (int(b.shape[1] * target_h / b.shape[0]), target_h),
                   interpolation=cv2.INTER_AREA)
    gap = np.full((target_h, 12, 3), 255, np.uint8)
    cv2.imwrite(out_p, np.hstack([a, gap, b]))
    print("wrote", out_p, target_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
