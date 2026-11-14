"""Build a coarse head-swap composite: the woman's head onto the man's body.

The live edit route takes exactly one reference image (serve/openai_images.cpp:201), so the face
transfer happens here in pixels and the diffusion pass is only asked to harmonize the seams.

OpenCV 5.0.0 dropped CascadeClassifier, so the geometry is given explicitly as boxes read off the
grid overlay (_grid.py). Usage:

  python -I _facepaste.py <man.jpg> <woman.jpg> <out.png> \
      --man-box x,y,w,h --woman-box x,y,w,h --head x,y,w,h [--debug boxes.png]

man-box / woman-box are the FACE boxes (brow to chin, ear to ear); head is the man's FULL head
(hair to chin) that the paste must cover.
"""
import sys

import cv2
import numpy as np


def parse_box(text):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("box must be x,y,w,h")
    return tuple(parts)


def crop_head(img, box, top_pad=0.85, side_pad=0.70, bottom_pad=0.45):
    """Crop around a face box, generous enough to carry the whole head and hair."""
    x, y, w, h = box
    h_img, w_img = img.shape[:2]
    x0 = max(0, int(x - side_pad * w))
    x1 = min(w_img, int(x + w + side_pad * w))
    y0 = max(0, int(y - top_pad * h))
    y1 = min(h_img, int(y + h + bottom_pad * h))
    return img[y0:y1, x0:x1], (x - x0, y - y0, w, h)


def harmonize(pasted, target_bgr, box, strength=0.30):
    """Pull the pasted head's Lab statistics toward the body skin tone, partially."""
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    patch = target_bgr[y:y + h, x:x + w]
    if patch.size == 0:
        return pasted
    src = cv2.cvtColor(pasted, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bs = cv2.cvtColor(cv2.resize(pasted, (w, h)), cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        s_mean, s_std = bs[..., c].mean(), bs[..., c].std() + 1e-6
        t_mean, t_std = a[..., c].mean(), a[..., c].std() + 1e-6
        src[..., c] = (src[..., c] - s_mean) * ((t_std / s_std) ** strength) \
            + s_mean + (t_mean - s_mean) * strength
    return cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def main() -> int:
    man_p, woman_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    man_box = parse_box(arg("--man-box"))
    woman_box = parse_box(arg("--woman-box"))
    head = parse_box(arg("--head"))
    debug_p = arg("--debug")

    man = cv2.imread(man_p, cv2.IMREAD_COLOR)
    woman = cv2.imread(woman_p, cv2.IMREAD_COLOR)
    if man is None or woman is None:
        print("READ_FAIL", man is None, woman is None)
        return 2
    print("man", man.shape[1], "x", man.shape[0], "woman", woman.shape[1], "x", woman.shape[0])
    print("man_face", man_box, "woman_face", woman_box, "man_head", head)

    scale_mult = float(arg("--scale-mult", "1.06"))
    tone = float(arg("--tone", "0.30"))
    crop, anchor = crop_head(woman, woman_box)
    scale = (man_box[2] * scale_mult) / float(woman_box[2])
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                      interpolation=cv2.INTER_CUBIC)
    anchor = (int(anchor[0] * scale), int(anchor[1] * scale),
              int(anchor[2] * scale), int(anchor[3] * scale))
    # place the scaled face centre on the man's face centre
    ax = int(man_box[0] + man_box[2] / 2.0 - (anchor[0] + anchor[2] / 2.0))
    ay = int(man_box[1] + man_box[3] / 2.0 - (anchor[1] + anchor[3] / 2.0))
    print("scale", round(scale, 3), "paste at", ax, ay, "crop", crop.shape[1], "x", crop.shape[0])

    crop = harmonize(crop, man, man_box, strength=tone)

    canvas = man.copy()
    x0, y0 = max(0, ax), max(0, ay)
    x1, y1 = min(canvas.shape[1], ax + crop.shape[1]), min(canvas.shape[0], ay + crop.shape[0])
    if x1 <= x0 or y1 <= y0:
        print("PASTE_OFF_CANVAS", ax, ay)
        return 4
    region = canvas[y0:y1, x0:x1]
    piece = crop[y0 - ay:y1 - ay, x0 - ax:x1 - ax]

    # the alpha covers the man's whole head box (soft), so his skull cannot peek out
    ecx = int(head[0] + head[2] / 2.0) - x0
    ecy = int(head[1] + head[3] / 2.0) - y0
    # analytic ellipse mask (OpenCV 5 dropped the ellipse overload this used)
    ax_a, ay_a = head[2] * 0.60, head[3] * 0.62
    yy, xx = np.mgrid[0:piece.shape[0], 0:piece.shape[1]]
    mask = (((xx - ecx) / ax_a) ** 2 + ((yy - ecy) / ay_a) ** 2 <= 1.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(4, int(head[2] * 0.10)))[..., None]

    blended = piece.astype(np.float32) * mask + region.astype(np.float32) * (1.0 - mask)
    canvas[y0:y1, x0:x1] = blended.astype(np.uint8)

    cv2.imwrite(out_p, canvas)
    print("wrote", out_p, canvas.shape[1], "x", canvas.shape[0])
    if debug_p:
        dbg = canvas.copy()
        cv2.line(dbg, (ecx + x0 - int(ax_a), ecy + y0), (ecx + x0 + int(ax_a), ecy + y0),
                 (0, 0, 255), 1)
        cv2.line(dbg, (ecx + x0, ecy + y0 - int(ay_a)), (ecx + x0, ecy + y0 + int(ay_a)),
                 (0, 0, 255), 1)
        for box, color in ((man_box, (0, 255, 0)), (head, (255, 0, 0))):
            cv2.rectangle(dbg, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 1)
        cv2.imwrite(debug_p, dbg)
        print("wrote debug", debug_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
