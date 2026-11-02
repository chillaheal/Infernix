"""Head swap composite, second try: keep her WHOLE hair, and grade the SKIN to HIS tanned body.

Why this file exists next to _facepaste.py (which the 21:59 run measured and STATE names, so it stays
untouched): faceswap/_p19_swap_tan.png shows the two defects, and both are pixels, not words.

  hair   the old alpha is an ellipse sized on HIS head box, so her hair -- far wider than his skull --
         is cut off at that boundary. Here the alpha is her own head+hair silhouette: GrabCut seeded
         with a colour prior, largest component, holes filled, and feathered ONLY at the outer boundary
         by a distance ramp, so the interior stays opaque and blonde strands are not blurred to nothing.
  tone   CLAUDE.md measures that text cannot re-grade skin in a preserved face. So the tan must be in
         the reference the model conditions on. The old harmonize() pulled Lab means 0.30 toward HIS
         FACE patch -- the patch about to be covered -- over the whole crop. Here the target is the
         body's skin, and the mapping runs on a skin mask only, with lightness and chroma moved by
         separate strengths, because a pale PINK face has to travel toward ORANGE chromaticity: scaling
         chroma along the existing hue angle cannot do that, and a blanket transform turns the face into
         an orange sheet and takes the lips with it.

  python -I _p20_paste.py MAN WOMAN OUT \
      --man-box x,y,w,h --woman-box x,y,w,h --head x,y,w,h --tone-box "x,y,w,h;x,y,w,h" \
      [--chroma-strength 0.75] [--light-strength 0.60] [--preserve 0.90] \
      [--scale-mult 1.06] [--hair-pad 1.15] [--feather 0.045] [--debug PREFIX]

man-box / woman-box are the FACE boxes (brow to chin, ear to ear); head is the man's FULL head the paste
must cover; tone-box is one or more clean skin patches on the man's body, in the man's coordinates.
"""
import sys

import cv2
import numpy as np


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def parse_box(text):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("box must be x,y,w,h, got %r" % text)
    return tuple(parts)


def robust(channel):
    """Statistics of the middle band of a distribution: the wall that leaks into a crop, a highlight or
    a stray earring must not be allowed to set the numbers the whole head is graded by."""
    low, high = np.percentile(channel, [5, 95])
    keep = channel[(channel >= low) & (channel <= high)]
    if keep.size < 64:
        keep = channel
    return float(np.median(keep)), float(keep.std())


def skin_stats(bgr, box, shrink=0.18):
    """Robust Lab statistics of the inner part of a box, where there is only skin."""
    x, y, w, h = box
    sx, sy = int(w * shrink), int(h * shrink)
    patch = bgr[y + sy:y + h - sy, x + sx:x + w - sx]
    if patch.size == 0:
        patch = bgr[y:y + h, x:x + w]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    return {c: robust(lab[..., c]) for c in range(3)}


def merged_stats(bgr, boxes):
    """Average the robust statistics of several body patches: no single patch of a chest carries the
    whole gradient of sunlight, highlights and shadow."""
    sets = [skin_stats(bgr, box) for box in boxes]
    return {c: (float(np.mean([s[c][0] for s in sets])), float(np.mean([s[c][1] for s in sets])))
            for c in range(3)}


def skin_alpha(crop_bgr, face_in_crop):
    """Where the tone mapping may run: cheeks, forehead, chin, neck. Not the hair, not the eyes,
    not the lips -- a face whose lips are also pushed toward the chest colour loses the mouth."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((hue <= 25) & (sat >= 35) & (sat <= 115) & (val >= 70)).astype(np.uint8)
    x, y, w, h = face_in_crop
    limit = np.zeros_like(m)
    limit[max(0, y - int(0.5 * h)):y + int(1.9 * h), max(0, x - int(0.45 * w)):x + int(1.45 * w)] = 1
    m = m * limit
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=3.0)


def grade(crop_bgr, src, dst, face_scaled, chroma_k, light_k, preserve):
    """Move the skin's Lab toward the body's, on the skin only, lightness and chroma at their own pace."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = lab.copy()
    for c in range(3):
        s_mean, s_std = src[c]
        t_mean, t_std = dst[c]
        if s_std < 1e-3:
            continue
        strength = light_k if c == 0 else chroma_k
        spread = (t_std / s_std) ** preserve
        out[..., c] = (lab[..., c] - s_mean) * spread + s_mean + (t_mean - s_mean) * strength
    mask = skin_alpha(crop_bgr, face_scaled)[..., None]
    blended = lab * (1.0 - mask) + out * mask
    print("grade dL=%+.1f dA=%+.1f dB=%+.1f skin=%.3f chroma_k=%.2f light_k=%.2f"
          % (dst[0][0] - src[0][0], dst[1][0] - src[1][0], dst[2][0] - src[2][0],
             float(mask.mean()), chroma_k, light_k), flush=True)
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def background_lab(crop_bgr):
    """The colour of her photo's background, read where the crop certainly has nothing but background."""
    h, w = crop_bgr.shape[:2]
    # The strip above her crown is nothing but her photo's wall, across the whole width: the median of it
    # is the background. Two corners were not enough -- the wall behind her head is a different, warmer
    # patch than the bright window at the corners, and that patch is exactly what came along as a grey
    # slab over her head.
    strip = crop_bgr[0:max(3, int(0.06 * h)), :, :]
    lab = strip.reshape(-1, 3).astype(np.float32)
    if lab.shape[0] < 32:
        return None
    bg = np.median(lab, axis=0)
    if bg[0] < 150.0:
        print("BACKGROUND_DOUBTFUL: the top strip is not bright (L=%.1f), trusting the colour prior alone"
              % bg[0], flush=True)
        return None
    return bg


def hair_alpha(crop_bgr, face_scaled, feather):
    """Her head and hair as one silhouette, feathered at its own edge only."""
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Blonde hair and fair skin are both chromatic; her blurred office wall and white blouse are not.
    chromatic = ((sat > 45) & (val > 40) & (hue > 3) & (hue < 60)).astype(np.uint8)
    mask = np.where(chromatic, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    # The wall is warm and grey-beige, which a hue and saturation rule alone cannot separate from blonde
    # hair, so compare against the wall itself instead: whatever matches the two corners the crop is
    # guaranteed to be background in is DEFINITE background, and the matte stops arguing about it.
    bg = background_lab(crop_bgr)
    if bg is not None:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        near_bg = np.linalg.norm(lab - bg[None, None, :], axis=-1) < 18.0
        mask[near_bg] = cv2.GC_BGD
    # Above the crown her photo is a bright, neutral office wall. Leaving it merely PR_BGD let GrabCut's
    # own model promote it: the first composite carried a grey arch over her head. Neutral and bright in
    # the upper part of the crop is definite background -- and only there, so the highlight on a forehead
    # or a neck cannot be punched out.
    neutral_bright = (sat <= 25) & (val >= 150)
    upper = np.zeros_like(neutral_bright)
    upper[:int(0.55 * h), :] = True
    mask[(neutral_bright & upper)] = cv2.GC_BGD
    x, y, fw, fh = face_scaled
    mask[y + int(0.15 * fh):y + int(0.85 * fh), x + int(0.15 * fw):x + int(0.85 * fw)] = cv2.GC_FGD
    edge = max(2, int(0.025 * max(h, w)))
    mask[:edge, :] = cv2.GC_BGD
    mask[-edge:, :] = cv2.GC_BGD
    mask[:, :edge] = cv2.GC_BGD
    mask[:, -edge:] = cv2.GC_BGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(crop_bgr, mask, (0, 0, w - 1, h - 1), bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if count > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (components == best).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    filled = fg.copy()
    cv2.floodFill(filled, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    fg = np.clip(fg + (1 - filled), 0, 1).astype(np.uint8)   # a hole inside the hair is hair, not wall
    # A distance ramp feathers the OUTLINE and leaves everything inside it opaque: a Gaussian on the
    # whole mask is what turns a blonde silhouette into a translucent one.
    ramp = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    alpha = np.clip(ramp / max(1.0, feather), 0.0, 1.0)
    print("alpha coverage=%.3f feather=%.1f px" % (float(fg.mean()), feather), flush=True)
    if float(fg.mean()) < 0.15 or float(fg.mean()) > 0.95:
        print("ALPHA_SUSPECT: %.3f of the crop is foreground, look at the debug image" % fg.mean(),
              flush=True)
    return alpha


def cut_above_hair(crop_bgr, alpha):
    """Drop the alpha above the first real strand in every column.

    Her photo's background is a bright, nearly neutral office wall, and GrabCut called the trapezoid of it
    above her crown foreground: three composites carried a grey slab over her head. Hair is darker and more
    chromatic than that wall, so per column the alpha ends where the first such pixel begins -- which is a
    statement about her silhouette rather than about a colour threshold the wall can imitate.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hairlike = (hsv[..., 2] < 205) & (hsv[..., 1] > 50)
    present = hairlike.any(axis=0)
    first = np.argmax(hairlike, axis=0)
    cut = np.where(present, first - 2, 0)
    rows = np.arange(alpha.shape[0])[:, None]
    dropped = rows < cut[None, :]
    removed = float((dropped & (alpha > 0.5)).mean())
    alpha = np.where(dropped, 0.0, alpha)
    print("hair cut: top row per column, %.3f of the crop's opaque area removed" % removed, flush=True)
    return alpha


def prefill_head(canvas, head, grow=8):
    """His own skull is removed before she arrives, so wherever her hair does not reach there is
    background instead of a dark patch of his scalp above her crown."""
    h, w = canvas.shape[:2]
    x, y, bw, bh = head
    y0 = max(0, y - grow)
    y1 = min(h, y + int(0.58 * bh))          # hair and scalp only; his jaw and traps stay untouched
    x0 = max(0, x - grow)
    x1 = min(w, x + bw + grow)
    span = x1 - x0
    source = x0 - span - 15
    if source >= 0:
        # Telea inpaint smears a flat grey over the area (visible as a slab over her head in the fourth
        # composite). The garden beside his head at the SAME rows is the honest content: real texture, the
        # right brightness, and the harmonize pass only has to continue it.
        out = canvas.copy()
        out[y0:y1, x0:x1] = canvas[y0:y1, source:source + span]
        return out
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(canvas, mask, 7, cv2.INPAINT_TELEA)


def main():
    man_p, woman_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    man_box = parse_box(arg("--man-box"))
    woman_box = parse_box(arg("--woman-box"))
    head = parse_box(arg("--head"))
    tone_boxes = [parse_box(part) for part in arg("--tone-box").split(";")]
    chroma_k = float(arg("--chroma-strength", "0.75"))
    light_k = float(arg("--light-strength", "0.60"))
    preserve = float(arg("--preserve", "0.90"))
    scale_mult = float(arg("--scale-mult", "1.06"))
    hair_pad = float(arg("--hair-pad", "1.15"))
    feather = float(arg("--feather", "0.045"))
    debug_p = arg("--debug")

    man = cv2.imread(man_p, cv2.IMREAD_COLOR)
    woman = cv2.imread(woman_p, cv2.IMREAD_COLOR)
    if man is None or woman is None:
        print("READ_FAIL man=%s woman=%s" % (man is None, woman is None))
        return 2
    print("man %dx%d woman %dx%d" % (man.shape[1], man.shape[0], woman.shape[1], woman.shape[0]), flush=True)
    print("man_face", man_box, "woman_face", woman_box, "man_head", head, "tone", tone_boxes, flush=True)

    x, y, w, h = woman_box
    h_img, w_img = woman.shape[:2]
    # The pads carry her crown, the longest hair and the neck, and stop there: a crop that reaches
    # further than her silhouette only invites the background into the matte.
    side = int(w * (0.70 + 0.35 * hair_pad))
    top = int(h * (0.30 + 0.45 * hair_pad))
    bottom = int(h * (0.40 + 0.40 * hair_pad))
    x0, x1 = max(0, x - side), min(w_img, x + w + side)
    y0, y1 = max(0, y - top), min(h_img, y + h + bottom)
    crop = woman[y0:y1, x0:x1]
    face_in_crop = (x - x0, y - y0, w, h)
    print("crop %dx%d at %d,%d" % (crop.shape[1], crop.shape[0], x0, y0), flush=True)

    scale = (man_box[2] * scale_mult) / float(woman_box[2])
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                      interpolation=cv2.INTER_CUBIC)
    face_scaled = (int(face_in_crop[0] * scale), int(face_in_crop[1] * scale),
                   int(face_in_crop[2] * scale), int(face_in_crop[3] * scale))
    ax = int(man_box[0] + man_box[2] / 2.0 - (face_scaled[0] + face_scaled[2] / 2.0))
    ay = int(man_box[1] + man_box[3] / 2.0 - (face_scaled[1] + face_scaled[3] / 2.0))
    print("scale %.3f paste at %d,%d" % (scale, ax, ay), flush=True)

    alpha = hair_alpha(crop, face_scaled, head[2] * feather)
    src = skin_stats(crop, face_scaled)
    dst = merged_stats(man, tone_boxes)
    crop = grade(crop, src, dst, face_scaled, chroma_k, light_k, preserve)

    canvas = prefill_head(man, head)
    x0p, y0p = max(0, ax), max(0, ay)
    x1p, y1p = min(canvas.shape[1], ax + crop.shape[1]), min(canvas.shape[0], ay + crop.shape[0])
    if x1p <= x0p or y1p <= y0p:
        print("PASTE_OFF_CANVAS %d,%d" % (ax, ay))
        return 4
    region = canvas[y0p:y1p, x0p:x1p]
    piece = crop[y0p - ay:y1p - ay, x0p - ax:x1p - ax]
    a = alpha[y0p - ay:y1p - ay, x0p - ax:x1p - ax].copy()

    # His own skull must not peek out anywhere, so his head box is inside the alpha too.
    ecx = int(head[0] + head[2] / 2.0) - x0p
    ecy = int(head[1] + head[3] / 2.0) - y0p
    yy, xx = np.mgrid[0:piece.shape[0], 0:piece.shape[1]]
    ell = (((xx - ecx) / (head[2] * 0.68)) ** 2 + ((yy - ecy) / (head[3] * 0.72)) ** 2 <= 1.0)
    a = np.maximum(a, ell.astype(np.float32))
    a = cut_above_hair(crop, a)
    # The head box can push the alpha up to the crop's own top edge, and a crop edge is a straight line:
    # that rectangle of hair floating over the garden is what read as a slab. Fade the top of the window.
    ramp = max(6, int(head[3] * 0.12))
    for i in range(min(ramp, a.shape[0])):
        a[i, :] *= (i / float(ramp)) ** 0.7
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=max(1.5, head[2] * 0.02), sigmaY=max(1.5, head[2] * 0.02))

    blended = piece.astype(np.float32) * a[..., None] + region.astype(np.float32) * (1.0 - a[..., None])
    canvas[y0p:y1p, x0p:x1p] = blended.astype(np.uint8)
    cv2.imwrite(out_p, canvas)
    print("wrote", out_p, canvas.shape[1], "x", canvas.shape[0], flush=True)
    if debug_p:
        cv2.imwrite(debug_p + "_alpha.png", (a * 255).astype(np.uint8))
        dbg = canvas.copy()
        for box, color in ((man_box, (0, 255, 0)), (head, (255, 0, 0))):
            cv2.rectangle(dbg, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 1)
        for box in tone_boxes:
            cv2.rectangle(dbg, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
        cv2.imwrite(debug_p + "_boxes.png", dbg)
        shown = (piece.astype(np.float32) * a[..., None]
                 + np.full_like(piece, 255, dtype=np.float32) * (1.0 - a[..., None])).astype(np.uint8)
        grid = shown.copy()
        for gx in range(0, grid.shape[1], 25):
            cv2.line(grid, (gx, 0), (gx, grid.shape[0]), (0, 255, 255), 1)
            cv2.putText(grid, str(gx), (gx + 1, 12), 0, 0.3, (0, 255, 255), 1)
        for gy in range(0, grid.shape[0], 25):
            cv2.line(grid, (0, gy), (grid.shape[1], gy), (0, 255, 255), 1)
            cv2.putText(grid, str(gy), (2, gy + 10), 0, 0.3, (0, 255, 255), 1)
        cv2.imwrite(debug_p + "_piece.png", grid)
        print("wrote debug", debug_p + "_alpha.png", debug_p + "_boxes.png", debug_p + "_piece.png",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
