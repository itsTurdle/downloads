#!/usr/bin/env python3
"""
Freckle counter.

Detects and counts freckles on a face photo using classic computer-vision
blob detection. Freckles are small, roughly circular spots that are darker
and redder than the surrounding skin, so the pipeline is:

  1. Build a skin mask (keep skin, drop hair / eyebrows / eyes / lips /
     background / clothing / earbuds).
  2. Inside the skin, run a black-top-hat to isolate small dark spots that
     stand out from the local skin tone.
  3. Threshold, then keep blobs whose size and shape look like freckles.

Run on one or more images; counts are summed across images. The two photos
of a face are assumed to show different (non-overlapping) sides, so the
total is just the sum.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def build_skin_mask(bgr):
    """Return a boolean mask of plausible skin pixels."""
    h, w = bgr.shape[:2]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Y, Cr, Cb = cv2.split(ycrcb)
    H, S, V = cv2.split(hsv)

    # Standard YCrCb skin gate.
    skin = (
        (Cr >= 133) & (Cr <= 183) &
        (Cb >= 77) & (Cb <= 127) &
        (Y >= 60)
    )

    # Drop very dark pixels (hair, eyebrows, pupils, deep shadow).
    skin &= V >= 60
    # Drop near-white background / earbuds (low saturation + very bright).
    skin &= ~((S < 40) & (V > 180))
    # Hue gate: facial skin is orange (OpenCV hue ~0-28). This rejects the
    # magenta/pink shirt (hue ~165-180) and most lip colour, which the
    # Cr/Cb gate lets through.
    skin &= (H <= 28) | (H >= 172)
    skin &= ~((H >= 172) & (S > 80))  # but kill saturated magenta (shirt)
    # Drop strongly saturated colour overall (lips).
    skin &= S < 150

    mask = skin.astype(np.uint8) * 255

    # Keep only the largest connected skin region, then fill holes so that
    # freckles inside the cheek are not excluded.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    # Erode the border inward so edge artifacts (hairline, jaw) don't fire.
    erode_px = max(5, int(0.012 * max(h, w)))
    mask = cv2.erode(mask,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (erode_px, erode_px)))
    # Fill interior holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask > 0


def detect_freckles(bgr, skin, min_area=3, max_area=120,
                    min_circularity=0.55, min_contrast=0.06, min_redness=1.6):
    """Return a list of (x, y, area) freckle detections inside the skin mask.

    A freckle is a small spot that is BOTH darker and redder than the skin
    immediately around it. To make the test robust to the strong lighting
    gradient across a face, we first *flatten the illumination*: estimate the
    local skin tone with a large blur, then measure each pixel's darkness as a
    fraction of that local tone. A pixel is freckle-like when

        relative darkness >= ``min_contrast``   (e.g. 8.5% darker than skin)
        AND local redness  >= ``min_redness``    (Lab a* above local skin)

    Working in *relative* contrast means a faint freckle on the bright
    forehead and a freckle in the shaded jaw are judged on the same scale.
    Lower ``min_contrast`` => more / fainter freckles.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)  # a*: green(-) .. red(+)

    # CLAHE equalizes local contrast so freckles read consistently whether the
    # skin patch is bright or dim, without blowing up global noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(lab[:, :, 0]).astype(np.float32)

    # Local skin tone (illumination + base colour) via a large blur. Sigma is
    # several freckle-widths so individual freckles don't pull their own
    # background down.
    bg_L = cv2.GaussianBlur(Lc, (0, 0), sigmaX=9)
    bg_A = cv2.GaussianBlur(A, (0, 0), sigmaX=9)

    # Relative darkness: how much darker than the surrounding skin (0..1).
    rel_dark = (bg_L - Lc) / (bg_L + 1e-3)
    rel_dark = np.clip(rel_dark, 0, None)
    # Local redness: a* above the surrounding skin.
    rel_red = np.clip(A - bg_A, 0, None)

    # Light smoothing so single-pixel noise doesn't form blobs.
    rel_dark = cv2.GaussianBlur(rel_dark, (3, 3), 0)
    rel_red = cv2.GaussianBlur(rel_red, (3, 3), 0)

    # Pull detections away from the skin border, where the hair/jaw/eyebrow
    # transition produces strong dark edges that are not freckles.
    core = cv2.erode(skin.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0

    response = np.where(core, rel_dark, 0.0).astype(np.float32)
    binimg = ((rel_dark >= min_contrast) & (rel_red >= min_redness) & core)
    binimg = binimg.astype(np.uint8) * 255

    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binimg, 8)
    freckles = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        wbb = stats[i, cv2.CC_STAT_WIDTH]
        hbb = stats[i, cv2.CC_STAT_HEIGHT]
        # Reject very elongated blobs (stray hairs / wrinkles / mask edges).
        if max(wbb, hbb) / max(1, min(wbb, hbb)) > 2.5:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        per = cv2.arcLength(cnts[0], True)
        if per == 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        if circ < min_circularity:
            continue
        cx, cy = centroids[i]
        freckles.append((float(cx), float(cy), int(area)))
    return freckles, _to_u8(response)


def _to_u8(img):
    """Normalize a float response map to a viewable 8-bit image."""
    m = float(img.max())
    if m <= 0:
        return np.zeros(img.shape, np.uint8)
    return np.clip(img / m * 255.0, 0, 255).astype(np.uint8)


def annotate(bgr, freckles, skin):
    out = bgr.copy()
    # Tint the non-skin area so it's clear what was excluded.
    overlay = out.copy()
    overlay[~skin] = (0, 0, 0)
    out = cv2.addWeighted(out, 0.75, overlay, 0.25, 0)
    for (x, y, _a) in freckles:
        cv2.circle(out, (int(round(x)), int(round(y))), 7, (0, 255, 0), 2)
    cv2.putText(out, f"count: {len(freckles)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)
    return out


WORK_WIDTH = 1100  # all detection runs at this width for stable parameters


def count_image(path, args):
    orig = cv2.imread(path)
    if orig is None:
        raise SystemExit(f"Could not read image: {path}")
    scale = WORK_WIDTH / orig.shape[1]
    bgr = cv2.resize(orig, (WORK_WIDTH, int(round(orig.shape[0] * scale))),
                     interpolation=cv2.INTER_AREA)

    skin = build_skin_mask(bgr)
    freckles, response = detect_freckles(
        bgr, skin,
        min_area=args.min_area, max_area=args.max_area,
        min_circularity=args.min_circularity,
        min_contrast=args.min_contrast, min_redness=args.min_redness)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.debug_dir, f"{base}_annotated.png"),
                    annotate(bgr, freckles, skin))
        cv2.imwrite(os.path.join(args.debug_dir, f"{base}_skin.png"),
                    (skin.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(args.debug_dir, f"{base}_response.png"), response)
    return len(freckles)


def main():
    p = argparse.ArgumentParser(description="Count freckles on face photos.")
    p.add_argument("images", nargs="+", help="One or more face image paths.")
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--max-area", type=int, default=120)
    p.add_argument("--min-circularity", type=float, default=0.55)
    p.add_argument("--min-contrast", type=float, default=0.06,
                   help="Min darkness vs. local skin, as a fraction (0.05-0.12). "
                        "LOWER = more / fainter freckles.")
    p.add_argument("--min-redness", type=float, default=1.6,
                   help="Min Lab a* above local skin (1.2-2.2). Higher rejects "
                        "stubble/hair (dark but not red). LOWER = more freckles.")
    p.add_argument("--debug-dir", default=None,
                   help="Write annotated / mask images here.")
    args = p.parse_args()

    total = 0
    for path in args.images:
        c = count_image(path, args)
        print(f"{os.path.basename(path)}: {c} freckles")
        total += c
    print(f"TOTAL (no crossover assumed): {total} freckles")


if __name__ == "__main__":
    sys.exit(main())
