#!/usr/bin/env python3
"""
Freckle counter.

Detects and counts freckles on a face photo using classic computer-vision
blob detection. A freckle is modelled by its *structure*, not just its
colour: a small spot whose centre is darker and redder than a surrounding
ring of lighter skin, at some characteristic size. The pipeline is:

  1. Build a skin mask (keep skin, drop hair / eyebrows / eyes / lips /
     background / clothing / earbuds).
  2. Build a "freckle signal" = local darkness gated by redness, so dark
     hairs/stubble (not red) are suppressed.
  3. Run a multi-scale Laplacian-of-Gaussian (a centre-surround blob
     operator) and keep its local maxima -- spots with a genuine lighter
     ring around them -- verified by an explicit centre-vs-surround test.

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


def detect_freckles(bgr, skin, sensitivity=2.5, redness_scale=6.0,
                    min_cs=3.0, scales=(1.6, 2.2, 3.0, 4.0)):
    """Return ``(detections, response_map)`` where each detection is
    ``(x, y, radius)``.

    Rather than thresholding on colour alone, this models what a freckle *is*:
    a small spot whose centre is darker/redder than a surrounding ring of
    lighter skin, at some characteristic size. That is a textbook blob, so we
    use a scale-normalised **Laplacian-of-Gaussian (LoG)** detector.

    Pipeline:
      1. Build a "freckle signal" S that is high where skin is dark AND red
         (CLAHE-equalised darkness + redness above the local skin tone). Hairs
         and stubble are dark but not red, so they score low.
      2. Run the LoG at several ``scales``. The LoG is a centre-surround
         operator: it fires only when a dark/red centre is wrapped by a lighter
         ring of the matching radius -- i.e. it understands the *shape* of a
         freckle, not just its colour. Taking the max across scales handles
         freckles of different sizes.
      3. Keep local maxima of that response (one peak per freckle), then
         verify each peak really is centre-darker-than-surround by an explicit
         ring test (``min_cs``). This rejects edges, ridges and plateaus that
         can still produce a stray LoG response.

    Lower ``sensitivity`` lowers the response threshold (more / fainter
    freckles). Higher ``redness_scale`` demands more redness before a spot
    counts, rejecting more grey hair/stubble. Higher ``min_cs`` demands a
    stronger lighter-ring around each spot.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1].astype(np.float32)  # a*: green(-) .. red(+)

    # CLAHE so faint freckles on bright and dim skin read on the same scale.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(lab[:, :, 0]).astype(np.float32)

    # Redness relative to the broad local skin tone.
    bg_A = cv2.GaussianBlur(A, (0, 0), sigmaX=12)
    rel_red = np.clip(A - bg_A, 0, None)

    # Freckle signal: darkness GATED by redness. A freckle is dark AND red, so
    # we modulate the darkness blob by how red the spot is. This is the key to
    # not counting stubble/hair: those are dark but not red, so the gate keeps
    # only ~20% of their darkness and they fall below threshold, while real
    # red-brown freckles keep their full amplitude. Gating (vs. adding redness)
    # is what makes darkness alone insufficient to trigger a detection.
    red_gate = np.clip(rel_red / max(1e-3, redness_scale), 0.0, 1.0)
    signal = (255.0 - Lc) * (0.2 + 0.8 * red_gate)
    signal = cv2.GaussianBlur(signal, (0, 0), sigmaX=0.8)  # kill pixel noise

    core = cv2.erode(skin.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0

    # --- multi-scale LoG blob response -------------------------------------
    # For a bright blob in `signal`, the Laplacian is negative at the centre,
    # so -(sigma^2 * Laplacian) peaks (positive) at freckle centres. The
    # sigma^2 factor makes responses comparable across scales.
    responses = []
    radii = []
    for s in scales:
        g = cv2.GaussianBlur(signal, (0, 0), sigmaX=s)
        log = cv2.Laplacian(g, cv2.CV_32F, ksize=1)
        responses.append(-(s * s) * log)
        radii.append(s * np.sqrt(2.0))  # blob radius for a LoG at scale s
    stack = np.stack(responses, axis=0)
    R = stack.max(axis=0)
    best_scale = stack.argmax(axis=0)
    R[~core] = 0.0

    # Threshold relative to the in-skin response distribution.
    vals = R[core & (R > 0)]
    if vals.size == 0:
        return [], _to_u8(R)
    thr = float(np.median(vals)) + sensitivity * float(vals.std())

    # Local maxima: a pixel that equals the local max and clears the threshold
    # is one freckle centre. The dilation radius sets the minimum spacing.
    local_max = cv2.dilate(R, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    peaks = (R >= local_max - 1e-5) & (R > thr) & core

    # Collapse clusters of equal-valued peak pixels to single centroids.
    n, _lab, _st, cents = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), 8)

    # Precompute centre/surround means of `signal` for the ring test.
    centre_mean = cv2.GaussianBlur(signal, (0, 0), sigmaX=1.5)
    surround_mean = cv2.GaussianBlur(signal, (0, 0), sigmaX=6.0)
    cs = centre_mean - surround_mean  # >0 where centre stands out from ring
    cs_thr = min_cs * float(cs[core].std())

    H, W = R.shape
    freckles = []
    for i in range(1, n):
        x, y = cents[i]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        # Ring test: centre must genuinely stand out from its lighter surround.
        if cs[yi, xi] < cs_thr:
            continue
        freckles.append((float(x), float(y), float(radii[best_scale[yi, xi]])))
    return freckles, _to_u8(R)


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
    for (x, y, r) in freckles:
        rad = max(4, int(round(r * 2.0)))
        cv2.circle(out, (int(round(x)), int(round(y))), rad, (0, 255, 0), 2)
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
        sensitivity=args.sensitivity, redness_scale=args.redness_scale,
        min_cs=args.min_cs)

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
    p.add_argument("--sensitivity", type=float, default=2.5,
                   help="LoG response threshold in std-devs above the median "
                        "(1.5-3.0). LOWER = more / fainter freckles.")
    p.add_argument("--redness-scale", type=float, default=6.0,
                   help="Lab a* (redness) needed for a spot to keep full weight. "
                        "HIGHER rejects more grey hair / stubble.")
    p.add_argument("--min-cs", type=float, default=0.6,
                   help="Centre-vs-surround ring strength a blob must clear "
                        "(in std-devs). Higher = stricter 'real spot' test.")
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
