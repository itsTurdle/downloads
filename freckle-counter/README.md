# Freckle Counter

Counts freckles on face photos using classic computer vision (OpenCV) — no
training data or ML model required. Give it one or more photos and it reports a
per-image count plus a total. The two photos of a face are assumed to show
**different, non-overlapping sides**, so the total is simply the sum (no
crossover / double counting).

## How it works

It detects freckles by their **structure**, not just their color: a freckle is
a small spot whose center is darker and redder than a *surrounding ring of
lighter skin*. Plain color thresholding can't tell a freckle from a shadow or a
smear of redness — modeling the dark-center / light-ring shape can.

1. **Skin mask** — keeps facial skin and drops everything else: hair,
   eyebrows, eyes, lips, background, earbuds, and clothing. Uses a YCrCb skin
   gate plus an HSV hue gate (facial skin is orange, hue ≈ 0–28; a pink shirt
   is magenta, hue ≈ 177, so it's rejected). Only the largest connected skin
   region is kept, then its border is eroded inward so the hairline/jaw edge
   doesn't masquerade as freckles.

2. **Freckle signal** — CLAHE equalizes local contrast, then darkness is
   **gated by redness**: a spot's darkness only counts to the extent it is also
   redder than the local skin. Hair and stubble are dark but *not* red, so they
   are suppressed — darkness alone can't trigger a detection.

3. **Multi-scale Laplacian-of-Gaussian (LoG) blob detector** — the LoG is a
   center-surround operator: at each size it responds only where a dark/red
   center is wrapped by a lighter ring of the matching radius. Running it at
   several scales finds freckles big and small; this is the step that
   "understands what a freckle is."

4. **Peak + ring verification** — local maxima of the LoG response give one
   point per freckle. Each is then confirmed by an explicit center-vs-surround
   test (`--min-cs`) so edges, ridges, and plateaus that sneak a stray response
   through are discarded.

## Usage

```bash
pip install -r requirements.txt

# Count both sides and print the total
python3 freckle_counter.py samples/left_side.jpeg samples/right_side.jpeg

# Write annotated / mask / response images for inspection
python3 freckle_counter.py samples/*.jpeg --debug-dir debug
```

Example output:

```
left_side.jpeg: 681 freckles
right_side.jpeg: 676 freckles
TOTAL (no crossover assumed): 1357 freckles
```

## Tuning

The count depends on how faint a spot you're willing to call a freckle, so it's
adjustable:

| Flag | Default | Effect |
|------|---------|--------|
| `--sensitivity` | `2.5` | LoG response threshold, in std-devs above the median. **Lower = more / fainter freckles.** |
| `--redness-scale` | `6.0` | Redness (Lab a\*) needed to keep a spot at full weight. Higher rejects more grey hair / stubble. |
| `--min-cs` | `3.0` | Required strength of the lighter ring around a spot. Higher = stricter "real spot" test. |

`--debug-dir` writes three images per input: `*_annotated.png` (green circles
sized to each detected blob, non-skin dimmed), `*_skin.png` (the skin mask),
and `*_response.png` (the LoG blob-response map). The fastest way to tune is to
eyeball `*_annotated.png` against the original.

## Notes / limitations

- It's an estimator, not ground truth — freckle counting is inherently fuzzy
  and the result moves with the flags above. Inspect `--debug-dir` and tune to
  taste.
- Designed for reasonably sharp, well-lit close-ups. Heavy makeup, motion blur,
  or strong shadows will reduce accuracy. Out-of-focus skin patches are dropped
  by the skin mask and won't be counted.
