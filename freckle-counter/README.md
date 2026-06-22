# Freckle Counter

Counts freckles on face photos using classic computer vision (OpenCV) — no
training data or ML model required. Give it one or more photos and it reports a
per-image count plus a total. The two photos of a face are assumed to show
**different, non-overlapping sides**, so the total is simply the sum (no
crossover / double counting).

## How it works

1. **Skin mask** — keeps facial skin and drops everything else: hair,
   eyebrows, eyes, lips, background, earbuds, and clothing. Uses a YCrCb skin
   gate plus an HSV hue gate (facial skin is orange, hue ≈ 0–28; a pink shirt
   is magenta, hue ≈ 177, so it's rejected). Only the largest connected skin
   region is kept, then its border is eroded inward so the hairline/jaw edge
   doesn't masquerade as freckles.

2. **Illumination flattening + contrast** — CLAHE equalizes local contrast, and
   a large blur estimates the local skin tone. Each pixel's darkness is then
   measured *relative* to that local tone, so a faint freckle on the bright
   forehead and one in the shaded jaw are judged on the same scale.

3. **Freckle test** — a pixel is freckle-like when it is both **darker** than
   the surrounding skin (`--min-contrast`) **and redder** in Lab a\*
   (`--min-redness`). Requiring both rejects plain shadows (dark, not red) and
   blush (red, not dark).

4. **Blob filtering** — connected blobs are kept only if their size, aspect
   ratio, and circularity look like a freckle (rejects pores, stray hairs, and
   wrinkles).

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
right_side.jpeg: 101 freckles
left_side.jpeg: 91 freckles
TOTAL (no crossover assumed): 192 freckles
```

## Tuning

The counts depend on how faint a spot you're willing to call a freckle. The
two knobs that matter most:

| Flag | Default | Effect |
|------|---------|--------|
| `--min-contrast` | `0.085` | Min darkness vs. local skin (fraction). **Lower = more / fainter freckles.** |
| `--min-redness` | `2.0` | Min Lab a\* above local skin. Lower = more freckles. |
| `--min-area` / `--max-area` | `3` / `120` | Allowed blob size (px at the 1100px working width). |
| `--min-circularity` | `0.55` | How round a blob must be. |

`--debug-dir` writes three images per input: `*_annotated.png` (green circles
on detections, non-skin dimmed), `*_skin.png` (the skin mask), and
`*_response.png` (the relative-darkness map).

## Notes / limitations

- It's an estimator, not ground truth — freckle counting is inherently fuzzy
  and the result moves with `--min-contrast`. Inspect `--debug-dir` and tune to
  taste.
- Designed for reasonably sharp, well-lit close-ups. Heavy makeup, motion blur,
  or strong shadows will reduce accuracy.
