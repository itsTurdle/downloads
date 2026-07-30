"""Control test: does the motion mask still FIRE on real motion?

A detector that always returns zero scores a perfect false-positive rate, so the
quiet-scene measurement alone proves nothing. This drives _motion_mask directly with
a synthetic sequence where the ground truth is known:

  * a static wall at 3 m with per-pixel noise
  * a deliberately unstable band (emulating a monitor or blank wall, the regions that
    made the old detector fire everywhere) with 6x the noise
  * after the background settles, a solid object appears at 2 m and moves across

Wanted: high detection inside the object, near-zero elsewhere -- including inside the
noisy band, which is the case the previous version failed.
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HOME", r"B:\hf-cache")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import depth_model as dm

H, W = 150, 200
NOISY = slice(20, 45)          # the "monitor" band
OBJ_H, OBJ_W = 40, 40


class Bare(dm.DepthEstimator):
    """Reuse the temporal logic without loading a model."""
    def __init__(self):
        self.smooth = 0.0
        self.smooth_static = 0.0
        self.jump_m = 0.25
        self.ref_alpha = 0.05
        self.max_corr = 1.5
        self.stationary = True
        self.bg_alpha = 0.02
        self.motion_m = 0.12
        self.dev_alpha = 0.03
        self.dev_k = 3.0
        self.dev_floor = 0.02
        self.motion_blur = 3
        self._state = {}


def frame(i, rng, with_object):
    d = np.full((H, W), 3.0, dtype=np.float32)
    d += rng.normal(0, 0.04, (H, W))                 # ordinary instability
    d[NOISY] += rng.normal(0, 0.24, (NOISY.stop - NOISY.start, W))   # unstable band
    d += rng.normal(0, 0.05)                          # whole-frame drift
    box = None
    if with_object:
        x = 20 + (i % 100)
        y = 90
        box = (slice(y, y + OBJ_H), slice(x, min(x + OBJ_W, W)))
        d[box] = 2.0 + rng.normal(0, 0.02, d[box].shape)
    return d, box


est = Bare()
rng = np.random.default_rng(7)

# Settle the background and the learned per-pixel noise on a static scene.
SETTLE = 150
for i in range(SETTLE):
    d, _ = frame(i, rng, with_object=False)
    est._motion_mask(d)

# Quiet baseline.
quiet = []
for i in range(40):
    d, _ = frame(SETTLE + i, rng, with_object=False)
    m = np.frombuffer(est._motion_mask(d), dtype=np.uint8).reshape(H, W)
    quiet.append(m)
quiet = np.stack(quiet)

# Now introduce a moving object.
det_in, fp_out, fp_noisy = [], [], []
for i in range(40):
    d, box = frame(i, rng, with_object=True)
    m = np.frombuffer(est._motion_mask(d), dtype=np.uint8).reshape(H, W)
    inside = m[box] > 40
    outside = np.ones((H, W), bool)
    outside[box] = False
    outside[NOISY] = False
    det_in.append(inside.mean())
    fp_out.append((m[outside] > 40).mean())
    fp_noisy.append((m[NOISY] > 40).mean())

print(f"static scene, background settled ({SETTLE} frames):")
print(f"  false positives overall      {100 * (quiet > 40).mean():6.2f}%")
print(f"  false positives in the noisy band "
      f"{100 * (quiet[:, NOISY] > 40).mean():6.2f}%   <- the old failure case")
print()
print("with a solid object moving across:")
print(f"  detected inside the object   {100 * np.mean(det_in):6.2f}%")
print(f"  false positives elsewhere    {100 * np.mean(fp_out):6.2f}%")
print(f"  false positives noisy band   {100 * np.mean(fp_noisy):6.2f}%")

ok = (np.mean(det_in) > 0.75 and (quiet > 40).mean() < 0.02
      and np.mean(fp_out) < 0.05)
print("\n" + ("PASS: sensitive to real motion, quiet otherwise"
              if ok else "FAIL: detector is not usable"))
sys.exit(0 if ok else 1)
