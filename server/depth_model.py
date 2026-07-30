"""Monocular metric depth on the GPU, for phones with no LiDAR scanner.

The iPhone Air has a single rear camera, so it cannot produce a live dense depth map
in hardware -- but ARKit world tracking still gives a real 6DoF pose. So the phone
sends colour + pose and this module supplies the depth, which puts the frame back
into exactly the shape the viewer already consumes.

Depth Anything V2 "Metric Indoor" outputs metres directly. Absolute scale from a
single image is inherently ambiguous, so `scale` is exposed for calibration: point
the phone at something you can measure and adjust until the readout matches.
"""

import os
import threading

os.environ.setdefault("HF_HOME", r"B:\hf-cache")

import io

import numpy as np
from PIL import Image

# Base rather than Small, at the checkpoint's native 518px: measured on an RTX PRO
# 6000 the two are 11.3 ms vs 8.9 ms per frame, and latency is nearly flat across
# input sizes -- the work is launch-overhead bound, not compute bound. Both are far
# above the ~20 fps the stream needs, so the better model costs nothing that matters.
DEFAULT_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"

# DINOv2 uses 14x14 patches, so model input dims must be multiples of 14.
PATCH = 14


def _round_to_patch(v: int) -> int:
    return max(PATCH, int(round(v / PATCH)) * PATCH)


class DepthEstimator:
    """Not thread-safe by design: the bridge drives it from a single worker thread
    and drops frames while it is busy, rather than queueing stale work."""

    def __init__(self, repo: str = DEFAULT_REPO, input_height: int = 518,
                 out_width: int = 512, out_height: int = 384,
                 scale: float = 1.0, device: str = "cuda"):
        import torch
        from transformers import AutoModelForDepthEstimation

        self.torch = torch
        self.repo = repo
        self.out_width = out_width
        self.out_height = out_height
        self.scale = scale
        self.input_height = _round_to_patch(input_height)
        self.lock = threading.Lock()

        # --- temporal stabilisation -----------------------------------------
        # Measured on a live stream from a *stationary* phone: per-pixel depth wanders
        # 0.055 m frame to frame, and the whole-frame median moves 1.03% on average
        # (4.18% worst). Only ~12% of that wobble is global scale; the rest is
        # per-pixel noise. Both are visible -- the global part as the cloud breathing
        # in and out, the per-pixel part as a boiling surface -- so both get treated.
        self.smooth = 0.6           # per-pixel EMA weight on the previous frame
        self.smooth_static = 0.88   # heavier when the camera is declared stationary
        self.jump_m = 0.25          # bigger change than this is real; do not blend it

        # Global scale: anchor a robust percentile to a slowly moving reference. Fast
        # flicker is removed while genuine approach/retreat still comes through, just
        # over about a second.
        self.ref_alpha = 0.05
        self.max_corr = 1.5         # never rescale a frame by more than this

        # Per-frame-shape temporal history; see _slot().
        self._state: dict = {}

        # --- stationary mode ------------------------------------------------
        # A fixed camera lets the average run much longer, which is worth roughly a
        # 3x noise reduction, and makes the difference from that average a clean mask
        # of whatever moved in front of an otherwise static scene.
        self.stationary = False
        self.bg_alpha = 0.02        # background adapts over ~5 s
        self.motion_m = 0.12        # depth change counting as full-strength motion

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device

        self.model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
        # Lets cuDNN pick fast algorithms; input shapes are constant frame to frame.
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.gpu_name = (torch.cuda.get_device_name(0)
                         if device == "cuda" else "cpu")

        # ImageNet normalisation, matching the checkpoint's image processor. Kept as
        # device tensors so the whole pre/post chain stays on the GPU -- doing the
        # resizes and normalisation with PIL/numpy instead measured 32 ms per frame
        # against 11 ms for the model, i.e. the plumbing cost 3x the inference.
        self.mean_t = torch.tensor([0.485, 0.456, 0.406],
                                   device=device).view(1, 3, 1, 1)
        self.std_t = torch.tensor([0.229, 0.224, 0.225],
                                  device=device).view(1, 3, 1, 1)

        self._warm()

    def reset_temporal(self):
        self._state.clear()

    def _slot(self, shape):
        """Temporal state, kept per frame shape.

        More than one source can be connected at once (a phone plus the synthetic
        tester, say), and their scenes have nothing to do with each other. Sharing one
        history across them would disable the EMA on every shape change and, worse,
        let one scene's scale reference drag the other's. Shape is a cheap stand-in for
        source identity.
        """
        s = self._state.get(shape)
        if s is None:
            s = {"prev": None, "bg": None, "ref": None}
            self._state[shape] = s
        return s

    def _stabilise(self, depth: np.ndarray) -> np.ndarray:
        valid = depth > 0.05
        if not valid.any():
            return depth
        s = self._slot(depth.shape)

        # 1. Global scale, against a slow reference. The 40th percentile is robust to
        #    outliers and weighted toward nearer content, which is the part the model
        #    predicts most reliably.
        m = float(np.percentile(depth[valid], 40))
        if m > 1e-3:
            if s["ref"] is None:
                s["ref"] = m
            else:
                s["ref"] += self.ref_alpha * (m - s["ref"])
            corr = float(np.clip(s["ref"] / m, 1.0 / self.max_corr, self.max_corr))
            depth = depth * corr

        # 2. Per-pixel EMA, skipped where the change is too large to be noise so that
        #    edges and real movement stay crisp.
        a = self.smooth_static if self.stationary else self.smooth
        prev = s["prev"]
        if prev is not None and prev.shape == depth.shape and a > 0:
            blended = a * prev + (1.0 - a) * depth
            depth = np.where(np.abs(depth - prev) < self.jump_m, blended, depth)

        s["prev"] = depth
        return depth

    def _motion_mask(self, depth: np.ndarray) -> bytes:
        """0..255 per pixel: how far this pixel is from the long-term background."""
        s = self._slot(depth.shape)
        if s["bg"] is None or s["bg"].shape != depth.shape:
            s["bg"] = depth.copy()
        else:
            s["bg"] += self.bg_alpha * (depth - s["bg"])
        diff = np.abs(depth - s["bg"])
        return (np.clip(diff / max(self.motion_m, 1e-3), 0, 1)
                * 255).astype(np.uint8).tobytes()

    def _warm(self):
        """Run once so kernel autotuning does not land on the first real frame."""
        h = self.input_height
        w = _round_to_patch(h * 4 / 3)
        self._run(np.zeros((h, w, 3), dtype=np.uint8))

    def _run(self, rgb: np.ndarray) -> tuple[bytes, int, int]:
        torch = self.torch
        F = torch.nn.functional
        ih, iw = rgb.shape[0], rgb.shape[1]

        with torch.inference_mode():
            t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float()
            t = t.div_(255.0)

            # Preserve aspect ratio, snap both dims to the patch grid.
            h = self.input_height
            w = _round_to_patch(iw * h / max(ih, 1))
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            t = (t - self.mean_t) / self.std_t

            with torch.autocast(self.device, dtype=torch.float16):
                pred = self.model(pixel_values=t).predicted_depth
            pred = pred.float()
            if pred.dim() == 3:
                pred = pred[:, None]        # (B,H,W) -> (B,1,H,W)

            # Resample to the output grid, keeping the source aspect so the cloud is
            # not stretched. (The bridge rescales fx and fy independently, so a
            # mismatched aspect would still unproject correctly, just waste pixels.)
            out_w = self.out_width
            out_h = max(1, int(round(out_w * ih / max(iw, 1))))
            pred = F.interpolate(pred, size=(out_h, out_w), mode="bilinear",
                                 align_corners=False)

            d = pred[0, 0] * self.scale
            # Non-finite predictions become "no reading" rather than a wild point.
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            depth = d.cpu().numpy()             # metres, float32

        depth = self._stabilise(depth)
        mask = self._motion_mask(depth) if self.stationary else b""

        mm = np.clip(depth * 1000.0, 0, 65535).astype("<u2")
        return mm.tobytes(), out_w, out_h, mask

    def infer(self, jpeg: bytes) -> tuple[bytes, int, int, bytes]:
        """JPEG -> (uint16 LE millimetre depth, width, height, motion mask or b'')."""
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        # np.array (not asarray): PIL hands back a read-only buffer, and wrapping that
        # with torch.from_numpy is documented undefined behaviour.
        return self._run(np.array(img))

    def describe(self) -> str:
        return (f"{self.repo.split('/')[-1]} @ {self.input_height}px in, "
                f"{self.out_width}px wide out, on {self.gpu_name}")
