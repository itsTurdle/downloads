"""Fake iPhone: streams a raytraced room so the display can be built and tested
without the phone.

    python synth_phone.py --host 127.0.0.1

Geometry is static and the camera orbits, which is exactly the condition that
exposes a wrong view matrix: if the unprojection or the ARKit axis convention is
off, accumulate mode smears the room instead of building one copy of it.

Matches the real app's conventions:
  * depth is PLANAR distance along camera -z (what ARKit sceneDepth gives), not
    radial distance
  * camera space is +x right, +y up, -z forward
  * the view matrix is camera->world, column-major
"""

import argparse
import asyncio
import io
import json
import math
import struct
import time

import numpy as np
from PIL import Image

import protocol

DW, DH = 256, 192
FX = FY = 221.7  # ~60 degrees horizontal, same ballpark as an iPhone depth map
CX, CY = DW / 2.0, DH / 2.0

ROOM_MIN = np.array([-2.5, -1.5, -5.0])
ROOM_MAX = np.array([2.5, 1.6, 1.5])

SPHERES = [
    (np.array([-0.9, -0.85, -2.2]), 0.55, np.array([220, 70, 60])),
    (np.array([1.1, -0.15, -3.6]), 0.75, np.array([70, 130, 230])),
]
CUBE_MIN = np.array([-0.45, -1.5, -3.9])
CUBE_MAX = np.array([0.5, -0.35, -3.0])
CUBE_COLOR = np.array([230, 200, 90])

WALL_COLORS = {
    0: np.array([150, 150, 158]),  # -x
    1: np.array([150, 150, 158]),  # +x
    2: np.array([120, 118, 112]),  # floor
    3: np.array([200, 200, 205]),  # ceiling
    4: np.array([165, 160, 150]),  # far wall
    5: np.array([165, 160, 150]),  # behind
}


def pixel_rays():
    """Camera-space ray directions with z == -1, so t is planar depth directly."""
    u, v = np.meshgrid(np.arange(DW, dtype=np.float32),
                       np.arange(DH, dtype=np.float32))
    x = (u + 0.5 - CX) / FX
    y = -(v + 0.5 - CY) / FY  # image v grows downward, camera y grows up
    z = -np.ones_like(x)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


RAYS = pixel_rays()


def configure(dw: int, dh: int, hfov_deg: float = 60.0):
    """Re-render at a different resolution. RGB-only mode wants a bigger frame,
    since the depth model works off the colour image rather than a 256x192 map."""
    global DW, DH, FX, FY, CX, CY, RAYS
    DW, DH = dw, dh
    FX = FY = (dw / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    CX, CY = dw / 2.0, dh / 2.0
    RAYS = pixel_rays()


def box_exit(o, d, lo, hi):
    """t at which a ray already inside an AABB leaves it, plus which face it hits."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (lo - o) / d
        t_hi = (hi - o) / d
    t_far = np.maximum(t_lo, t_hi)
    t_exit = np.nanmin(t_far, axis=1)
    axis = np.nanargmin(t_far, axis=1)
    face = axis * 2 + (d[np.arange(len(d)), axis] > 0).astype(np.int64)
    return t_exit, face


def box_enter(o, d, lo, hi):
    """t at which a ray outside an AABB enters it; inf when it misses."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (lo - o) / d
        t_hi = (hi - o) / d
    t_near = np.nanmax(np.minimum(t_lo, t_hi), axis=1)
    t_far = np.nanmin(np.maximum(t_lo, t_hi), axis=1)
    hit = (t_far >= np.maximum(t_near, 0)) & (t_near > 0)
    return np.where(hit, t_near, np.inf)


def sphere_hit(o, d, center, radius):
    oc = o - center
    a = np.einsum("ij,ij->i", d, d)
    b = 2.0 * np.einsum("ij,ij->i", oc, d)
    c = np.einsum("ij,ij->i", oc, oc) - radius * radius
    disc = b * b - 4 * a * c
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t = (-b - sq) / (2 * a)
    return np.where(ok & (t > 0), t, np.inf)


def render(view: np.ndarray, frame_i: int):
    """Raytrace one depth+colour frame from a camera->world matrix."""
    rot = view[:3, :3]
    origin = view[:3, 3]

    d = RAYS @ rot.T
    o = np.broadcast_to(origin, d.shape)

    t, face = box_exit(o, d, ROOM_MIN, ROOM_MAX)
    normal = np.zeros_like(d)
    axis = face // 2
    normal[np.arange(len(d)), axis] = np.where(face % 2 == 1, -1.0, 1.0)
    color = np.stack([WALL_COLORS[int(f)] for f in range(6)])[face].astype(np.float32)

    for center, radius, col in SPHERES:
        ts = sphere_hit(o, d, center, radius)
        closer = ts < t
        if closer.any():
            hit_pt = o[closer] + ts[closer, None] * d[closer]
            n = hit_pt - center
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            normal[closer] = n
            color[closer] = col
            t = np.where(closer, ts, t)

    tc = box_enter(o, d, CUBE_MIN, CUBE_MAX)
    closer = tc < t
    if closer.any():
        # Face normal = whichever slab produced t_near.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (CUBE_MIN - o[closer]) / d[closer]
            t_hi = (CUBE_MAX - o[closer]) / d[closer]
        which = np.nanargmax(np.minimum(t_lo, t_hi), axis=1)
        n = np.zeros((closer.sum(), 3), dtype=np.float32)
        n[np.arange(len(n)), which] = -np.sign(d[closer][np.arange(len(n)), which])
        normal[closer] = n
        color[closer] = CUBE_COLOR
        t = np.where(closer, tc, t)

    # Shade so the RGB channel carries real structure to look at.
    light = np.array([0.35, 0.9, 0.25])
    light = light / np.linalg.norm(light)
    lam = np.clip(np.einsum("ij,j->i", normal, light), 0.0, 1.0)
    shade = (0.35 + 0.65 * lam)[:, None]
    rgb = np.clip(color * shade, 0, 255).astype(np.uint8).reshape(DH, DW, 3)

    depth = t.astype(np.float32)

    # Sensor character: range-dependent noise, and dropouts on grazing surfaces and
    # far corners the way a real LiDAR loses return.
    rng = np.random.default_rng(1234 + frame_i)
    depth += rng.normal(0.0, 0.004 + 0.006 * depth, depth.shape).astype(np.float32)
    grazing = np.abs(np.einsum("ij,ij->i", normal, d / np.linalg.norm(d, axis=1, keepdims=True)))
    invalid = (rng.random(depth.shape) < 0.015) | (grazing < 0.12) | (depth > 9.0)
    depth[invalid] = 0.0

    conf = np.full(depth.shape, 2, dtype=np.uint8)
    conf[depth > 3.5] = 1
    conf[grazing < 0.3] = 1
    conf[depth > 6.0] = 0
    conf[invalid] = 0

    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype("<u2").reshape(DH, DW)
    return depth_mm, conf.reshape(DH, DW), rgb


def orbit_view(t: float) -> np.ndarray:
    """Camera->world for a slow orbit that also drifts in height and yaw."""
    radius = 1.25 + 0.35 * math.sin(t * 0.31)
    ang = t * 0.28
    eye = np.array([radius * math.sin(ang), 0.12 * math.sin(t * 0.53), -1.1 + radius * math.cos(ang) * 0.5])
    target = np.array([0.0, -0.55, -3.1])

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -fwd  # camera looks down -z
    view[:3, 3] = eye
    return view


async def run(host: str, port: int, fps: float, jpeg_quality: int,
              rgb_only: bool = False, still: bool = False):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"[synth] connected to {host}:{port}"
          f" ({'rgb only' if rgb_only else 'lidar'})", flush=True)

    hello = {
        "name": "Synthetic iPhone" + (" (no scanner)" if rgb_only else ""),
        "model": "synth",
        "src": "rgb" if rgb_only else "lidar",
        "dw": DW,
        "dh": DH,
        "note": "raytraced test pattern",
    }
    hb = json.dumps(hello).encode()
    writer.write(protocol.HELLO + struct.pack("<I", len(hb)) + hb)
    await writer.drain()

    t0 = time.monotonic()
    period = 1.0 / fps
    i = 0
    try:
        while True:
            started = time.monotonic()
            t = started - t0
            # A fixed pose would otherwise produce byte-identical JPEGs, which the
            # model maps to identical depth -- no jitter to stabilise, so no test.
            # Real grain is what makes the prediction wander.
            view = orbit_view(0.0 if still else t)
            depth_mm, conf, rgb = render(view, 0 if still else i)

            if still:
                grain = np.random.default_rng(i).normal(0, 2.5, rgb.shape)
                rgb = np.clip(rgb.astype(np.float32) + grain, 0, 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=jpeg_quality)

            header = {
                "t": round(t, 4),
                # Withholding depth is exactly what a phone with no scanner does, so
                # this exercises the bridge's inference path for real.
                "dw": 0 if rgb_only else DW,
                "dh": 0 if rgb_only else DH,
                "iw": DW,
                "ih": DH,
                "fx": FX, "fy": FY, "cx": CX, "cy": CY,
                # Column-major, matching what ARKit's simd_float4x4 memory looks like.
                "view": [float(x) for x in view.T.reshape(-1)],
                "src": "rgb" if rgb_only else "lidar",
                "track": "normal",
            }
            writer.write(protocol.pack_frame(
                header,
                b"" if rgb_only else protocol.deflate(depth_mm.tobytes()),
                b"" if rgb_only else protocol.deflate(conf.tobytes()),
                buf.getvalue(),
            ))
            await writer.drain()

            i += 1
            if i % 60 == 0:
                print(f"[synth] {i} frames", flush=True)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
    except (ConnectionResetError, BrokenPipeError):
        print("[synth] server closed the connection", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--rgb-only", action="store_true",
                    help="send colour + pose only, like a phone with no LiDAR")
    ap.add_argument("--still", action="store_true",
                    help="hold the camera fixed, with camera-like grain per frame, "
                         "for A/B testing temporal stabilisation on identical input")
    ap.add_argument("--width", type=int, default=0,
                    help="render width (defaults to 256, or 640 with --rgb-only)")
    a = ap.parse_args()

    width = a.width or (640 if a.rgb_only else 256)
    configure(width, int(round(width * 3 / 4)))

    try:
        asyncio.run(run(a.host, a.port, a.fps, a.quality, a.rgb_only, a.still))
    except KeyboardInterrupt:
        pass
