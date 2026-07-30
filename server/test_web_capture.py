"""End-to-end check of the browser capture path (web/capture.html -> bridge -> viewer).

Mimics exactly what the page does: connect over WSS to /phone, send a hello, then send
`u32 headerLen | headerJSON | jpeg` frames with no depth at all. The bridge must infer
depth on the GPU and publish a frame the viewer can render.

Validates the certificate against the local CA rather than skipping verification --
that is the same chain the phone checks once ca.crt is installed, so a broken chain
fails here instead of on the phone.

Run the bridge first, then:

    python test_web_capture.py
"""

import asyncio
import io
import json
import ssl
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.asyncio.client import connect

CA = Path(__file__).resolve().parent / "_certs" / "ca.crt"
PHONE_URL = "wss://127.0.0.1:8444/phone"
VIEW_URL = "wss://127.0.0.1:8444/"


def room_jpeg(w=640, h=480, shift=0.0):
    """A believable indoor-ish frame, so the depth model has real structure to work
    with rather than noise."""
    import synth_phone as sp
    sp.configure(w, h)
    view = sp.orbit_view(shift)
    _, _, rgb = sp.render(view, 0)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=70)
    return buf.getvalue(), view


def web_frame(jpeg: bytes, view, w: int, h: int, t: float) -> bytes:
    hfov = 64 * np.pi / 180
    fx = (w / 2) / np.tan(hfov / 2)
    header = {
        "t": t,
        "dw": 0, "dh": 0,          # no depth: the bridge must supply it
        "iw": w, "ih": h,
        "fx": float(fx), "fy": float(fx), "cx": w / 2, "cy": h / 2,
        "view": [float(x) for x in view.T.reshape(-1)],
        "src": "web",
        "track": "orientation",
    }
    hb = json.dumps(header).encode()
    return struct.pack("<I", len(hb)) + hb + jpeg


def parse_viewer_frame(buf: bytes):
    (hlen,) = struct.unpack("<I", buf[:4])
    header = json.loads(buf[4:4 + hlen])
    off = 4 + hlen
    depth = buf[off:off + header["dlen"]]
    return header, depth


async def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    if not CA.exists():
        sys.exit(f"no CA at {CA} - start the bridge once to generate it")

    # The chain the phone will verify after installing ca.crt.
    ctx = ssl.create_default_context(cafile=str(CA))

    W, H = 640, 480
    async with connect(VIEW_URL, ssl=ctx, max_size=None) as viewer:
        async with connect(PHONE_URL, ssl=ctx, max_size=None) as phone:
            await phone.send(json.dumps({
                "type": "hello", "name": "Safari capture (test)",
                "model": "iPhone (web)", "src": "web", "app": "capture.html",
            }))
            check("WSS handshake verified against the local CA", True)

            for i in range(4):
                jpeg, view = room_jpeg(W, H, shift=i * 0.6)
                await phone.send(web_frame(jpeg, view, W, H, i * 0.08))
                await asyncio.sleep(0.35)   # let inference finish; busy frames are dropped

            frames = []
            while len(frames) < 2:
                try:
                    msg = await asyncio.wait_for(viewer.recv(), timeout=8)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    frames.append(msg)

    print(f"\nviewer received {len(frames)} rendered frame(s)\n")
    check("bridge published inferred frames", len(frames) >= 1, f"got {len(frames)}")
    if not frames:
        print("\nno frames -- is the depth model loaded? check the bridge log")
        sys.exit(1)

    header, depth = parse_viewer_frame(frames[0])
    check("source marked as web + inference", header.get("src") == "web+ml",
          str(header.get("src")))
    dw, dh = header.get("dw"), header.get("dh")
    check("depth dimensions filled in by the bridge", bool(dw and dh), f"{dw}x{dh}")
    check("aspect preserved from the 4:3 source",
          abs((dw / dh) - (W / H)) < 0.02, f"{dw / dh:.3f} vs {W / H:.3f}")
    check("depth payload matches its dimensions", len(depth) == dw * dh * 2,
          f"{len(depth)} bytes")

    mm = np.frombuffer(depth, dtype="<u2")
    valid = mm[mm > 0]
    check("depth is populated", valid.size > mm.size * 0.5,
          f"{100 * valid.size / mm.size:.0f}% non-zero")
    check("depths are plausible indoor metres", valid.size > 0
          and 0.1 < valid.min() / 1000 < 12 and 0.2 < valid.max() / 1000 < 30,
          f"{valid.min() / 1000:.2f}..{valid.max() / 1000:.2f} m")

    # Intrinsics must follow the rescale from JPEG grid to depth grid, or the cloud
    # comes out with the wrong field of view.
    expect_fx = ((W / 2) / np.tan(64 * np.pi / 180 / 2)) * (dw / W)
    check("fx rescaled to the depth grid",
          abs(header["fx"] - expect_fx) < 0.5,
          f"{header['fx']:.1f} vs {expect_fx:.1f}")
    check("cx sits at the depth-map centre", abs(header["cx"] - dw / 2) < 0.5,
          f"{header['cx']:.1f} vs {dw / 2}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {', '.join(failures)}")
        sys.exit(1)
    print("browser capture path works end to end.")


asyncio.run(main())
