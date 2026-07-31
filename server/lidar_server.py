"""Receive depth (or colour) from a phone and fan it out to browser clients.

    python lidar_server.py

Ports:

    8770   pages + WebSocket, plaintext
    8443   the same, over TLS -- iOS grants the camera only on a secure origin
    8771   raw TCP, frames in from the native iOS app

Pages and the WebSocket deliberately share a port: a quick tunnel
(`cloudflared tunnel --url http://localhost:8770`) forwards exactly one, and a tunnel
is the only way to get a genuinely trusted certificate for a LAN address.

Depth from the native app arrives raw-DEFLATE and leaves raw; inflating here costs
~0.3 ms of native zlib and saves shipping an inflate implementation to the page. A
phone with no scanner sends only colour, and depth_model.py supplies the depth.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import mimetypes
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.parse
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidMessage
from websockets.http11 import Response

import protocol


class _QuietHandshakeNoise(logging.Filter):
    """Drop the traceback websockets logs when something opens a TCP connection to the
    WebSocket port and closes it without speaking HTTP -- a port scan, a health check,
    or a browser probing. It is not a fault, and a stack trace per occurrence buries
    the errors that are.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, (InvalidMessage, EOFError))

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

HTTP_PORT = 8770
PHONE_PORT = 8771
WS_PORT = 8772
# The capture page needs a secure origin or iOS refuses it the camera outright, so the
# same content is served again over TLS on these.
HTTPS_PORT = 8443
WSS_PORT = 8444

CA_PATH = None      # set at startup when TLS is available

# Two frames of slack per client. A browser that cannot keep up should show recent
# data with gaps, never a growing backlog of stale frames.
CLIENT_QUEUE_DEPTH = 2


class Hub:
    """Fans frames out to browser clients and keeps the rolling stats for the HUD."""

    def __init__(self):
        self.clients: dict[object, asyncio.Queue] = {}
        self.device: dict = {}
        self.frames = 0
        self.dropped = 0
        self.bytes_in = 0
        self._window_start = time.monotonic()
        self._window_frames = 0
        self._window_bytes = 0
        self.fps = 0.0
        self.kbps = 0.0
        self.last_frame_at = 0.0

    def add(self, ws) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_DEPTH)
        self.clients[ws] = q
        return q

    def remove(self, ws):
        self.clients.pop(ws, None)

    def publish(self, payload: bytes):
        for q in self.clients.values():
            if q.full():
                # Drop the oldest so the newest still gets through.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)

    def note_frame(self, nbytes: int):
        self.frames += 1
        self.bytes_in += nbytes
        self._window_frames += 1
        self._window_bytes += nbytes
        self.last_frame_at = time.time()
        elapsed = time.monotonic() - self._window_start
        if elapsed >= 1.0:
            self.fps = self._window_frames / elapsed
            self.kbps = self._window_bytes / elapsed / 1024.0
            self._window_start = time.monotonic()
            self._window_frames = 0
            self._window_bytes = 0

    def stats(self) -> dict:
        s = {
            "device": self.device,
            "frames": self.frames,
            "dropped": self.dropped,
            "fps": round(self.fps, 1),
            "kbps": round(self.kbps, 1),
            "clients": len(self.clients),
            "stale": time.time() - self.last_frame_at if self.last_frame_at else None,
        }
        if depth_worker is not None:
            s["depth"] = depth_worker.stats()
        if hand_worker is not None:
            s["hands"] = hand_worker.stats()
        return s


hub = Hub()


class DepthWorker:
    """Runs monocular depth on one background thread and drops frames while busy.

    Queueing would be worse than dropping: the phone keeps sending, so a backlog only
    grows and every frame it eventually renders is already stale.
    """

    def __init__(self, estimator):
        self.est = estimator
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth")
        self.lock = threading.Lock()
        self.busy = False
        self.last_ms = 0.0
        self.inferred = 0
        self.skipped = 0

    async def infer(self, jpeg: bytes):
        with self.lock:
            if self.busy:
                self.skipped += 1
                return None
            self.busy = True
        try:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(self.pool, self.est.infer, jpeg)
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            self.inferred += 1
            return result
        finally:
            with self.lock:
                self.busy = False

    def stats(self) -> dict:
        return {"ms": round(self.last_ms, 1), "inferred": self.inferred,
                "skipped": self.skipped}


depth_worker: DepthWorker | None = None


class HandWorker:
    """Hand tracking on its own thread, so it overlaps the GPU depth pass.

    Hand landmarks run on the CPU (~17 ms) while depth runs on the GPU (~19 ms); in
    one thread they would add up and halve the frame rate. Results are cached and
    attached to whichever frame is current, so a detection may lag by one frame --
    invisible for a visual effect, and far better than stalling the stream.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")
        self.lock = threading.Lock()
        self.busy = False
        self.latest: dict = {}
        self.last_ms = 0.0
        self.detected = 0
        self.frames = 0

    def submit(self, jpeg: bytes):
        with self.lock:
            if self.busy:
                return
            self.busy = True
        asyncio.get_running_loop().run_in_executor(self.pool, self._run, jpeg)

    def _run(self, jpeg: bytes):
        import io

        import numpy as np
        from PIL import Image
        try:
            t0 = time.perf_counter()
            rgb = np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            res = self.tracker.detect(rgb)
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            self.frames += 1
            if res.get("frame"):
                self.detected += 1
            self.latest = res
        except Exception as e:
            print(f"[hands] {type(e).__name__}: {e}", flush=True)
            self.latest = {}
        finally:
            with self.lock:
                self.busy = False

    def stats(self) -> dict:
        return {"ms": round(self.last_ms, 1), "frames": self.frames,
                "framed": self.detected}


hand_worker: HandWorker | None = None


def rescale_intrinsics(header: dict, from_w: int, from_h: int, to_w: int, to_h: int):
    """Move fx/fy/cx/cy from one image grid to another, in place.

    The phone's intrinsics describe the JPEG it sent; the viewer unprojects the depth
    map. When those differ the intrinsics must follow, or the cloud comes out with the
    wrong field of view.
    """
    if not from_w or not from_h:
        return
    sx = to_w / from_w
    sy = to_h / from_h
    for key, s in (("fx", sx), ("cx", sx), ("fy", sy), ("cy", sy)):
        if key in header:
            header[key] = header[key] * s


async def ingest(header: dict, depth: bytes, conf: bytes, rgb: bytes, raw_in: int):
    """Shared tail for every source: supply depth if the sender had none, sanity check
    it, then fan out. Keeps the native app and the browser capture page on identical
    handling so the viewer cannot tell them apart."""
    # Kick hand tracking off first so it runs alongside the depth pass rather than
    # after it; whatever it last produced is attached below.
    if rgb and hand_worker is not None:
        hand_worker.submit(rgb)

    if not depth and rgb and depth_worker is not None:
        result = await depth_worker.infer(rgb)
        if result is None:
            return                          # GPU still busy; skip rather than lag
        depth, dw, dh, mask = result
        rescale_intrinsics(header, header.get("iw", dw), header.get("ih", dh), dw, dh)
        header["dw"], header["dh"] = dw, dh
        header["src"] = f"{header.get('src') or 'rgb'}+ml"
        # Inference has no confidence channel, so in stationary mode that slot carries
        # the motion mask instead. The header says which, so the viewer never has to
        # guess what the byte means.
        conf = mask
        header["mask"] = "motion" if mask else "none"

    if not depth:
        return

    expect = header.get("dw", 0) * header.get("dh", 0) * 2
    if len(depth) != expect:
        print(
            f"[ingest] depth size mismatch: got {len(depth)} want {expect}"
            f" for {header.get('dw')}x{header.get('dh')} - dropping frame",
            flush=True,
        )
        return

    if hand_worker is not None and hand_worker.latest:
        h = hand_worker.latest
        if h.get("hands"):
            header["hands"] = h["hands"]
        if h.get("frame"):
            header["frame"] = h["frame"]

    hub.note_frame(raw_in)
    hub.publish(protocol.pack_browser_frame(header, depth, conf, rgb))


async def handle_web_phone(ws):
    """Ingest from web/capture.html. A browser cannot open a raw TCP socket, so the
    capture page speaks the same frames over this WebSocket instead:

        text   -> {"type":"hello", ...} once
        binary -> u32 headerLen | headerJSON | jpeg
    """
    peer = "web"
    print(f"[web] capture page connected", flush=True)
    try:
        async for msg in ws:
            if isinstance(msg, str):
                info = json.loads(msg)
                if info.get("type") == "hello":
                    hub.device = info
                    print(f"[web] hello {json.dumps(info)}", flush=True)
                continue

            if len(msg) < 4:
                continue
            (hlen,) = struct.unpack_from("<I", msg, 0)
            if hlen > protocol.MAX_HEADER or 4 + hlen > len(msg):
                print(f"[web] bad header length {hlen}", flush=True)
                continue
            header = json.loads(bytes(msg[4:4 + hlen]))
            rgb = bytes(msg[4 + hlen:])
            await ingest(header, b"", b"", rgb, len(msg))
    except ConnectionClosed:
        pass
    except (json.JSONDecodeError, struct.error) as e:
        print(f"[web] malformed frame from {peer}: {e}", flush=True)
    finally:
        print("[web] capture page disconnected", flush=True)


async def ws_router(ws):
    """One port serves both directions; the path picks which."""
    path = ""
    with contextlib.suppress(Exception):
        path = ws.request.path or ""
    if "phone" in path:
        await handle_web_phone(ws)
    else:
        await handle_browser(ws)


async def handle_phone(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    addr = f"{peer[0]}:{peer[1]}" if peer else "?"
    print(f"[phone] connected {addr}", flush=True)
    try:
        hello = await protocol.read_hello(reader)
        hub.device = hello
        print(f"[phone] hello {json.dumps(hello)}", flush=True)

        while True:
            header, cdepth, cconf, rgb = await protocol.read_frame(reader)
            raw_in = len(cdepth) + len(cconf) + len(rgb)
            depth = protocol.inflate(cdepth) if cdepth else b""
            conf = protocol.inflate(cconf) if cconf else b""
            await ingest(header, depth, conf, rgb, raw_in)
    except asyncio.IncompleteReadError:
        print(f"[phone] {addr} disconnected", flush=True)
    except protocol.ProtocolError as e:
        print(f"[phone] protocol error from {addr}: {e}", flush=True)
    except zlib.error as e:
        print(f"[phone] bad zlib payload from {addr}: {e}", flush=True)
    except (ConnectionResetError, OSError) as e:
        print(f"[phone] {addr} connection error: {e}", flush=True)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


SETTABLE = {"smooth", "smooth_static", "stationary", "bg_alpha", "motion_m",
            "ref_alpha", "max_corr", "jump_m", "scale",
            "dev_alpha", "dev_k", "dev_floor", "motion_blur"}


def apply_settings(patch: dict) -> dict:
    """Let the viewer retune the depth pipeline live. Restarting the bridge to try a
    different smoothing value means reloading the model, which is far too slow to
    iterate with."""
    est = depth_worker.est if depth_worker else None
    if est is None:
        return {}
    applied = {}
    for k, v in patch.items():
        if k not in SETTABLE:
            continue
        cur = getattr(est, k, None)
        try:
            v = bool(v) if isinstance(cur, bool) else float(v)
        except (TypeError, ValueError):
            continue
        setattr(est, k, v)
        applied[k] = v
    if "stationary" in applied:
        # Averages built for one camera pose are meaningless for another.
        est.reset_temporal()
    return applied


async def handle_browser(ws):
    q = hub.add(ws)
    print(f"[ws] browser connected ({len(hub.clients)} total)", flush=True)

    async def pump():
        while True:
            await ws.send(await q.get())

    async def control():
        async for msg in ws:
            if not isinstance(msg, str):
                continue
            try:
                m = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if m.get("type") == "set":
                applied = apply_settings(m)
                if applied:
                    print(f"[set] {applied}", flush=True)
            elif m.get("type") == "reset" and depth_worker:
                depth_worker.est.reset_temporal()
                print("[set] temporal state reset", flush=True)

    try:
        await ws.send(json.dumps({"type": "hello", **hub.stats()}))
        # Reading and writing concurrently: the viewer both receives frames and sends
        # settings, and a single loop would block one on the other.
        await asyncio.gather(pump(), control())
    except ConnectionClosed:
        pass
    finally:
        hub.remove(ws)
        print(f"[ws] browser gone ({len(hub.clients)} left)", flush=True)


async def stats_ticker():
    """Push a small JSON stats blob once a second; the page shows it in the HUD."""
    while True:
        await asyncio.sleep(1.0)
        if not hub.clients:
            continue
        msg = json.dumps({"type": "stats", **hub.stats()})
        for ws in list(hub.clients):
            with contextlib.suppress(Exception):
                await ws.send(msg)


def _resp(status: int, reason: str, body: bytes, ctype: str,
          extra: dict | None = None) -> Response:
    headers = Headers({
        "Content-Type": ctype,
        "Content-Length": str(len(body)),
        # The pages are edited constantly; caching only confuses.
        "Cache-Control": "no-store",
    })
    for k, v in (extra or {}).items():
        headers[k] = v
    return Response(status, reason, headers, body)


def serve_static(raw_path: str) -> Response:
    clean = urllib.parse.urlparse(raw_path).path
    if clean in ("", "/"):
        clean = "/index.html"

    if clean == "/ips":
        return _resp(200, "OK", json.dumps({
            "ips": lan_ips(), "port": PHONE_PORT, "tls": CA_PATH is not None,
        }).encode(), "application/json")

    # Installing this on the phone AND enabling full trust for it is what makes the
    # camera work -- iOS refuses media capture on any origin with a certificate error,
    # even after you tap through Safari's warning.
    if clean == "/ca.crt":
        if not (CA_PATH and CA_PATH.exists()):
            return _resp(404, "Not Found", b"no CA (TLS disabled)\n", "text/plain")
        return _resp(200, "OK", CA_PATH.read_bytes(),
                     # This type makes iOS offer it as an installable profile.
                     "application/x-x509-ca-cert",
                     {"Content-Disposition": 'attachment; filename="lidar-ca.crt"'})

    root = WEB_DIR.resolve()
    target = (root / clean.lstrip("/")).resolve()
    # Keep path traversal inside the web directory.
    if root not in target.parents and target != root:
        return _resp(403, "Forbidden", b"forbidden\n", "text/plain")
    if not target.is_file():
        return _resp(404, "Not Found", b"not found\n", "text/plain")
    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype == "application/javascript":
        ctype += "; charset=utf-8"
    return _resp(200, "OK", target.read_bytes(), ctype)


def process_request(connection, request):
    """Serve pages from the same port that carries the WebSocket.

    A WebSocket handshake is just an HTTP request with `Upgrade: websocket`, so anything
    without it is a page load and gets answered here. One origin for both matters
    because a quick tunnel (cloudflared/ngrok) forwards a single port -- and a tunnel is
    the only way to get a genuinely trusted certificate on a LAN address.
    """
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    try:
        return serve_static(request.path)
    except OSError as e:
        return _resp(500, "Internal Server Error", f"{e}\n".encode(), "text/plain")


def lan_ips() -> list[str]:
    """Best-effort list of this machine's LAN addresses, for the QR/connect hint."""
    ips = set()
    with contextlib.suppress(Exception):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    # getaddrinfo misses some adapters; a dummy connect reveals the default route.
    with contextlib.suppress(Exception):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    return sorted(ips)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http-port", type=int, default=HTTP_PORT)
    ap.add_argument("--phone-port", type=int, default=PHONE_PORT)
    ap.add_argument("--ws-port", type=int, default=WS_PORT)
    ap.add_argument("--https-port", type=int, default=HTTPS_PORT)
    ap.add_argument("--wss-port", type=int, default=WSS_PORT)
    ap.add_argument("--no-tls", action="store_true",
                    help="skip HTTPS; the browser capture page will not work")
    ap.add_argument("--no-hands", action="store_true",
                    help="skip hand tracking (saves a CPU core)")
    ap.add_argument("--no-depth-model", action="store_true",
                    help="do not load monocular depth; LiDAR phones only")
    ap.add_argument("--depth-input", type=int, default=518,
                    help="model input height (518 is the checkpoint's native size)")
    ap.add_argument("--depth-out", type=int, default=512,
                    help="width of the inferred depth map sent to the viewer")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="multiplier on inferred metres, for calibration")
    args = ap.parse_args()

    if not WEB_DIR.is_dir():
        sys.exit(f"web directory missing: {WEB_DIR}")

    global depth_worker, hand_worker
    if not args.no_hands:
        try:
            import hands as hands_mod
            hand_worker = HandWorker(hands_mod.HandTracker())
            print("[hands] MediaPipe hand landmarker ready", flush=True)
        except Exception as e:
            print(f"[hands] unavailable ({type(e).__name__}: {e})", flush=True)

    if not args.no_depth_model:
        try:
            import depth_model
            est = depth_model.DepthEstimator(
                input_height=args.depth_input,
                out_width=args.depth_out,
                out_height=int(round(args.depth_out * 3 / 4)),
                scale=args.depth_scale,
            )
            depth_worker = DepthWorker(est)
            print(f"[depth] {est.describe()}", flush=True)
        except Exception as e:
            # A LiDAR phone needs none of this, so this is a warning, not fatal.
            print(f"[depth] unavailable ({type(e).__name__}: {e})", flush=True)
            print("[depth] phones without a scanner will send colour that "
                  "cannot be turned into depth", flush=True)

    logging.getLogger("websockets.server").addFilter(_QuietHandshakeNoise())

    ips = lan_ips()
    lan = [ip for ip in ips if not ip.startswith(("100.", "172.2"))]

    phone_srv = await asyncio.start_server(handle_phone, "0.0.0.0", args.phone_port)
    # Pages and WebSocket share each port, so a single-port tunnel carries both.
    plain = await serve(ws_router, "0.0.0.0", args.http_port,
                        process_request=process_request, max_size=None)

    global CA_PATH
    tls_ok = False
    if not args.no_tls:
        try:
            import tls as tlsmod
            cert, key, ca = tlsmod.ensure_cert(ips)
            CA_PATH = ca
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            await serve(ws_router, "0.0.0.0", args.https_port,
                        process_request=process_request, ssl=ctx, max_size=None)
            tls_ok = True
        except Exception as e:
            print(f"[tls] unavailable ({type(e).__name__}: {e})", flush=True)

    host = lan[0] if lan else "localhost"
    print("=" * 72)
    print("  iPhone bridge is up")
    print(f"  viewer (this PC)    http://localhost:{args.http_port}")
    if tls_ok:
        print(f"  phone capture       https://{host}:{args.https_port}/capture.html")
        print(f"  install this first  http://{host}:{args.http_port}/ca.crt")
        print("                      ...then Settings > General > About >")
        print("                      Certificate Trust Settings > enable it.")
        print("                      iOS blocks the camera on an untrusted cert even")
        print("                      after you tap through Safari's warning.")
    print()
    print("  No cert hassle? Expose one port through a tunnel for a real")
    print("  certificate (note: frames then transit that provider):")
    print(f"    cloudflared tunnel --url http://localhost:{args.http_port}")
    print(f"  native app ->       {host}:{args.phone_port}")
    print("=" * 72, flush=True)

    async with phone_srv, plain:
        await asyncio.gather(phone_srv.serve_forever(), stats_ticker())


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
