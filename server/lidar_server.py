"""Receive an iPhone LiDAR stream and fan it out to browser clients.

    python lidar_server.py

Listens on three ports:

    8770  HTTP     the point-cloud display (../web)
    8771  TCP      frames in from the iPhone app
    8772  WS       frames out to the browser

Depth arrives zlib-compressed and leaves raw: inflating here costs ~0.3 ms of native
zlib per frame and saves shipping an inflate implementation to the page.
"""

import argparse
import asyncio
import contextlib
import json
import socket
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import protocol

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

HTTP_PORT = 8770
PHONE_PORT = 8771
WS_PORT = 8772

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

            # No scanner on the phone: infer depth here from the colour frame.
            if not depth and rgb and depth_worker is not None:
                result = await depth_worker.infer(rgb)
                if result is None:
                    continue                    # GPU still busy; skip rather than lag
                depth, dw, dh = result
                rescale_intrinsics(header,
                                   header.get("iw", dw), header.get("ih", dh), dw, dh)
                header["dw"], header["dh"] = dw, dh
                header["src"] = "rgb+ml"
                conf = b""                      # inference has no confidence channel

            if not depth:
                continue

            expect = header.get("dw", 0) * header.get("dh", 0) * 2
            if len(depth) != expect:
                print(
                    f"[phone] depth size mismatch: got {len(depth)} want {expect}"
                    f" for {header.get('dw')}x{header.get('dh')} - dropping frame",
                    flush=True,
                )
                continue

            hub.note_frame(raw_in)
            hub.publish(protocol.pack_browser_frame(header, depth, conf, rgb))
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


async def handle_browser(ws):
    q = hub.add(ws)
    print(f"[ws] browser connected ({len(hub.clients)} total)", flush=True)
    try:
        await ws.send(json.dumps({"type": "hello", **hub.stats()}))
        while True:
            payload = await q.get()
            await ws.send(payload)
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


class WebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    def do_GET(self):
        # So the page can tell you which address to type into the phone, instead of
        # you having to go read ipconfig.
        if self.path == "/ips":
            body = json.dumps({"ips": lan_ips(), "port": PHONE_PORT}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def handle_one_request(self):
        # A browser closing a keep-alive socket makes ThreadingHTTPServer dump a full
        # traceback. Left alone it buries the log in noise that looks like a fault.
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def end_headers(self):
        # The page is edited constantly during development; caching only confuses.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "200" not in (args[1] if len(args) > 1 else ""):
            print(f"[http] {fmt % args}", flush=True)


def start_http():
    srv = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), WebHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


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
    ap.add_argument("--no-depth-model", action="store_true",
                    help="do not load monocular depth; LiDAR phones only")
    ap.add_argument("--depth-input", type=int, default=518,
                    help="model input height (518 is the checkpoint's native size)")
    ap.add_argument("--depth-out", type=int, default=384,
                    help="width of the inferred depth map sent to the viewer")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="multiplier on inferred metres, for calibration")
    args = ap.parse_args()

    if not WEB_DIR.is_dir():
        sys.exit(f"web directory missing: {WEB_DIR}")

    global depth_worker
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

    start_http()
    phone_srv = await asyncio.start_server(handle_phone, "0.0.0.0", args.phone_port)
    ws_srv = await serve(handle_browser, "0.0.0.0", args.ws_port, max_size=None)

    ips = lan_ips()
    print("=" * 62)
    print("  iPhone LiDAR bridge is up")
    print(f"  display    http://localhost:{args.http_port}")
    for ip in ips:
        print(f"  phone -> {ip}:{args.phone_port}")
    print("=" * 62, flush=True)

    async with phone_srv, ws_srv:
        await asyncio.gather(phone_srv.serve_forever(), stats_ticker())


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
