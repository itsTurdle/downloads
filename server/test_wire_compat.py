"""End-to-end wire check that emulates the Swift client byte for byte.

The Swift app builds its frames by hand -- magic, little-endian u32 lengths, a
JSONEncoder header, raw DEFLATE payloads -- so this test does the same rather than
reusing protocol.pack_frame(). That way a mismatch between what FrameStreamer.swift
emits and what the bridge expects fails here instead of on the phone, where the only
symptom would be a silent stall.

Run the bridge first, then:

    python test_wire_compat.py
"""

import asyncio
import json
import struct
import sys
import zlib

from websockets.asyncio.client import connect

HOST = "127.0.0.1"
PHONE_PORT = 8771
WS_URL = "ws://127.0.0.1:8772"

DW, DH = 256, 192


def swift_raw_deflate(data: bytes) -> bytes:
    """What compression_encode_buffer(..., COMPRESSION_ZLIB) produces: RFC 1951."""
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def make_depth() -> bytes:
    """A depth ramp with a known value at every pixel, plus a hole to prove 0 passes
    through as 'no reading' rather than being clamped to something."""
    vals = bytearray()
    for y in range(DH):
        for x in range(DW):
            mm = 0 if (x // 16 + y // 16) % 17 == 0 else 500 + (x * 7 + y * 13) % 6000
            vals += struct.pack("<H", mm)
    return bytes(vals)


def expected_mm(x: int, y: int) -> int:
    return 0 if (x // 16 + y // 16) % 17 == 0 else 500 + (x * 7 + y * 13) % 6000


def make_conf() -> bytes:
    return bytes(((x + y) % 3) for y in range(DH) for x in range(DW))


# A one-pixel JPEG: contents are irrelevant, but the bytes must survive untouched.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    "0806060706050807070709090807"
) + b"\xff\xd9"

VIEW = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.25, -0.5, 1.75, 1.0,     # translation in the last column (column-major)
]


def swift_frame(depth_raw: bytes, conf_raw: bytes, jpeg: bytes, t: float) -> bytes:
    depth = swift_raw_deflate(depth_raw)
    conf = swift_raw_deflate(conf_raw)
    header = {
        "t": t,
        "dw": DW, "dh": DH,
        "fx": 221.7, "fy": 221.7, "cx": 128.0, "cy": 96.0,
        "view": VIEW,
        "src": "lidar",
        "track": "normal",
        "dlen": len(depth), "clen": len(conf), "rlen": len(jpeg),
    }
    hb = json.dumps(header).encode()
    return (b"LFRM" + struct.pack("<I", len(hb)) + hb + depth + conf + jpeg)


def swift_hello() -> bytes:
    info = {"name": "Wire Test", "model": "iPhone15,2", "src": "lidar",
            "app": "LidarStream"}
    j = json.dumps(info).encode()
    return b"LIDRHELO" + struct.pack("<I", len(j)) + j


def parse_browser_frame(buf: bytes):
    (hlen,) = struct.unpack("<I", buf[:4])
    header = json.loads(buf[4:4 + hlen])
    off = 4 + hlen
    depth = buf[off:off + header["dlen"]]; off += header["dlen"]
    conf = buf[off:off + header["clen"]]; off += header["clen"]
    rgb = buf[off:off + header["rlen"]]; off += header["rlen"]
    return header, depth, conf, rgb, off, (4 + hlen) % 4


async def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    depth_raw = make_depth()
    conf_raw = make_conf()

    async with connect(WS_URL, max_size=None) as ws:
        reader, writer = await asyncio.open_connection(HOST, PHONE_PORT)
        writer.write(swift_hello())
        await writer.drain()

        # Paced like a real 20 fps phone. Bursting them back to back would instead
        # exercise the bridge's drop-oldest backpressure (queue depth 2, newest
        # wins) and legitimately lose the first frame.
        SENT = 5
        for i in range(SENT):
            writer.write(swift_frame(depth_raw, conf_raw, TINY_JPEG, 100.0 + i))
            await writer.drain()
            await asyncio.sleep(0.05)

        frames = []
        deadline = asyncio.get_running_loop().time() + 10
        while len(frames) < SENT and asyncio.get_running_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                frames.append(msg)

        print(f"\nreceived {len(frames)} relayed frame(s)\n")
        check(f"bridge relayed all {SENT} paced frames", len(frames) == SENT,
              f"got {len(frames)}")
        if not frames:
            writer.close()
            print("\nno frames relayed - is the bridge running?")
            sys.exit(1)

        header, depth, conf, rgb, consumed, align = parse_browser_frame(frames[0])

        check("payload offset is 4-byte aligned", align == 0,
              f"depth starts at offset {4 + len(json.dumps(header))} -> mod4={align}")
        check("no trailing bytes", consumed == len(frames[0]),
              f"consumed {consumed} of {len(frames[0])}")
        check("dims preserved", (header["dw"], header["dh"]) == (DW, DH),
              f"{header['dw']}x{header['dh']}")
        check("depth inflated to raw uint16", len(depth) == DW * DH * 2,
              f"{len(depth)} bytes")
        check("conf inflated to raw uint8", len(conf) == DW * DH, f"{len(conf)} bytes")
        check("jpeg passed through byte-exact", rgb == TINY_JPEG,
              f"{len(rgb)} of {len(TINY_JPEG)} bytes")
        check("intrinsics preserved",
              (header["fx"], header["cx"], header["cy"]) == (221.7, 128.0, 96.0))
        check("view matrix preserved (column-major)", header["view"] == VIEW)
        check("translation lands in the last column",
              header["view"][12:15] == [0.25, -0.5, 1.75])
        check("tracking + source preserved",
              header["src"] == "lidar" and header["track"] == "normal")

        # Every depth sample must survive the mm -> deflate -> inflate round trip.
        got = struct.unpack(f"<{DW * DH}H", depth)
        bad = [(x, y, got[y * DW + x], expected_mm(x, y))
               for y in range(0, DH, 7) for x in range(0, DW, 5)
               if got[y * DW + x] != expected_mm(x, y)]
        check("all sampled depth values round-trip exactly", not bad,
              f"{len(bad)} mismatches" if bad else "")
        holes = sum(1 for v in got if v == 0)
        check("zero-depth holes preserved", holes > 0, f"{holes} hole pixels")

        conf_got = conf[0], conf[1], conf[2]
        check("confidence values round-trip", conf_got == (0, 1, 2), str(conf_got))

        writer.close()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {', '.join(failures)}")
        sys.exit(1)
    print("wire format is compatible end to end.")


asyncio.run(main())
