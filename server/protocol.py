"""Wire protocol for iPhone LiDAR streaming.

Two hops:

  iPhone --(TCP 8771, zlib depth)--> lidar_server.py --(WS 8772, raw depth)--> browser

Phone -> PC
-----------
Handshake once per connection:

    b"LIDRHELO"  u32 json_len  json

Then repeating frames:

    b"LFRM"  u32 header_len  header_json  depth  conf  rgb

`header_json` carries the payload byte counts so the reader never has to guess:

    {
      "t":    float,          # frame timestamp, seconds, phone clock
      "dw":   int, "dh": int, # depth map dimensions
      "fx":   float, "fy": float, "cx": float, "cy": float,
                              # pinhole intrinsics ALREADY SCALED to depth pixels
      "view": [16 floats],    # camera->world, column-major (ARKit convention)
      "src":  "lidar" | "truedepth",
      "track":"normal" | "limited" | "notAvailable",
      "dlen": int,            # zlib(uint16 LE millimetres, row major); 0 = no reading
      "clen": int,            # zlib(uint8 confidence 0..2); may be 0
      "rlen": int             # JPEG bytes; may be 0
    }

Depth is millimetres in uint16 so a frame is 96 KiB before compression and ~25 KiB
after. Float32 would double that for precision we cannot see at these ranges.

PC -> browser
-------------
Same shape, minus the magic, with depth/conf already inflated so the page does not
have to. `dlen`/`clen`/`rlen` are rewritten to the decompressed sizes.

    u32 header_len  header_json  depth  conf  rgb
"""

import json
import struct
import zlib

HELLO = b"LIDRHELO"
FRAME_MAGIC = b"LFRM"

# Raw DEFLATE with no zlib/gzip wrapper. This is what Apple's Compression framework
# emits for COMPRESSION_ZLIB, so the iOS app needs no third-party zlib to speak it.
RAW_DEFLATE_WBITS = -15

# Guard rails so a desynced stream fails loudly instead of trying to allocate 4 GiB.
MAX_HEADER = 8192
MAX_PAYLOAD = 32 << 20


class ProtocolError(Exception):
    pass


def deflate(data: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level, zlib.DEFLATED, RAW_DEFLATE_WBITS)
    return c.compress(data) + c.flush()


def inflate(data: bytes) -> bytes:
    return zlib.decompress(data, RAW_DEFLATE_WBITS)


def pack_frame(header: dict, depth: bytes, conf: bytes, rgb: bytes) -> bytes:
    """Serialise one phone->PC frame (magic included)."""
    header = dict(header)
    header["dlen"] = len(depth)
    header["clen"] = len(conf)
    header["rlen"] = len(rgb)
    hb = json.dumps(header, separators=(",", ":")).encode()
    return b"".join(
        (FRAME_MAGIC, struct.pack("<I", len(hb)), hb, depth, conf, rgb)
    )


def pack_browser_frame(header: dict, depth: bytes, conf: bytes, rgb: bytes) -> bytes:
    """Serialise one PC->browser frame (no magic; lengths describe raw payloads).

    The header is space-padded so the depth payload starts 4-byte aligned: the page
    reads it as a Uint16Array view directly onto the received ArrayBuffer, and a
    typed-array view at an odd byte offset throws.
    """
    header = dict(header)
    header["dlen"] = len(depth)
    header["clen"] = len(conf)
    header["rlen"] = len(rgb)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-(len(hb) + 4) % 4)  # trailing space is legal JSON whitespace
    return b"".join((struct.pack("<I", len(hb)), hb, depth, conf, rgb))


async def read_exactly(reader, n: int) -> bytes:
    if n < 0 or n > MAX_PAYLOAD:
        raise ProtocolError(f"refusing to read {n} bytes")
    if n == 0:
        return b""
    return await reader.readexactly(n)


async def read_hello(reader) -> dict:
    magic = await read_exactly(reader, len(HELLO))
    if magic != HELLO:
        raise ProtocolError(f"bad hello magic {magic!r}")
    (n,) = struct.unpack("<I", await read_exactly(reader, 4))
    if n > MAX_HEADER:
        raise ProtocolError(f"hello header too large ({n})")
    return json.loads(await read_exactly(reader, n))


async def read_frame(reader):
    """Read one phone->PC frame. Returns (header, depth, conf, rgb) still compressed."""
    magic = await read_exactly(reader, len(FRAME_MAGIC))
    if magic != FRAME_MAGIC:
        # Losing sync means every subsequent length is garbage, so stop rather than
        # try to resynchronise on the magic.
        raise ProtocolError(f"bad frame magic {magic!r} (stream desynced)")
    (hlen,) = struct.unpack("<I", await read_exactly(reader, 4))
    if hlen > MAX_HEADER:
        raise ProtocolError(f"frame header too large ({hlen})")
    header = json.loads(await read_exactly(reader, hlen))
    depth = await read_exactly(reader, header.get("dlen", 0))
    conf = await read_exactly(reader, header.get("clen", 0))
    rgb = await read_exactly(reader, header.get("rlen", 0))
    return header, depth, conf, rgb
