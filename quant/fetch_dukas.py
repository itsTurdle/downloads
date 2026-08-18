"""Fetch USA30IDXUSD (Dow CFD) minute candles from Dukascopy day-files,
decode bi5 (LZMA), resample to 30m and 1h closes (+volume), with retries,
checkpointing, and price-scale sanity checks."""
import calendar
import json
import lzma
import os
import random
import ssl
import struct
import sys
import time
import urllib.request

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"
OUT30 = f"{SP}/dow/dow30m.json"
CKPT = f"{SP}/dow/dukas_ckpt.json"
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
BASE = "https://datafeed.dukascopy.com/datafeed/USA30IDXUSD/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45, context=CTX) as r:
                blob = r.read()
            if blob[:5] == b"<html":
                raise IOError("html error page")
            return blob
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # no data that day (weekend/holiday)
            time.sleep(2**attempt + random.random())
        except Exception:
            time.sleep(2**attempt + random.random())
    return "FAIL"


def decode(blob):
    d = None
    for fmt in (lzma.FORMAT_AUTO, lzma.FORMAT_ALONE):
        try:
            d = lzma.LZMADecompressor(format=fmt).decompress(blob)
            break
        except Exception:
            continue
    if d is None:
        try:
            d = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 23}]).decompress(blob)
        except Exception:
            return None
    n = len(d)//24
    return [struct.unpack(">5if", d[i*24:(i+1)*24]) for i in range(n)]


def main():
    state = {"done": {}, "scale": None}          # done: "yyyy-mm-dd" -> [[bucket_epoch, close, vol], ...]
    if os.path.exists(CKPT):
        state = json.load(open(CKPT))
    done = state["done"]
    days = []
    today = time.gmtime()
    for y in range(2012, 2027):
        for m in range(1, 13):
            for dd in range(1, calendar.monthrange(y, m)[1]+1):
                if (y, m, dd) >= (today.tm_year, today.tm_mon, today.tm_mday): break
                if calendar.weekday(y, m, dd) >= 5: continue     # skip Sat/Sun
                days.append((y, m, dd))
    todo = [x for x in days if f"{x[0]}-{x[1]:02d}-{x[2]:02d}" not in done]
    print(f"{len(days)} weekdays, {len(todo)} to fetch", flush=True)
    fails = 0
    for k, (y, m, dd) in enumerate(todo):
        key = f"{y}-{m:02d}-{dd:02d}"
        blob = get(BASE.format(y=y, m=m-1, d=dd))    # months 0-indexed!
        if blob == "FAIL":
            fails += 1
            if fails > 60:
                print("too many failures, checkpoint & abort", flush=True); break
            continue
        if blob is None:
            done[key] = []                        # no data that day
        else:
            recs = decode(blob)
            if not recs:
                done[key] = []
            else:
                if state["scale"] is None:
                    p0 = recs[0][1]
                    for sc in (1000, 100, 10, 1):
                        if 5000 < p0/sc < 60000: state["scale"] = sc; break
                    print("price scale:", state["scale"], "sample:", p0, flush=True)
                sc = state["scale"]
                day_epoch = calendar.timegm((y, m, dd, 0, 0, 0))
                buckets = {}
                for (sec, o, c, lo, hi, vol) in recs:
                    if c <= 0: continue
                    b = day_epoch + (sec//1800)*1800
                    if b not in buckets: buckets[b] = [c/sc, 0.0]
                    buckets[b][0] = c/sc
                    buckets[b][1] += float(vol)
                done[key] = [[b, round(v[0], 2), round(v[1], 2)] for b, v in sorted(buckets.items())]
        if (k+1) % 250 == 0:
            json.dump(state, open(CKPT, "w"))
            print(f"checkpoint {k+1}/{len(todo)}", flush=True)
        time.sleep(0.25 + random.random()*0.2)
    json.dump(state, open(CKPT, "w"))
    rows = []
    for key in sorted(done):
        rows.extend(done[key])
    rows.sort()
    nonempty = [k for k in done if done[k]]
    if rows:
        t0 = rows[0][0]
        out = dict(t0=t0, dt=[(r[0]-t0)//1800 for r in rows],
                   e=[r[1] for r in rows], v=[r[2] for r in rows])
        json.dump(out, open(OUT30, "w"))
        print(f"days with data: {len(nonempty)} | 30m bars: {len(rows)} | "
              f"{min(nonempty)} -> {max(nonempty)}", flush=True)
    else:
        print("no data collected", flush=True)


main()
