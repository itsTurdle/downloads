"""Fetch EUR/USD M1 from histdata.com (yearly zips + current-year monthlies),
resample to H1 closes (UTC), splice into pack.json as the 'hr' block."""
import calendar
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"
BASE = "https://www.histdata.com"
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


def get(url, data=None, referer=None):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=180, context=CTX).read()


def hidden(html, name):
    m = re.search(r'id="%s"[^>]*value="([^"]*)"' % name, html)
    if not m:
        m = re.search(r'name="%s"[^>]*value="([^"]*)"' % name, html)
    return m.group(1) if m else None


def download_zip(page_url):
    html = get(page_url).decode("utf-8", "ignore")
    fields = {}
    for f in ("tk", "date", "datemonth", "platform", "timeframe", "fxpair"):
        v = hidden(html, f)
        if v is None:
            return None
        fields[f] = v
    post = urllib.parse.urlencode(fields).encode()
    blob = get(BASE + "/get.php", data=post, referer=page_url)
    if len(blob) < 1000 or blob[:2] != b"PK":
        return None
    return blob


hourly = {}          # "YYYYMMDD HH" (EST) -> last close seen


def ingest(blob):
    zf = zipfile.ZipFile(io.BytesIO(blob))
    n = 0
    for nm in zf.namelist():
        if not nm.lower().endswith(".csv"):
            continue
        for line in io.TextIOWrapper(zf.open(nm), encoding="utf-8", errors="ignore"):
            # 20150101 170000;open;high;low;close;vol
            parts = line.rstrip("\n").split(";")
            if len(parts) < 5 or len(parts[0]) < 11:
                continue
            hourly[parts[0][:11] + ("0" if parts[0][10:12] < "30" else "3")] = parts[4]
            n += 1
    return n


def main():
    years = range(2010, 2026)
    ok, fail = [], []
    for y in years:
        url = f"{BASE}/download-free-forex-data/?/ascii/1-minute-bar-quotes/spxusd/{y}"
        try:
            blob = download_zip(url)
            if blob:
                n = ingest(blob)
                ok.append(f"{y}:{n}")
            else:
                fail.append(str(y))
        except Exception as e:
            fail.append(f"{y}({e})")
        print(f"year {y} done", flush=True)
        time.sleep(0.8)
    for m in range(1, 13):
        url = f"{BASE}/download-free-forex-data/?/ascii/1-minute-bar-quotes/spxusd/2026/{m}"
        try:
            blob = download_zip(url)
            if blob:
                n = ingest(blob)
                ok.append(f"2026-{m}:{n}")
        except Exception:
            pass
        time.sleep(0.6)
    print("ok:", len(ok), "fail:", fail, flush=True)

    # EST (UTC-5, no DST) -> UTC epoch, hour buckets
    rows = []
    for k in sorted(hourly):
        st = time.strptime(k[:11], "%Y%m%d %H"); half = 1800 if k[11] == "3" else 0
        epoch = calendar.timegm(st) + 5 * 3600 + half
        rows.append((epoch, float(hourly[k])))
    print("hourly bars:", len(rows),
          time.strftime("%Y-%m-%d %H", time.gmtime(rows[0][0])), "->",
          time.strftime("%Y-%m-%d %H", time.gmtime(rows[-1][0])), flush=True)

    pack = {}
    t0 = rows[0][0]
    pack["hr"] = dict(t0=t0, dt=[(t - t0) // 1800 for t, _ in rows],
                      e=[round(c, 5) for _, c in rows])
    js = json.dumps(pack, separators=(",", ":"))
    open(f"{SP}/dow/spx30.json", "w").write(js)
    print("pack:", len(js) // 1024, "KB", flush=True)


main()
