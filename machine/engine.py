#!/usr/bin/env python3
"""THE MACHINE — daily signal engine for the gated 3x ladder.

Zero dependencies: standard library only. Runs on any Python 3.8+.
    py machine\\engine.py --balance 3000 --held 0          (Windows)
    python3 machine/engine.py --balance 3000 --held 0      (Mac/Linux)

What it does, once per day after market close:
  1. Downloads full daily history for the underlying index ETF (QQQ and/or SPY).
  2. Computes three regime detectors (exact rules below, nothing else):
       BAND  200-day SMA state machine with a 2% hysteresis band:
             risk-ON when close > 1.02 x SMA200, risk-OFF when close < 0.98 x SMA200,
             otherwise hold previous state.  OFF means: everything to cash. Always.
       VOL   20-day realized volatility vs its own expanding median (needs 252+ days
             of history). High vol -> halve exposure.
       CHOP  10-day SMA below 100-day SMA -> halve exposure.
  3. Multiplies them into ONE exposure level: 0, 0.25, 0.5, or 1.0
     = the fraction of the account that belongs in the 3x fund (TQQQ for QQQ,
       UPRO for SPY). The rest sits in cash / a treasury money-market fund.
  4. Prints the level, whether it CHANGED vs the previous trading day, and the
     exact dollar order to place if you pass --balance / --held.

The only network call is price data. Every rule is fixed in this file.
No prediction, no news, no opinions. The machine does not know what you hope.
"""
import argparse
import bisect
import json
import math
import os
import ssl
import sys
import urllib.request

ENGINES = {
    "tqqq": {"under": "QQQ", "fund": "TQQQ", "er": 0.0095},
    "upro": {"under": "SPY", "fund": "UPRO", "er": 0.0091},
}
API = "https://stockanalysis.com/api/symbol/s/{sym}/history?range=Max&period=Daily"
LEVEL_NAMES = {1.0: "FULL", 0.5: "HALF", 0.25: "QUARTER", 0.0: "CASH"}


def fetch(sym):
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
    req = urllib.request.Request(API.format(sym=sym),
                                 headers={"User-Agent": "Mozilla/5.0 (signal-engine)"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        payload = json.load(r)
    rows = payload["data"]
    rows.sort(key=lambda x: x["t"])                      # oldest first
    dates = [x["t"] for x in rows]
    close = [float(x["c"]) for x in rows]
    adj = [float(x["a"]) for x in rows]
    return dates, close, adj


def sma(vals, n):
    """rolling mean, None until n values seen"""
    out = [None] * len(vals)
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def compute_levels(dates, close, adj):
    """returns per-day exposure level + detector detail, same-day semantics:
    level[i] is computed from data through close of day i and applies to the
    NEXT session. (In backtests this is the standard one-day signal lag.)"""
    n = len(close)
    rets = [None] + [adj[i] / adj[i - 1] - 1 for i in range(1, n)]
    sma200 = sma(close, 200)
    sma10 = sma(close, 10)
    sma100 = sma(close, 100)

    # 20-day realized vol (sample stdev, annualized)
    rv = [None] * n
    for i in range(20, n):
        window = rets[i - 19:i + 1]
        m = sum(window) / 20
        var = sum((x - m) ** 2 for x in window) / 19
        rv[i] = math.sqrt(var) * math.sqrt(252)

    levels, detail = [None] * n, [None] * n
    state = 1.0
    rv_sorted = []                                        # expanding median store
    for i in range(n):
        if sma200[i] is not None:
            if state == 1.0 and close[i] < sma200[i] * 0.98:
                state = 0.0
            elif state == 0.0 and close[i] > sma200[i] * 1.02:
                state = 1.0
        band = state if sma200[i] is not None else None

        lowv = False
        if rv[i] is not None:
            bisect.insort(rv_sorted, rv[i])
            k = len(rv_sorted)
            if k >= 252:
                med = (rv_sorted[k // 2] if k % 2 else
                       (rv_sorted[k // 2 - 1] + rv_sorted[k // 2]) / 2)
                lowv = rv[i] < med

        chop = (sma10[i] is not None and sma100[i] is not None
                and sma10[i] < sma100[i])

        if band is None:
            continue
        levels[i] = band * (1.0 if lowv else 0.5) * (0.5 if chop else 1.0)
        detail[i] = {"band": band, "lowv": lowv, "chop": chop,
                     "close": close[i], "sma200": sma200[i], "rv": rv[i]}
    return levels, detail


def backtest(dates, adj, levels, er, years=None):
    """gated ladder vs raw 3x vs underlying, flat 4.5% rate assumption.
    Approximation for self-audit; the research numbers used daily EFFR."""
    rf = 0.045
    start = 0
    if years:
        target = str(int(dates[-1][:4]) - years) + dates[-1][4:]
        while start < len(dates) and dates[start] < target:
            start += 1
    eq_g = eq_r = eq_u = 1.0
    pk_g = pk_r = pk_u = 1.0
    dd_g = dd_r = dd_u = 0.0
    prev_lvl = 0.0
    ndays = 0
    for i in range(max(start, 1), len(dates)):
        if levels[i - 1] is None or adj[i - 1] <= 0:
            continue
        r1 = adj[i] / adj[i - 1] - 1
        letf = 3 * r1 - 2 * (rf + 0.005) / 360 - er / 252
        f = levels[i - 1]                                 # signal lag: prior close
        g = f * letf + (1 - f) * rf / 360 - abs(f - prev_lvl) * 3 * 5e-4
        prev_lvl = f
        eq_g *= 1 + g; eq_r *= 1 + letf; eq_u *= 1 + r1
        pk_g = max(pk_g, eq_g); dd_g = min(dd_g, eq_g / pk_g - 1)
        pk_r = max(pk_r, eq_r); dd_r = min(dd_r, eq_r / pk_r - 1)
        pk_u = max(pk_u, eq_u); dd_u = min(dd_u, eq_u / pk_u - 1)
        ndays += 1
    yrs = ndays / 252
    out = []
    for name, eq, dd in [("gated ladder", eq_g, dd_g),
                         ("raw 3x buy&hold", eq_r, dd_r),
                         ("underlying 1x", eq_u, dd_u)]:
        cagr = (eq ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        out.append((name, cagr, dd * 100, eq))
    return yrs, out


def run_engine(key, balance, held, do_backtest):
    cfg = ENGINES[key]
    dates, close, adj = fetch(cfg["under"])
    levels, detail = compute_levels(dates, close, adj)
    today, yday = levels[-1], levels[-2]
    d = detail[-1]
    changed = today != yday

    print("=" * 62)
    print(f"{cfg['under']} -> {cfg['fund']}   data through {dates[-1]}")
    print("=" * 62)
    print(f"  BAND  : {'risk-ON' if d['band'] else 'RISK-OFF'}   "
          f"(close {d['close']:.2f} vs SMA200 {d['sma200']:.2f}; "
          f"flip at {d['sma200'] * (0.98 if d['band'] else 1.02):.2f})")
    print(f"  VOL   : {'low (full)' if d['lowv'] else 'HIGH (halve)'}   "
          f"(20d realized {d['rv'] * 100:.1f}%)")
    print(f"  CHOP  : {'clear (full)' if not d['chop'] else 'CHOPPY (halve)'}")
    print(f"  LEVEL : {today:.2f}  [{LEVEL_NAMES[today]}]"
          + (f"   <<< CHANGED (was {yday:.2f} [{LEVEL_NAMES[yday]}])" if changed
             else f"   (unchanged since prior session)"))
    if balance is not None:
        target = today * balance
        delta = target - (held or 0.0)
        print(f"  TARGET: ${target:,.0f} in {cfg['fund']}, "
              f"${balance - target:,.0f} in cash/treasury MMF")
        if abs(delta) < max(0.01 * balance, 1):
            print("  ORDER : none - you are on target")
        else:
            side = "BUY" if delta > 0 else "SELL"
            print(f"  ORDER : {side} ${abs(delta):,.0f} of {cfg['fund']}")
    if do_backtest:
        for yrs_req in (None, 10):
            yrs, rows = backtest(dates, adj, levels, cfg["er"], years=yrs_req)
            label = "full history" if yrs_req is None else f"last {yrs_req}y"
            print(f"  self-audit ({label}, {yrs:.1f}y, flat-rate approx):")
            for name, cagr, dd, eq in rows:
                print(f"    {name:<18} CAGR {cagr:>6.1f}%   maxDD {dd:>6.0f}%   "
                      f"$1 -> {eq:,.2f}")
    return changed, today


def main():
    p = argparse.ArgumentParser(description="daily signal for the gated 3x ladder")
    p.add_argument("--engine", choices=["tqqq", "upro", "both"], default="both")
    p.add_argument("--balance", type=float, default=None,
                   help="total account value in dollars (fund + cash)")
    p.add_argument("--held", type=float, default=0.0,
                   help="dollars currently in the 3x fund")
    p.add_argument("--backtest", action="store_true",
                   help="print self-audit backtest from the same data")
    a = p.parse_args()
    keys = ["tqqq", "upro"] if a.engine == "both" else [a.engine]
    bal = a.balance / len(keys) if (a.balance and a.engine == "both") else a.balance
    any_change = False
    for k in keys:
        changed, _ = run_engine(k, bal, a.held, a.backtest)
        any_change = any_change or changed
    print()
    print("RULES: trade only when LEVEL changes. Deposits buy at the current level.")
    print("Never override the band. The machine is only as good as your obedience.")
    sys.exit(2 if any_change else 0)


if __name__ == "__main__":
    main()
