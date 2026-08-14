#!/usr/bin/env python3
"""Audit: recompute the exposure levels with an independent pandas implementation
(the original research code, vectorized) and compare to engine.py's stdlib math
on the exact same downloaded data. Prints any mismatching days.

Needs pandas+numpy (dev-only; engine.py itself needs nothing).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import engine as eng


def research_levels(dates, close, adj):
    df = pd.DataFrame({"c": close, "a": adj}, index=pd.to_datetime(dates))
    ret = df.a.pct_change()
    rv = ret.rolling(20).std() * np.sqrt(252)
    ma = df.c.rolling(200).mean()
    up = (df.c > ma * 1.02).values
    dn = (df.c < ma * 0.98).values
    st = np.zeros(len(df)); cur = 1.0
    for i in range(len(df)):
        if cur == 1.0 and dn[i]: cur = 0.0
        elif cur == 0.0 and up[i]: cur = 1.0
        st[i] = cur
    band = pd.Series(st, index=df.index)                       # same-day state
    chop = (df.c.rolling(10).mean() < df.c.rolling(100).mean())
    med = rv.expanding(252).median()
    lowv = (rv < med).fillna(False)
    f = band * np.where(lowv, 1.0, 0.5) * np.where(chop, 0.5, 1.0)
    f[ma.isna()] = np.nan                                      # warmup undefined
    return f


def main():
    ok = True
    for key in ("tqqq", "upro"):
        sym = eng.ENGINES[key]["under"]
        dates, close, adj = eng.fetch(sym)
        lv, _ = eng.compute_levels(dates, close, adj)
        ref = research_levels(dates, close, adj)
        n = mism = 0
        for i, d in enumerate(dates):
            a, b = lv[i], ref.iloc[i]
            if a is None or np.isnan(b):
                continue
            n += 1
            if abs(a - b) > 1e-12:
                mism += 1
                if mism <= 10:
                    print(f"  MISMATCH {sym} {d}: engine={a} research={b}")
        print(f"{sym}: {n} comparable days, {mism} mismatches "
              f"({100 * (1 - mism / max(n, 1)):.3f}% agreement)")
        ok = ok and mism == 0
    print("VERIFIED: engine.py reproduces the research signal exactly." if ok
          else "FAILED: implementations disagree — do not trade this.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
