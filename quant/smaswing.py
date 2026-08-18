"""SMA swing systems on the Dow (DIA daily, 1998-2026) — pre-registered.

Protocol fixed before running:
  IS = through 2012-12-31 (grid search, pick top Sharpe per family)
  OOS = 2013-01-01 onward (untouched; reported as-is)
  Costs: 1bp per unit turnover. Financing: (L-1)*5%/360 per day. Idle cash: 0%.
  Families:
    A trend cross: long when SMA_f > SMA_s (variants: long/flat, long/short)
    B price-vs-SMA band: long when c > SMA_n*(1+b), flat below
    C dip-buy swing: enter when c < SMA_n*(1-z), exit at SMA touch or 10d;
      variant gated by c > SMA200 (only dips inside uptrends)
    D panic bounce: buy close after a day <= -k%, hold m days (gated variant same)
  Leverage ladder on the OOS survivor: 1/2/3/5/10x with single-day ruin check
  (day loss <= -95% of equity => account dead, stays dead).
"""
import json
import numpy as np

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"
rows = json.load(open(f"{SP}/dow/DIA.json"))["data"]
rows.sort(key=lambda r: r["t"])
D = np.array([int(r["t"].replace("-", "")) for r in rows])
A = np.array([float(r["a"]) for r in rows])
C = np.array([float(r["c"]) for r in rows])
N = len(A)
R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1

def sma(x, n):
    out = np.full(len(x), np.nan)
    c = np.cumsum(x)
    out[n-1:] = (c[n-1:] - np.concatenate(([0], c[:-n]))) / n
    return out

SMAS = {n: sma(C, n) for n in (5, 10, 20, 50, 100, 200)}
IS_END = 20130101
COST = 1e-4

def run(expo, mask):
    """expo[i] = exposure held during day i's return (decided at close i-1)."""
    e = expo[mask]; r = R[mask]
    tr = np.abs(np.diff(np.concatenate(([0.0], e))))
    strat = e*r - tr*COST
    eq = np.cumprod(1+strat)
    yrs = len(r)/252
    pk = np.maximum.accumulate(eq)
    dd = (eq/pk-1).min()
    sh = strat.mean()/strat.std()*np.sqrt(252) if strat.std() > 0 else 0
    return dict(cagr=(eq[-1]**(1/yrs)-1)*100, dd=dd*100, sh=sh,
                expo=np.abs(e).mean(), tpy=tr.sum()/2/yrs)

def split_masks():
    warm = np.arange(N) >= 210
    return warm & (D < IS_END), warm & (D >= IS_END)

mIS, mOOS = split_masks()

def lag(sig):
    """decision at close t applies to day t+1"""
    e = np.zeros(N); e[1:] = sig[:-1]; return e

candidates = {}

# A: trend crosses
for f in (5, 10, 20, 50):
    for s in (50, 100, 200):
        if f >= s: continue
        base = (SMAS[f] > SMAS[s]).astype(float)
        candidates[f"A cross {f}/{s} L/F"] = lag(np.nan_to_num(base))
        candidates[f"A cross {f}/{s} L/S"] = lag(np.nan_to_num(base*2-1))
# B: price vs SMA band
for n in (20, 50, 100, 200):
    for b in (0.0, 0.01, 0.02):
        base = (C > SMAS[n]*(1+b)).astype(float)
        candidates[f"B band {n} +{b*100:.0f}% L/F"] = lag(np.nan_to_num(base))
# C: dip-buy swing (stateful)
for n in (5, 10, 20):
    for z in (0.01, 0.02, 0.03):
        for gate in (False, True):
            sig = np.zeros(N); pos = 0.0; age = 0
            for i in range(210, N):
                if pos == 0.0:
                    if C[i] < SMAS[n][i]*(1-z) and (not gate or C[i] > SMAS[200][i]):
                        pos = 1.0; age = 0
                else:
                    age += 1
                    if C[i] >= SMAS[n][i] or age >= 10: pos = 0.0
                sig[i] = pos
            candidates[f"C dip {n} -{z*100:.0f}%{' gated' if gate else ''}"] = lag(sig)
# D: panic bounce
for k in (0.015, 0.02, 0.03):
    for m in (1, 2, 3, 5):
        for gate in (False, True):
            sig = np.zeros(N); left = 0
            for i in range(210, N):
                if R[i] <= -k and (not gate or C[i] > SMAS[200][i]): left = m
                elif left > 0: left -= 1
                sig[i] = 1.0 if left > 0 else 0.0
            candidates[f"D panic {k*100:.1f}%/{m}d{' gated' if gate else ''}"] = lag(sig)

# grade all: IS pick per family, OOS verdict
fams = {}
for name, e in candidates.items():
    isr = run(e, mIS); oos = run(e, mOOS)
    fams.setdefault(name[0], []).append((name, isr, oos))

bh_is, bh_oos = run(np.ones(N), mIS), run(np.ones(N), mOOS)
print(f"{'':<26}{'IS CAGR':>8}{'IS DD':>7}{'IS Sh':>6} | {'OOS CAGR':>8}{'OOS DD':>7}{'OOS Sh':>7}{'expo':>6}{'tr/yr':>6}")
print(f"{'buy & hold DIA':<26}{bh_is['cagr']:>7.1f}%{bh_is['dd']:>6.0f}%{bh_is['sh']:>6.2f} | "
      f"{bh_oos['cagr']:>7.1f}%{bh_oos['dd']:>6.0f}%{bh_oos['sh']:>7.2f}{1.0:>6.2f}{0:>6}")
winners = {}
for fam in "ABCD":
    ranked = sorted(fams[fam], key=lambda x: -x[1]["sh"])
    for name, isr, oos in ranked[:3]:
        tag = " <-- IS pick" if name == ranked[0][0] else ""
        print(f"{name:<26}{isr['cagr']:>7.1f}%{isr['dd']:>6.0f}%{isr['sh']:>6.2f} | "
              f"{oos['cagr']:>7.1f}%{oos['dd']:>6.0f}%{oos['sh']:>7.2f}{oos['expo']:>6.2f}{oos['tpy']:>6.1f}{tag}")
    winners[fam] = ranked[0]
    print()

# leverage ladder on each family's IS pick, full period OOS, with ruin
print("LEVERAGE LADDER — OOS only (2013-2026), financing (L-1)*5%/360, ruin if day <= -95%")
print(f"{'strategy':<26}{'L':>4}{'CAGR':>8}{'maxDD':>7}{'$1k ->':>9}{'ruined?':>8}{'worst day':>10}")
for fam in "ABCD":
    name, _, _ = winners[fam]
    e_full = candidates[name]
    for L in (1, 3, 5, 10):
        e = e_full[mOOS]*L; r = R[mOOS]
        tr = np.abs(np.diff(np.concatenate(([0.0], e))))
        fin = np.maximum(np.abs(e)-1, 0)*0.05/360
        strat = e*r - tr*COST - fin
        eq = 1.0; pk = 1.0; dd = 0.0; dead = False; worst = 0.0
        for s_ in strat:
            worst = min(worst, s_)
            if dead: continue
            if s_ <= -0.95: eq = 0.0; dead = True; continue
            eq *= 1+s_
            pk = max(pk, eq); dd = min(dd, eq/pk-1)
        yrs = len(r)/252
        cg = (eq**(1/yrs)-1)*100 if eq > 0 else -100.0
        print(f"{name:<26}{L:>3}x{cg:>7.1f}%{dd*100:>6.0f}%{eq*1000:>9,.0f}"
              f"{'DEAD' if dead else 'no':>8}{worst*100:>9.1f}%")
    print()
