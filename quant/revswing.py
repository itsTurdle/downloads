"""Mean-reversion swing battery on the Dow — the tools that MATCH the market's
personality (VR 0.74, AC1 -0.097). Same protocol: IS through 2012, OOS 2013+,
1bp per unit turnover, pick per family by IS Sharpe, report OOS untouched.

  E  RSI(2) dip-buy: RSI2 < X at close -> long; exit close > SMA5 or RSI2 > 65
  F  IBS fade: (C-L)/(H-L) < x -> long; exit IBS > y or 5d
  G  N-day-low pullback (only above SMA200): close = lowest of N -> long;
     exit close > prior day's high or 5d
  H  gap fade, intraday only: open <= prev close * (1-g) -> long at open,
     flat at close same day (no overnight exposure)
Variants: each also 'gated' = only when close > SMA200 (uptrend regime).
Then: leverage ladder on OOS survivors; hourly annex on 2y of ^DJI.
"""
import json
import numpy as np

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"
rows = json.load(open(f"{SP}/dow/DIA.json"))["data"]
rows.sort(key=lambda r: r["t"])
D = np.array([int(r["t"].replace("-", "")) for r in rows])
O = np.array([float(r["o"]) for r in rows]); H = np.array([float(r["h"]) for r in rows])
L = np.array([float(r["l"]) for r in rows]); C = np.array([float(r["c"]) for r in rows])
A = np.array([float(r["a"]) for r in rows])
N = len(A)
R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1                      # close-to-close total
RID = (C-O)/O                                                # intraday open->close

def sma(x, n):
    out = np.full(len(x), np.nan); c = np.cumsum(x)
    out[n-1:] = (c[n-1:]-np.concatenate(([0], c[:-n])))/n
    return out
S5, S200 = sma(C, 5), sma(C, 200)

def rsi(x, n=2):
    d = np.diff(x, prepend=x[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = np.zeros(len(x)); ad = np.zeros(len(x))
    au[0] = up[0]; ad[0] = dn[0]
    a = 1/n
    for i in range(1, len(x)):
        au[i] = (1-a)*au[i-1] + a*up[i]
        ad[i] = (1-a)*ad[i-1] + a*dn[i]
    rs = au/np.maximum(ad, 1e-12)
    return 100 - 100/(1+rs)
RSI2 = rsi(C, 2)
IBS = np.where(H > L, (C-L)/(H-L), 0.5)

IS_END = 20130101; COST = 1e-4
warm = np.arange(N) >= 210
mIS = warm & (D < IS_END); mOOS = warm & (D >= IS_END)

def run(e_next, ret, mask):
    e = e_next[mask]; r = ret[mask]
    tr = np.abs(np.diff(np.concatenate(([0.0], e))))
    strat = e*r - tr*COST
    eq = np.cumprod(1+strat); yrs = len(r)/252
    pk = np.maximum.accumulate(eq)
    sh = strat.mean()/strat.std()*np.sqrt(252) if strat.std() > 0 else 0
    return dict(cagr=(eq[-1]**(1/yrs)-1)*100, dd=((eq/pk)-1).min()*100, sh=sh,
                expo=np.abs(e).mean(), tpy=tr.sum()/2/yrs)

def lag(sig):
    e = np.zeros(N); e[1:] = sig[:-1]; return e

cands = {}   # name -> (exposure_series, return_series)
for X in (5, 10, 15, 25):
    for gate in (False, True):
        sig = np.zeros(N); pos = 0.0
        for i in range(210, N):
            if pos == 0.0:
                if RSI2[i] < X and (not gate or C[i] > S200[i]): pos = 1.0
            else:
                if C[i] > S5[i] or RSI2[i] > 65: pos = 0.0
            sig[i] = pos
        cands[f"E RSI2<{X}{' gated' if gate else ''}"] = (lag(sig), R)
for x in (0.1, 0.2, 0.3):
    for y in (0.7, 0.8, 0.9):
        for gate in (False, True):
            sig = np.zeros(N); pos = 0.0; age = 0
            for i in range(210, N):
                if pos == 0.0:
                    if IBS[i] < x and (not gate or C[i] > S200[i]): pos = 1.0; age = 0
                else:
                    age += 1
                    if IBS[i] > y or age >= 5: pos = 0.0
                sig[i] = pos
            cands[f"F IBS<{x:.1f}/>{y:.1f}{' g' if gate else ''}"] = (lag(sig), R)
for n_ in (5, 7, 10):
    for gate in (True,):                       # pullback family pre-gated by design
        sig = np.zeros(N); pos = 0.0; age = 0
        for i in range(210, N):
            if pos == 0.0:
                lowN = C[i] <= C[max(0, i-n_+1):i+1].min()
                if lowN and C[i] > S200[i]: pos = 1.0; age = 0
            else:
                age += 1
                if C[i] > H[i-1] or age >= 5: pos = 0.0
            sig[i] = pos
        cands[f"G {n_}d-low pullback g"] = (lag(sig), R)
for gpc in (0.003, 0.005, 0.01):
    for gate in (False, True):
        sig = np.zeros(N)
        for i in range(210, N):
            if O[i] <= C[i-1]*(1-gpc) and (not gate or C[i-1] > S200[i-1]): sig[i] = 1.0
        cands[f"H gap{gpc*100:.1f}%{' g' if gate else ''}"] = (sig, RID)  # same-day, known at open

fams = {}
for name, (e, ret) in cands.items():
    fams.setdefault(name[0], []).append((name, run(e, ret, mIS), run(e, ret, mOOS)))

bh_oos = run(np.ones(N), R, mOOS)
print(f"{'':<24}{'IS CAGR':>8}{'IS Sh':>6} | {'OOS CAGR':>8}{'OOS DD':>7}{'OOS Sh':>7}{'expo':>6}{'tr/yr':>6}")
print(f"{'buy & hold DIA (OOS)':<24}{'':>8}{'':>6} | {bh_oos['cagr']:>7.1f}%{bh_oos['dd']:>6.0f}%{bh_oos['sh']:>7.2f}{1.0:>6.2f}")
winners = {}
for fam in "EFGH":
    ranked = sorted(fams[fam], key=lambda t: -t[1]["sh"])
    for name, isr, oos in ranked[:3]:
        tag = "  <-- IS pick" if name == ranked[0][0] else ""
        print(f"{name:<24}{isr['cagr']:>7.1f}%{isr['sh']:>6.2f} | {oos['cagr']:>7.1f}%"
              f"{oos['dd']:>6.0f}%{oos['sh']:>7.2f}{oos['expo']:>6.2f}{oos['tpy']:>6.1f}{tag}")
    winners[fam] = ranked[0]; print()

# per-exposure honesty + leverage on IS picks that stayed positive OOS
print("LEVERAGE — OOS, IS-picked survivors, financing 5% on excess, ruin at -95%/day")
print(f"{'strategy':<24}{'L':>4}{'CAGR':>8}{'maxDD':>7}{'$1k ->':>9}{'worst day':>10}")
for fam in "EFGH":
    name, isr, oos = winners[fam]
    if oos["cagr"] <= 0: continue
    e_full, ret = cands[name]
    for Lv in (1, 2, 3, 5):
        e = e_full[mOOS]*Lv; r = ret[mOOS]
        tr = np.abs(np.diff(np.concatenate(([0.0], e))))
        fin = np.maximum(np.abs(e)-1, 0)*0.05/360
        strat = e*r - tr*COST - fin
        eq = 1.0; pk = 1.0; dd = 0.0; worst = 0.0; dead = False
        for s_ in strat:
            worst = min(worst, s_)
            if dead: continue
            if s_ <= -0.95: eq = 0; dead = True; continue
            eq *= 1+s_; pk = max(pk, eq); dd = min(dd, eq/pk-1)
        yrs = len(r)/252
        cg = (eq**(1/yrs)-1)*100 if eq > 0 else -100
        print(f"{name:<24}{Lv:>3}x{cg:>7.1f}%{dd*100:>6.0f}%{eq*1000:>9,.0f}{worst*100:>9.1f}%")
    print()

# ---------------- hourly annex: 2y ^DJI ----------------
r_ = json.load(open(f"{SP}/dow/dji_h1.json"))["chart"]["result"][0]
cl = r_["indicators"]["quote"][0]["close"]
hc = np.array([c for c in cl if c])
hn = len(hc); hr = np.zeros(hn); hr[1:] = hc[1:]/hc[:-1]-1
hrsi = rsi(hc, 2)
hs5 = sma(hc, 5)
half = hn//2
print(f"HOURLY ANNEX — ^DJI 60m, {hn} bars (~2y). half1 = pick, half2 = verdict. cost 1bp/turn")
for X in (5, 10, 15):
    sig = np.zeros(hn); pos = 0.0
    for i in range(10, hn):
        if pos == 0.0:
            if hrsi[i] < X: pos = 1.0
        elif hc[i] > hs5[i] or hrsi[i] > 65: pos = 0.0
        sig[i] = pos
    e = np.zeros(hn); e[1:] = sig[:-1]
    for nm, sl in (("h1", slice(10, half)), ("h2", slice(half, hn))):
        ee = e[sl]; rr = hr[sl]
        tr = np.abs(np.diff(np.concatenate(([0.0], ee))))
        st = ee*rr - tr*COST
        ann = 252*6.5
        sh = st.mean()/st.std()*np.sqrt(ann) if st.std() > 0 else 0
        tot = (np.cumprod(1+st)[-1]-1)*100
        print(f"  RSI2<{X} {nm}: total {tot:+6.1f}%  Sh {sh:+.2f}  expo {ee.mean():.2f}", end="")
    print()
