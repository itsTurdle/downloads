"""Modern out-of-sample scoreboard — CAGR on data the parameters never saw,
era by era, so regime drift is visible.

Unseen-status ledger (strict):
  * IBS swing on DIA (enter IBS<0.2, exit IBS>0.7 or 5d): parameters chosen on
    1998-2012 ONLY -> every year >=2013 is genuinely unseen.  CLEAN.
  * 200d band switch on 3x (UPRO/TQQQ proxies): the rule (200d MA, hysteresis,
    risk-off to cash) was published pre-2016 (Gayed) and has no tuned knobs ->
    2016+ is unseen by parameter selection; earlier years shown for context. CLEAN-ISH.
  * Full car (band + vol gear + chop gear): gears were validated on the full
    1998-2026 window during this project -> era table shows STABILITY, not proof.
    FLAGGED.
  * IBS + ibs>=0.045 entry filter: the loss-autopsy analysts saw losses from all
    years incl. post-2013 -> its 2013+ numbers are NOT clean. FLAGGED, shown last.
Costs 1bp/turn; 3x financing (L-1)*5%/360; cash idle 0%.
"""
import json
import numpy as np

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"

def load(sym):
    rows = json.load(open(f"{SP}/dow/{sym}.json"))["data"]; rows.sort(key=lambda r: r["t"])
    D = np.array([int(r["t"].replace("-", "")) for r in rows])
    O = np.array([float(r["o"]) for r in rows]); Hh = np.array([float(r["h"]) for r in rows])
    Ll = np.array([float(r["l"]) for r in rows]); C = np.array([float(r["c"]) for r in rows])
    A = np.array([float(r["a"]) for r in rows])
    return D, O, Hh, Ll, C, A

def sma(x, n):
    out = np.full(len(x), np.nan); c = np.cumsum(x)
    out[n-1:] = (c[n-1:]-np.concatenate(([0], c[:-n])))/n
    return out

ERAS = [(20130101, 20170101, "2013-16"), (20170101, 20200101, "2017-19"),
        (20200101, 20230101, "2020-22"), (20230101, 20270101, "2023-26"),
        (20250815, 20270101, "last 12m")]

def era_stats(D, ret, lo, hi):
    m = (D >= lo) & (D < hi)
    r = ret[m]
    if len(r) < 40: return None
    eq = np.cumprod(1+r); yrs = len(r)/252
    pk = np.maximum.accumulate(eq)
    return (eq[-1]**(1/yrs)-1)*100, ((eq/pk)-1).min()*100

def row(name, D, ret):
    cells = []
    for lo, hi, _ in ERAS:
        s = era_stats(D, ret, lo, hi)
        cells.append(f"{s[0]:>7.1f}% {s[1]:>5.0f}%" if s else f"{'—':>14}")
    print(f"{name:<34}" + "".join(cells))

COST = 1e-4
def band_engine(D, C, A, L, gears=False):
    N = len(A)
    R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1
    ma = sma(C, 200)
    st = np.ones(N); cur = 1.0
    for i in range(200, N):
        if cur == 1.0 and C[i] < ma[i]*0.98: cur = 0.0
        elif cur == 0.0 and C[i] > ma[i]*1.02: cur = 1.0
        st[i] = cur
    f = st.copy()
    if gears:
        rv = np.full(N, np.nan)
        for i in range(20, N): rv[i] = R[i-19:i+1].std()*np.sqrt(252)
        med = np.full(N, np.nan)
        for i in range(252, N): med[i] = np.nanmedian(rv[20:i+1])
        lowv = rv < med
        chop = sma(C, 10) < sma(C, 100)
        f = st*np.where(lowv, 1.0, 0.5)*np.where(np.nan_to_num(chop, nan=0) > 0, 0.5, 1.0)
    f_lag = np.zeros(N); f_lag[1:] = f[:-1]
    letf = L*R - (L-1)*0.05/360 - 0.0095/252
    strat = f_lag*letf - np.abs(np.diff(f_lag, prepend=0))*L*COST
    return strat

def ibs_engine(D, O, Hh, Ll, C, A, L, ibs_floor=None):
    N = len(A)
    R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1
    IBS = np.where(Hh > Ll, (C-Ll)/(Hh-Ll), 0.5)
    sig = np.zeros(N); pos = 0.0; age = 0
    for i in range(210, N):
        if pos == 0.0:
            if IBS[i] < 0.2 and (ibs_floor is None or IBS[i] >= ibs_floor): pos = 1.0; age = 0
        else:
            age += 1
            if IBS[i] > 0.7 or age >= 5: pos = 0.0
        sig[i] = pos
    e = np.zeros(N); e[1:] = sig[:-1]*L
    fin = np.maximum(e-1, 0)*0.05/360
    return e*R - np.abs(np.diff(e, prepend=0))*COST - fin

Ds, Os, Hs, Ls_, Cs, As = load("SPY")
Dq, Oq, Hq, Lq, Cq, Aq = load("QQQ")
Dd, Od, Hd, Ld, Cd, Ad = load("DIA")

hdr = f"{'strategy (CAGR% maxDD% per era)':<34}" + "".join(f"{n:>14}" for _, _, n in ERAS)
print(hdr); print("-"*len(hdr))
Rs = np.zeros(len(As)); Rs[1:] = As[1:]/As[:-1]-1
Rq = np.zeros(len(Aq)); Rq[1:] = Aq[1:]/Aq[:-1]-1
Rd = np.zeros(len(Ad)); Rd[1:] = Ad[1:]/Ad[:-1]-1
row("SPY buy & hold", Ds, Rs)
row("QQQ buy & hold", Dq, Rq)
print()
row("CLEAN: IBS swing DIA 1x", Dd, ibs_engine(Dd, Od, Hd, Ld, Cd, Ad, 1))
row("CLEAN: IBS swing DIA 3x", Dd, ibs_engine(Dd, Od, Hd, Ld, Cd, Ad, 3))
row("CLEAN-ISH: 200d band UPRO 3x", Ds, band_engine(Ds, Cs, As, 3))
row("CLEAN-ISH: 200d band TQQQ 3x", Dq, band_engine(Dq, Cq, Aq, 3))
print()
row("FLAGGED: full car TQQQ 3x", Dq, band_engine(Dq, Cq, Aq, 3, gears=True))
row("FLAGGED: IBS+floor filter 3x", Dd, ibs_engine(Dd, Od, Hd, Ld, Cd, Ad, 3, 0.045))
