"""Replica v2 — situational read on 1h bars, fast loss exits, confidence-scaled wins.

Per approved plan:
  target: next-1d smoothed direction (SMA50 course, adaptive reversal tolerance)
  model : GBM walk-forward, 6-month refits; 4h-scale escalation in the dead-band
  policy: scaled entry, quick exits (loss cut m*sigma; TP 1.5*m*(0.75+0.5*conv);
          staleness H; p-flip/decay), 3 scalars tuned on VAL, TEST untouched
  claim : ~90% WR at ~1.5:1 shape (expectancy ~ +1.25R). Controls: always-long
          with same exits; shuffled-p. No-edge WR at this shape ~ 40%.

usage: replica2.py <pack.json> <val_start> <test_start>   (years)
"""
import json
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

PACK_PATH, VAL_Y, TEST_Y = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pk = json.load(open(PACK_PATH))
if "hr" in pk: pk = pk["hr"]
T30 = np.array(pk["dt"], dtype=np.int64)*1800 + pk["t0"]
E30 = np.array(pk["e"], dtype=float)
V30 = np.array(pk.get("v", np.zeros(len(E30))), dtype=float)

# ---- aggregate 30m -> 1h (close = last in bucket, vol = sum) ----
hb = T30 // 3600
idx = np.nonzero(np.diff(hb, append=hb[-1]+1))[0]      # last 30m bar of each hour
T = hb[idx]*3600; C = E30[idx]
V = np.add.reduceat(V30, np.concatenate(([0], idx[:-1]+1)))
N = len(C)
R = np.zeros(N); R[1:] = C[1:]/C[:-1]-1
YEAR = np.array([1970 + t//31556952 for t in T])       # approx; refined below
import time as _time
YEAR = np.array([_time.gmtime(t).tm_year for t in T])
HOUR = np.array([_time.gmtime(t).tm_hour + _time.gmtime(t).tm_min/60 for t in T])
DOW = np.array([_time.gmtime(t).tm_wday for t in T])

bars_day = np.median(np.bincount((T//86400).astype(int) - int(T[0]//86400))[
    np.bincount((T//86400).astype(int) - int(T[0]//86400)) > 0])
H = int(round(bars_day))
WIN = int(252*bars_day)                                 # trailing "1y" in bars
print(f"bars: {N}  {_time.strftime('%Y-%m-%d', _time.gmtime(T[0]))} -> "
      f"{_time.strftime('%Y-%m-%d', _time.gmtime(T[-1]))}  bars/day~{H}  1y~{WIN}")

def sma(x, n):
    out = np.full(len(x), np.nan); c = np.cumsum(x)
    out[n-1:] = (c[n-1:]-np.concatenate(([0], c[:-n])))/n
    return out
def rollstd(x, n):
    c1 = np.cumsum(x); c2 = np.cumsum(x*x)
    out = np.full(len(x), np.nan)
    s1 = c1[n-1:]-np.concatenate(([0], c1[:-n])); s2 = c2[n-1:]-np.concatenate(([0], c2[:-n]))
    out[n-1:] = np.sqrt(np.maximum(s2/n-(s1/n)**2, 1e-18))
    return out
def zsc(x, n):
    m = sma(np.nan_to_num(x), n); s = rollstd(np.nan_to_num(x), n)
    return (x-m)/np.maximum(s, 1e-12)
def rollmin(x, n):
    out = np.full(len(x), np.nan)
    from collections import deque
    dq = deque()
    for i, v in enumerate(x):
        while dq and x[dq[-1]] >= v: dq.pop()
        dq.append(i)
        if dq[0] <= i-n: dq.popleft()
        if i >= n-1: out[i] = x[dq[0]]
    return out
def rollmax(x, n): return -rollmin(-x, n)

S50, S200 = sma(C, 50), sma(C, 200)
SIG1 = rollstd(R, WIN)                                  # trailing-1y hourly ret vol
PS = C/S50-1; PL = C/S200-1; SPR = S50/S200-1
slope50 = np.full(N, np.nan); slope50[6:] = S50[6:]/S50[:-6]-1
curv50 = np.full(N, np.nan); curv50[6:] = slope50[6:]-slope50[:-6]
slope200 = np.full(N, np.nan); slope200[H:] = S200[H:]/S200[:-H]-1
cross = np.sign(S50-S200); since = np.zeros(N)
for i in range(1, N):
    since[i] = 0 if cross[i] != cross[i-1] else since[i-1]+1
rv_s, rv_l = rollstd(R, H), rollstd(R, 5*H)
lo_d, hi_d = rollmin(C, H), rollmax(C, H)
lo_w, hi_w = rollmin(C, 5*H), rollmax(C, 5*H)
def retz(k):
    r = np.full(N, np.nan); r[k:] = C[k:]/C[:-k]-1
    return zsc(r, WIN)

F1 = np.column_stack([np.nan_to_num(f) for f in [
    zsc(PS, WIN), zsc(PL, WIN), zsc(SPR, WIN), zsc(slope50, WIN), zsc(curv50, WIN),
    zsc(slope200, WIN), np.log1p(since)/8*np.sign(cross), rv_s/np.maximum(rv_l, 1e-12),
    (C-lo_d)/np.maximum(hi_d-lo_d, 1e-12), (C-lo_w)/np.maximum(hi_w-lo_w, 1e-12),
    retz(1), retz(3), retz(6), retz(H),
    np.sin(2*np.pi*HOUR/24), np.cos(2*np.pi*HOUR/24), DOW.astype(float),
    zsc(V, WIN) if V.std() > 0 else np.zeros(N)]])

# 4h-scale escalation features (computed on 4h closes, mapped back)
h4 = T // 14400
i4 = np.nonzero(np.diff(h4, append=h4[-1]+1))[0]
C4 = C[i4]; n4 = len(C4)
S50_4, S200_4 = sma(C4, 50), sma(C4, 200)
sl4 = np.full(n4, np.nan); sl4[6:] = S50_4[6:]/S50_4[:-6]-1
W4 = max(WIN//4, 300)
F2s = np.column_stack([np.nan_to_num(f) for f in [
    zsc(C4/S50_4-1, W4), zsc(C4/S200_4-1, W4), zsc(S50_4/S200_4-1, W4), zsc(sl4, W4)]])
map4 = np.searchsorted(i4, np.arange(N), side="left")   # latest complete 4h bar index
map4 = np.clip(map4-1, 0, n4-1)
F2 = F2s[map4]

# ---- target: clean smoothed direction over next H bars ----
KAPPA = 0.5
dser = np.full(N, np.nan); dser[H:] = S50[H:]-S50[:-H]  # realized (causal) H-bar SMA50 change
sigD = rollstd(np.nan_to_num(dser), WIN)
fwd = np.full(N, np.nan); fwd[:-H] = S50[H:]-S50[:-H]
minpath = np.full(N, np.nan); maxpath = np.full(N, np.nan)
S50p = np.nan_to_num(S50, nan=np.inf)
for i in range(N-H):
    w = S50[i+1:i+H+1]
    minpath[i] = np.nanmin(w)-S50[i]; maxpath[i] = np.nanmax(w)-S50[i]
lab = np.full(N, 9)                                     # 9 = unlabeled/messy
ok = (~np.isnan(fwd)) & (~np.isnan(sigD)) & (sigD > 0) & (~np.isnan(S50))
up = ok & (fwd > 0) & (minpath > -KAPPA*sigD)
dn = ok & (fwd < 0) & (maxpath < KAPPA*sigD)
lab[up] = 1; lab[dn] = 0
print(f"labels: clean-up {up.sum()}  clean-down {dn.sum()}  messy/other {(lab==9).sum()}")

# ---- walk-forward GBM, 6-month refits ----
P1 = np.full(N, np.nan); P2 = np.full(N, np.nan)
halves = []
for y in range(YEAR.min()+3, YEAR.max()+1):
    halves += [(y, 1), (y, 7)]
MON = np.array([_time.gmtime(t).tm_mon for t in T])
for (y, mth) in halves:
    cut = np.searchsorted(YEAR*100+MON, y*100+mth)
    end = np.searchsorted(YEAR*100+MON, (y*100+mth)+6 if mth == 1 else (y+1)*100+1)
    if end <= cut: continue
    trmask = (np.arange(N) + H < cut) & (lab != 9) & (np.arange(N) >= WIN)
    if trmask.sum() < 2000: continue
    tr = np.nonzero(trmask)[0][-30000:]
    te = np.arange(cut, min(end, N))
    m1 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06,
                                    subsample=0.8, random_state=7)
    m1.fit(F1[tr], lab[tr]); P1[te] = m1.predict_proba(F1[te])[:, 1]
    m2 = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.06,
                                    subsample=0.8, random_state=7)
    m2.fit(F2[tr], lab[tr]); P2[te] = m2.predict_proba(F2[te])[:, 1]

oosm = ~np.isnan(P1)
for nm, P in (("1h read", P1), ("4h read", P2)):
    mv = oosm & (lab != 9)
    print(f"AUC {nm} (clean labels, all OOS): {roc_auc_score(lab[mv], P[mv]):.3f}", end="   ")
print()

# ---- policy ----
COST = 1e-4
def run_policy(days_idx, d_ent, d_exf, m_cut, mode="cascade"):
    d_exit = d_ent*d_exf
    trades = []
    pos = 0.0; entry_px = 0.0; entry_i = 0; conv = 0.0; dirn = 0
    for i in days_idx:
        p1 = P1[i]
        if np.isnan(p1): continue
        if mode == "always": p = 0.99
        elif mode == "oracle": p = 0.99 if lab[i] == 1 else (0.01 if lab[i] == 0 else 0.5)
        elif mode == "shuffle": p = PSH[i]
        elif mode == "noesc": p = p1
        else: p = P2[i] if abs(p1-0.5) < d_ent else p1
        if pos != 0.0:
            adv = (C[i]/entry_px-1)*dirn
            held = i-entry_i
            w_tp = 1.5*m_cut*(0.75+0.5*conv)
            exit_now = (adv <= -m_cut*SIG1[entry_i]*np.sqrt(H)
                        or adv >= w_tp*SIG1[entry_i]*np.sqrt(H)
                        or held >= H
                        or (mode != "always" and (abs(p-0.5) < d_exit or np.sign(p-0.5) != dirn)))
            if exit_now:
                pnl = adv*abs(pos) - 2*COST*abs(pos)
                trades.append((pnl, adv, dirn, held))
                pos = 0.0
                continue
        if pos == 0.0 and i < N-1:
            if abs(p-0.5) >= d_ent:
                dirn = int(np.sign(p-0.5))
                conv = min(abs(p-0.5)/0.35, 1.0)
                pos = conv*dirn
                entry_px = C[i]; entry_i = i
    return trades

def tstats(trades):
    if not trades: return None
    pn = np.array([t[0] for t in trades]); adv = np.array([t[1] for t in trades])
    wr = (pn > 0).mean()*100
    aw = adv[adv > 0].mean() if (adv > 0).any() else 0
    al = adv[adv <= 0].mean() if (adv <= 0).any() else -1e-9
    return dict(n=len(pn), wr=wr, ratio=aw/abs(al) if al else 0,
                expR=(pn.mean()/abs(al*np.mean([abs(t[0])/max(abs(t[1]),1e-9) for t in trades[:1]]) if al else 1)) if al else 0,
                exp=pn.mean()*100, tot=(np.cumprod(1+pn)[-1]-1)*100,
                aw=aw*100, al=al*100)

VALI = np.nonzero(oosm & (YEAR >= VAL_Y) & (YEAR < TEST_Y))[0]
TESTI = np.nonzero(oosm & (YEAR >= TEST_Y))[0]
rng = np.random.RandomState(7)
PSH = P1.copy(); v = PSH[oosm]; rng.shuffle(v); PSH[oosm] = v

print("\nTUNE on VAL:")
best = None
for d_ent in (0.06, 0.10, 0.14):
    for d_exf in (0.3, 0.6):
        for m_cut in (0.75, 1.25, 2.0):
            s = tstats(run_policy(VALI, d_ent, d_exf, m_cut))
            if not s or s["n"] < 60: continue
            key = (s["wr"] >= 90, s["wr"] >= 70, s["exp"])
            if best is None or key > best[0]: best = (key, (d_ent, d_exf, m_cut), s)
(d_ent, d_exf, m_cut) = best[1]; s = best[2]
print(f"  picked d_ent={d_ent} d_exf={d_exf} m={m_cut} -> VAL n={s['n']} WR={s['wr']:.1f}% "
      f"shape={s['ratio']:.2f}:1 exp={s['exp']:+.3f}%")

print(f"\n=== TEST {TEST_Y}+ (untouched) ===")
for label, mode in [("replica (cascade)", "cascade"), ("no escalation", "noesc"),
                    ("always-long, same exits", "always"), ("shuffled-p (dead eyes)", "shuffle"),
                    ("ORACLE (sees the true label)", "oracle")]:
    s = tstats(run_policy(TESTI, d_ent, d_exf, m_cut, mode))
    if not s: print(f"  {label:<26} no trades"); continue
    print(f"  {label:<26} n={s['n']:>4}  WR={s['wr']:.1f}%  shape={s['ratio']:.2f}:1  "
          f"avg win {s['aw']:+.2f}% avg loss {s['al']:+.2f}%  exp {s['exp']:+.3f}%  tot {s['tot']:+.1f}%")

# leak test: all features lagged one extra bar
F1L = np.roll(F1, 1, axis=0); F1L[0] = 0
yv = YEAR.max()-1
cut = np.searchsorted(YEAR, yv)
trmask = (np.arange(N)+H < cut) & (lab != 9) & (np.arange(N) >= WIN)
tr = np.nonzero(trmask)[0][-30000:]; te = np.arange(cut, N)
te = te[(lab[te] != 9)]
m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.06,
                               subsample=0.8, random_state=7)
m.fit(F1[tr], lab[tr]); a1 = roc_auc_score(lab[te], m.predict_proba(F1[te])[:, 1])
m.fit(F1L[tr], lab[tr]); a2 = roc_auc_score(lab[te], m.predict_proba(F1L[te])[:, 1])
print(f"\nleak test (final-year holdout): normal AUC {a1:.3f} vs +1-bar-lag AUC {a2:.3f} "
      f"(mild drop = clean; collapse = leak)")
