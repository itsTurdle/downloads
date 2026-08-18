"""Replicating the human: 'red or green below the line' with escalating context.

Architecture (fixed before running):
  1. Three walk-forward GBM classifiers on the Dow (DIA daily), target = sign of
     forward 5d return. Same model, three CONTEXT SIZES:
       S: short features (<=20d)   M: adds medium (50-100d)   L: adds long (200-252d)
     Refit each January on all data through the prior year (like the Markov test).
     Scored by AUC per era.
  2. Escalation, like the human: read pS; if |pS-0.5| < gap, ask for more data (pM);
     still unsure -> pL. Trade only if the final read clears conviction theta.
  3. Trade like the human: enter at close in the read's direction; book the trade
     at the first close in profit >= wintake; confidence decays -> time-stop at T
     days. No price stop (the human never mentioned one). Costs 1bp/side.
  4. Tune (theta, gap, wintake, T) on VAL years 2005-2012 targeting WR>=90 then max
     expectancy. Grade untouched TEST years 2013-2026.
  5. Ablations: no-escalation (pS only) and no-classifier (enter long every day the
     policy is flat, same exits) — to attribute the win rate honestly.
"""
import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"
rows = json.load(open(f"{SP}/dow/DIA.json"))["data"]; rows.sort(key=lambda r: r["t"])
D = np.array([int(r["t"].replace("-", "")) for r in rows])
O = np.array([float(r["o"]) for r in rows]); Hh = np.array([float(r["h"]) for r in rows])
Ll = np.array([float(r["l"]) for r in rows]); C = np.array([float(r["c"]) for r in rows])
A = np.array([float(r["a"]) for r in rows]); V = np.array([float(r["v"]) for r in rows])
N = len(A)
R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1

def sma(x, n):
    out = np.full(len(x), np.nan); c = np.cumsum(x)
    out[n-1:] = (c[n-1:]-np.concatenate(([0], c[:-n])))/n
    return out
def roll_std(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n-1, len(x)): out[i] = x[i-n+1:i+1].std()
    return out
S5, S20, S50, S100, S200 = (sma(C, n) for n in (5, 20, 50, 100, 200))
VM20, VM60 = sma(V, 20), sma(V, 60)
RV5, RV20, RV60 = roll_std(R, 5), roll_std(R, 20), roll_std(R, 60)
TR = np.maximum(Hh-Ll, np.maximum(np.abs(Hh-np.roll(C, 1)), np.abs(Ll-np.roll(C, 1))))
ATR14 = sma(TR, 14)
HI252 = np.array([C[max(0, i-251):i+1].max() for i in range(N)])
IBS = np.where(Hh > Ll, (C-Ll)/(Hh-Ll), 0.5)
def retn(n):
    out = np.full(N, np.nan); out[n:] = C[n:]/C[:-n]-1; return out
def slope(s, n):
    out = np.full(N, np.nan); out[n:] = s[n:]/s[:-n]-1; return out

FS = [retn(1), retn(2), retn(3), retn(5), retn(10), C/S5-1, C/S20-1, slope(S20, 5),
      V/VM20, IBS, TR/ATR14, RV5/RV20]
FM = FS + [C/S50-1, slope(S50, 10), retn(20), V/VM60, RV20*np.sqrt(252), RV5/RV60]
FL = FM + [C/S200-1, slope(S200, 20), retn(60), retn(120), C/HI252-1]
def matrix(fl): return np.column_stack([np.nan_to_num(f, nan=0.0) for f in fl])
X = {"S": matrix(FS), "M": matrix(FM), "L": matrix(FL)}
FWD = np.full(N, np.nan); FWD[:-5] = A[5:]/A[:-5]-1
Y = (FWD > 0).astype(int)

# walk-forward probabilities for the three context sizes (cached)
import os
P = {k: np.full(N, np.nan) for k in "SML"}
valid = ~np.isnan(FWD)
CACHE = f"{SP}/dow/replica_P.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    if len(z["S"]) == N:
        P = {k: z[k] for k in "SML"}
for k in "SML":
    if not np.all(np.isnan(P[k])): continue
    for yr in range(2005, 2027):
        tr = (D < yr*10000) & valid & (np.arange(N) >= 260)
        te = (D >= yr*10000) & (D < (yr+1)*10000)
        if te.sum() == 0 or tr.sum() < 800: continue
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=7)
        m.fit(X[k][tr], Y[tr])
        P[k][te] = m.predict_proba(X[k][te])[:, 1]
np.savez(CACHE, **P)

mask_oos = ~np.isnan(P["S"])
for k in "SML":
    mv = mask_oos & valid
    print(f"AUC {k} (all OOS 2005-26): {roc_auc_score(Y[mv], P[k][mv]):.3f}", end="  ")
print()

def escalate(i, gap):
    p = P["S"][i]
    if abs(p-0.5) < gap: p = P["M"][i]
    if abs(p-0.5) < gap: p = P["L"][i]
    return p

COST = 1e-4
def policy(days, theta, gap, wintake, T, mode="cascade"):
    """returns trade list [(pnl_after_cost, dir, hold)]"""
    trades = []; i_pos = None; dir_ = 0; entry = 0.0; held = 0
    for i in days:
        if i_pos is not None:
            pnl = (A[i]/entry-1)*dir_
            held += 1
            if pnl >= wintake or held >= T:
                trades.append((pnl - 2*COST, dir_, held))
                i_pos = None
        if i_pos is None and not np.isnan(P["S"][i]) and i < N-1:
            if mode == "cascade": p = escalate(i, gap)
            elif mode == "noesc": p = P["S"][i]
            else: p = 1.0                       # no-classifier: always long
            if p >= 0.5+theta: dir_ = 1
            elif p <= 0.5-theta and mode != "always": dir_ = -1
            else: continue
            i_pos = i; entry = A[i]; held = 0
    return trades

def stats(trades):
    if not trades: return None
    p = np.array([t[0] for t in trades])
    wr = (p > 0).mean()*100
    return dict(n=len(p), wr=wr, exp=p.mean()*100, tot=(np.cumprod(1+p)[-1]-1)*100,
                aw=p[p > 0].mean()*100 if (p > 0).any() else 0,
                al=p[p <= 0].mean()*100 if (p <= 0).any() else 0)

VAL = [i for i in range(N) if mask_oos[i] and 20050101 <= D[i] < 20130101]
TEST = [i for i in range(N) if mask_oos[i] and D[i] >= 20130101]

print("\nTUNE on 2005-2012 (WR>=90 first, then expectancy):")
best = None
for theta in (0.03, 0.05, 0.08, 0.12):
    for gap in (0.03, 0.06, 0.10):
        for wintake in (0.0005, 0.001, 0.0025):
            for T in (8, 12, 16, 20):
                s = stats(policy(VAL, theta, gap, wintake, T))
                if not s or s["n"] < 40: continue
                key = (s["wr"] >= 90, s["wr"] >= 85, s["exp"])
                if best is None or key > best[0]: best = (key, (theta, gap, wintake, T), s)
(theta, gap, wintake, T) = best[1]
s = best[2]
print(f"  picked theta={theta} gap={gap} wintake={wintake*100:.2f}% T={T} -> "
      f"VAL: n={s['n']} WR={s['wr']:.1f}% exp={s['exp']:+.3f}%/trade")

print("\n=== TEST 2013-2026 (untouched) ===")
for label, mode in [("cascade (the replica)", "cascade"), ("no escalation (pS only)", "noesc"),
                    ("no classifier (always long)", "always")]:
    s = stats(policy(TEST, theta, gap, wintake, T, mode))
    print(f"  {label:<28} n={s['n']:>4}  WR={s['wr']:.1f}%  avg win {s['aw']:+.2f}%  "
          f"avg loss {s['al']:+.2f}%  exp {s['exp']:+.3f}%/tr  total {s['tot']:+.1f}%")
s = stats(policy(VAL, theta, gap, wintake, T, "cascade"))
print(f"  (VAL cascade, for reference)  n={s['n']}  WR={s['wr']:.1f}%  exp {s['exp']:+.3f}%/tr")

# how often escalation changed the action, TEST days
ch = 0; tot = 0
for i in TEST:
    if np.isnan(P["S"][i]): continue
    pS = P["S"][i]; pC = escalate(i, gap)
    aS = 1 if pS >= 0.5+theta else (-1 if pS <= 0.5-theta else 0)
    aC = 1 if pC >= 0.5+theta else (-1 if pC <= 0.5-theta else 0)
    tot += 1; ch += aS != aC
print(f"\nescalation changed the action on {ch}/{tot} days ({ch/tot*100:.0f}%)")
