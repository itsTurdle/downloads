"""Pre-registered predictability battery: is the Dow (DIA) easier for machines
than SPY / QQQ / EURUSD?

Fixed before running (no shopping after):
  A. AC(1),AC(2),AC(5) of daily returns, SE = 1/sqrt(n)
  B. Variance ratios VR(5), VR(20)  (>1 trending, <1 mean-reverting, 1 random walk)
  C. Momentum carryover: corr(past-20d return, next-5d return), non-overlapping next windows
  D. Conditionals: next-day mean after up day / down day / <=-2% panic day (bps)
  E. Walk-forward ML: GradBoost + Logistic, features = lagged rets(1,2,3,5,10,20),
     dist-from-SMA200, rv20, day-of-week; refit each Jan, OOS 2005->2026 (2010-> for FX);
     score = accuracy vs always-up baseline, AUC
  F. Vol predictability: corr(rv20_t, realized vol next 20d)  [the control: this SHOULD work]
  G. The car (band+gears) on DIA 3x proxy vs buy-hold  [structure transfer, not prediction]
"""
import json, math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"

def load_sa(sym):
    rows = json.load(open(f"{SP}/dow/{sym}.json"))["data"]
    rows.sort(key=lambda r: r["t"])
    d = np.array([int(r["t"].replace("-","")) for r in rows])
    a = np.array([float(r["a"]) for r in rows])
    c = np.array([float(r["c"]) for r in rows])
    return d, a, c

def load_fx():
    pack = json.load(open(f"{SP}/fx/pack.json"))
    d = np.array(pack["d"]); e = np.array(pack["e"])
    return d, e, e

ASSETS = {}
for s in ("DIA","SPY","QQQ"): ASSETS[s] = load_sa(s)
ASSETS["EURUSD"] = load_fx()

def acf(r, k):
    x = r - r.mean()
    return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))

def vr(r, q):
    x = r - r.mean()
    n = len(x)
    rq = np.convolve(x, np.ones(q), "valid")          # overlapping q-sums
    return float(rq.var() / (q * x.var()))

print(f"{'':<8}{'n':>6}{'drift':>7}{'AC1':>7}{'AC2':>7}{'AC5':>7}{'VR5':>6}{'VR20':>6}"
      f"{'mom20->5':>9}{'afterUP':>8}{'afterDN':>8}{'panic+1':>8}{'volR':>6}")
results = {}
for name,(d,a,c) in ASSETS.items():
    r = a[1:]/a[:-1]-1
    n = len(r); se = 1/math.sqrt(n)
    # C: past-20 vs next-5, stepping 5 to avoid overlap of targets
    p20, n5 = [], []
    for i in range(20, n-5, 5):
        p20.append(a[i]/a[i-20]-1); n5.append(a[i+5]/a[i]-1)
    mom = float(np.corrcoef(p20, n5)[0,1])
    up = r[:-1] > 0
    a_up = r[1:][up].mean()*1e4; a_dn = r[1:][~up].mean()*1e4
    panic = r[:-1] <= -0.02
    a_pan = r[1:][panic].mean()*1e4 if panic.sum() >= 20 else float("nan")
    rv = np.array([r[max(0,i-19):i+1].std() for i in range(n)])
    fwd = np.array([r[i+1:i+21].std() if i+21 <= n else np.nan for i in range(n)])
    m = ~np.isnan(fwd) & (np.arange(n) >= 20)
    volr = float(np.corrcoef(rv[m], fwd[m])[0,1])
    row = dict(n=n, drift=r.mean()*1e4, ac1=acf(r,1), ac2=acf(r,2), ac5=acf(r,5),
               vr5=vr(r,5), vr20=vr(r,20), mom=mom, a_up=a_up, a_dn=a_dn,
               a_pan=a_pan, volr=volr, se=se)
    results[name] = row
    star = lambda v: "*" if abs(v) > 2*se else " "
    print(f"{name:<8}{n:>6}{row['drift']:>6.1f}b{row['ac1']:>6.3f}{star(row['ac1'])}"
          f"{row['ac2']:>6.3f}{star(row['ac2'])}{row['ac5']:>6.3f}{star(row['ac5'])}"
          f"{row['vr5']:>6.2f}{row['vr20']:>6.2f}{mom:>9.3f}{a_up:>7.1f}b{a_dn:>7.1f}b"
          f"{row['a_pan']:>7.1f}b{volr:>6.2f}")
print(f"(* = |AC| > 2 standard errors; drift/conditionals in bps/day)")

# ---------------- E: walk-forward ML ----------------
print("\nWALK-FORWARD ML — predict next-day direction, refit each January, all OOS")
print(f"{'':<8}{'OOS days':>9}{'always-up':>10}{'GBoost':>8}{'Logistic':>9}{'GB AUC':>8}{'edge/day':>9}")
for name,(d,a,c) in ASSETS.items():
    r = a[1:]/a[:-1]-1
    dd = d[1:]
    n = len(r)
    # features at day i predict sign(r[i+1])
    F, Y, YD, RN = [], [], [], []
    sma = np.convolve(a, np.ones(200)/200, "valid")   # sma[i] = mean a[i..i+199]
    for i in range(220, n-1):
        rv20 = r[i-19:i+1].std()
        f = [r[i], r[i-1], r[i-2],
             a[i+1]/a[i-4]-1, a[i+1]/a[i-9]-1, a[i+1]/a[i-19]-1,
             a[i+1]/sma[i+1-199]-1, rv20, (dd[i]//1) % 7]
        F.append(f); Y.append(1 if r[i+1] > 0 else 0); YD.append(dd[i]); RN.append(r[i+1])
    F = np.array(F); Y = np.array(Y); YD = np.array(YD); RN = np.array(RN)
    y0 = 2005 if name != "EURUSD" else 2010
    accs_gb, accs_lr, aucs, base, nn, pnl = [], [], [], [], 0, []
    for yr in range(y0, 2027):
        tr = YD < yr*10000; te = (YD >= yr*10000) & (YD < (yr+1)*10000)
        if te.sum() < 50 or tr.sum() < 800: continue
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                        learning_rate=0.05, subsample=0.8, random_state=7)
        gb.fit(F[tr], Y[tr])
        p = gb.predict_proba(F[te])[:,1]
        lr = LogisticRegression(max_iter=500).fit(F[tr], Y[tr])
        pl = lr.predict(F[te])
        accs_gb.append(((p > 0.5) == Y[te]).mean()*te.sum())
        accs_lr.append((pl == Y[te]).mean()*te.sum())
        try: aucs.append(roc_auc_score(Y[te], p)*te.sum())
        except Exception: pass
        base.append(Y[te].mean()*te.sum()); nn += te.sum()
        pnl.append(np.where(p > 0.5, RN[te], 0).sum() - RN[te].clip(min=0).sum()*0)
    gbA = sum(accs_gb)/nn*100; lrA = sum(accs_lr)/nn*100; bA = sum(base)/nn*100
    auc = sum(aucs)/nn
    edge = (gbA-bA)/100 * 1  # not meaningful in bps; report accuracy delta
    print(f"{name:<8}{nn:>9}{bA:>9.1f}%{gbA:>7.1f}%{lrA:>8.1f}%{auc:>8.3f}{gbA-bA:>+8.1f}pp")

# ---------------- G: car on DIA ----------------
d,a,c = ASSETS["DIA"]
r = a[1:]/a[:-1]-1
ma = np.convolve(c, np.ones(200)/200, "valid")
st = np.ones(len(c)); cur = 1.0
for i in range(199, len(c)):
    m = ma[i-199]
    if cur == 1.0 and c[i] < m*0.98: cur = 0.0
    elif cur == 0.0 and c[i] > m*1.02: cur = 1.0
    st[i] = cur
band = st[:-1]                                   # applied to next day's return
letf = 3*r - 2*0.05/360 - 0.0095/252
eq = np.cumprod(1 + band[199:]*letf[199:] + (1-band[199:])*0.04/360)
bh = np.cumprod(1 + letf[199:])
def stats(x):
    yrs = len(x)/252; pk = np.maximum.accumulate(x)
    return (x[-1]**(1/yrs)-1)*100, ((x/pk)-1).min()*100
cg, dd_ = stats(eq); cb, db = stats(bh)
print(f"\nCAR ON DOW 3x: gated {cg:.1f}%/yr maxDD {dd_:.0f}%  vs raw 3x {cb:.1f}%/yr maxDD {db:.0f}%")
