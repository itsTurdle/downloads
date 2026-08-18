"""Validate candidate loss-filters on the IBS system. Usage:
  venv/bin/python validate_filters.py '[{"feature":"vol20","op":">","threshold":25}, ...]'
Grades each single filter and the top pair, IS (98-12) / OOS (13-26), vs unfiltered and QQQ.
"""
import json, sys
import numpy as np

SP = "/tmp/claude-0/-home-user-downloads/f780e50d-87ae-526c-abad-4e61fb91143e/scratchpad"

def load(sym):
    rows = json.load(open(f"{SP}/dow/{sym}.json"))["data"]; rows.sort(key=lambda r: r["t"])
    return rows

rows = load("DIA")
T = [r["t"] for r in rows]
D = np.array([int(r["t"].replace("-", "")) for r in rows])
O = np.array([float(r["o"]) for r in rows]); Hh = np.array([float(r["h"]) for r in rows])
Ll = np.array([float(r["l"]) for r in rows]); C = np.array([float(r["c"]) for r in rows])
A = np.array([float(r["a"]) for r in rows]); N = len(A)
R = np.zeros(N); R[1:] = A[1:]/A[:-1]-1

def sma(x, n):
    out = np.full(len(x), np.nan); c = np.cumsum(x)
    out[n-1:] = (c[n-1:]-np.concatenate(([0], c[:-n])))/n
    return out
S50, S200 = sma(C, 50), sma(C, 200)
def rsi2f(x):
    d = np.diff(x, prepend=x[0]); up = np.where(d > 0, d, 0.); dn = np.where(d < 0, -d, 0.)
    au = np.zeros(len(x)); ad = np.zeros(len(x)); au[0] = up[0]; ad[0] = dn[0]; a = 0.5
    for i in range(1, len(x)):
        au[i] = (1-a)*au[i-1]+a*up[i]; ad[i] = (1-a)*ad[i-1]+a*dn[i]
    return 100-100/(1+au/np.maximum(ad, 1e-12))
RSI = rsi2f(C)
IBS = np.where(Hh > Ll, (C-Ll)/(Hh-Ll), 0.5)
TR = np.maximum(Hh-Ll, np.maximum(np.abs(Hh-np.roll(C, 1)), np.abs(Ll-np.roll(C, 1))))
ATR14 = np.array([TR[max(0, i-13):i+1].mean() for i in range(N)])
HI252 = np.array([C[max(0, i-251):i+1].max() for i in range(N)])
VOL20 = np.array([R[max(0, i-19):i+1].std()*np.sqrt(252)*100 for i in range(N)])
RV5 = np.array([R[max(0, i-4):i+1].std() for i in range(N)])
RV60 = np.array([R[max(0, i-59):i+1].std() for i in range(N)])
DS = np.zeros(N)
for i in range(1, N): DS[i] = DS[i-1]+1 if C[i] < C[i-1] else 0
WD = np.array([np.datetime64(t).astype('datetime64[D]').astype(object).weekday() for t in T])

FEAT = dict(
    ibs=IBS, ret1=R*100,
    gap=np.concatenate(([0], (O[1:]/C[:-1]-1)*100)),
    vol20=VOL20, rvratio=RV5/np.maximum(RV60, 1e-12),
    dist200=(C/S200-1)*100, dist50=(C/S50-1)*100, rsi2=RSI,
    downstreak=DS, dow=WD.astype(float), atrratio=TR/np.maximum(ATR14, 1e-9),
    mom20=np.concatenate((np.zeros(20), (C[20:]/C[:-20]-1)*100)),
    ddhigh=(C/HI252-1)*100,
    prioribs=np.concatenate(([0.5], IBS[:-1])),
)

def system(block):
    """block[i]=True -> skip new entries on day i. Returns exposure (lagged)."""
    sig = np.zeros(N); pos = 0.; age = 0
    for i in range(210, N):
        if pos == 0.:
            if IBS[i] < 0.2 and not block[i]: pos = 1.; age = 0
        else:
            age += 1
            if IBS[i] > 0.7 or age >= 5: pos = 0.
        sig[i] = pos
    e = np.zeros(N); e[1:] = sig[:-1]
    return e

warm = np.arange(N) >= 210
mIS = warm & (D < 20130101); mOOS = warm & (D >= 20130101)
COST = 1e-4

def grade(e_full, mask, L=1.0, fin=0.05):
    e = e_full[mask]*L; r = R[mask]
    tr = np.abs(np.diff(np.concatenate(([0.], e))))
    f = np.maximum(np.abs(e)-1, 0)*fin/360
    st = e*r - tr*COST - f
    eq = np.cumprod(1+st); yrs = len(r)/252
    pk = np.maximum.accumulate(eq)
    return dict(cagr=(eq[-1]**(1/yrs)-1)*100, dd=((eq/pk)-1).min()*100,
                sh=st.mean()/st.std()*np.sqrt(252) if st.std() > 0 else 0,
                expo=np.abs(e).mean())

def cond(rule):
    x = FEAT[rule["feature"]]
    return (x > rule["threshold"]) if rule["op"] == ">" else (x < rule["threshold"])

rules = json.loads(sys.argv[1])
base = system(np.zeros(N, bool))
qq = load("QQQ")
QA = np.array([float(r["a"]) for r in qq]); QD = np.array([int(r["t"].replace("-", "")) for r in qq])
QR = np.zeros(len(QA)); QR[1:] = QA[1:]/QA[:-1]-1
qm = QD >= 20130101
qeq = np.cumprod(1+QR[qm]); qyrs = qm.sum()/252
qpk = np.maximum.accumulate(qeq)
print(f"{'variant':<44}{'IS CAGR':>8}{'IS Sh':>6} | {'OOS CAGR':>9}{'OOS DD':>7}{'OOS Sh':>7}{'expo':>6}")
print(f"{'QQQ buy & hold (the bar)':<44}{'':>8}{'':>6} | {(qeq[-1]**(1/qyrs)-1)*100:>8.1f}%{((qeq/qpk)-1).min()*100:>6.0f}%{'':>7}{1.0:>6.2f}")
gi, go = grade(base, mIS), grade(base, mOOS)
print(f"{'IBS unfiltered':<44}{gi['cagr']:>7.1f}%{gi['sh']:>6.2f} | {go['cagr']:>8.1f}%{go['dd']:>6.0f}%{go['sh']:>7.2f}{go['expo']:>6.2f}")
graded = []
for rule in rules:
    e = system(cond(rule))
    gi, go = grade(e, mIS), grade(e, mOOS)
    nm = f"skip if {rule['feature']} {rule['op']} {rule['threshold']}"
    graded.append((rule, gi, go))
    print(f"{nm:<44}{gi['cagr']:>7.1f}%{gi['sh']:>6.2f} | {go['cagr']:>8.1f}%{go['dd']:>6.0f}%{go['sh']:>7.2f}{go['expo']:>6.2f}")
# best pair by IS Sharpe
best = sorted(graded, key=lambda t: -t[1]["sh"])[:2]
if len(best) == 2:
    r1, r2 = best[0][0], best[1][0]
    e = system(cond(r1) | cond(r2))
    gi, go = grade(e, mIS), grade(e, mOOS)
    print(f"{'PAIR (top-2 by IS): skip if either':<44}{gi['cagr']:>7.1f}%{gi['sh']:>6.2f} | {go['cagr']:>8.1f}%{go['dd']:>6.0f}%{go['sh']:>7.2f}{go['expo']:>6.2f}")
    print("\nLEVERAGE on the pair, OOS:")
    for L in (2, 3, 4, 5):
        g = grade(e, mOOS, L)
        print(f"  {L}x: CAGR {g['cagr']:>6.1f}%  maxDD {g['dd']:>5.0f}%  Sh {g['sh']:.2f}")
