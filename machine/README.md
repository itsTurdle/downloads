# THE MACHINE

A daily signal engine for a regime-gated leveraged index strategy. One file,
zero dependencies, runs on any Python 3.8+ (Windows, Mac, Linux, phone IDEs).

It answers exactly one question each day: **what fraction of the account
belongs in the 3x fund right now — 0, 25, 50, or 100%?** You place the trade
in your broker app only on the days the answer changes. Typical trading
frequency: a handful of times per year.

## Daily use

Run any time after US market close (~4pm ET):

```
python3 machine/engine.py --balance 3000 --held 0
```

- `--balance` — total account value (fund + cash)
- `--held`    — dollars currently sitting in the 3x fund
- `--engine tqqq|upro|both` — which ladder (default both, balance split 50/50)
- `--backtest` — self-audit: gated vs raw 3x vs index on the same data it
  just downloaded, so the strategy's justification is reproducible from
  scratch on your own machine, from primary price data, with no external
  claims involved.

Output ends with a `LEVEL` line per engine and, when it differs from the
previous session, a `<<< CHANGED` marker plus the exact BUY/SELL order.
Exit code is 2 when any level changed, 0 otherwise (for automation).

## The rules (fixed — this is the whole strategy)

1. **BAND** — 200-day SMA state machine, 2% hysteresis. Close above
   1.02x SMA200 = risk-ON; close below 0.98x = risk-OFF (everything to
   cash) until it recloses above. This is the master switch; it is the only
   thing that saved every backtested variant from -95%+ drawdowns.
2. **VOL** — 20-day realized volatility above its own all-history median
   halves exposure.
3. **CHOP** — 10-day SMA below 100-day SMA halves exposure again.
4. Level = BAND x VOL x CHOP ∈ {0, 0.25, 0.5, 1.0} = fraction in
   TQQQ (QQQ engine) or UPRO (SPY engine). Remainder in cash or a
   treasury money-market fund.
5. **Trade only on level changes. Deposits buy in at the current level.
   Never override the band.** Every discretionary override is a bet that
   you know more than the tested system; the record says you don't.

All positions are plain long ETFs — works in a cash account, no margin,
no options, no minimum.

## What this is not

It does not predict anything. It does not know about news, earnings, or
your feelings. It is a drawdown-control system wrapped around leveraged
index exposure: it accepts slightly less return than raw buy-and-hold in
the best decades in exchange for surviving the worst ones. Expected
long-run range is roughly 13-22%/yr with drawdowns to about -45 to -65%
depending on the engine; single years can be far outside that in both
directions. Nothing here is a guarantee.

## Audit

`machine/verify.py` (needs pandas/numpy) recomputes every day's level with
an independent vectorized implementation and diffs it against `engine.py`.
Last run: 14,943 trading days, 100.000% agreement, both engines.
