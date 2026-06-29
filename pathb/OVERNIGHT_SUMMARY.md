# Overnight Work — End-of-Night Summary

Two tasks completed in order. Path B numbers left untouched as the anchor. Nothing pushed.

## Task 1 — Scheduling design audit (read-only) → `pathb/AUDIT_scheduling.md`
1. **SRPT normalization** — denominator runs on **true** sizes (oracle optimum); policies use
   predictions. ρ_TC is a standard competitive ratio vs the clairvoyant optimum — **design choice**, not a bug.
2. **Parallel span as serial size** — `p* ` (fork-join span) used as single-machine size; tail is
   extreme (P99/P50 ≈ 152×), which amplifies ratios — **design choice (justify in paper)**.
3. **Log vs linear** — every scheduler entry verified **linear** (worked example: p*=24 s, S*=648.9,
   d=15,572.8 s). The 169.6 / 2931 max-stretch outliers are **non-predictive FIFO/LAS baselines** —
   **no bug / honest**.
4. **Vs FIFO** — predictor policies cut ΣC by **39–63%** vs FIFO (best M5 PRR 62.9%) — ranking
   genuinely helps; the large ρ is the Point-1 normalization. **Honest result.**
   → No invariant tripped; nothing changed.

## Task 2 — Guarded cold-start improvement → `pathb/EXPLORATION_coldstart.md`
Validation-selected method (quantile-0.5 objective + val-only per-segment multiplicative
calibration, direct submit-time features). Test read once.

| Population | Cov@25 val → **test** | Path B anchor |
|---|---|---|
| Cold-start (count==0) | 17.35 → **16.20** | New 12.74 → **+3.46 pp** |
| Seen (count≥1) | 44.05 → 44.45 | — |
| All | 33.74 → 30.89 | base 29.11 / Meta+Cal 29.92 |

- **Best cold-start delta: +3.46 pp on test (12.74 → 16.20)** — modest, within the stated
  information ceiling (< 30 ⇒ no leakage).
- Driver = median objective + val-only segment calibration; **hierarchical priors did NOT help**
  (winner is direct-only) — honest negative result.
- **Cost:** worse RMSLE (1.447 → 1.513) and lower global Spearman (0.720 → 0.668) — coverage traded
  for ranking; this calibrated predictor should **not** feed the schedulers (they use the
  uncalibrated base ranking).

## Invariants
I1 (temporal split), I2/I3 (train-only / submit-time), I4 (shift(1) histories) re-checked at
feature build and finalize — **all held every round**. Leakage checks: test_seen − val_seen =
+0.40 pp (OK); cold test < cold val (selection-optimism direction, not inflation).

## STOP events
**None.** No invariant violation, no suspicious gain. Both tasks ran to completion.

## Not pushed
Per instruction, nothing was committed or pushed. New files only: `pathb/AUDIT_scheduling.md`,
`pathb/EXPLORATION_coldstart.md`, this summary, and `pathb/improve/` (features/search/finalize +
artifacts). Path B's REPORT, tables, figures, and base_preds.pkl are unchanged.
