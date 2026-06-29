# Path B — ATLAS/LASched Re-baseline: REPORT

Corrected, leakage-free re-run of the ATLAS prediction + LASched scheduling benchmark. Executed strictly in order **0 → A → (B, C)** with a single Meta base established in Part A, persisted to `pathb/artifacts/base_preds.pkl`, and loaded (never re-trained) by B and C.

**[STOP] markers:** none. Egress to aliyuncs stayed blocked (403); the three trace tarballs were taken from the repo `main` root, extracted to the repo root, and `load_tables()` was pointed there.

## Headline

- **Determinism (V5):** two seeded Meta-base runs are **identical** (Cov@25 29.1123 == 29.1123, `array_equal=True`). The pipeline is deterministic/seeded and the base is reproducible.
- **33.4 NOT reproduced → superseded.** The corrected Meta base Cov@25 = **X = 29.1123**. The prior 33.4 was produced by the *buggy* (scrambled-test) pipeline; after the Part-0 fix the honest test-window Cov@25 is 29.11. **All downstream numbers use this new base; prior handoff numbers are NOT reused.**
- **Meta+Cal (isotonic, val-only):** All Cov@25 = 29.92 (base 29.11, gain **+0.80pp**, modest, no leak flag). Rank-preserving (Spearman base↔cal = 0.9998).

## Part 0 — index-misalignment bug fix

**Bug.** In cell 4, `add_causal_histories_no_leak()` did `df = df.sort_values("submit_time").reset_index(drop=True)` **after** `time_split()` had computed `idx_tr/va/te` on the *pre-sort* frame, and returned that re-indexed frame. `prepare_matrices()` / the schedulers then did `df.loc[idx_te]` on a frame whose labels no longer matched the split → a **scrambled** test set (temporal leakage, meaningless test metrics).

**Fix (invariant I1).** Sort by `r_j` (submit_time) **once, up front** (stable mergesort), compute the split indices on the sorted frame, and remove the internal re-sort. Indices stay aligned to `df` everywhere downstream.

**Proof (V2).** `mean(r_j[idx_te] ≥ t_val) = 1.000000` (required 1.0).

## Part A — Meta base, determinism, calibration

- **A.1** Corrected pipeline (cells 1→4) on the real trace; LGBM-Meta base test Cov@25 = **X = 29.1123**.
- **A.2** Determinism gate: ran the base twice with the existing config (`random_state=42`, `np.random.seed(42)`); predictions are **bit-identical** (`array_equal=True`), Cov@25 29.1123 both runs ⇒ deterministic. No re-seeding was required for reproducibility; 33.4 is nonetheless not reproduced because the *bug fix* (not the seed) changed the test set. Decision: `deterministic_existing`.
- **A.3** Persisted base val/test log-predictions → `pathb/artifacts/base_preds.pkl`.
- **A.4** Calibration spec: IsotonicRegression(increasing=True, out_of_bounds='clip') of true log-span on Meta base predicted log-span, fit on VALIDATION only, applied to test.
  Meta+Cal — All Cov@25 29.92 / Rec 47.48 / New 12.74; Cov@50 All 49.31 / Rec 71.38; RMSLE 1.454; Spearman 0.720.

### tab:results-main — Prediction (FULL new numbers; supersede prior handoff)

All numbers from the **corrected** pipeline (Part-0 fix). The prior 33.4 anchor was produced by the buggy (scrambled-test) pipeline and is **superseded**.

| Method | Cov@25 All | Cov@25 Rec | Cov@25 New | Cov@50 All | Cov@50 Rec | RMSLE | Spearman ρ |
|---|---|---|---|---|---|---|---|
| M1 CQR | 20.56 | 33.64 | 7.45 | 37.74 | 58.72 | 1.579 | 0.660 |
| M3 HRAS | 19.05 | 32.55 | 6.54 | 36.63 | 59.44 | 1.678 | 0.572 |
| M4 Isotonic | 23.12 | 37.22 | 8.46 | 40.22 | 62.09 | 1.591 | 0.644 |
| M5 Meta (base) | 29.11 | 46.30 | 12.39 | 48.17 | 70.24 | 1.447 | 0.720 |
| M6 TwoStage | 25.12 | 43.27 | 6.35 | 41.41 | 66.01 | 1.657 | 0.565 |
| M7 Recency | 24.30 | 41.30 | 6.66 | 40.53 | 62.80 | 1.676 | 0.580 |
| **Meta+Cal (M5+isotonic)** | **29.92** | 47.48 | 12.74 | 49.31 | 71.38 | 1.454 | 0.720 |

Rec = training signature count ≥ 5; New = unseen (count = 0).


## Part B — recurrence figure (panel b)

Cov@25 & Cov@50 of **Meta+Cal** by training-recurrence bucket; teal/CDF house style. Files: `fig_recurrence.{pdf,png,svg}`.

| Bucket (train sig. count) | n | Cov@25 | Cov@50 |
|---|---|---|---|
| New | 52,752 | 12.74 | 27.51 |
| 1-4 | 4,918 | 27.71 | 49.00 |
| 5-49 | 43,052 | 47.18 | 70.15 |
| >=50 | 9,132 | 48.88 | 77.18 |

Coverage rises monotonically with recurrence — the expected signal (more training history ⇒ better calibrated predictions). Panel (a) is full-trace, unchanged.

## Part C — scheduling on the uncalibrated base

Schedulers consume the **uncalibrated** base ranking `predictions['M5']` (guard V8: `max|expm1(base_log_te) − M5| = 0.0e+00`, and M5 ≠ Meta+Cal). If A.2 had re-baselined the model, predictor-fed rows would be regenerated; here the base is deterministic, and the **whole table is regenerated** under Path B regardless. Non-predictive baselines (SRPT/FIFO/RR/LAS/McNaughton/OPT) are objective-defined and unaffected.

### tab:unified_all_methods — Scheduling (predictor rows NEW; baselines unchanged)

**C.1 Total completion ΣC_j (1 machine, online, preemptive) — normalized by SRPT (n=10,000, PRR λ=0.7).**

| Policy | ρ_TC = ΣC/SRPT |
|---|---|
| SRPT (baseline) | 1.0000 |
| SJF (baseline) | 1.0001 |
| RR (baseline) | 1.9984 |
| FIFO (baseline) | 6.1712 |
| M1 CQR — SPJF | 3.1185 |
| M1 CQR — PRR | 2.6991 |
| M3 HRAS — SPJF | 3.3099 |
| M3 HRAS — PRR | 2.6166 |
| M4 Isotonic — SPJF | 2.8730 |
| M4 Isotonic — PRR | 2.5606 |
| M5 Meta (base) — SPJF | 2.6275 |
| M5 Meta (base) — PRR | 2.2880 |
| M6 TwoStage — SPJF | 3.7762 |
| M6 TwoStage — PRR | 2.7417 |
| M7 Recency — SPJF | 3.6407 |
| M7 Recency — PRR | 2.6969 |

**C.2 Max-stretch S_max (1 machine, online, preemptive) — normalized by offline S\* (EDF bisection; n=5,000, S\*=648.867, S_emp=648.442).**

| Policy | ρ_S,max | ρ_S,99 | ρ_S,med |
|---|---|---|---|
| OPT (EDF at S*) (baseline) | 1.000 | 0.985 | 0.041 |
| SRPT (true) (baseline) | 1.186 | 1.090 | 0.013 |
| FIFO (baseline) | 2931.741 | 1287.149 | 25.728 |
| LAS/FB (baseline) | 169.637 | 132.238 | 3.556 |
| M1 CQR — SPRPT | 43.639 | 28.145 | 0.390 |
| M1 CQR — EDF-P | 71.565 | 53.890 | 1.517 |
| M3 HRAS — SPRPT | 53.749 | 33.037 | 0.471 |
| M3 HRAS — EDF-P | 96.748 | 70.345 | 1.644 |
| M4 Isotonic — SPRPT | 50.697 | 27.235 | 0.457 |
| M4 Isotonic — EDF-P | 63.438 | 47.617 | 0.553 |
| M5 Meta (base) — SPRPT | 27.451 | 21.335 | 0.293 |
| M5 Meta (base) — EDF-P | 29.932 | 22.545 | 0.689 |
| M6 TwoStage — SPRPT | 51.680 | 28.667 | 0.417 |
| M6 TwoStage — EDF-P | 86.882 | 66.223 | 1.577 |
| M7 Recency — SPRPT | 46.385 | 26.844 | 0.408 |
| M7 Recency — EDF-P | 65.571 | 53.150 | 1.120 |

**C.3 Makespan C_max (m machines, batch, non-preemptive) — normalized by McNaughton OPT_pre = max(Σp/m, max_j p_j).**

| m | OPT_pre | LPT ρ | SPT ρ | Rand ρ | M1 CQR LPPT ρ | M3 HRAS LPPT ρ | M4 Isotonic LPPT ρ | M5 Meta (base) LPPT ρ | M6 TwoStage LPPT ρ | M7 Recency LPPT ρ |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 85028.4 | 1.0001 | 1.3084 | 1.6315 | 1.4806 | 1.2406 | 1.2641 | 1.2763 | 1.2724 | 1.5272 |
| 10 | 138422.3 | 1.0000 | 1.3049 | 1.8335 | 1.7372 | 1.4513 | 1.7492 | 1.5277 | 1.8544 | 1.3936 |
| 20 | 183117.2 | 1.0000 | 1.5540 | 2.0064 | 1.6274 | 1.3516 | 1.5725 | 1.7771 | 1.8751 | 1.5813 |
| 50 | 236903.8 | 1.0000 | 1.4243 | 1.9673 | 1.8395 | 1.9261 | 2.0624 | 1.8118 | 1.7869 | 1.6390 |
| 100 | 283942.3 | 1.0000 | 1.6629 | 2.1490 | 2.0411 | 2.3407 | 2.0707 | 1.8951 | 2.5472 | 2.1396 |

(SPPT ρ per method/m is in `pathb/artifacts/sched_results.json` and `makespan_results.csv`.)


## Verification — V1…V8

- **V1** — ✅ PASS: train.max < val.min and val.max < test.min (ordered, non-overlapping)
- **V2** — ✅ PASS: mean(r_j[idx_te] >= t_val) must equal 1.0 (the bug fix)
- **V3** — ✅ PASS: target excluded; gro_hist_mean == strictly-prior expanding mean (shift1); isotonic fit on validation-sized vector only.
- **V4** — ✅ PASS: n_test=109,854 (exp 109,854); Rec(>=5)=47.5% (exp 47.5); New(==0)=48.0% (exp 48.0); resid(1-4)=4.5% (exp 4.5)
- **V5** — ✅ PASS: two seeded base runs identical -> deterministic; new anchor X=29.1123; 33.4 NOT reproduced (superseded by bug fix)
- **V6** — ✅ PASS: Meta+Cal Cov@25 >= base, gain modest (<=6pp), isotonic monotone => rank ~unchanged
- **V7** — ✅ PASS: p_star == max_t e_t - min_t s_t (fork-join span), not max_t d_t
- **V8** — ✅ PASS: ranking source == uncalibrated Meta base; normalizations confirmed; full table regenerated

**ALL_PASS = True**

Selected values: V1 train[542323,5020578] val[5020579,5764865] test[5764873,6451081]; V2 frac=1.000000; V4 n_test=109,854, Rec=47.5%, New=48.0%, resid=4.5%; V6 gain=+0.80pp, ρ(base,cal)=0.9998; V7 spot-check max|Δp*|=0.0e+00 over 300 jobs.

## Deviations from the verbatim notebook (all non-semantic)

1. **Part-0 fix** (sort `r_j` first; remove internal re-sort) — the requested correction.
2. **Vectorized causal histories.** The original val/test "frozen stats" step looped row-by-row with scalar `df.loc[i, col]` assignments (~275k iterations; impractically slow under pandas 3.0). Replaced with the numerically-identical vectorized `.map()` form, and train-side `expanding().mean()/.count().shift(1)` replaced by the equivalent cumsum identity (verified `np.allclose` to 1e-12; ewm kept as per-group `ewm(span=10,adjust=False).mean().shift(1)`). **Numbers unchanged.**
3. **Stable sort** (`kind='mergesort'`) for reproducible tie ordering; **`to_pickle`** instead of parquet (no pyarrow available). No effect on results.

## Reproduce

```
python3 pathb/build_data.py   # Part 0: load+sort+split+features (cache)
python3 pathb/predict.py      # Part A: methods + determinism gate + calibration
python3 pathb/figure.py       # Part B: fig_recurrence.{pdf,png,svg}
python3 pathb/sched_run.py    # Part C: scheduling (uncalibrated base)
python3 pathb/verify.py       # V1..V8
python3 pathb/report.py       # this REPORT.md + tables
```
