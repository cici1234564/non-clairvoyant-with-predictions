# ATLAS / LASched (Path B) — Data Summary + Plotting Handoff (self-contained)

This single file contains **all the result data inline** (so you can read/plot even without the
CSVs) **plus** instructions for an assistant with a charting skill. Companion CSVs (same numbers)
live in `pathb/figures/`.

## How to use
Upload the 10 CSVs and paste this file as context. Ask: "Using my charting skill, make figures
A–E per the *suggested plot* column; teal house style; export png+pdf; keep the caveats on each."

## Global conventions
- Pipeline is **leakage-free**, time-split 70/15/15 by submit time; **test used for final reporting only**.
- Units: `p*` / job sizes = **seconds**; `cov*` = **percent**; `rho`/`spearman` = Spearman rank corr.
- **Two recurrence definitions — DO NOT MIX:**
  - Prediction table → **Rec = train signature count ≥ 5**, **New = count == 0**.
  - Cold-start table → **seen = count ≥ 1**, **cold = count == 0**.
  - Count is per test job, computed on TRAIN only (known at submit time).
- n_test = 109,854 (cold/New = 52,752 ≈ 48%; seen ≥1 = 57,102; Rec ≥5 = 52,184 ≈ 47.5%).

---

## 1. Prediction methods — `data_prediction_methods.csv`
*Plot: grouped bars of Cov@25 by All/Rec/New across methods; M5 Meta is the base.*

| method | Cov25 All | Cov25 Rec | Cov25 New | Cov50 All | Cov50 Rec | RMSLE | Spearman |
|---|---|---|---|---|---|---|---|
| M1 CQR | 20.56 | 33.64 | 7.45 | 37.74 | 58.72 | 1.579 | 0.660 |
| M3 HRAS | 19.05 | 32.55 | 6.54 | 36.63 | 59.44 | 1.678 | 0.572 |
| M4 Isotonic | 23.12 | 37.22 | 8.46 | 40.22 | 62.09 | 1.591 | 0.644 |
| M5 Meta (base) | 29.11 | 46.30 | 12.39 | 48.17 | 70.24 | 1.447 | 0.720 |
| M6 TwoStage | 25.12 | 43.27 | 6.35 | 41.41 | 66.01 | 1.657 | 0.565 |
| M7 Recency | 24.30 | 41.30 | 6.66 | 40.53 | 62.80 | 1.676 | 0.580 |
| Meta+Cal | 29.92 | 47.48 | 12.74 | 49.31 | 71.38 | 1.454 | 0.720 |

## 2. Coverage vs training-recurrence bucket — `C_cov_by_recurrence.csv`
*Plot: bars; coverage rises monotonically with history; New is the bottleneck.*

| bucket | n | base Cov25 | base Cov50 | Meta+Cal Cov25 | Meta+Cal Cov50 |
|---|---|---|---|---|---|
| New(0) | 52,752 | 12.39 | 26.41 | 12.74 | 27.51 |
| 1–4 | 4,918 | 26.15 | 47.54 | 27.71 | 49.00 |
| 5–49 | 43,052 | 46.35 | 69.44 | 47.18 | 70.15 |
| ≥50 | 9,132 | 46.05 | 73.97 | 48.88 | 77.18 |

## 3. Coverage vs true-size decile — `D_cov_by_size_decile.csv`
*Plot: line of Cov@25 vs decile (x = median p*). Note: error is LOWEST at the smallest jobs, not the tail.*

| decile | median p*(s) | n | base Cov25 | Meta+Cal Cov25 | Meta+Cal Cov50 |
|---|---|---|---|---|---|
| D1 | 25 | 11,097 | 14.81 | 18.86 | 27.28 |
| D2 | 53 | 10,906 | 13.80 | 18.83 | 35.76 |
| D3 | 96 | 10,995 | 20.60 | 23.07 | 38.34 |
| D4 | 216 | 11,064 | 40.85 | 41.46 | 58.65 |
| D5 | 365 | 10,876 | 38.72 | 39.04 | 53.33 |
| D6 | 728 | 10,991 | 30.19 | 33.70 | 55.42 |
| D7 | 1,413 | 10,974 | 35.63 | 31.16 | 56.62 |
| D8 | 3,004 | 10,981 | 26.64 | 23.27 | 57.61 |
| D9 | 6,304 | 10,984 | 35.29 | 34.30 | 60.72 |
| D10 | 18,809 | 10,986 | 34.65 | 35.51 | 49.54 |

## 4. Job-size distribution — `A_pstar_percentiles.csv` (+ `A_pstar_cdf.csv` for full CDF points)
*Plot: CDF (semilog-x) and log-log CCDF tail; mark P50/P90/P99/max.*

min=4 · **P50=515** · **P90=10,432** · **P99=56,447** · max=535,085 · mean=4,147 (seconds). P99/P50 ≈ **110×** (full test; ~152× on the 10k ΣC sample).

## 5. Total completion ΣC_j — `B_fifo_vs_srpt.csv`
*Plot: two panels — competitive ratio vs SRPT (oracle=1) and ΣC % reduction vs FIFO. Single machine, sample 10,000.*

Baselines (/SRPT): SRPT 1.000 · SJF 1.0001 · RR 1.998 · **FIFO 6.171**.

| method | spearman (10k) | SPJF /SRPT | PRR /SRPT | PRR reduction vs FIFO |
|---|---|---|---|---|
| M1 | 0.596 | 3.119 | 2.699 | 56.3% |
| M3 | 0.539 | 3.310 | 2.617 | 57.6% |
| M4 | 0.586 | 2.873 | 2.561 | 58.5% |
| M5 | 0.501 | 2.627 | 2.288 | 62.9% |
| M6 | 0.422 | 3.776 | 2.742 | 55.6% |
| M7 | 0.440 | 3.641 | 2.697 | 56.3% |

## 6. Max-stretch — `data_maxstretch.csv`
*Plot: bars of rho_max per policy. Single machine, sample 5,000; S*=648.87; normalize by offline S*.*

Baselines (rho_max/99/med): OPT 1.000/0.985/0.041 · SRPT(true) 1.186/1.090/0.013 · FIFO 2931.7/1287.1/25.73 · LAS/FB 169.6/132.2/3.56.

| method | SPRPT rho_max | rho_99 | rho_med | EDF-P rho_max | rho_99 | rho_med |
|---|---|---|---|---|---|---|
| M1 | 43.64 | 28.15 | 0.39 | 71.57 | 53.89 | 1.52 |
| M3 | 53.75 | 33.04 | 0.47 | 96.75 | 70.35 | 1.64 |
| M4 | 50.70 | 27.24 | 0.46 | 63.44 | 47.62 | 0.55 |
| M5 | 27.45 | 21.34 | 0.29 | 29.93 | 22.55 | 0.69 |
| M6 | 51.68 | 28.67 | 0.42 | 86.88 | 66.22 | 1.58 |
| M7 | 46.39 | 26.84 | 0.41 | 65.57 | 53.15 | 1.12 |

## 7. Makespan — `data_makespan.csv`
*Plot: line of LPPT ratio vs m per method; LPT(oracle)≈1.0. Normalize by McNaughton OPT_pre.*

Baselines by m (LPT / SPT / Random ratio):
| m | OPT_pre | LPT | SPT | Random |
|---|---|---|---|---|
| 5 | 85,028 | 1.0001 | 1.308 | 1.632 |
| 10 | 138,422 | 1.000 | 1.305 | 1.834 |
| 20 | 183,117 | 1.000 | 1.554 | 2.006 |
| 50 | 236,904 | 1.000 | 1.424 | 1.967 |
| 100 | 283,942 | 1.000 | 1.663 | 2.149 |

M5 (LPPT / SPPT) by m: 5→1.276/1.308 · 10→1.528/1.524 · 20→1.777/1.546 · 50→1.812/1.640 · 100→1.895/1.889. (All 6 methods’ LPPT/SPPT are in the CSV.)

## 8. Cold-start exploration (Task 2) — `data_coldstart_exploration.csv`
*Plot: val-vs-test bars by population. Method = quantile-median + val-only segment calibration.*

| split | population | n | Cov25 | Cov50 | RMSLE | Spearman |
|---|---|---|---|---|---|---|
| val_sel | all | 43,942 | 33.74 | 54.27 | 1.309 | 0.782 |
| val_sel | seen | 26,975 | 44.05 | 67.22 | 0.820 | 0.902 |
| val_sel | cold | 16,967 | 17.35 | 33.69 | 1.836 | 0.584 |
| test | all | 109,854 | 30.89 | 49.11 | 1.513 | 0.668 |
| test | seen | 57,102 | 44.45 | 67.01 | 0.925 | 0.874 |
| test | cold | 52,752 | 16.20 | 29.73 | 1.961 | 0.429 |

Cold-start improved **12.74 → 16.20 (+3.46pp)** on test, within the information ceiling (no leakage).

## 9. Calibration / ranking trade-off — `E_calibration_tradeoff.csv`
*Plot: 3 panels (Cov@25 ↑, RMSLE ↓, Spearman ↑) for the three models.*

| model | Cov25 | RMSLE | Spearman |
|---|---|---|---|
| Meta-base (M5, uncalibrated) | 29.11 | 1.447 | 0.720 |
| Meta+Cal (isotonic) | 29.92 | 1.454 | 0.720 |
| Coverage-opt (Task 2) | 30.89 | 1.513 | 0.668 |

---

## Honest caveats (put these on the figures)
1. Competitive ratios normalize by the **clairvoyant** optimum (SRPT on true sizes / offline S* /
   McNaughton OPT_pre); ρ > 1 is expected — which is why predictors still beat FIFO by 56–63%.
2. Single-machine sims use the **parallel fork-join span p*** as a serial size; the heavy tail
   (P99/P50 ≈ 110×) structurally inflates ratios.
3. Cov@25 is a **relative-error** metric, so the smallest jobs (deciles D1–D2) get the lowest
   coverage — a metric artifact, not necessarily larger absolute error.
4. The coverage-optimized predictor raises Cov@25 but lowers Spearman (0.720 → 0.668); schedulers
   should use the **uncalibrated** ranking, not this calibrated predictor.
