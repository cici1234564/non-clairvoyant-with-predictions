# Plotting handoff — ATLAS / LASched (Path B)

Paste this whole file to the assistant and upload the CSVs in `pathb/figures/`. All numbers are
from a leakage-free, time-split (70/15/15 by submit time) pipeline; **test was used only for final
reporting**. Units: `p*` / sizes are **seconds**; `cov*` are **percent**; `rho`/`spearman` are
Spearman rank correlation.

## Two recurrence definitions — DO NOT MIX
- **Prediction table** uses Path B buckets: **Rec = training signature count ≥ 5**, **New = count == 0**.
- **Cold-start exploration** uses: **seen = count ≥ 1**, **cold = count == 0**.
The signature count is per test job, computed on TRAIN only (known at submit time).

## Files and what to plot

| CSV | columns | suggested plot |
|---|---|---|
| `data_prediction_methods.csv` | method, cov25_all/rec/new, cov50_all/rec, rmsle_all, spearman_all | grouped bars of Cov@25 by All/Rec/New across 6 methods + Meta+Cal; M5 Meta is the base. |
| `C_cov_by_recurrence.csv` | bucket {New,1-4,5-49,>=50}, n, base/metacal cov25/cov50 | bars: coverage rises monotonically with history; New (48% of test) is the bottleneck. |
| `D_cov_by_size_decile.csv` | decile, p_lo/hi/median, n, base/metacal cov25/cov50 | line of Cov@25 vs size decile (x = median p*); note error is LOWEST at the smallest jobs, NOT the tail. |
| `A_pstar_percentiles.csv` + `A_pstar_cdf.csv` | percentiles; sorted p* + cdf | CDF (semilog-x) and log-log CCDF tail; mark P50=515s, P90=10,432s, P99=56,447s, max=535,085s (P99/P50≈110x). |
| `B_fifo_vs_srpt.csv` | method, spearman, spjf/prr ratio vs SRPT, %reduction vs FIFO | two panels: competitive ratio vs SRPT (oracle=1) and ΣC %reduction vs FIFO (56–63%). |
| `data_maxstretch.csv` | policy, rho_max/99/med | bars of rho_max per policy; baselines OPT=1, SRPT(true)=1.19, FIFO=2932, LAS=170; best predictor M5 SPRPT=27.5. |
| `data_makespan.csv` | method, m, OPT_pre, LPT/SPT/Random/LPPT/SPPT ratios | line of LPPT ratio vs m∈{5,10,20,50,100} per method; LPT(oracle)≈1.0. |
| `data_coldstart_exploration.csv` | split{val_sel,test}, population{all,seen,cold}, n, cov25/50, rmsle, spearman | val-vs-test bars by population; cold-start 12.74→16.20 on test (+3.46pp). |
| `E_calibration_tradeoff.csv` | model, cov25, rmsle, spearman | 3 panels (Cov@25↑, RMSLE↓, Spearman↑) for Meta-base / Meta+Cal / Coverage-opt: coverage up but RMSLE/rank degrade. |

## Caveats to keep honest (state these on the figures)
1. Competitive ratios normalize by the **clairvoyant** optimum (SRPT on true sizes / offline S* /
   McNaughton OPT_pre), so ρ>1 is expected; that's why predictors still beat FIFO by 56–63%.
2. The single-machine sims treat the **parallel fork-join span p*** as a serial size; the heavy
   tail (P99/P50≈110x) structurally inflates ratios.
3. Cov@25 is a **relative-error** metric — tiny jobs are penalized by small absolute errors, which
   is why decile D1–D2 (smallest) have the lowest Cov@25.
4. The coverage-optimized predictor (Task 2) raises Cov@25 but lowers Spearman (0.720→0.668);
   schedulers should use the **uncalibrated** ranking, not this calibrated predictor.

## Anchor numbers
- Prediction base (M5 Meta) test: Cov@25 All 29.11 / Rec 46.30 / New 12.39; RMSLE 1.447; ρ 0.720.
- Cold-start improved 12.74 → 16.20 (+3.46pp), within the information ceiling (no leakage).
