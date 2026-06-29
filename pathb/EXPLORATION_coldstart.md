# Task 2 — Guarded Cold-Start Prediction Improvement

**Goal.** Improve prediction accuracy, primary target = the **cold-start** slice
(signature count == 0), whose Cov@25 was ~12.7, **without any leakage** and strictly
**non-clairvoyant**. Test was read **exactly once** (finalize), after the method was chosen on
validation. Path B numbers are untouched; all work is in new files under `pathb/improve/`.

Populations (work-order definition): **All / Seen (train signature count ≥ 1) / Cold-start
(count == 0)**. Test: cold = 52,752 jobs (48%), seen = 57,102.

---

## Protocol (how leakage was prevented)

- **Reused the verified temporal split** (`splits.npz`); re-checked **I1** at every stage:
  `max(r_j train)=5,020,578 < min(r_j val)=5,020,579 < min(r_j test)=5,764,873`, and
  `mean(r_j[idx_te] ≥ t_val) = 1.0` (the original index-scramble bug stays fixed).
- **Validation-internal temporal holdout for honest selection:** val split into
  `val_es` = earliest 60% (LightGBM early stopping **and** calibrator fit) and
  `val_sel` = latest 40% (**held-out** selection metric). Test never used for any decision.
- **All stats train-only (I2/I3):** every prior is a TRAIN target mean, computed **out-of-fold
  (5-fold) on train rows** (no row sees its own label) and **frozen full-train** for val/test
  (train precedes val/test, so it is causal). Label encoders fit on train. Calibrators fit on
  `val_es` only. Target `log1p(p*)` is never a feature (asserted: no feature equals `y` on train).
- **Submit-time only (I3):** features are resource plan/logs/ratios, parallelism (num_tasks,
  inst_num, inst-per-task), temporal (hour/dow/cyclic), gpu_type, train-only causal **shift(1)**
  group/user histories (**I4**, reused from Path B), and the hierarchical priors below. No
  utilization/placement/queueing/realized-start signal.
- **227 search trials**, each scored on `val_sel` **cold-start Cov@25** (guard: reject configs
  whose All Cov@25 < 26 to prevent degenerate cold-boosting). Best config persisted continuously
  (`best_config.json`), full trace in `trials.jsonl`.

**Leakage red-flag rule (G6).** Cold-start Cov@25 jumping > 30% ⇒ stop & investigate. It did
not: best was ~17% on val, 16.2% on test — squarely within the stated information ceiling.

---

## What I tried, and what the validation selection said

Hierarchical OOF train-only priors at 19 granularities (coarse-signature backoff):
`sig8 → user|group|workload → group|workload → workload → user → group → gpu_type →
workload|gpu → single resource buckets (cpu/gpu/mem/inst) → num_tasks bucket → 2-way resource
shapes (gpu×inst, gpu_type×gpu, cpu×mem, workload×inst)`, each EB-shrunk with a log-count feature.
Search over LightGBM hyperparameters, objective {L2, quantile@0.5, Huber}, calibration {none,
isotonic, segment-multiplicative, iso+segmult}, and feature groups {direct only / +prior means /
+prior counts}.

**Ablation (all from `val_sel`, the held-out validation slice — these are selection numbers, not test):**

| Variant | cold Cov@25 | seen Cov@25 | all Cov@25 | all RMSLE |
|---|---|---|---|---|
| L2 + no calibration (≈ baseline family) | 14.68 | 33.48 | 26.22 | 1.298 |
| **quantile(0.5) + no calibration** | **16.53** | 37.69 | 29.52 | 1.316 |
| isotonic only | 14.62 | 34.92 | 27.08 | 1.268 |
| **quantile + segment-mult calibration (SELECTED)** | **17.35** | 44.05 | 33.74 | 1.309 |
| iso + segment-mult | 15.38 | 40.15 | 30.59 | 1.236 |
| best **direct-only** (no priors) | 17.35 | 44.05 | 33.74 | 1.309 |
| best **with all priors** | 16.11 | 44.94 | 33.81 | 1.346 |

**Selected by validation:** `objective=quantile(0.5)`, `calib=segment-multiplicative`,
`groups=direct-only`, `num_leaves=127, min_child=200, lr=0.07, max_depth=12, subsample=0.7`
(best_iter=1427). Selection used `val_sel` cold Cov@25 only; **test was not consulted**.

---

## Final method — validation vs test, by population (test read once)

| Population (n_test) | Cov@25 val / **test** | Cov@50 val / test | RMSLE val / test | Spearman val / test |
|---|---|---|---|---|
| All (109,854) | 33.74 / **30.89** | 54.27 / 49.11 | 1.309 / 1.513 | 0.782 / 0.668 |
| Seen ≥1 (57,102) | 44.05 / **44.45** | 67.22 / 67.01 | 0.820 / 0.925 | 0.902 / 0.874 |
| **Cold-start =0 (52,752)** | 17.35 / **16.20** | 33.69 / 29.73 | 1.836 / 1.961 | 0.584 / 0.429 |

**vs Path B anchor** (M5 Meta base / Meta+Cal): All Cov@25 29.11 / 29.92 → **30.89**;
New(cold) Cov@25 **12.74 → 16.20 = +3.46 pp**.

**Leakage checks (G1/G6):**
- `test_seen − val_seen` Cov@25 = **+0.40 pp** (not test ≫ val) ⇒ **no leakage flag**.
- cold test 16.20 < val_sel 17.35 — the expected, honest selection-optimism direction (we picked
  the max of 227 val trials), **not** inflation.
- cold test 16.20 < 30 ⇒ within the information ceiling, no red flag.

**Invariants I1–I4:** re-checked at feature-build and at finalize — all held every round.

---

## Honest assessment — what helped, what did not

- **What helped (legitimately, leakage-free):**
  1. **Quantile(0.5) / conditional-median objective** is the real lever. Cov@25 is a *relative*
     error metric; the median minimizes relative-coverage loss better than the squared-log mean
     the Path B Meta-stack effectively targets. Cold +1.85 pp on val (14.68 → 16.53) with **no
     calibration at all**.
  2. **Validation-only per-segment multiplicative calibration** adds ~0.8 pp on cold (→17.35) and
     lifts seen/all coverage substantially. It fits one scalar per segment {cold, seen} on
     `val_es` to maximize Cov@25; segment membership is the train-derived signature count, known
     at submit time, so it is non-clairvoyant.
- **What did NOT help (honest negative result):** the **hierarchical OOF priors / coarse-signature
  backoff did not move cold-start** beyond direct submit-time features — the validation-selected
  winner is **direct-only**, and prior-rich configs were competitive (16.1) but never better. The
  cold-start information ceiling is real: an unseen signature whose user/group/workload are also
  thin has little train signal beyond resource shape, which the direct features already carry.
- **The cost (must be stated):** coverage was bought at the expense of **RMSLE** (All 1.447 →
  1.513) and **global ranking** (All Spearman 0.720 → 0.668). The segment scaling re-weights cold
  vs seen, lowering cross-segment rank correlation; within-segment ranking is preserved. Because
  Path B's schedulers consume the **uncalibrated** base ranking, this calibrated predictor is a
  better *coverage* estimator but **not** a drop-in replacement for the scheduling ranking source —
  it should not be fed to the schedulers.

**Bottom line.** Cold-start Cov@25 improved **+3.46 pp (12.74 → 16.20)** honestly and within the
information ceiling, driven by a median objective + val-only segment calibration, **not** by
feature engineering. This is a modest, genuine gain — consistent with the stated reality that
unseen-signature jobs carry limited submit-time signal. No invariant tripped; nothing was tuned on
test; Path B numbers are untouched.

### Reproduce
```
python3 pathb/improve/features_cs.py   # leakage-safe enriched features (OOF priors)
python3 pathb/improve/search_cs.py 7200 300   # val-only iterative search (test untouched)
python3 pathb/improve/finalize_cs.py   # evaluate the val-selected config on test ONCE
```
Artifacts: `pathb/improve/artifacts/{trials.jsonl, best_config.json, final_cs.json}`.
