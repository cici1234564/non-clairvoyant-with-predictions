# Full Scheduling Regeneration — random multi-seed, all objectives × populations

Mode=full, seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_per_draw=10000. Algorithms unchanged; only sampler (random multi-seed) + population filter changed. Train-only signature counts; submit-time safe.

## Part 0 — Sampling audit

| objective | sampler (original) | random? | seed effective? | action |
|---|---|---|---|---|
| Total completion ΣC_j | `test_df.iloc[:N]` (time prefix) | NO | no-op | **fixed → random multi-seed** |
| Max-stretch S_max | `rng.choice` in `_prepare_jobs` | yes | yes, but wrapper used fixed seed=42 | **redone with multi-seed** |
| Makespan C_max | `rng.choice` in `_sample_jobs` | yes | yes, but `evaluate_makespan` used fixed seed=42 | **redone with multi-seed** |

## Part 1 — Scheduling leakage audit (S1–S6)

- **S1 PASS** — schedulers consume `predictions[*]` produced by Part-A models trained on `idx_tr` (early-stop `idx_va`) and inferred on `idx_te`; no test backflow. (provenance: `predictions.pkl`).
- **S2 PASS** — recurrence/seen population uses TRAIN-only signature count: verified `sig_count_te == count of each test signature among idx_tr rows` (excludes test-window occurrences).
- **S3 PASS** — predictor policies rank by PREDICTED size only: SPJF/PRR use `predicted_size`; SPRPT/EDF-P use `q=clip(pred)`; LPPT/SPPT use `pred_size`. True sizes appear ONLY in the normalizers SRPT / offline S* / McNaughton OPT_pre (intended clairvoyant denominators).
- **S4 PASS** — λ is fixed a priori in code (default 0.7); the λ-sweep below is reported with error bars and the Part-3 recommendation is principled (consistency/robustness), not test-selected.
- **S5 PASS** — each seed draws an independent `RandomState(base+seed).choice(...)` subsample (no iloc prefix, no cross-run leakage).
- **S6 PASS** — per (population, seed) the SRPT/FIFO/RR (ΣC), S*/SRPT/FIFO/LAS (stretch), OPT_pre/LPT/SPT/Random (makespan) normalizers are computed on the SAME sampled job set as the policies.

## Part 2 — Population: All  (n_qualifying=109,854, draw n=10,000)

### (2a) Total completion ρ_TC (vs SRPT=1). Refs: RR=1.971±0.001, SJF=1.001±0.001, FIFO ratio≈5.80, SPJF(M5)=2.042±0.057.

| λ | ρ_TC (PRR-M5) | imp vs FIFO % | ρ_TC ≤ RR? |
|---|---|---|---|
| 0.1 | 2.021±0.010 | 65.1±0.9 | no |
| 0.2 | 2.073±0.015 | 64.2±0.9 | no |
| 0.3 | 2.124±0.021 | 63.3±0.8 | no |
| 0.4 | 2.158±0.030 | 62.8±0.7 | no |
| 0.5 | 2.156±0.039 | 62.8±0.5 | no |
| 0.6 | 2.125±0.045 | 63.3±0.5 | no |
| 0.7 | 2.075±0.047 | 64.2±0.4 | no |
| 0.8 | 2.023±0.048 | 65.1±0.4 | no |
| 0.9 | 1.984±0.049 | 65.8±0.4 | no |

### (2b) Max-stretch ρ_S (vs offline S*). [ρ_max | ρ_99 | ρ_med], mean±std

| policy | ρ_S,max | ρ_S,99 | ρ_S,med |
|---|---|---|---|
| OPT (EDF@S*) | 1.00±0.00 | 0.99±0.00 | 0.511±0.014 |
| SRPT (true) | 1.13±0.00 | 1.08±0.01 | 0.387±0.023 |
| FIFO | 3702.61±773.01 | 1252.62±55.93 | 23.474±1.508 |
| LAS/FB | 508.28±50.28 | 320.15±16.43 | 7.181±0.314 |
| M1 SPRPT | 42.55±0.47 | 29.41±0.76 | 0.670±0.036 |
| M1 EDFP | 48.01±0.97 | 36.57±0.92 | 1.099±0.016 |
| M3 SPRPT | 51.77±1.19 | 35.48±1.01 | 0.644±0.025 |
| M3 EDFP | 59.28±1.93 | 47.75±0.97 | 0.936±0.026 |
| M4 SPRPT | 50.07±1.66 | 24.80±0.81 | 1.036±0.050 |
| M4 EDFP | 53.54±2.10 | 33.20±1.12 | 1.198±0.040 |
| M5 SPRPT | 25.41±0.70 | 19.91±0.54 | 0.810±0.038 |
| M5 EDFP | 26.12±0.64 | 20.70±0.51 | 0.934±0.029 |
| M6 SPRPT | 49.97±1.01 | 31.98±0.67 | 0.605±0.036 |
| M6 EDFP | 55.70±1.54 | 45.32±1.05 | 0.902±0.022 |
| M7 SPRPT | 45.40±0.71 | 25.57±0.48 | 0.671±0.041 |
| M7 EDFP | 51.06±1.29 | 35.07±0.67 | 0.890±0.034 |

(mean S* = 1370.9)

### (2c) Makespan ρ (vs McNaughton OPT_pre), per m. mean±std

| m | LPT | SPT | Random | M1 LPPT | M3 LPPT | M4 LPPT | M5 LPPT | M6 LPPT | M7 LPPT |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1.000±0.000 | 1.020±0.008 | 1.081±0.011 | 1.059±0.024 | 1.061±0.024 | 1.070±0.023 | 1.055±0.020 | 1.070±0.038 | 1.049±0.019 |
| 10 | 1.000±0.000 | 1.047±0.018 | 1.157±0.015 | 1.107±0.027 | 1.131±0.031 | 1.146±0.052 | 1.117±0.037 | 1.130±0.049 | 1.109±0.024 |
| 20 | 1.000±0.000 | 1.110±0.038 | 1.299±0.049 | 1.233±0.061 | 1.199±0.054 | 1.226±0.078 | 1.243±0.064 | 1.232±0.070 | 1.258±0.061 |
| 50 | 1.000±0.000 | 1.307±0.104 | 1.598±0.049 | 1.478±0.101 | 1.500±0.104 | 1.515±0.079 | 1.546±0.098 | 1.442±0.059 | 1.532±0.153 |
| 100 | 1.000±0.000 | 1.591±0.104 | 1.998±0.124 | 1.817±0.198 | 1.897±0.157 | 1.883±0.336 | 1.928±0.200 | 1.802±0.187 | 1.982±0.373 |

## Part 2 — Population: Seen>=1  (n_qualifying=57,102, draw n=10,000)

### (2a) Total completion ρ_TC (vs SRPT=1). Refs: RR=1.974±0.001, SJF=1.000±0.000, FIFO ratio≈5.17, SPJF(M5)=1.237±0.022.

| λ | ρ_TC (PRR-M5) | imp vs FIFO % | ρ_TC ≤ RR? |
|---|---|---|---|
| 0.1 | 1.954±0.013 | 62.2±0.2 | yes |
| 0.2 | 1.876±0.022 | 63.7±0.3 | yes |
| 0.3 | 1.771±0.024 | 65.8±0.3 | yes |
| 0.4 | 1.665±0.024 | 67.8±0.3 | yes |
| 0.5 | 1.566±0.023 | 69.7±0.3 | yes |
| 0.6 | 1.478±0.022 | 71.4±0.3 | yes |
| 0.7 | 1.398±0.021 | 73.0±0.3 | yes |
| 0.8 | 1.326±0.020 | 74.4±0.3 | yes |
| 0.9 | 1.267±0.020 | 75.5±0.3 | yes |

### (2b) Max-stretch ρ_S (vs offline S*). [ρ_max | ρ_99 | ρ_med], mean±std

| policy | ρ_S,max | ρ_S,99 | ρ_S,med |
|---|---|---|---|
| OPT (EDF@S*) | 1.00±0.00 | 0.99±0.00 | 0.591±0.013 |
| SRPT (true) | 1.20±0.01 | 1.15±0.01 | 0.510±0.016 |
| FIFO | 3810.76±787.60 | 1082.58±50.08 | 20.505±0.734 |
| LAS/FB | 680.68±57.02 | 305.83±14.81 | 7.646±0.426 |
| M1 SPRPT | 15.82±0.47 | 11.38±0.31 | 0.674±0.024 |
| M1 EDFP | 14.92±0.45 | 10.84±0.31 | 0.784±0.028 |
| M3 SPRPT | 24.46±0.93 | 12.54±0.42 | 0.678±0.020 |
| M3 EDFP | 23.15±1.20 | 12.07±0.48 | 0.806±0.030 |
| M4 SPRPT | 32.32±1.26 | 17.81±0.63 | 0.823±0.018 |
| M4 EDFP | 20.69±1.08 | 12.28±0.21 | 1.025±0.022 |
| M5 SPRPT | 16.78±0.57 | 12.44±0.34 | 0.668±0.020 |
| M5 EDFP | 15.02±0.52 | 10.97±0.37 | 0.806±0.023 |
| M6 SPRPT | 17.50±0.85 | 11.73±0.28 | 0.653±0.020 |
| M6 EDFP | 17.75±1.16 | 10.57±0.39 | 0.731±0.026 |
| M7 SPRPT | 16.38±0.64 | 11.47±0.28 | 0.640±0.017 |
| M7 EDFP | 16.08±0.55 | 9.66±0.35 | 0.711±0.027 |

(mean S* = 1458.4)

### (2c) Makespan ρ (vs McNaughton OPT_pre), per m. mean±std

| m | LPT | SPT | Random | M1 LPPT | M3 LPPT | M4 LPPT | M5 LPPT | M6 LPPT | M7 LPPT |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1.000±0.000 | 1.018±0.006 | 1.083±0.009 | 1.042±0.022 | 1.041±0.023 | 1.044±0.021 | 1.034±0.012 | 1.039±0.016 | 1.036±0.007 |
| 10 | 1.000±0.000 | 1.045±0.013 | 1.157±0.018 | 1.083±0.043 | 1.095±0.033 | 1.108±0.050 | 1.057±0.019 | 1.073±0.020 | 1.073±0.018 |
| 20 | 1.000±0.000 | 1.106±0.029 | 1.249±0.021 | 1.134±0.028 | 1.155±0.038 | 1.185±0.056 | 1.143±0.033 | 1.164±0.029 | 1.153±0.042 |
| 50 | 1.000±0.000 | 1.303±0.074 | 1.554±0.062 | 1.332±0.087 | 1.407±0.167 | 1.374±0.115 | 1.401±0.110 | 1.355±0.085 | 1.324±0.084 |
| 100 | 1.000±0.000 | 1.614±0.080 | 1.994±0.075 | 1.582±0.138 | 1.777±0.101 | 1.732±0.153 | 1.655±0.117 | 1.636±0.147 | 1.704±0.134 |

## Part 2 — Population: Seen>=5  (n_qualifying=52,184, draw n=10,000)

### (2a) Total completion ρ_TC (vs SRPT=1). Refs: RR=1.974±0.001, SJF=1.000±0.000, FIFO ratio≈5.22, SPJF(M5)=1.226±0.018.

| λ | ρ_TC (PRR-M5) | imp vs FIFO % | ρ_TC ≤ RR? |
|---|---|---|---|
| 0.1 | 1.949±0.014 | 62.7±0.4 | yes |
| 0.2 | 1.869±0.020 | 64.2±0.4 | yes |
| 0.3 | 1.762±0.023 | 66.3±0.3 | yes |
| 0.4 | 1.654±0.023 | 68.3±0.3 | yes |
| 0.5 | 1.555±0.022 | 70.2±0.3 | yes |
| 0.6 | 1.465±0.021 | 71.9±0.3 | yes |
| 0.7 | 1.384±0.020 | 73.5±0.3 | yes |
| 0.8 | 1.313±0.019 | 74.8±0.3 | yes |
| 0.9 | 1.253±0.018 | 76.0±0.3 | yes |

### (2b) Max-stretch ρ_S (vs offline S*). [ρ_max | ρ_99 | ρ_med], mean±std

| policy | ρ_S,max | ρ_S,99 | ρ_S,med |
|---|---|---|---|
| OPT (EDF@S*) | 1.00±0.00 | 0.99±0.00 | 0.585±0.011 |
| SRPT (true) | 1.20±0.01 | 1.15±0.01 | 0.512±0.014 |
| FIFO | 4211.89±787.82 | 1093.46±34.61 | 20.396±0.800 |
| LAS/FB | 713.27±55.74 | 308.15±14.75 | 7.223±0.237 |
| M1 SPRPT | 15.58±0.56 | 11.46±0.33 | 0.668±0.013 |
| M1 EDFP | 14.67±0.45 | 10.71±0.28 | 0.782±0.014 |
| M3 SPRPT | 22.33±0.60 | 12.87±0.25 | 0.647±0.013 |
| M3 EDFP | 21.18±0.68 | 11.72±0.37 | 0.783±0.018 |
| M4 SPRPT | 31.35±1.82 | 18.25±0.41 | 0.820±0.014 |
| M4 EDFP | 21.54±0.97 | 12.29±0.22 | 1.023±0.027 |
| M5 SPRPT | 17.41±0.58 | 12.58±0.24 | 0.665±0.015 |
| M5 EDFP | 14.57±0.49 | 10.99±0.22 | 0.802±0.018 |
| M6 SPRPT | 18.34±0.40 | 11.80±0.21 | 0.639±0.014 |
| M6 EDFP | 19.12±0.54 | 10.19±0.38 | 0.734±0.017 |
| M7 SPRPT | 17.03±0.45 | 11.54±0.30 | 0.632±0.016 |
| M7 EDFP | 16.51±0.38 | 9.35±0.33 | 0.719±0.018 |

(mean S* = 1460.1)

### (2c) Makespan ρ (vs McNaughton OPT_pre), per m. mean±std

| m | LPT | SPT | Random | M1 LPPT | M3 LPPT | M4 LPPT | M5 LPPT | M6 LPPT | M7 LPPT |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1.000±0.000 | 1.015±0.003 | 1.075±0.009 | 1.034±0.015 | 1.033±0.013 | 1.033±0.011 | 1.039±0.023 | 1.032±0.020 | 1.035±0.012 |
| 10 | 1.000±0.000 | 1.037±0.008 | 1.141±0.013 | 1.062±0.022 | 1.080±0.029 | 1.087±0.025 | 1.071±0.028 | 1.066±0.026 | 1.074±0.031 |
| 20 | 1.000±0.000 | 1.085±0.019 | 1.250±0.028 | 1.123±0.033 | 1.140±0.024 | 1.162±0.050 | 1.136±0.049 | 1.125±0.037 | 1.131±0.030 |
| 50 | 1.000±0.000 | 1.246±0.055 | 1.524±0.056 | 1.269±0.085 | 1.335±0.060 | 1.319±0.080 | 1.293±0.075 | 1.308±0.085 | 1.299±0.096 |
| 100 | 1.000±0.000 | 1.527±0.116 | 1.891±0.105 | 1.491±0.130 | 1.654±0.189 | 1.681±0.166 | 1.513±0.140 | 1.564±0.119 | 1.566±0.172 |

## Part 3 — Diagnostic summary & principled λ

- **Corrected sampling**: ΣC_j was time-prefix sampled (seed was a no-op); now random multi-seed. S_max and C_max were random but single-seed (fixed seed=42 in the wrappers); now multi-seed. Every number above is mean±std over independent random draws.

**Headline.** Learning-augmentation pays off on jobs the model has seen before. On **Seen≥1 / Seen≥5**, PRR-M5 beats Round-Robin at *every* λ and prediction-trust helps monotonically; on **All** (≈48% never-seen signatures) it never beats RR — the cold-start tail dominates. The right deployment is a recurrence-gated λ, not one global λ.

Robustness budget: RR is 2-competitive; we cap PRR's worst-case at 2/(1−λ) ≤ 4 (≈2× RR), i.e. λ ≤ 0.5. The **balanced** λ is the largest λ that beats RR *and* stays in budget; the **aggressive** λ is the best observed average (ignores worst-case).

| population | RR ρ | λ-shape | beats RR | balanced λ (ρ, gain, worst-case) | aggressive λ (ρ, gain, worst-case) |
|---|---|---|---|---|---|
| All | 1.971±0.001 | best at largest λ (trust helps; non-monotone) | **never** | **λ→0.1** (RR fallback, 2.2×) | λ=0.9 (ρ=1.984, -1%, 20.0×) |
| Seen>=1 | 1.974±0.001 | decreasing in λ — more prediction-trust is better | all 9 λ | **λ=0.5** (ρ=1.566, +21%, 4.0×) | λ=0.9 (ρ=1.267, +36%, 20.0×) |
| Seen>=5 | 1.974±0.001 | decreasing in λ — more prediction-trust is better | all 9 λ | **λ=0.5** (ρ=1.555, +21%, 4.0×) | λ=0.9 (ρ=1.253, +36%, 20.0×) |

**Consistency–robustness frontier (Seen>=1).** Each step of λ buys average-case ΣC at a monotonically worsening worst-case bound:
| λ | ρ_TC | gain vs RR | worst-case 2/(1−λ) | in budget (≤4)? |
|---|---|---|---|---|
| 0.1 | 1.954 | +1.0% | 2.22× | yes |
| 0.2 | 1.876 | +4.9% | 2.50× | yes |
| 0.3 | 1.771 | +10.3% | 2.86× | yes |
| 0.4 | 1.665 | +15.6% | 3.33× | yes |
| 0.5 | 1.566 | +20.6% | 4.00× | yes |
| 0.6 | 1.478 | +25.1% | 5.00× | no |
| 0.7 | 1.398 | +29.2% | 6.67× | no |
| 0.8 | 1.326 | +32.8% | 10.00× | no |
| 0.9 | 1.267 | +35.8% | 20.00× | no |

**Principled recommendation.** The worst-case multiplier 2/(1−λ) explodes past the budget (λ=0.7→6.7×, 0.8→10×, 0.9→20×) while the marginal average-case gain flattens (~+3–4 pp per 0.1 step beyond λ=0.6). Pushing λ to its best-looking value (0.9) buys a few extra points of average ΣC for a 5× heavier tail — not a defensible trade. The budgeted λ captures most of the achievable gain at a worst-case comparable to RR's own.
- **Operational λ (recurrence-gated)**: on the *full* stream PRR-M5 never beats RR → λ→0.1 (behave like the 2-competitive RR; worst-case 2.2×). On *recurrent* jobs (Seen≥1, where predictions are reliable) → **λ=0.5**: +21% ΣC vs RR at a worst-case 4.0× (only if you accept a 20× tail does the aggressive λ=0.9 [+36%] pay). A recurrence-gated λ dominates any single global λ.
- **Other objectives**: max-stretch and makespan predictor policies (SPRPT/EDF-P; LPPT/SPPT) are reported vs the offline S*, FIFO/LAS, and McNaughton OPT_pre / LPT-SPT-Random with across-seed error bars; the Seen-population tables are the fair test of prediction value there too.
- **Ordering-change flag**: prefix→random sampling shifts ΣC values materially; any method ordering taken from the old deterministic-prefix ΣC table must be re-read from the random multi-seed tables above.
