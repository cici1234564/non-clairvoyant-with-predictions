# Task 1 — Scheduling Benchmark Design Audit (read-only)

**Scope.** Decide whether the large competitive ratios (ρ_TC = 2.288, ρ_S,max = 27.45,
max-stretch outliers 169.6 / 2931, …) come from honest causes or from benchmark **design
choices**. Nothing was changed, retuned, or "fixed". All code references are to the verbatim
scheduling cells as committed in `pathb/sched_tc_lib.py` (cell 6), `pathb/sched_ms_lib.py`
(cell 8), `pathb/sched_mk_lib.py` (cell 10). Numeric evidence reproduced read-only via
`pathb/improve/audit_inspect.py` (writes nothing).

---

## Point 1 — Normalization baseline: SRPT on **true** sizes

**What the code does.**
- Baselines are built with the **true** size as the key: `baseline_df['y_pred'] = baseline_df['p_star']` (`sched_tc_lib.py:267`), and `simulate_srpt` pushes `J[i].true_size` onto the ready heap (`sched_tc_lib.py:48`). So the SRPT denominator runs on **true remaining processing time**.
- Predictor policies use predictions in the numerator: `sched_df['y_pred'] = y_pred` (`:334`); `simulate_sjf_or_spjf(..., use_predictions=True)` keys on `J[i].predicted_size` (`:106`); PRR keys on `active[idx][0].predicted_size` (`:181`).

**Evidence.** `ratio_vs_opt` is normalized by SRPT = 1.0000; baselines `{SRPT:1.0, SJF:1.0001, RR:1.998, FIFO:6.171}`.

**Verdict — design choice (standard, justify in paper).** SRPT is the *offline clairvoyant optimum* for `1|r_j,pmtn|ΣC_j`, so ρ_TC is a textbook competitive ratio vs the optimum, **not** a bug. The number to state honestly: ρ_TC = 2.288 means the prediction-fed online policy is 2.29× the **clairvoyant optimum** — it is not "2.29× the best achievable online." That framing belongs in the paper.

---

## Point 2 — Job "size" semantics: parallel fork-join span used as a single-machine serial size

**What the code does.** Each job's single-machine processing time is exactly `p_star`
(`true_size=float(row.p_star)`, `sched_tc_lib.py:29`; makespan `true_sizes = test_df['p_star']`,
`sched_mk_lib.py:120`). `p_star = max_t e_t − min_t s_t` is the **fork-join envelope across all
tasks/instances** — an inherently parallel quantity.

**Evidence (tail of the sizes the scheduler actually uses).**
- Total-completion sample (first 10,000 by r_j), `p_star` seconds: **P50 = 434, P90 = 10,821, P99 = 65,898, max = 371,209** → **P99/P50 ≈ 152×**.
- Max-stretch sample (5,000): P50 = 540, P90 = 10,717, P99 = 56,891, max = 185,065.

**Verdict — design choice (justify in paper).** Using a parallel span as a serial size is a
modeling decision, not a coding bug. But the **extreme heavy tail (P99 ≈ 150× the median)** is
the amplification mechanism: SRPT/SPJF with release dates and ΣC_j are dominated by correct
ordering of the few giant jobs, so any ranking error on the tail inflates the ratio. This must be
acknowledged when interpreting ρ_TC and ρ_S,max.

---

## Point 3 — Log vs linear domain at every scheduler entry

**What the code does.** The `predictions` dict is already in **linear seconds** (each method
returns `np.expm1(...).clip(min=0)`). Schedulers consume it directly:
- TC: `predicted_size = float(row.y_pred)` (`sched_tc_lib.py:30`), `y_pred` = linear predictions (`:325,334`).
- Max-stretch: `q = np.maximum(y_pred, 1.0)` then `q = np.clip(q, 0.1·p, 10·p)` (`sched_ms_lib.py:38,42`); deadlines `d = r + S·p` (`:107,129`), EDF-P `d = r + factor·p_pred` (`:242`).
- Makespan: `pred_size = float(y_pred[i])` (`sched_mk_lib.py:185`).

**Evidence (worked numeric example).**
- `predictions['M5']` range = **11.8 … 103,340 s** (linear); `p_star` range = 3 … 626,384 s — same order of magnitude / same unit.
- One max-stretch job: `r_j = 0.0`, `p_j* = 24.0 s`, `q_j = 113.8 s` (clipped expm1 prediction), `S* = 648.867`, `d_j = r_j + S*·p_j* = 15,572.8 s`. If a log-domain size had leaked in, `p_j*` would read ≈ `log1p(24) = 3.22`; it reads **24.0 s**, so the entry is **linear**.

**Verdict — no bug (honest).** Every scheduler entry point is linear; V8's `expm1` is applied
everywhere. The blown-up **max-stretch outliers (FIFO ρ_max = 2931, LAS/FB ρ_max = 169.6) are
NON-predictive baselines** — they never touch a prediction, so their magnitudes are intrinsic
policy behavior (FIFO/LAS are catastrophic for a worst-case max-stretch objective under heavy
tails), not a log/linear bug and not prediction error. The best predictor policy (M5 SPRPT
ρ_max = 27.45) is ~100× better than FIFO, exactly what correct linear sizes should yield.

---

## Point 4 — Sanity vs FIFO (does the ranking actually help?)

**Evidence (ΣC_j reduction vs FIFO on the same workload; FIFO = 6.171× SRPT).**

| Method | SPJF ρ/SRPT | ΣC reduction vs FIFO | PRR ρ/SRPT | ΣC reduction vs FIFO | Spearman |
|---|---|---|---|---|---|
| M1 CQR | 3.119 | 49.5% | 2.699 | 56.3% | 0.596 |
| M3 HRAS | 3.310 | 46.4% | 2.617 | 57.6% | 0.539 |
| M4 Iso | 2.873 | 53.4% | 2.561 | 58.5% | 0.586 |
| **M5 Meta** | **2.627** | **57.4%** | **2.288** | **62.9%** | 0.501 |
| M6 TwoStage | 3.776 | 38.8% | 2.742 | 55.6% | 0.422 |
| M7 Recency | 3.641 | 41.0% | 2.697 | 56.3% | 0.440 |

(Non-predictive RR = 1.998× SRPT.)

**Verdict — honest + reassuring.** Every predictor policy reduces total completion time vs the
natural online baseline FIFO by **39–63%** (best: M5 PRR, 62.9%). So the predicted ranking is
genuinely useful; the large ρ vs SRPT is the **oracle-normalization choice of Point 1**, not a
sign the corrected predictor is broken.

---

## Summary of verdicts

| Point | Verdict |
|---|---|
| 1. SRPT on true sizes | **Design choice (standard).** ρ_TC = competitive ratio vs clairvoyant optimum; correct, state framing in paper. |
| 2. Parallel span as serial size | **Design choice (justify).** Heavy tail P99/P50 ≈ 152× amplifies ratios; not a bug. |
| 3. Log vs linear | **No bug (honest).** All entries linear (worked example confirms); max-stretch FIFO/LAS outliers are non-predictive baselines. |
| 4. Vs FIFO | **Honest result.** Predictors cut ΣC by 39–63% vs FIFO — ranking helps; large ρ is the Point-1 normalization. |

**Bottom line.** The poor-looking competitive ratios are explained by two deliberate
normalization/modeling choices (clairvoyant-optimum denominator; heavy-tailed parallel span used
as serial size), **not** by a log/linear unit bug or an index/misranking error. No invariant
violation found. No fix is required by this audit — only that the two design choices be justified
when the numbers are reported. Nothing was changed.
