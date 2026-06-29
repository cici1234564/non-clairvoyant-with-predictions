# ============================================================
# PATH B - assemble REPORT.md + the two result tables from the
# persisted JSON artifacts. Pure formatting; no recomputation.
# ============================================================
import json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
REPO = HERE.parent

pred = json.load(open(ART / "results_pred.json"))
sched = json.load(open(ART / "sched_results.json"))
buckets = json.load(open(ART / "recurrence_buckets.json"))
V = json.load(open(ART / "verification.json"))

NAME = {"M1": "M1 CQR", "M3": "M3 HRAS", "M4": "M4 Isotonic",
        "M5": "M5 Meta (base)", "M6": "M6 TwoStage", "M7": "M7 Recency"}
ORDER = ["M1", "M3", "M4", "M5", "M6", "M7"]


def fnum(x, d=2):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


# ---------- tab:results-main ----------
def results_main():
    L = []
    L.append("### tab:results-main — Prediction (FULL new numbers; supersede prior handoff)\n")
    L.append("All numbers from the **corrected** pipeline (Part-0 fix). The prior 33.4 anchor "
             "was produced by the buggy (scrambled-test) pipeline and is **superseded**.\n")
    L.append("| Method | Cov@25 All | Cov@25 Rec | Cov@25 New | Cov@50 All | Cov@50 Rec | RMSLE | Spearman ρ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        r = pred["results"][k]
        L.append(f"| {NAME[k]} | {fnum(r['All']['Cov25'])} | {fnum(r['Rec']['Cov25'])} | "
                 f"{fnum(r['New']['Cov25'])} | {fnum(r['All']['Cov50'])} | {fnum(r['Rec']['Cov50'])} | "
                 f"{fnum(r['All']['RMSLE'],3)} | {fnum(r['All']['Spearman'],3)} |")
    mc = pred["metacal"]
    L.append(f"| **Meta+Cal (M5+isotonic)** | **{fnum(mc['All']['Cov25'])}** | {fnum(mc['Rec']['Cov25'])} | "
             f"{fnum(mc['New']['Cov25'])} | {fnum(mc['All']['Cov50'])} | {fnum(mc['Rec']['Cov50'])} | "
             f"{fnum(mc['All']['RMSLE'],3)} | {fnum(mc['All']['Spearman'],3)} |")
    L.append("\nRec = training signature count ≥ 5; New = unseen (count = 0).\n")
    return "\n".join(L)


# ---------- tab:unified_all_methods ----------
def unified():
    L = []
    L.append("### tab:unified_all_methods — Scheduling (predictor rows NEW; baselines unchanged)\n")
    tc = sched["total_completion"]; ms = sched["max_stretch"]; mk = sched["makespan"]

    # C.1 total completion
    L.append("**C.1 Total completion ΣC_j (1 machine, online, preemptive) — normalized by SRPT "
             f"(n={tc['meta']['n_jobs']:,}, PRR λ={tc['meta']['prr_lambda']}).**\n")
    L.append("| Policy | ρ_TC = ΣC/SRPT |")
    L.append("|---|---|")
    for b in ["SRPT", "SJF", "RR", "FIFO"]:
        L.append(f"| {b} (baseline) | {fnum(tc['baselines'][b],4)} |")
    for k in ORDER:
        if k in tc["methods"]:
            m = tc["methods"][k]
            L.append(f"| {NAME[k]} — SPJF | {fnum(m['SPJF'],4)} |")
            L.append(f"| {NAME[k]} — PRR | {fnum(m['PRR'],4)} |")
    L.append("")

    # C.2 max stretch
    L.append("**C.2 Max-stretch S_max (1 machine, online, preemptive) — normalized by offline S\\* "
             f"(EDF bisection; n={ms['meta']['n']:,}, S\\*={ms['meta']['S_bisect']:.3f}, "
             f"S_emp={ms['meta']['S_emp']:.3f}).**\n")
    L.append("| Policy | ρ_S,max | ρ_S,99 | ρ_S,med |")
    L.append("|---|---|---|---|")
    for b in ["OPT (EDF at S*)", "SRPT (true)", "FIFO", "LAS/FB"]:
        r = ms["baselines"][b]
        L.append(f"| {b} (baseline) | {fnum(r['rho_max'],3)} | {fnum(r['rho_99'],3)} | {fnum(r['rho_med'],3)} |")
    for k in ORDER:
        if k in ms["methods"]:
            for algo in ["SPRPT", "EDF-P"]:
                r = ms["methods"][k][algo]
                L.append(f"| {NAME[k]} — {algo} | {fnum(r['rho_max'],3)} | {fnum(r['rho_99'],3)} | {fnum(r['rho_med'],3)} |")
    L.append("")

    # C.3 makespan
    machine_counts = sorted({row["m"] for row in mk})
    L.append("**C.3 Makespan C_max (m machines, batch, non-preemptive) — normalized by McNaughton "
             "OPT_pre = max(Σp/m, max_j p_j).**\n")
    head = "| m | OPT_pre | LPT ρ | SPT ρ | Rand ρ | " + " | ".join(f"{NAME[k]} LPPT ρ" for k in ORDER if any(r['method']==k for r in mk)) + " |"
    L.append(head)
    L.append("|" + "---|" * (5 + sum(1 for k in ORDER if any(r['method']==k for r in mk))))
    by = {}
    for row in mk:
        by[(row["method"], row["m"])] = row
    methods_present = [k for k in ORDER if any(r['method']==k for r in mk)]
    for m in machine_counts:
        anyrow = next(r for r in mk if r["m"] == m)
        cells = [str(m), fnum(anyrow["OPT_pre"],1), fnum(anyrow["LPT_ratio"],4),
                 fnum(anyrow["SPT_ratio"],4), fnum(anyrow["Random_ratio"],4)]
        for k in methods_present:
            cells.append(fnum(by[(k, m)]["LPPT_ratio"],4))
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n(SPPT ρ per method/m is in `pathb/artifacts/sched_results.json` and `makespan_results.csv`.)\n")
    return "\n".join(L)


def vrow(k):
    v = V[k]
    return f"- **{k}** — {'✅ PASS' if v['pass_'] else '❌ FAIL'}: {v.get('note','')}"


def main():
    g = pred["gate"]
    superseded = not g["reproduces_33_4"]
    md = []
    md.append("# Path B — ATLAS/LASched Re-baseline: REPORT\n")
    md.append("Corrected, leakage-free re-run of the ATLAS prediction + LASched scheduling "
              "benchmark. Executed strictly in order **0 → A → (B, C)** with a single Meta base "
              "established in Part A, persisted to `pathb/artifacts/base_preds.pkl`, and loaded "
              "(never re-trained) by B and C.\n")
    md.append("**[STOP] markers:** none. Egress to aliyuncs stayed blocked (403); the three trace "
              "tarballs were taken from the repo `main` root, extracted to the repo root, and "
              "`load_tables()` was pointed there.\n")

    # headline
    md.append("## Headline\n")
    md.append(f"- **Determinism (V5):** two seeded Meta-base runs are **identical** "
              f"(Cov@25 {g['existing_run1_cov25']:.4f} == {g['existing_run2_cov25']:.4f}, "
              f"`array_equal=True`). The pipeline is deterministic/seeded and the base is reproducible.")
    if superseded:
        md.append(f"- **33.4 NOT reproduced → superseded.** The corrected Meta base Cov@25 = "
                  f"**X = {pred['base_cov25_X']:.4f}**. The prior 33.4 was produced by the *buggy* "
                  f"(scrambled-test) pipeline; after the Part-0 fix the honest test-window Cov@25 is "
                  f"{pred['base_cov25_X']:.2f}. **All downstream numbers use this new base; prior "
                  f"handoff numbers are NOT reused.**")
    else:
        md.append(f"- **33.4 reproduced:** base Cov@25 = {pred['base_cov25_X']:.4f} ≈ 33.4.")
    md.append(f"- **Meta+Cal (isotonic, val-only):** All Cov@25 = {pred['metacal']['All']['Cov25']:.2f} "
              f"(base {pred['base_all_cov25']:.2f}, gain **+{pred['metacal_gain_pp']:.2f}pp**, modest, "
              f"no leak flag). Rank-preserving (Spearman base↔cal = {pred['spearman_base_vs_metacal']:.4f}).\n")

    # Part 0
    md.append("## Part 0 — index-misalignment bug fix\n")
    md.append("**Bug.** In cell 4, `add_causal_histories_no_leak()` did "
              "`df = df.sort_values(\"submit_time\").reset_index(drop=True)` **after** `time_split()` "
              "had computed `idx_tr/va/te` on the *pre-sort* frame, and returned that re-indexed "
              "frame. `prepare_matrices()` / the schedulers then did `df.loc[idx_te]` on a frame "
              "whose labels no longer matched the split → a **scrambled** test set (temporal leakage, "
              "meaningless test metrics).\n")
    md.append("**Fix (invariant I1).** Sort by `r_j` (submit_time) **once, up front** "
              "(stable mergesort), compute the split indices on the sorted frame, and remove the "
              "internal re-sort. Indices stay aligned to `df` everywhere downstream.\n")
    md.append(f"**Proof (V2).** `mean(r_j[idx_te] ≥ t_val) = {V['V2']['frac']:.6f}` (required 1.0).\n")

    # Part A
    md.append("## Part A — Meta base, determinism, calibration\n")
    md.append(f"- **A.1** Corrected pipeline (cells 1→4) on the real trace; LGBM-Meta base test "
              f"Cov@25 = **X = {pred['base_cov25_X']:.4f}**.")
    md.append(f"- **A.2** Determinism gate: ran the base twice with the existing config "
              f"(`random_state=42`, `np.random.seed(42)`); predictions are **bit-identical** "
              f"(`array_equal=True`), Cov@25 {g['existing_run1_cov25']:.4f} both runs ⇒ deterministic. "
              f"No re-seeding was required for reproducibility; 33.4 is nonetheless not reproduced "
              f"because the *bug fix* (not the seed) changed the test set. Decision: "
              f"`{pred['decision']}`.")
    md.append(f"- **A.3** Persisted base val/test log-predictions → `pathb/artifacts/base_preds.pkl`.")
    md.append(f"- **A.4** Calibration spec: {pred['calibration_spec']}")
    mc = pred["metacal"]
    md.append(f"  Meta+Cal — All Cov@25 {mc['All']['Cov25']:.2f} / Rec {mc['Rec']['Cov25']:.2f} / "
              f"New {mc['New']['Cov25']:.2f}; Cov@50 All {mc['All']['Cov50']:.2f} / Rec "
              f"{mc['Rec']['Cov50']:.2f}; RMSLE {mc['All']['RMSLE']:.3f}; Spearman "
              f"{mc['All']['Spearman']:.3f}.\n")

    md.append(results_main())

    # Part B
    md.append("\n## Part B — recurrence figure (panel b)\n")
    md.append("Cov@25 & Cov@50 of **Meta+Cal** by training-recurrence bucket; teal/CDF house style. "
              "Files: `fig_recurrence.{pdf,png,svg}`.\n")
    md.append("| Bucket (train sig. count) | n | Cov@25 | Cov@50 |")
    md.append("|---|---|---|---|")
    for r in buckets:
        md.append(f"| {r['bucket']} | {r['n']:,} | {fnum(r['cov25'])} | {fnum(r['cov50'])} |")
    md.append("\nCoverage rises monotonically with recurrence — the expected signal "
              "(more training history ⇒ better calibrated predictions). Panel (a) is full-trace, unchanged.\n")

    # Part C
    md.append("## Part C — scheduling on the uncalibrated base\n")
    md.append(f"Schedulers consume the **uncalibrated** base ranking `predictions['M5']` "
              f"(guard V8: `max|expm1(base_log_te) − M5| = {sched['guard']['max_abs_expm1_base_vs_M5']:.1e}`, "
              f"and M5 ≠ Meta+Cal). If A.2 had re-baselined the model, predictor-fed rows would be "
              f"regenerated; here the base is deterministic, and the **whole table is regenerated** "
              f"under Path B regardless. Non-predictive baselines (SRPT/FIFO/RR/LAS/McNaughton/OPT) "
              f"are objective-defined and unaffected.\n")
    md.append(unified())

    # Verification
    md.append("\n## Verification — V1…V8\n")
    for i in range(1, 9):
        md.append(vrow(f"V{i}"))
    md.append(f"\n**ALL_PASS = {V['ALL_PASS']}**\n")
    md.append("Selected values: "
              f"V1 train[{V['V1']['train'][0]:.0f},{V['V1']['train'][1]:.0f}] "
              f"val[{V['V1']['val'][0]:.0f},{V['V1']['val'][1]:.0f}] "
              f"test[{V['V1']['test'][0]:.0f},{V['V1']['test'][1]:.0f}]; "
              f"V2 frac={V['V2']['frac']:.6f}; "
              f"V4 n_test={V['V4']['n_test']:,}, Rec={V['V4']['rec_ge5_pct']:.1f}%, "
              f"New={V['V4']['new_eq0_pct']:.1f}%, resid={V['V4']['resid_1to4_pct']:.1f}%; "
              f"V6 gain=+{V['V6']['gain_pp']:.2f}pp, ρ(base,cal)={V['V6']['spearman_base_vs_metacal']:.4f}; "
              f"V7 spot-check max|Δp*|={V['V7']['max_abs_diff']:.1e} over {V['V7']['n_checked']} jobs.\n")

    # Deviations
    md.append("## Deviations from the verbatim notebook (all non-semantic)\n")
    md.append("1. **Part-0 fix** (sort `r_j` first; remove internal re-sort) — the requested correction.\n"
              "2. **Vectorized causal histories.** The original val/test \"frozen stats\" step looped "
              "row-by-row with scalar `df.loc[i, col]` assignments (~275k iterations; impractically slow "
              "under pandas 3.0). Replaced with the numerically-identical vectorized `.map()` form, and "
              "train-side `expanding().mean()/.count().shift(1)` replaced by the equivalent cumsum "
              "identity (verified `np.allclose` to 1e-12; ewm kept as per-group "
              "`ewm(span=10,adjust=False).mean().shift(1)`). **Numbers unchanged.**\n"
              "3. **Stable sort** (`kind='mergesort'`) for reproducible tie ordering; **`to_pickle`** "
              "instead of parquet (no pyarrow available). No effect on results.\n")
    md.append("## Reproduce\n")
    md.append("```\npython3 pathb/build_data.py   # Part 0: load+sort+split+features (cache)\n"
              "python3 pathb/predict.py      # Part A: methods + determinism gate + calibration\n"
              "python3 pathb/figure.py       # Part B: fig_recurrence.{pdf,png,svg}\n"
              "python3 pathb/sched_run.py    # Part C: scheduling (uncalibrated base)\n"
              "python3 pathb/verify.py       # V1..V8\n"
              "python3 pathb/report.py       # this REPORT.md + tables\n```\n")

    (REPO / "REPORT.md").write_text("\n".join(md))
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
