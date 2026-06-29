# ============================================================
# PATH B - PART C : scheduling on the UNCALIBRATED base ranking.
# Uses the verbatim cell-6/8/10 scheduling code (sched_*_lib.py),
# the corrected df/idx_te, and the predictions dict from Part A.
#
# V8 guard: schedulers consume predictions['M5'] (uncalibrated base),
# NOT Meta+Cal.  We assert expm1(base_log_te).clip(0) == predictions['M5'].
# Normalizations are unchanged: total-completion / SRPT, makespan /
# McNaughton OPT_pre, max-stretch / offline S* (EDF bisection).
# ============================================================
import sys, json, pickle, pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CACHE = HERE / "cache"
ART = HERE / "artifacts"

import sched_tc_lib as tc      # total completion (cell 6)
import sched_ms_lib as ms      # max stretch     (cell 8)
import sched_mk_lib as mk      # makespan        (cell 10)


def load():
    df = pd.read_pickle(CACHE / "feat_df.pkl")
    sp = np.load(CACHE / "splits.npz")
    idx_te = pd.Index(sp["idx_te"])
    with open(ART / "base_preds.pkl", "rb") as f:
        base = pickle.load(f)
    with open(ART / "predictions.pkl", "rb") as f:
        P = pickle.load(f)
    return df, idx_te, base, P


def guard(base, P):
    m5 = np.asarray(P["predictions"]["M5"], dtype=float)
    recon = np.expm1(base["base_log_te"]).clip(min=0)
    dmax = float(np.max(np.abs(recon - m5)))
    metacal = np.asarray(P["metacal_raw"], dtype=float)
    diff_from_cal = float(np.max(np.abs(m5 - metacal)))
    g = dict(max_abs_expm1_base_vs_M5=dmax,
             guard_pass=bool(dmax <= 1e-6),
             m5_differs_from_metacal=bool(diff_from_cal > 1e-9),
             max_abs_M5_vs_metacal=diff_from_cal)
    print(f"[V8 guard] max|expm1(base_log_te)-M5| = {dmax:.2e}  -> "
          f"{'PASS' if g['guard_pass'] else 'FAIL'}")
    print(f"[V8 guard] M5 (uncalibrated) differs from Meta+Cal: {g['m5_differs_from_metacal']} "
          f"(max abs diff {diff_from_cal:.3f})")
    assert g["guard_pass"], "scheduler base ranking != uncalibrated Meta base!"
    return g


def main():
    df, idx_te, base, P = load()
    predictions = P["predictions"]      # all UNCALIBRATED raw predictions
    g = guard(base, P)

    print("\n==== TOTAL COMPLETION (1|r,pmtn|sum C_j), sample=10000, PRR lambda=0.7 ====")
    tc_res = tc.evaluate_total_completion_all_methods(
        df=df, idx_te=idx_te, predictions=predictions,
        sample_size=10000, prr_lambda=0.7, seed=42, verbose=True)

    print("\n==== MAX-STRETCH (1|r,pmtn|S_max), sample=5000 ====")
    ms_res = ms.run_max_stretch_all_methods(
        df=df, idx_te=idx_te, predictions=predictions,
        sample_size=5000, clip_ratio=(0.1, 10.0), verbose=True)

    print("\n==== MAKESPAN (P||C_max), m in {5,10,20,50,100} ====")
    mk_df = mk.evaluate_makespan(
        df=df, idx_te=idx_te, predictions=predictions,
        machine_counts=[5, 10, 20, 50, 100], n_random_trials=10, seed=42, verbose=True)

    # ---- distill to JSON for the unified table ----
    tc_out = {"baselines": {k: tc_res["baselines"][k]["ratio_vs_opt"]
                            for k in ["SRPT", "SJF", "RR", "FIFO"]},
              "methods": {m: {"SPJF": tc_res["methods"][m]["SPJF"]["ratio_vs_opt"],
                              "PRR":  tc_res["methods"][m]["PRR"]["ratio_vs_opt"],
                              "spearman": tc_res["methods"][m]["prediction_quality"]}
                          for m in tc_res["methods"]},
              "meta": tc_res["meta"]}

    ms_out = {"meta": ms_res["_meta"],
              "baselines": {k: {"rho_max": ms_res["baselines"][k]["max_over_OPT"],
                                "rho_99": ms_res["baselines"][k]["p99_over_OPT"],
                                "rho_med": ms_res["baselines"][k]["med_over_OPT"]}
                            for k in ms_res["baselines"]},
              "methods": {m: {"SPRPT": {"rho_max": ms_res["methods"][m]["SPRPT"]["max_over_OPT"],
                                        "rho_99": ms_res["methods"][m]["SPRPT"]["p99_over_OPT"],
                                        "rho_med": ms_res["methods"][m]["SPRPT"]["med_over_OPT"]},
                              "EDF-P": {"rho_max": ms_res["methods"][m]["EDF-P"]["max_over_OPT"],
                                        "rho_99": ms_res["methods"][m]["EDF-P"]["p99_over_OPT"],
                                        "rho_med": ms_res["methods"][m]["EDF-P"]["med_over_OPT"]}}
                          for m in ms_res["methods"]}}

    mk_rows = mk_df.to_dict("records")

    with open(ART / "sched_results.json", "w") as f:
        json.dump(dict(guard=g, total_completion=tc_out, max_stretch=ms_out,
                       makespan=mk_rows), f, indent=2)
    print("\n[done] wrote artifacts/sched_results.json")


if __name__ == "__main__":
    main()
