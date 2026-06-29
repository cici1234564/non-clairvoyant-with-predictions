# Task 1 (read-only) — numeric evidence for the scheduling design audit.
# Reads cache + predictions + sched_results.json. Writes NOTHING to committed artifacts.
import sys, json, pickle, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent           # pathb/
sys.path.insert(0, str(ROOT))
import sched_ms_lib as ms

df = pd.read_pickle(ROOT / "cache/feat_df.pkl")
sp = np.load(ROOT / "cache/splits.npz")
idx_te = pd.Index(sp["idx_te"])
P = pickle.load(open(ROOT / "artifacts/predictions.pkl", "rb"))
S = json.load(open(ROOT / "artifacts/sched_results.json"))
m5 = np.asarray(P["predictions"]["M5"], float)

def pct(a):
    a = np.asarray(a, float)
    return dict(p50=float(np.percentile(a,50)), p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)), mx=float(a.max()), mn=float(a.min()))

print("="*70)
print("POINT 2 — p_star (job span) used as single-machine serial size")
# TC sample: df.loc[idx_te] -> reset -> sort by submit_time -> first 10000
te = df.loc[idx_te, ["submit_time","p_star"]].reset_index(drop=True)
te = te.sort_values("submit_time").reset_index(drop=True).iloc[:10000]
print("  TC sample (first 10,000 by r_j) p_star seconds:", pct(te["p_star"].values))
# MS sample via the library's own preparation (5000, with clip on q)
r,p,q = ms._prepare_jobs(df, idx_te, m5, sample_size=5000, clip_ratio=(0.1,10.0))
print("  MS sample (5,000) p (true size) seconds:", pct(p))
print("  ratio p99/p50 (TC):", round(pct(te['p_star'].values)['p99']/pct(te['p_star'].values)['p50'],1))

print("="*70)
print("POINT 1 — SRPT denominator uses TRUE sizes; policies use predictions")
print("  TC baselines ratio_vs_opt:", S["total_completion"]["baselines"])
print("  (SRPT==1.0 by construction => denominator is the TRUE-size optimum)")

print("="*70)
print("POINT 3 — linear vs log domain at scheduler entry (worked example)")
print(f"  predictions['M5'] range seconds: min={m5.min():.1f} max={m5.max():.1f} (linear, expm1 already applied)")
print(f"  p_star (true) range seconds: min={df['p_star'].min():.1f} max={df['p_star'].max():.1f}")
Sstar = S["max_stretch"]["meta"]["S_bisect"]
j = 0
print(f"  Worked max-stretch example (job idx {j} in MS sample):")
print(f"    r_j   = {r[j]:.1f} s (relative)")
print(f"    p_j*  = {p[j]:.1f} s   <-- true size, seconds")
print(f"    q_j   = {q[j]:.1f} s   <-- predicted size fed to SPRPT/EDF-P (clipped expm1)")
print(f"    S*    = {Sstar:.3f}")
print(f"    d_j = r_j + S*·p_j* = {r[j] + Sstar*p[j]:.1f} s  (deadline, seconds — NOT log)")
print(f"    sanity: log1p(p_j*) would be ~{np.log1p(p[j]):.2f}; deadline uses {p[j]:.1f}, so LINEAR.")

print("="*70)
print("POINT 4 — improvement over FIFO (total completion), same workload")
b = S["total_completion"]["baselines"]; meth = S["total_completion"]["methods"]
fifo = b["FIFO"]  # ratio vs SRPT
print(f"  FIFO ratio_vs_SRPT = {fifo:.4f}")
for mname, d in meth.items():
    spjf, prr = d["SPJF"], d["PRR"]
    print(f"  {mname}: SPJF ratio={spjf:.3f} -> reduce ΣC vs FIFO by {100*(1-spjf/fifo):.1f}% ; "
          f"PRR ratio={prr:.3f} -> reduce vs FIFO by {100*(1-prr/fifo):.1f}%  (spearman={d['spearman']:.3f})")
print("  RR baseline ratio_vs_SRPT =", b["RR"], "(non-predictive sharing)")

print("="*70)
print("MAX-STRETCH baseline outliers (are they predictor-driven?)")
print("  baselines:", S["max_stretch"]["baselines"])
print("  -> FIFO/LAS are NON-predictive; their huge rho is policy behavior, not prediction error.")
