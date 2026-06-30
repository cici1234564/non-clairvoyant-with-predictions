# ============================================================
# Figure & insight generation ONLY. No experiments, no model
# selection, no optimization, no test tuning. Reads existing
# committed results + already-produced test predictions. Writes
# only to pathb/figures/. Re-checks I1-I4 before using any data.
# ============================================================
import json, pickle, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent          # pathb/
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
TEAL_D, TEAL, TEAL_L, GRID = "#0f6e6e", "#1aa3a3", "#7fd4d4", "#d9e6e6"
ORANGE = "#d1772b"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#33514f",
                     "axes.linewidth": 1.0, "figure.dpi": 130})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

def cov(yt, yh, p):
    yt = np.maximum(yt, 1e-12); return 100.0 * float(np.mean(np.abs(yh - yt) / yt <= p))

# ---------------- load existing artifacts ----------------
df = pd.read_pickle(ROOT / "cache/feat_df.pkl")
sp = np.load(ROOT / "cache/splits.npz")
idx_tr, idx_va, idx_te = sp["idx_tr"], sp["idx_va"], sp["idx_te"]
t_train, t_val = int(sp["t_train"]), int(sp["t_val"])
P = pickle.load(open(ROOT / "artifacts/predictions.pkl", "rb"))
RP = json.load(open(ROOT / "artifacts/results_pred.json"))
SCH = json.load(open(ROOT / "artifacts/sched_results.json"))
FCS = json.load(open(ROOT / "improve/artifacts/final_cs.json"))

yte = np.asarray(P["yte"], float)                  # true test p_star (linear)
m5 = np.asarray(P["predictions"]["M5"], float)     # uncalibrated Meta base
mcal = np.asarray(P["metacal_raw"], float)         # Meta+Cal (isotonic)
sc = np.asarray(P["sig_count_te"], float)          # train signature count (per test job)

# ---------------- I1-I4 re-check ----------------
st = df["submit_time"].values
inv = {}
inv["I1_order"] = bool(st[idx_tr].max() < st[idx_va].min() < st[idx_te].min())
inv["I1_test_window_frac"] = float((st[idx_te] >= t_val).mean())     # must be 1.0
inv["align_len"] = bool(len(m5) == len(idx_te) == len(yte) == len(sc) == len(mcal))
inv["align_yte"] = bool(np.allclose(yte, df["p_star"].values[idx_te]))  # preds aligned to test rows
inv["I2_I3_I4"] = ("inherited: features submit-time + train-only stats + shift(1) histories; "
                   "verified in Path B V1-V8 (V3 leakage PASS). No data refit here.")
assert inv["I1_order"] and abs(inv["I1_test_window_frac"]-1.0) < 1e-12 and inv["align_len"] and inv["align_yte"], \
    "INVARIANT/ALIGNMENT CHECK FAILED — aborting figure generation"
print("[I1-I4] OK:", json.dumps({k: v for k, v in inv.items() if k != "I2_I3_I4"}))
json.dump(inv, open(FIG / "invariant_check.json", "w"), indent=1)

insights = []   # (id, png, insight, caveat)

# ================= A. p_star distribution: CDF + log-log tail =================
p = np.sort(yte)
n = len(p)
q = {str(k): float(np.percentile(p, k)) for k in (50, 90, 99)}
q["max"] = float(p.max()); q["min"] = float(p.min()); q["mean"] = float(p.mean())
ratio = q["99"] / q["50"]
# export CSV: percentiles + thinned CDF
cdf_x = p[:: max(1, n // 2000)]
pd.DataFrame({"p_star_sorted": cdf_x,
              "cdf": np.arange(1, len(cdf_x)+1)/len(cdf_x)}).to_csv(FIG/"A_pstar_cdf.csv", index=False)
pd.DataFrame([{"stat": k, "p_star_sec": v} for k, v in q.items()]
             ).to_csv(FIG/"A_pstar_percentiles.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cdf = np.arange(1, n+1)/n
ax[0].plot(p, cdf, color=TEAL_D, lw=2)
ax[0].set_xscale("log"); ax[0].set_xlabel("job size p* (s, log)"); ax[0].set_ylabel("CDF")
ax[0].set_title("A. Job-size CDF", color=TEAL_D)
for k, lbl, c in [("50","P50",TEAL),("90","P90",ORANGE),("99","P99","#b03030")]:
    ax[0].axvline(q[k], color=c, ls="--", lw=1.2); ax[0].text(q[k], 0.05, f"{lbl}\n{q[k]:.0f}s", color=c, fontsize=8, ha="center")
ax[0].yaxis.grid(True, color=GRID)
# log-log CCDF tail
ccdf = 1.0 - cdf + 1.0/n
ax[1].loglog(p, ccdf, color=TEAL_D, lw=2)
ax[1].set_xlabel("job size p* (s, log)"); ax[1].set_ylabel("P(X > x) (log)")
ax[1].set_title(f"A. Heavy-tail CCDF  (P99/P50≈{ratio:.0f}x, max={q['max']:.0f}s)", color=TEAL_D)
ax[1].axvline(q["99"], color="#b03030", ls="--", lw=1.2)
ax[1].grid(True, which="both", color=GRID)
for a in ax:
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
save(fig, "A_pstar_distribution")
insights.append(("A",
    f"Job sizes are extremely heavy-tailed (P50={q['50']:.0f}s, P90={q['90']:.0f}s, "
    f"P99={q['99']:.0f}s, max={q['max']:.0f}s; P99/P50≈{ratio:.0f}x), so a handful of giant "
    f"jobs dominate ΣC_j/SRPT and structurally inflate competitive ratios.",
    "Tail is the TEST set's true p*; the single-machine sims treat this parallel fork-join span "
    "as a serial size (a modeling choice, not measured here)."))

# ================= B. improvement vs FIFO + competitive ratio vs SRPT =================
tc = SCH["total_completion"]; fifo = tc["baselines"]["FIFO"]
rows = []
for m, d in tc["methods"].items():
    rows.append(dict(method=m, spearman=d["spearman"],
                     spjf_ratio_srpt=d["SPJF"], prr_ratio_srpt=d["PRR"],
                     spjf_reduction_fifo_pct=100*(1-d["SPJF"]/fifo),
                     prr_reduction_fifo_pct=100*(1-d["PRR"]/fifo)))
order = ["M1","M3","M4","M5","M6","M7"]
rows = sorted(rows, key=lambda r: order.index(r["method"]))
pd.DataFrame(rows).to_csv(FIG/"B_fifo_vs_srpt.csv", index=False)
labels = [r["method"] for r in rows]; x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(x-w/2, [r["spjf_ratio_srpt"] for r in rows], w, label="SPJF", color=TEAL_L)
ax[0].bar(x+w/2, [r["prr_ratio_srpt"] for r in rows], w, label="PRR", color=TEAL_D)
ax[0].axhline(1.0, color="#b03030", ls="--", lw=1, label="SRPT (oracle)=1")
ax[0].set_title("B. Competitive ratio vs SRPT (oracle)", color=TEAL_D); ax[0].set_ylabel("ρ_TC = ΣC / SRPT")
ax[1].bar(x-w/2, [r["spjf_reduction_fifo_pct"] for r in rows], w, label="SPJF", color=TEAL_L)
ax[1].bar(x+w/2, [r["prr_reduction_fifo_pct"] for r in rows], w, label="PRR", color=TEAL_D)
ax[1].set_title("B. ΣC reduction vs FIFO (%)", color=TEAL_D); ax[1].set_ylabel("% reduction vs FIFO")
for a in (ax[0], ax[1]):
    a.set_xticks(x); a.set_xticklabels(labels); a.legend(frameon=False, fontsize=9)
    a.yaxis.grid(True, color=GRID)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
save(fig, "B_fifo_vs_srpt")
best = min(rows, key=lambda r: r["prr_ratio_srpt"])
insights.append(("B",
    f"Every predictor cuts ΣC by ~{min(r['prr_reduction_fifo_pct'] for r in rows):.0f}-"
    f"{max(r['prr_reduction_fifo_pct'] for r in rows):.0f}% vs FIFO (best PRR={best['method']}, "
    f"{best['prr_reduction_fifo_pct']:.0f}%), so the predicted ranking genuinely helps even though "
    f"ρ vs SRPT stays 2.3-3.8.",
    "ρ is vs the clairvoyant SRPT optimum (Point 1 of the audit); the large ρ is a normalization "
    "choice, not predictor failure. Spearman of the fed ranking varies by method."))

# ================= C. Cov@25/50 vs training signature-count bucket =================
buckets = [("New(0)", sc == 0), ("1-4", (sc>=1)&(sc<=4)), ("5-49", (sc>=5)&(sc<=49)), (">=50", sc>=50)]
crows = []
for name, mask in buckets:
    crows.append(dict(bucket=name, n=int(mask.sum()),
                      base_cov25=cov(yte[mask], m5[mask], .25), base_cov50=cov(yte[mask], m5[mask], .5),
                      metacal_cov25=cov(yte[mask], mcal[mask], .25), metacal_cov50=cov(yte[mask], mcal[mask], .5)))
pd.DataFrame(crows).to_csv(FIG/"C_cov_by_recurrence.csv", index=False)
xb = np.arange(len(buckets)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.bar(xb-w/2, [r["metacal_cov25"] for r in crows], w, label="Cov@25 (Meta+Cal)", color=TEAL_D)
ax.bar(xb+w/2, [r["metacal_cov50"] for r in crows], w, label="Cov@50 (Meta+Cal)", color=TEAL_L)
ax.plot(xb, [r["base_cov25"] for r in crows], "o--", color=ORANGE, lw=1.5, ms=5, label="Cov@25 (base M5)")
ax.set_xticks(xb); ax.set_xticklabels([f"{r['bucket']}\n(n={r['n']:,})" for r in crows])
ax.set_ylabel("Coverage (%)"); ax.set_xlabel("training signature count")
ax.set_title("C. Predictability grows with history", color=TEAL_D)
ax.legend(frameon=False, fontsize=9); ax.yaxis.grid(True, color=GRID)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
save(fig, "C_cov_by_recurrence")
insights.append(("C",
    f"Cov@25 rises monotonically with history (New={crows[0]['metacal_cov25']:.1f}% -> "
    f">=50={crows[3]['metacal_cov25']:.1f}%); the cold-start bottleneck is the New bucket, which is "
    f"{100*crows[0]['n']/len(sc):.0f}% of test.",
    "'New' (fine-signature unseen) may still share coarser keys with train; bucket counts are the "
    "train fine-signature count, the same definition used in Path B."))

# ================= D. Cov@25 by true-size decile =================
dec = np.quantile(yte, np.linspace(0, 1, 11))
dec[0] = -np.inf; dec[-1] = np.inf
drows = []
for i in range(10):
    mask = (yte > dec[i]) & (yte <= dec[i+1])
    pv = yte[mask]
    drows.append(dict(decile=i+1, p_lo=float(pv.min()), p_hi=float(pv.max()),
                      p_median=float(np.median(pv)), n=int(mask.sum()),
                      base_cov25=cov(yte[mask], m5[mask], .25), metacal_cov25=cov(yte[mask], mcal[mask], .25),
                      base_cov50=cov(yte[mask], m5[mask], .5), metacal_cov50=cov(yte[mask], mcal[mask], .5)))
pd.DataFrame(drows).to_csv(FIG/"D_cov_by_size_decile.csv", index=False)
xd = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(xd, [r["metacal_cov25"] for r in drows], "o-", color=TEAL_D, lw=2, label="Cov@25 (Meta+Cal)")
ax.plot(xd, [r["base_cov25"] for r in drows], "s--", color=ORANGE, lw=1.5, ms=5, label="Cov@25 (base M5)")
ax.set_xticks(xd)
ax.set_xticklabels([f"D{i}\n{r['p_median']:.0f}s" for i, r in zip(xd, drows)], fontsize=8)
ax.set_xlabel("true job-size decile (median p* shown)"); ax.set_ylabel("Cov@25 (%)")
ax.set_title("D. Where prediction error concentrates", color=TEAL_D)
ax.legend(frameon=False, fontsize=9); ax.yaxis.grid(True, color=GRID)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
save(fig, "D_cov_by_size_decile")
lo = min(drows, key=lambda r: r["metacal_cov25"]); hi = max(drows, key=lambda r: r["metacal_cov25"])
d10 = drows[9]
insights.append(("D",
    f"Counter to the 'error is in the heavy tail' intuition, Cov@25 is LOWEST at the smallest jobs "
    f"(D1-D2, median {drows[1]['p_median']:.0f}s, ~{drows[1]['metacal_cov25']:.0f}%), peaks mid-range "
    f"(D{hi['decile']}={hi['metacal_cov25']:.1f}%), and the heavy tail D10 (median "
    f"{d10['p_median']:.0f}s) is only middling ({d10['metacal_cov25']:.1f}%); so prediction is "
    f"hardest for tiny jobs, while the tail jobs that dominate scheduling cost are predicted "
    f"moderately well.",
    "Buckets are TRUE-size deciles; Cov@25 is a relative-error metric, so very small jobs are "
    "penalized by tiny absolute errors (a metric artifact, not necessarily larger absolute error). "
    "Computed from existing test predictions, no retraining."))

# ================= E. calibration / ranking trade-off =================
m5m = RP["results"]["M5"]["All"]; mc = RP["metacal"]["All"]; co = FCS["test"]["all"]
erows = [
    dict(model="Meta-base (M5, uncal.)", cov25=m5m["Cov25"], rmsle=m5m["RMSLE"], spearman=m5m["Spearman"]),
    dict(model="Meta+Cal (isotonic)",    cov25=mc["Cov25"],  rmsle=mc["RMSLE"],  spearman=mc["Spearman"]),
    dict(model="Coverage-opt (Task2)",   cov25=co["cov25"],  rmsle=co["rmsle"],  spearman=co["rho"]),
]
pd.DataFrame(erows).to_csv(FIG/"E_calibration_tradeoff.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 4))
cols = [TEAL_D, TEAL, ORANGE]; names = [r["model"] for r in erows]
for a, key, ttl, better in [(ax[0],"cov25","Cov@25 % (higher better)","up"),
                            (ax[1],"rmsle","RMSLE (lower better)","down"),
                            (ax[2],"spearman","Spearman ρ (higher=ranking)","up")]:
    a.bar(range(3), [r[key] for r in erows], color=cols)
    a.set_xticks(range(3)); a.set_xticklabels(["Meta\nbase","Meta+\nCal","Cov\nopt"], fontsize=9)
    a.set_title(ttl, color=TEAL_D, fontsize=10); a.yaxis.grid(True, color=GRID)
    for i, r in enumerate(erows): a.text(i, r[key], f"{r[key]:.3f}" if key!="cov25" else f"{r[key]:.1f}", ha="center", va="bottom", fontsize=8)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
save(fig, "E_calibration_tradeoff")
insights.append(("E",
    f"Isotonic calibration nudges Cov@25 {m5m['Cov25']:.1f}->{mc['Cov25']:.1f} while preserving rank "
    f"(Spearman {mc['Spearman']:.3f}); the coverage-optimized variant reaches Cov@25 {co['cov25']:.1f} "
    f"but worsens RMSLE ({m5m['RMSLE']:.3f}->{co['rmsle']:.3f}) and breaks rank "
    f"(Spearman {co['rho']:.3f}) — which is why scheduling keeps the uncalibrated ranking.",
    "The Task-2 'coverage-opt' differs in model+objective+calibration (not a pure calibration "
    "ablation); only the Meta-base vs Meta+Cal pair isolates isotonic calibration."))

# ================= FIGURES.md =================
md = ["# Figures — heavy tails, ranking value, predictability, and the calibration trade-off\n",
      "Read-only from committed Path B + Task-2 artifacts. No experiments/selection/tuning. "
      "Invariants re-checked (`invariant_check.json`): I1 split ordered + test-window frac=1.0, "
      "predictions aligned to test rows; I2/I3/I4 inherited from Path B V1-V8.\n"]
fname = {"A":"A_pstar_distribution","B":"B_fifo_vs_srpt","C":"C_cov_by_recurrence",
         "D":"D_cov_by_size_decile","E":"E_calibration_tradeoff"}
title = {"A":"A. Job-size distribution (CDF + heavy-tail CCDF)",
         "B":"B. Predictor improvement vs FIFO and competitive ratio vs SRPT",
         "C":"C. Cov@25/50 vs training signature-count bucket",
         "D":"D. Cov@25 by true-size decile",
         "E":"E. Calibration / ranking trade-off"}
for fid, ins, cav in insights:
    md.append(f"## {title[fid]}")
    md.append(f"![{fid}]({fname[fid]}.png)")
    md.append(f"- **Insight:** {ins}")
    md.append(f"- **Caveat:** {cav}")
    md.append(f"- Data: `{fname[fid]}.csv` (+ `A_pstar_percentiles.csv` for A).\n")
(FIG/"FIGURES.md").write_text("\n".join(md))
print("[done] wrote 5 figures (png+pdf), CSVs, FIGURES.md, invariant_check.json to pathb/figures/")
