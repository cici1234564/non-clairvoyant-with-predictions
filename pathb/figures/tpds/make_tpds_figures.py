# ============================================================
# TPDS (double-column, textwidth ~7.16in) figures from existing
# Path B + Task-2 results. Read-only: no recompute of models.
# 4 figures x 3 panels, consistent IEEE-ish serif style.
# ============================================================
import json, pickle, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = pathlib.Path(__file__).resolve().parent.parent          # pathb/figures
PB = ROOT.parent                                               # pathb/
OUT = ROOT / "tpds"; OUT.mkdir(exist_ok=True)
TW = 7.16   # IEEE double-column textwidth (inches)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
    "figure.dpi": 200, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
})
# consistent method palette (M5 = the base, given the strongest hue)
MC = {"M1": "#4C72B0", "M3": "#DD8452", "M4": "#55A868", "M5": "#C44E52",
      "M6": "#8172B3", "M7": "#64B5CD"}
MLAB = {"M1": "M1 CQR", "M3": "M3 HRAS", "M4": "M4 Iso", "M5": "M5 Meta",
        "M6": "M6 2-Stage", "M7": "M7 Recency"}
ORD = ["M1", "M3", "M4", "M5", "M6", "M7"]
TEAL_D, TEAL_M, TEAL_L = "#0f6e6e", "#1aa3a3", "#7fd4d4"

def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
def despine(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)

# ---------------- load data ----------------
P = pickle.load(open(PB / "artifacts/predictions.pkl", "rb"))
yte = np.asarray(P["yte"], float); sc = np.asarray(P["sig_count_te"], float)
pred_df = pd.read_csv(ROOT / "data_prediction_methods.csv")
recb = pd.read_csv(ROOT / "C_cov_by_recurrence.csv")
decd = pd.read_csv(ROOT / "D_cov_by_size_decile.csv")
bfs = pd.read_csv(ROOT / "B_fifo_vs_srpt.csv").set_index("method")
msd = pd.read_csv(ROOT / "data_maxstretch.csv").set_index("policy")
mkd = pd.read_csv(ROOT / "data_makespan.csv")
csx = pd.read_csv(ROOT / "data_coldstart_exploration.csv")
cal = pd.read_csv(ROOT / "E_calibration_tradeoff.csv")
FIFO_TC = 6.171

# ===================== FIGURE 1: workload =====================
fig, ax = plt.subplots(1, 3, figsize=(TW, 2.35))
p = np.sort(yte); n = len(p); cdf = np.arange(1, n+1)/n
P50, P90, P99 = np.percentile(p, [50, 90, 99])
# (a) CDF
ax[0].plot(p, cdf, color=TEAL_D)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"job size $p^*$ (s)"); ax[0].set_ylabel("CDF")
ax[0].set_title("Job-size CDF")
for v, l, c, yy in [(P50,"P50",TEAL_M,0.05),(P90,"P90","#d1772b",0.30),(P99,"P99","#b03030",0.55)]:
    ax[0].axvline(v, color=c, ls="--", lw=0.9)
    ax[0].text(v, yy, f"{l}={v:.0f}s", color=c, fontsize=6, ha="center", va="bottom")
panel_label(ax[0], "(a)"); despine(ax[0])
# (b) CCDF log-log
ccdf = 1 - cdf + 1.0/n
ax[1].loglog(p, ccdf, color=TEAL_D)
ax[1].set_xlabel(r"job size $p^*$ (s)"); ax[1].set_ylabel(r"$P(X>x)$")
ax[1].set_title(f"Heavy tail  (P99/P50$\\approx${P99/P50:.0f}$\\times$)")
ax[1].axvline(P99, color="#b03030", ls="--", lw=0.9)
ax[1].text(0.96, 0.92, f"max={p.max():.0f}s", transform=ax[1].transAxes, ha="right", fontsize=6.5)
panel_label(ax[1], "(b)"); despine(ax[1])
# (c) recurrence composition (stacked %)
segs = [("New (0)", int((sc==0).sum()), "#b03030"),
        ("1–4", int(((sc>=1)&(sc<=4)).sum()), "#d1772b"),
        ("5–49", int(((sc>=5)&(sc<=49)).sum()), TEAL_M),
        ("≥50", int((sc>=50).sum()), TEAL_D)]
tot = sum(s[1] for s in segs); left = 0
for name, cnt, c in segs:
    frac = 100*cnt/tot
    ax[2].barh(0, frac, left=left, color=c, edgecolor="white", lw=0.6)
    if frac > 3:
        ax[2].text(left+frac/2, 0, f"{name}\n{frac:.0f}%", ha="center", va="center",
                   color="white", fontsize=6, fontweight="bold")
    left += frac
ax[2].set_xlim(0, 100); ax[2].set_ylim(-0.5, 0.5); ax[2].set_yticks([])
ax[2].set_xlabel("share of test jobs (%)"); ax[2].set_title("Recurrence composition")
ax[2].grid(False)
panel_label(ax[2], "(c)"); despine(ax[2])
fig.tight_layout(w_pad=1.6)
save(fig, "fig1_workload")

# ===================== FIGURE 2: prediction =====================
fig, ax = plt.subplots(1, 3, figsize=(TW, 2.45))
m = pred_df[pred_df.method.str.startswith(("M1","M3","M4","M5","M6","M7"))].copy()
x = np.arange(len(m)); w = 0.26
sh = [TEAL_D, TEAL_M, TEAL_L]
for i, (col, lab) in enumerate([("cov25_all","All"),("cov25_rec","Rec(≥5)"),("cov25_new","New(0)")]):
    ax[0].bar(x+(i-1)*w, m[col], w, color=sh[i], label=lab, edgecolor="white", lw=0.4)
ax[0].set_xticks(x); ax[0].set_xticklabels([t.split()[0] for t in m.method], rotation=0)
ax[0].set_ylabel("Cov@25 (%)"); ax[0].set_title("Prediction accuracy by method")
ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)
panel_label(ax[0], "(a)"); despine(ax[0])
# (b) by recurrence bucket
xb = np.arange(len(recb)); w = 0.38
ax[1].bar(xb-w/2, recb.metacal_cov25, w, color=TEAL_D, label="Cov@25")
ax[1].bar(xb+w/2, recb.metacal_cov50, w, color=TEAL_L, label="Cov@50")
ax[1].set_xticks(xb); ax[1].set_xticklabels(recb.bucket)
ax[1].set_xlabel("train signature count"); ax[1].set_ylabel("coverage (%)")
ax[1].set_title("Predictability vs history"); ax[1].legend(handlelength=1.0)
panel_label(ax[1], "(b)"); despine(ax[1])
# (c) by size decile
ax[2].plot(decd.decile, decd.metacal_cov25, "o-", color=TEAL_D, ms=3.5, label="Meta+Cal")
ax[2].plot(decd.decile, decd.base_cov25, "s--", color="#d1772b", ms=3, lw=1.0, label="base M5")
ax[2].set_xticks(decd.decile)
ax[2].set_xticklabels([f"{v:.0f}" for v in decd.p_median], rotation=45, fontsize=5.6)
ax[2].set_xlabel(r"true-size decile (median $p^*$, s)"); ax[2].set_ylabel("Cov@25 (%)")
ax[2].set_title("Accuracy vs job size"); ax[2].legend(handlelength=1.4)
panel_label(ax[2], "(c)"); despine(ax[2])
fig.tight_layout(w_pad=1.6)
save(fig, "fig2_prediction")

# ===================== FIGURE 3: scheduling =====================
fig, ax = plt.subplots(1, 3, figsize=(TW, 2.5))
# (a) total completion ratio vs SRPT
x = np.arange(len(ORD)); w = 0.38
ax[0].bar(x-w/2, [bfs.loc[f"{k} {MLAB[k].split()[1]}" if False else k if k in bfs.index else k, "spjf_ratio_srpt"] if False else bfs.loc[k,"spjf_ratio_srpt"] for k in ORD], w,
          color="#b9c6d6", label="SPJF", edgecolor="white", lw=0.4)
ax[0].bar(x+w/2, [bfs.loc[k,"prr_ratio_srpt"] for k in ORD], w,
          color=[MC[k] for k in ORD], label="PRR", edgecolor="white", lw=0.4)
ax[0].axhline(1.0, color="#444", ls="-", lw=0.8); ax[0].text(len(ORD)-1, 1.06, "SRPT=1 (opt)", fontsize=6, ha="right")
ax[0].axhline(1.998, color="#888", ls=":", lw=0.9); ax[0].text(0, 2.05, "RR", fontsize=6)
ax[0].text(0.02, 0.96, f"FIFO={FIFO_TC:.1f} (off-scale)", transform=ax[0].transAxes, fontsize=6, va="top")
ax[0].set_xticks(x); ax[0].set_xticklabels(ORD)
ax[0].set_ylabel(r"$\rho_{TC}=\Sigma C/\mathrm{SRPT}$"); ax[0].set_title(r"Total completion $\Sigma C_j$")
ax[0].legend(handlelength=1.0, loc="upper right")
panel_label(ax[0], "(a)"); despine(ax[0])
# (b) max-stretch rho_max (methods linear; baselines off-scale annotated)
sprpt = [msd.loc[f"{k} SPRPT","rho_max"] for k in ORD]
edfp  = [msd.loc[f"{k} EDF-P","rho_max"] for k in ORD]
ax[1].bar(x-w/2, sprpt, w, color="#b9c6d6", label="SPRPT", edgecolor="white", lw=0.4)
ax[1].bar(x+w/2, edfp, w, color=[MC[k] for k in ORD], label="EDF-P", edgecolor="white", lw=0.4)
ax[1].axhline(1.0, color="#444", lw=0.8); ax[1].text(len(ORD)-1, 2.5, "OPT=1", fontsize=6, ha="right")
ax[1].set_xticks(x); ax[1].set_xticklabels(ORD)
ax[1].set_ylabel(r"$\rho_{S,\max}=S_{\max}/S^*$"); ax[1].set_title("Max-stretch")
ax[1].text(0.02, 0.97, "baselines off-scale:\nLAS/FB=170, FIFO=2932", transform=ax[1].transAxes,
           fontsize=5.8, va="top")
ax[1].legend(handlelength=1.0, loc="upper right")
panel_label(ax[1], "(b)"); despine(ax[1])
# (c) makespan LPPT ratio vs m
for k in ORD:
    sub = mkd[mkd.method == k].sort_values("m")
    ax[2].plot(sub.m, sub.LPPT_ratio, "o-", color=MC[k], ms=3, label=MLAB[k])
ax[2].axhline(1.0, color="#444", ls="--", lw=0.8); ax[2].text(5, 1.02, "LPT=1 (opt)", fontsize=6)
ax[2].set_xscale("log"); ax[2].set_xticks([5,10,20,50,100]); ax[2].set_xticklabels([5,10,20,50,100])
ax[2].set_xlabel("machines $m$"); ax[2].set_ylabel(r"$\rho=C_{\max}/\mathrm{OPT_{pre}}$")
ax[2].set_title("Makespan (LPPT)")
ax[2].legend(ncol=2, handlelength=1.2, columnspacing=0.8, loc="upper left", fontsize=5.8)
panel_label(ax[2], "(c)"); despine(ax[2])
fig.tight_layout(w_pad=1.7)
save(fig, "fig3_scheduling")

# ===================== FIGURE 4: interpretation =====================
fig, ax = plt.subplots(1, 3, figsize=(TW, 2.45))
# (a) reduction vs FIFO
red = [100*(1 - bfs.loc[k,"prr_ratio_srpt"]/FIFO_TC) for k in ORD]
ax[0].bar(x, red, 0.6, color=[MC[k] for k in ORD], edgecolor="white", lw=0.4)
for xi, v in zip(x, red): ax[0].text(xi, v+0.5, f"{v:.0f}", ha="center", fontsize=6)
ax[0].set_xticks(x); ax[0].set_xticklabels(ORD); ax[0].set_ylim(0, max(red)*1.18)
ax[0].set_ylabel(r"$\Sigma C$ reduction vs FIFO (%)"); ax[0].set_title("Predictions help (PRR)")
panel_label(ax[0], "(a)"); despine(ax[0])
# (b) calibration / ranking tradeoff (relative to base)
base = cal.iloc[0]
mets = [("cov25","Cov@25"),("rmsle","RMSLE"),("spearman","Spearman")]
xb = np.arange(len(mets)); w = 0.26
mods = [("Meta-base",cal.iloc[0],"#888888"),("Meta+Cal",cal.iloc[1],TEAL_M),("Cov-opt",cal.iloc[2],"#b03030")]
for i,(nm,row,c) in enumerate(mods):
    vals = [row[k]/base[k] for k,_ in mets]
    ax[1].bar(xb+(i-1)*w, vals, w, color=c, label=nm, edgecolor="white", lw=0.4)
ax[1].axhline(1.0, color="#444", lw=0.8)
ax[1].set_xticks(xb); ax[1].set_xticklabels([l for _,l in mets])
ax[1].set_ylabel("relative to Meta-base"); ax[1].set_ylim(0.85, 1.10)
ax[1].set_title("Calibration trade-off"); ax[1].legend(handlelength=1.0, loc="lower left")
ax[1].text(1.0, 1.085, "↑ better cov\n↑ worse RMSLE\n↓ worse rank", fontsize=5.4, ha="center", va="top")
panel_label(ax[1], "(b)"); despine(ax[1])
# (c) cold-start val vs test by population
pops = ["all","seen","cold"]; xb = np.arange(len(pops)); w = 0.38
v_val = [csx[(csx.split=="val_sel")&(csx.population==p)]["cov25"].values[0] for p in pops]
v_te  = [csx[(csx.split=="test")&(csx.population==p)]["cov25"].values[0] for p in pops]
ax[2].bar(xb-w/2, v_val, w, color=TEAL_L, label="val (held-out)")
ax[2].bar(xb+w/2, v_te, w, color=TEAL_D, label="test")
ax[2].axhline(12.74, color="#b03030", ls="--", lw=0.9)
ax[2].text(-0.45, 13.4, "Path B base New=12.7", color="#b03030", fontsize=5.6, ha="left", va="bottom")
for xi, vv in zip(xb-w/2, v_val): ax[2].text(xi, vv+0.6, f"{vv:.0f}", ha="center", fontsize=5.8)
for xi, vv in zip(xb+w/2, v_te): ax[2].text(xi, vv+0.6, f"{vv:.0f}", ha="center", fontsize=5.8)
ax[2].set_xticks(xb); ax[2].set_xticklabels(["All","Seen\n(≥1)","Cold\n(0)"])
ax[2].set_ylabel("Cov@25 (%)"); ax[2].set_title("Cold-start gain (Task 2)")
ax[2].legend(handlelength=1.0, loc="upper left"); ax[2].set_ylim(0, max(v_val+v_te)*1.28)
panel_label(ax[2], "(c)"); despine(ax[2])
fig.tight_layout(w_pad=1.7)
save(fig, "fig4_interpretation")

print("[done] wrote fig1_workload, fig2_prediction, fig3_scheduling, fig4_interpretation (pdf+png) to pathb/figures/tpds/")
