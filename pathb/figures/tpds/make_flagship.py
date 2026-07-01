# ============================================================
# Flagship composite figure (journal style), TPDS double-column.
# textwidth ~7.16in x ~3.25in (~1/3 page). One big main panel +
# two small supporting panels. Fresh blue/teal/yellow palette
# (Water Research 2026 "双一区三色" + light companions).
# Read-only: uses existing committed results.
# ============================================================
import pickle, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = pathlib.Path(__file__).resolve().parent.parent
PB = ROOT.parent
OUT = ROOT / "tpds"; OUT.mkdir(exist_ok=True)
TW = 7.16

# ---- fresh palette ----
BLUE, BLUE_L = "#4683B4", "#CBDDEB"
TEAL, TEAL_L = "#67B3AD", "#EAF5E2"
GOLD, GOLD_L = "#F3AF44", "#FCF0D5"
INK = "#33414a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
    "figure.dpi": 220, "axes.axisbelow": True, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#7f8c93",
    "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight",
})

def despine(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
def plabel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="right", color=INK)

# ---- data ----
P = pickle.load(open(PB / "artifacts/predictions.pkl", "rb"))
yte = np.asarray(P["yte"], float)
pred = pd.read_csv(ROOT / "data_prediction_methods.csv")
recb = pd.read_csv(ROOT / "C_cov_by_recurrence.csv")
m = pred[pred.method.str.startswith(("M1","M3","M4","M5","M6","M7"))].reset_index(drop=True)
short = [t.split()[0] for t in m.method]

fig = plt.figure(figsize=(TW, 3.25))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1],
                       wspace=0.28, hspace=0.55)
axM = fig.add_subplot(gs[:, 0])     # big main (spans both rows)
axT = fig.add_subplot(gs[0, 1])     # small top-right
axB = fig.add_subplot(gs[1, 1])     # small bottom-right

# ===== (a) MAIN: Cov@25 by method x population =====
x = np.arange(len(m)); w = 0.26
bars = [("cov25_all", "All", BLUE), ("cov25_rec", "Rec (≥5)", TEAL), ("cov25_new", "New (0)", GOLD)]
for i, (col, lab, c) in enumerate(bars):
    axM.bar(x + (i-1)*w, m[col], w, color=c, label=lab, edgecolor="white", lw=0.5, zorder=3)
# highlight the base method M5
m5i = short.index("M5")
axM.axvspan(m5i-0.45, m5i+0.45, color="#f2f4f5", zorder=0)
axM.set_xticks(x); axM.set_xticklabels(short)
axM.set_ylabel("Cov@25 (%)")
axM.set_title("Prediction accuracy by method and recurrence", pad=8)
axM.set_ylim(0, max(m.cov25_rec)*1.16)
axM.yaxis.grid(True, color="#e6e9ea", lw=0.6)
axM.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           handlelength=1.1, columnspacing=1.0, handleheight=0.9)
axM.text(m5i, max(m.cov25_rec)*1.02, "base", ha="center", fontsize=6.2, color="#6b7a82")
plabel(axM, "(a)"); despine(axM)

# ===== (b) small top-right: heavy-tail CCDF =====
p = np.sort(yte); n = len(p); cdf = np.arange(1, n+1)/n; ccdf = 1 - cdf + 1.0/n
P50, P99 = np.percentile(p, [50, 99])
axT.loglog(p, ccdf, color=TEAL, lw=1.5)
axT.fill_between(p, ccdf, 1e-6, color=TEAL_L, alpha=0.55, zorder=0)
axT.axvline(P99, color=GOLD, ls="--", lw=1.0)
axT.set_xlabel(r"job size $p^*$ (s)", labelpad=1); axT.set_ylabel(r"$P(X{>}x)$", labelpad=1)
axT.set_title(f"Heavy tail  (P99/P50$\\approx${P99/P50:.0f}$\\times$)", fontsize=7.6, pad=4)
axT.text(0.95, 0.9, f"max={p.max():.0f}s", transform=axT.transAxes, ha="right", fontsize=6)
axT.tick_params(labelsize=6.2)
plabel(axT, "(b)"); despine(axT)

# ===== (c) small bottom-right: coverage vs recurrence bucket =====
xb = np.arange(len(recb)); w2 = 0.4
axB.bar(xb - w2/2, recb.metacal_cov25, w2, color=BLUE, label="Cov@25", edgecolor="white", lw=0.4, zorder=3)
axB.bar(xb + w2/2, recb.metacal_cov50, w2, color=GOLD, label="Cov@50", edgecolor="white", lw=0.4, zorder=3)
axB.set_xticks(xb); axB.set_xticklabels(["New", "1–4", "5–49", "≥50"], fontsize=6.2)
axB.set_xlabel("train signature count", labelpad=1); axB.set_ylabel("coverage (%)", labelpad=1)
axB.set_title("Predictability vs history", fontsize=7.6, pad=4)
axB.yaxis.grid(True, color="#e6e9ea", lw=0.6)
axB.legend(ncol=2, loc="upper left", handlelength=1.0, columnspacing=0.8, fontsize=6)
axB.tick_params(labelsize=6.2)
plabel(axB, "(c)"); despine(axB)

fig.savefig(OUT / "flagship_prediction.pdf")
fig.savefig(OUT / "flagship_prediction.png")
plt.close(fig)
print("[done] wrote flagship_prediction.{pdf,png}")
