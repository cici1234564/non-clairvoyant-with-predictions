# ============================================================
# Advanced chart types on the real data (read-only):
#   (a) violin: log(pred/true) by recurrence bucket
#   (b) Sankey: test jobs -> recurrence bucket -> Cov@25 hit/miss
#   (c) waterfall+line: cold-start Cov@25 build-up (validation)
# Desaturated fresh palette (HSV saturation ~0.62).
# ============================================================
import pickle, pathlib, warnings, colorsys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parent.parent
PB = ROOT.parent; OUT = ROOT / "tpds"; OUT.mkdir(exist_ok=True)
TW = 7.16; INK = "#3a464d"

def desat(hexc, s=0.62, v=1.05):
    r, g, b = mcolors.to_rgb(hexc); h, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, ss*s, min(1.0, vv*v))
BLUE, TEAL, GOLD = desat("#4683B4"), desat("#67B3AD"), desat("#F3AF44")
RED = desat("#C44E52"); GREY = "#b9c2c7"
BUCKET_C = [desat("#C44E52"), GOLD, TEAL, BLUE]   # New,1-4,5-49,>=50

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.3, "axes.labelsize": 8, "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.7,
    "figure.dpi": 220, "axes.axisbelow": True, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#8a969c",
    "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight",
})
def despine(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def plabel(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")

# ---- data ----
P = pickle.load(open(PB / "artifacts/predictions.pkl", "rb"))
yte = np.asarray(P["yte"], float); mcal = np.asarray(P["metacal_raw"], float)
sc = np.asarray(P["sig_count_te"], float)
recb = pd.read_csv(ROOT / "C_cov_by_recurrence.csv")
masks = [("New (0)", sc == 0), ("1–4", (sc >= 1) & (sc <= 4)),
         ("5–49", (sc >= 5) & (sc <= 49)), ("≥50", sc >= 50)]

fig = plt.figure(figsize=(TW, 3.8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1, 1],
                       wspace=0.28, hspace=0.62)
axV = fig.add_subplot(gs[0, 0])
axW = fig.add_subplot(gs[0, 1])
axS = fig.add_subplot(gs[1, :])

# ===== (a) violin: log(pred/true) by recurrence bucket =====
data = []
for _, mk in masks:
    lr = np.log(np.maximum(mcal[mk], 1.0) / np.maximum(yte[mk], 1.0))
    data.append(np.clip(lr, -4, 4))
band = np.log(1.25)
axV.axhspan(-band, band, color="#eef2f3", zorder=0)
axV.axhline(0, color="#8a969c", lw=0.7, ls="--", zorder=1)
parts = axV.violinplot(data, positions=range(len(masks)), widths=0.82,
                       showmeans=False, showextrema=False, showmedians=True)
for i, b in enumerate(parts["bodies"]):
    b.set_facecolor(BUCKET_C[i]); b.set_edgecolor("white"); b.set_alpha(0.85); b.set_zorder=2
parts["cmedians"].set_color(INK); parts["cmedians"].set_linewidth(1.1)
axV.set_xticks(range(len(masks))); axV.set_xticklabels([m[0] for m in masks])
axV.set_ylabel(r"$\log(\hat p / p^*)$  (0 = exact)")
axV.set_xlabel("train signature count")
axV.set_title("Prediction error spread by recurrence")
axV.text(0.02, 0.97, r"shaded: $\pm\ln 1.25$ (Cov@25 zone)", transform=axV.transAxes,
         fontsize=5.8, va="top", color="#6b7a82")
axV.set_ylim(-4.2, 4.2)
plabel(axV, "(a)"); despine(axV)

# ===== (b) waterfall + cumulative line: cold-start Cov@25 (validation) =====
# honest ablation from the Task-2 search (val_sel cold Cov@25, direct features):
labels = ["LightGBM\n(L2)", "+ median\nobjective", "+ segment\ncalibration", "final"]
cum = [14.68, 16.53, 17.35]                      # cumulative tops
base = cum[0]; inc1 = cum[1]-cum[0]; inc2 = cum[2]-cum[1]
xs = np.arange(4)
axW.bar(0, base, 0.62, color=GREY, edgecolor="white", lw=0.5, zorder=3)
axW.bar(1, inc1, 0.62, bottom=base, color=GOLD, edgecolor="white", lw=0.5, zorder=3)
axW.bar(2, inc2, 0.62, bottom=cum[1], color=TEAL, edgecolor="white", lw=0.5, zorder=3)
axW.bar(3, cum[2], 0.62, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
# connector line over the three build-up steps (not the summary bar)
axW.plot([0, 1, 2], [base, cum[1], cum[2]], "o-", color=INK, ms=3, lw=1.0, zorder=4)
for x, t in [(0, base), (1, cum[1]), (2, cum[2]), (3, cum[2])]:
    axW.text(x, t+0.45, f"{t:.1f}", ha="center", fontsize=6.4)
axW.annotate(f"+{inc1:.1f}", (1, cum[1]), (0.62, cum[1]+1.5), fontsize=6, color=desat("#F3AF44",1.0,0.85),
             ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaa"))
axW.annotate(f"+{inc2:.1f}", (2, cum[2]), (1.68, cum[2]+1.5), fontsize=6, color=TEAL,
             ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaa"))
axW.axhline(12.74, color=RED, ls="--", lw=0.9)
axW.text(3.3, 12.74, "Path B base 12.7", color=RED, fontsize=5.6, ha="right", va="bottom")
axW.set_xticks(xs); axW.set_xticklabels(labels, fontsize=6.2)
axW.set_ylabel("cold-start Cov@25 (%)"); axW.set_ylim(0, 20)
axW.set_title("Where the cold-start gain comes from (val)")
axW.yaxis.grid(True, color="#eceff0", lw=0.6)
plabel(axW, "(b)"); despine(axW)

# ===== (c) Sankey: jobs -> recurrence bucket -> Cov@25 hit/miss =====
axS.set_xlim(0, 1); axS.set_ylim(0, 1); axS.axis("off")
plabel(axS, "(c)")
axS.set_title("Where Cov@25 hits come from: jobs → recurrence → covered", pad=2)
counts = np.array([int(mk.sum()) for _, mk in masks], float)
cov = recb.set_index("bucket")
cov25 = []
for name, _ in masks:
    key = {"New (0)": "New(0)", "1–4": "1-4", "5–49": "5-49", "≥50": ">=50"}[name]
    cov25.append(float(cov.loc[key, "metacal_cov25"]))
cov25 = np.array(cov25)/100.0
hit = counts*cov25; miss = counts-hit
total = counts.sum()
GAP = 0.02
# left node geometry (recurrence buckets, top->bottom)
def stack(vals, gap, top=1.0):
    h = (vals/total)*(1 - gap*(len(vals)-1)); ys = []; cur = top
    for hi in h:
        ys.append((cur-hi, cur)); cur -= hi+gap
    return ys, h
Lys, Lh = stack(counts, GAP)
# right nodes: Covered (top), Missed (bottom)
Rvals = np.array([hit.sum(), miss.sum()])
Rys, Rh = stack(Rvals, 0.04)
xL0, xL1, xR0, xR1 = 0.04, 0.13, 0.87, 0.96
def ribbon(y0t, y0b, y1t, y1b, color):
    verts = [(xL1, y0t), (0.5, y0t), (0.5, y1t), (xR0, y1t),
             (xR0, y1b), (0.5, y1b), (0.5, y0b), (xL1, y0b), (xL1, y0t)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    axS.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=0.55, zorder=1))
# cursors on right nodes
cur_hit = Rys[0][1]; cur_miss = Rys[1][1]
for i, (name, _) in enumerate(masks):
    yb0, yb1 = Lys[i]; bh = yb1-yb0
    f_hit = (hit[i]/counts[i])*bh; f_miss = bh-f_hit
    # hit ribbon (top part of bucket -> covered node)
    ribbon(yb1, yb1-f_hit, cur_hit, cur_hit-f_hit, BLUE); cur_hit -= f_hit
    # miss ribbon (bottom part -> missed node)
    ribbon(yb1-f_hit, yb0, cur_miss, cur_miss-f_miss, RED); cur_miss -= f_miss
    # left node rectangle + label
    axS.add_patch(Rectangle((xL0, yb0), xL1-xL0, bh, facecolor=BUCKET_C[i], edgecolor="white", lw=0.5, zorder=3))
    axS.text(xL0-0.008, (yb0+yb1)/2, f"{name}\n{counts[i]/total*100:.0f}%", ha="right", va="center", fontsize=6)
# right node rectangles + labels
for (y0, y1), lab, c, val in [(Rys[0], "Covered@25", BLUE, hit.sum()), (Rys[1], "Missed", RED, miss.sum())]:
    axS.add_patch(Rectangle((xR0, y0), xR1-xR0, y1-y0, facecolor=c, edgecolor="white", lw=0.5, zorder=3))
    axS.text(xR1+0.008, (y0+y1)/2, f"{lab}\n{val/total*100:.0f}%", ha="left", va="center", fontsize=6)

fig.savefig(OUT / "advanced_charts.pdf"); fig.savefig(OUT / "advanced_charts.png")
plt.close(fig)
print("[done] wrote advanced_charts.{pdf,png}")
