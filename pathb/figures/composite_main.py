# ============================================================
# ONE publication-grade composite figure (panels a-f), ATLAS/TPDS.
# WIDE cross-column figure*: left block = 6 quantitative panels with
# room to breathe; right HALF = the 3-stage Sankey, stages spread out.
# Nature Evo1.5 teal palette. Real per-job Path B data only.
# d & e recomputed from raw committed test predictions.
# ============================================================
import pickle, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from scipy.stats import gaussian_kde

ROOT = pathlib.Path(__file__).resolve().parent
PB = ROOT.parent
W, H = 11.5, 6.0

C = dict(teal_pale="#d5eada", teal_bright="#33c5b2", teal_mid="#32a4b4", teal_dark="#024e52",
         pale_teal2="#9ad8c8", soft_blue="#b2c9ce", near_white="#f3f3f3", grey_light="#d9dbdd",
         grey_mid="#8f9092", grey_soft="#a3b1ae", violet="#8a83a0", mauve="#776f76")
INK = "#2b3a3f"
TEAL_CMAP = mcolors.LinearSegmentedColormap.from_list("teal", [C["near_white"], C["pale_teal2"],
                                                               C["teal_bright"], C["teal_dark"]])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.axisbelow": True, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#6f7c80",
    "xtick.color": INK, "ytick.color": INK,
})
def despine(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def tag(ax, s):
    ax.text(-0.06, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")

# ---------------- data + leakage re-verification ----------------
P = pickle.load(open(PB / "artifacts/predictions.pkl", "rb"))
sp = np.load(PB / "cache/splits.npz"); dfm = pd.read_pickle(PB / "cache/feat_df.pkl")
st = dfm["submit_time"].values
assert st[sp["idx_tr"]].max() < st[sp["idx_va"]].min() < st[sp["idx_te"]].min(), "I1 temporal order FAILED"
yte = np.asarray(P["yte"], float)
assert np.allclose(yte, dfm["p_star"].values[sp["idx_te"]]), "predictions not aligned to test rows"
mcal = np.asarray(P["metacal_raw"], float)
sc = np.asarray(P["sig_count_te"], float)        # train-only signature count (submit-time)
NT = len(yte)
print(f"[leakage] I1 temporal order OK; test-window frac={float((st[sp['idx_te']]>=int(sp['t_val'])).mean()):.3f}; "
      f"sig_count is train-derived; n_test={NT}")

pred_df = pd.read_csv(ROOT / "data_prediction_methods.csv")
recb = pd.read_csv(ROOT / "C_cov_by_recurrence.csv")
cdf_df = pd.read_csv(ROOT / "A_pstar_cdf.csv")
pct = pd.read_csv(ROOT / "A_pstar_percentiles.csv").set_index("stat")["p_star_sec"]
bfs = pd.read_csv(ROOT / "B_fifo_vs_srpt.csv").set_index("method")
msd = pd.read_csv(ROOT / "data_maxstretch.csv").set_index("policy")
mkd = pd.read_csv(ROOT / "data_makespan.csv")
FIFO = 6.171                       # total-completion FIFO ratio vs SRPT
FIFO_SMAX = float(msd.loc["FIFO", "rho_max"])   # max-stretch FIFO (=2931.7)
relerr = (mcal - yte) / np.maximum(yte, 1e-9)
hit = np.abs(relerr) <= 0.25

# ---------------- scaffold: left block (3x2) | Sankey (half width) ----------------
fig = plt.figure(figsize=(W, H))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10, figure=fig)
Lg = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=outer[0, 0],
                                      height_ratios=[1, 1, 1.0], hspace=0.95, wspace=0.95)
axA = fig.add_subplot(Lg[0, 0:3]); axB = fig.add_subplot(Lg[0, 3:6])
axC = fig.add_subplot(Lg[1, 0:3]); axD = fig.add_subplot(Lg[1, 3:6])
axF = fig.add_subplot(Lg[2, 0:2]); axG = fig.add_subplot(Lg[2, 2:4]); axH = fig.add_subplot(Lg[2, 4:6])
axE = fig.add_subplot(outer[0, 1])

# ===================== (a) job-size CDF =====================
axA.plot(cdf_df.p_star_sorted, cdf_df.cdf, color=C["teal_dark"], lw=1.5)
for k, c in [("50", C["teal_mid"]), ("90", C["grey_mid"]), ("99", C["violet"])]:
    axA.axvline(pct[k], color=c, ls="--", lw=0.9)
axA.set_xscale("log"); axA.set_ylabel("CDF"); axA.set_xlabel(r"job size $p^*$ (s)")
axA.set_title("Job-size distribution")
axA.text(0.04, 0.82, f"P99/P50\n$\\approx${pct['99']/pct['50']:.0f}$\\times$", transform=axA.transAxes,
         fontsize=6.5, color=C["violet"])
axA.text(pct["50"], 0.03, "P50", fontsize=6, color=C["teal_mid"], ha="center")
axA.text(pct["90"], 0.55, "P90", fontsize=6, color=C["grey_mid"], ha="center")
axA.text(pct["99"], 0.30, "P99", fontsize=6, color=C["violet"], ha="center")
axA.set_ylim(0, 1.03); axA.yaxis.grid(True, color=C["near_white"], lw=0.7)
tag(axA, "a"); despine(axA)

# ===================== (b) coverage by recurrence bucket =====================
xb = np.arange(len(recb)); w = 0.4
c25 = [C["violet"], C["teal_mid"], C["teal_mid"], C["teal_mid"]]
c50 = [mcolors.to_rgba(C["violet"], 0.45)] + [mcolors.to_rgba(C["teal_bright"], 0.55)]*3
axB.bar(xb-w/2, recb.metacal_cov25, w, color=c25, edgecolor="white", lw=0.5, label="Cov@25")
axB.bar(xb+w/2, recb.metacal_cov50, w, color=c50, edgecolor="white", lw=0.5, label="Cov@50")
axB.set_xticks(xb); axB.set_xticklabels(["New", "1–4", "5–49", "≥50"])
axB.set_ylabel("coverage (%)"); axB.set_xlabel("train signature count")
axB.set_title("Predictability vs history")
axB.annotate("cold-start\n48% of jobs", (0, recb.metacal_cov25[0]), (0.35, 62),
             fontsize=6, color=C["violet"], ha="center",
             arrowprops=dict(arrowstyle="->", color=C["violet"], lw=0.7))
axB.set_ylim(0, 90); axB.yaxis.grid(True, color=C["near_white"], lw=0.7)
axB.legend(loc="upper right", handlelength=1.1, fontsize=6)
tag(axB, "b"); despine(axB)

# ===================== (c) heatmap methods x metrics =====================
rows = list(pred_df.method)
metrics = [("C@25", pred_df.cov25_all, "{:.1f}"), ("C@25\nnew", pred_df.cov25_new, "{:.1f}"),
           ("C@50", pred_df.cov50_all, "{:.1f}"), ("1/\nRMSLE", 1.0/pred_df.rmsle_all, None),
           (r"$\rho$", pred_df.spearman_all, "{:.2f}")]
M = np.zeros((len(rows), len(metrics))); raw = np.empty((len(rows), len(metrics)), object)
for j, (name, col, fmt) in enumerate(metrics):
    v = np.asarray(col, float); M[:, j] = (v - v.min())/(v.max()-v.min()+1e-12)
    for i in range(len(rows)):
        raw[i, j] = (f"{pred_df.rmsle_all[i]:.2f}" if fmt is None else fmt.format(v[i]))
axC.imshow(M, cmap=TEAL_CMAP, aspect="auto", vmin=0, vmax=1)
axC.set_xticks(range(len(metrics))); axC.set_xticklabels([m[0] for m in metrics], fontsize=6)
axC.tick_params(axis="x", pad=2)
axC.set_yticks(range(len(rows)))
axC.set_yticklabels([r.split()[0] if not r.startswith("Meta+") else "M+Cal" for r in rows], fontsize=6.5)
for i in range(len(rows)):
    for j in range(len(metrics)):
        axC.text(j, i, raw[i, j], ha="center", va="center", fontsize=6,
                 color=(INK if M[i, j] < 0.6 else "white"))
m5r = [i for i, r in enumerate(rows) if r.startswith("M5")][0]
axC.add_patch(Rectangle((-0.5, m5r-0.5), len(metrics), 1, fill=False, edgecolor=C["violet"], lw=1.4))
axC.set_title("Prediction metrics", pad=6)
tag(axC, "c")
for s in axC.spines.values(): s.set_visible(False)

# ===================== (d) half-violin + half-box error distribution =====================
seen = sc >= 1; cold = sc == 0
med_seen = np.median(yte[seen])
groups = [("cold\n(new)", cold, C["violet"]),
          ("seen\nsmall", seen & (yte <= med_seen), C["teal_mid"]),
          ("seen\nlarge", seen & (yte > med_seen), C["teal_dark"])]
axD.axhspan(-0.25, 0.25, color=C["near_white"], zorder=0)
axD.axhline(0, color=C["grey_mid"], lw=0.7, ls="--")
print("=== (d) error-distribution group stats: signed (pred-true)/true ===")
print(f"    seen median p* split = {med_seen:.0f}s")
for i, (name, mk, col) in enumerate(groups):
    e = relerr[mk]; ed = np.clip(e, -1, 4)
    n = int(mk.sum()); q1, md, q3 = np.percentile(e, [25, 50, 75])
    print(f"  {name.replace(chr(10),' '):11s} n={n:6d}  median={md:+.3f}  IQR=[{q1:+.3f}, {q3:+.3f}]")
    kde = gaussian_kde(ed); ys = np.linspace(-1, 4, 240); dens = kde(ys); dens = dens/dens.max()*0.38
    axD.fill_betweenx(ys, i, i+dens, color=col, alpha=0.55, lw=0, zorder=2)
    bx = axD.boxplot([np.clip(e, -1, 4)], positions=[i-0.14], widths=0.2, vert=True, patch_artist=True,
                     showfliers=False, manage_ticks=False, zorder=3)
    for b in bx["boxes"]: b.set(facecolor=mcolors.to_rgba(col, 0.5), edgecolor=col, lw=0.8)
    for w_ in bx["whiskers"]+bx["caps"]: w_.set(color=col, lw=0.8)
    for me in bx["medians"]: me.set(color=INK, lw=1.1)
axD.set_xticks(range(len(groups))); axD.set_xticklabels([g[0] for g in groups], fontsize=6.5)
axD.set_xlim(-0.5, len(groups)-0.2)
axD.set_ylabel(r"signed rel. error $(\hat p{-}p^*)/p^*$"); axD.set_ylim(-1.15, 4.15)
axD.set_title("Error spread: cold-start is widest")
axD.text(0.98, 0.05, "band: ±0.25 (Cov@25);  y clipped to [−1,4]", transform=axD.transAxes,
         fontsize=5.6, va="bottom", ha="right", color=C["grey_mid"])
tag(axD, "d"); despine(axD)

# ===================== SCHEDULING ROW: f, g, h (vs FIFO; M5 highlighted) =====================
ORD = ["M1", "M3", "M4", "M5", "M6", "M7"]
schcol = [C["teal_dark"] if k == "M5" else C["teal_mid"] for k in ORD]

# (f) total completion: x lower ΣC than FIFO  (FIFO/method on the SRPT-normalized ratios)
foldF = [FIFO/bfs.loc[k, "prr_ratio_srpt"] for k in ORD]
axF.bar(np.arange(len(ORD)), foldF, 0.62, color=schcol, edgecolor="white", lw=0.5)
for i, v in enumerate(foldF): axF.text(i, v+0.04, f"{v:.1f}", ha="center", fontsize=6.2)
axF.set_xticks(range(len(ORD))); axF.set_xticklabels(ORD, fontsize=6.5)
axF.set_ylabel(r"$\times$ lower than FIFO"); axF.set_ylim(0, max(foldF)*1.18)
axF.set_title(r"Total completion $\Sigma C_j$", fontsize=7.5)
axF.yaxis.grid(True, color=C["near_white"], lw=0.7)
tag(axF, "f"); despine(axF)

# (g) max-stretch: x lower S_max than FIFO (SPRPT)
foldG = [FIFO_SMAX/msd.loc[f"{k} SPRPT", "rho_max"] for k in ORD]
axG.bar(np.arange(len(ORD)), foldG, 0.62, color=schcol, edgecolor="white", lw=0.5)
for i, v in enumerate(foldG): axG.text(i, v+1.5, f"{v:.0f}", ha="center", fontsize=6.2)
axG.set_xticks(range(len(ORD))); axG.set_xticklabels(ORD, fontsize=6.5)
axG.set_ylabel(r"$\times$ lower than FIFO"); axG.set_ylim(0, max(foldG)*1.18)
axG.set_title(r"Max-stretch $S_{\max}$", fontsize=7.5)
axG.yaxis.grid(True, color=C["near_white"], lw=0.7)
tag(axG, "g"); despine(axG)

# (h) makespan: LPPT ratio vs m (lines per method; lower=better, LPT oracle=1)
MLC = {"M1": C["teal_mid"], "M3": C["violet"], "M4": C["teal_bright"],
       "M5": C["teal_dark"], "M6": C["mauve"], "M7": C["soft_blue"]}
for k in ORD:
    sub = mkd[mkd.method == k].sort_values("m")
    axH.plot(sub.m, sub.LPPT_ratio, "o-", color=MLC[k], ms=2.6,
             lw=(2.0 if k == "M5" else 1.0), label=k, zorder=(5 if k == "M5" else 3))
axH.axhline(1.0, color=C["grey_mid"], ls="--", lw=0.8); axH.text(5, 1.03, "LPT=1 (opt)", fontsize=5.4)
axH.set_xscale("log"); axH.set_xticks([5, 10, 20, 50, 100]); axH.set_xticklabels([5, 10, 20, 50, 100], fontsize=6.2)
axH.set_xlabel(r"machines $m$"); axH.set_ylabel(r"makespan $/\,\mathrm{OPT_{pre}}$")
axH.set_title("Makespan (LPPT)", fontsize=7.5)
axH.legend(ncol=2, fontsize=5.0, handlelength=1.0, columnspacing=0.7, loc="upper left")
axH.yaxis.grid(True, color=C["near_white"], lw=0.7)
tag(axH, "h"); despine(axH)

# ===================== (e) 3-stage alluvial Sankey (right half) =====================
axE.set_xlim(0, 1); axE.set_ylim(0, 1); axE.axis("off")
axE.set_title("Where Cov@25 misses come from: recurrence → size → outcome", pad=8, fontsize=8.8)
B = ["New", "1–4", "5–49", "≥50"]
Bmask = {"New": sc == 0, "1–4": (sc >= 1) & (sc <= 4), "5–49": (sc >= 5) & (sc <= 49), "≥50": sc >= 50}
q33, q67 = np.quantile(yte, [1/3, 2/3])
S = ["small", "medium", "large"]
Smask = {"small": yte <= q33, "medium": (yte > q33) & (yte <= q67), "large": yte > q67}
O = ["hit", "miss"]; Omask = {"hit": hit, "miss": ~hit}
Bcol = {"New": C["violet"], "1–4": C["teal_bright"], "5–49": C["teal_mid"], "≥50": C["teal_dark"]}
print(f"=== (e) size terciles of test p*: small<= {q33:.0f}s, medium<= {q67:.0f}s, large> {q67:.0f}s ===")
tri = {}
for b in B:
    for s in S:
        for o in O:
            tri[(b, s, o)] = int((Bmask[b] & Smask[s] & Omask[o]).sum())
assert sum(tri.values()) == NT, f"flow {sum(tri.values())} != {NT}"
print("=== (e) Sankey flow table (recurrence, size, outcome, count) ===")
for k in sorted(tri):
    if tri[k]: print(f"  {k[0]:5s} -> {k[1]:6s} -> {k[2]:4s} : {tri[k]:6d}")
print(f"  TOTAL = {sum(tri.values())} (reconciles to n_test={NT})")

def layout(node_keys, ncount, order_fn, gap=0.04, top=1.0):
    usable = 1 - gap*(len(node_keys)-1); npos, tband = {}, {}; cur = top
    for nk in node_keys:
        h = ncount[nk]/NT*usable; npos[nk] = (cur-h, cur); yc = cur
        for tk in order_fn(nk):
            th = tri[tk]/NT*usable; tband[tk] = (yc-th, yc); yc -= th
        cur -= h + gap
    return npos, tband
c1 = {b: sum(tri[(b, s, o)] for s in S for o in O) for b in B}
c2 = {s: sum(tri[(b, s, o)] for b in B for o in O) for s in S}
c3 = {o: sum(tri[(b, s, o)] for b in B for s in S) for o in O}
pos1, t1 = layout(B, c1, lambda b: [(b, s, o) for s in S for o in O])
pos2, t2 = layout(S, c2, lambda s: [(b, s, o) for b in B for o in O])
pos3, t3 = layout(O, c3, lambda o: [(b, s, o) for b in B for s in S])
x1a, x1b, x2a, x2b, x3a, x3b = 0.11, 0.16, 0.49, 0.54, 0.86, 0.91
def band(x0, x1, y0b, y0t, y1b, y1t, color):
    v = [(x0, y0t), ((x0+x1)/2, y0t), ((x0+x1)/2, y1t), (x1, y1t),
         (x1, y1b), ((x0+x1)/2, y1b), ((x0+x1)/2, y0b), (x0, y0b), (x0, y0t)]
    co = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
          MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    axE.add_patch(PathPatch(MPath(v, co), facecolor=color, edgecolor="none", alpha=0.55, zorder=1))
for tk in tri:
    if not tri[tk]: continue
    col = Bcol[tk[0]]
    band(x1b, x2a, *t1[tk], *t2[tk], col)
    band(x2b, x3a, *t2[tk], *t3[tk], col)
allc = {**c1, **c2, **c3}
def node(x0, x1, posd, labels, side, colmap):
    for nk in posd:
        yb, yt = posd[nk]; cnt = allc[nk]
        axE.add_patch(Rectangle((x0, yb), x1-x0, yt-yb, facecolor=colmap[nk], edgecolor="white", lw=0.6, zorder=4))
        txt = f"{labels[nk]}\n{cnt:,} ({cnt/NT*100:.0f}%)"
        if side == "left":
            axE.text(x0-0.015, (yb+yt)/2, txt, ha="right", va="center", fontsize=6.3)
        elif side == "right":
            axE.text(x1+0.015, (yb+yt)/2, txt, ha="left", va="center", fontsize=6.3)
        else:
            axE.text((x0+x1)/2, (yb+yt)/2, txt, ha="center", va="center", fontsize=5.6,
                     color=INK, zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78))
node(x1a, x1b, pos1, {b: b for b in B}, "left", Bcol)
node(x2a, x2b, pos2, {s: s for s in S}, "mid", {s: C["grey_soft"] for s in S})
node(x3a, x3b, pos3, {"hit": "hit", "miss": "miss"}, "right", {"hit": C["teal_mid"], "miss": C["grey_mid"]})
for xx, lab, ha in [((x1a+x1b)/2, "recurrence", "center"), ((x2a+x2b)/2, "true size", "center"),
                    ((x3a+x3b)/2, "Cov@25", "center")]:
    axE.text(xx, -0.015, lab, fontsize=6.5, color=C["grey_mid"], ha=ha, va="top")
tag(axE, "e")

fig.subplots_adjust(left=0.055, right=0.95, top=0.91, bottom=0.08)
for ext in ("pdf", "svg"): fig.savefig(ROOT / f"composite_main.{ext}")
fig.savefig(ROOT / "composite_main.png", dpi=600)
plt.close(fig)
print("\n[done] wrote composite_main.{pdf,svg,png} to pathb/figures/")
