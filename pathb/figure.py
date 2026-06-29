# ============================================================
# PATH B - PART B : recurrence figure (panel b).
# Cov@25 & Cov@50 of Meta+Cal by TRAINING-recurrence bucket
# {New=0, 1-4, 5-49, >=50}. Teal / CDF house style.
# Outputs fig_recurrence.{pdf,png,svg}.
# ============================================================
import pickle, pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE / "artifacts"
REPO = HERE.parent

# teal house palette
TEAL_DARK = "#0f6e6e"
TEAL = "#1aa3a3"
TEAL_LIGHT = "#7fd4d4"
GRID = "#d9e6e6"

BUCKETS = [("New", lambda c: c == 0),
           ("1-4", lambda c: (c >= 1) & (c <= 4)),
           ("5-49", lambda c: (c >= 5) & (c <= 49)),
           (">=50", lambda c: c >= 50)]


def cov(y, yhat, pct):
    y = np.maximum(y, 1e-12)
    return 100.0 * float(np.mean(np.abs(yhat - y) / y <= pct))


def main():
    with open(ART / "predictions.pkl", "rb") as f:
        P = pickle.load(f)
    yte = np.asarray(P["yte"], dtype=float)
    metacal = np.asarray(P["metacal_raw"], dtype=float)
    sc = np.asarray(P["sig_count_te"], dtype=float)

    labels, c25, c50, ns = [], [], [], []
    for name, fn in BUCKETS:
        m = fn(sc)
        n = int(m.sum())
        labels.append(name); ns.append(n)
        c25.append(cov(yte[m], metacal[m], 0.25) if n else float("nan"))
        c50.append(cov(yte[m], metacal[m], 0.50) if n else float("nan"))

    bucket_tbl = [dict(bucket=labels[i], n=ns[i], cov25=c25[i], cov50=c50[i]) for i in range(len(labels))]
    with open(ART / "recurrence_buckets.json", "w") as f:
        json.dump(bucket_tbl, f, indent=2)
    for r in bucket_tbl:
        print(f"  bucket {r['bucket']:>5s}  n={r['n']:>7,}  Cov@25={r['cov25']:5.2f}  Cov@50={r['cov50']:5.2f}")

    # ---- figure: grouped bars, teal/CDF house style ----
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#33514f",
                         "axes.linewidth": 1.0, "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(labels)); w = 0.38
    b1 = ax.bar(x - w / 2, c25, w, label="Cov@25%", color=TEAL_DARK, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, c50, w, label="Cov@50%", color=TEAL_LIGHT, edgecolor="white", zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9, color="#234")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, ns)])
    ax.set_xlabel("Training-recurrence bucket (signature count in training)")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, max([v for v in c50 if np.isfinite(v)] + [1]) * 1.18)
    ax.set_title("Meta+Cal coverage by training recurrence (panel b)", color=TEAL_DARK)
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(REPO / f"fig_recurrence.{ext}", bbox_inches="tight")
    print("[done] wrote fig_recurrence.{pdf,png,svg} and recurrence_buckets.json")


if __name__ == "__main__":
    main()
