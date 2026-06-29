# ============================================================
# TASK 2 - leakage-safe enriched features for cold-start.
# Reuses the VERIFIED temporal split (splits.npz) and raw fields
# from feat_df.pkl. Adds hierarchical TRAIN-ONLY priors at many
# granularities (coarse-signature backoff), each:
#   - TRAIN rows: out-of-fold (K-fold) target mean  (NO self-leakage)
#   - VAL/TEST  : frozen FULL-train mean            (train < val < test => causal)
#   - EB-shrunk toward the global train mean, plus a log-count feature.
# Non-target features (logs/ratios/temporal/encoders/causal shift(1)
# histories) are taken directly. Target y never enters features.
#
# Invariants asserted here: I1 (temporal order), I2/I3 (train-only,
# submit-time only), I4 (causal histories are shift(1), reused as-is).
# ============================================================
import json, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
SEED = 42; K = 5

df = pd.read_pickle(ROOT / "cache/feat_df.pkl")
sp = np.load(ROOT / "cache/splits.npz")
tr = np.asarray(sp["idx_tr"]); va = np.asarray(sp["idx_va"]); te = np.asarray(sp["idx_te"])
t_train = int(sp["t_train"]); t_val = int(sp["t_val"]); gml = float(sp["global_mean_log"])
N = len(df)

# ---- I1: temporal split intact (no re-sort scrambled indices) ----
st = df["submit_time"].values
assert st[tr].max() < st[va].min() < st[te].min(), "I1 FAILED: temporal split not ordered"
assert abs((st[te] >= t_val).mean() - 1.0) < 1e-12, "I1 FAILED: idx_te not all in test window"
print(f"[I1] OK  train.max={st[tr].max():.0f} < val.min={st[va].min():.0f} < test.min={st[te].min():.0f}")

y = df["p_star_log"].values.astype(np.float64)        # = log1p(p_star); TARGET (never a feature)
assert np.allclose(y, np.log1p(df["p_star"].values)), "target def mismatch"

# ---------- helper: OOF-on-train / frozen-full-train EB prior ----------
def eb_prior(key_arr, lam=15.0):
    mean = np.full(N, np.nan); cnt = np.zeros(N)
    ktr = key_arr[tr]; ytr = y[tr]
    # full-train stats (for val/test, and to define folds' complement maps)
    full = pd.DataFrame({"k": ktr, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
    full_mean = full["sum"] / full["count"]
    # out-of-fold for train
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, K, size=len(tr))
    for f in range(K):
        inf = fold == f
        oth = ~inf
        g = pd.DataFrame({"k": ktr[oth], "y": ytr[oth]}).groupby("k")["y"].agg(["sum", "count"])
        gm = g["sum"] / g["count"]
        kk = pd.Series(ktr[inf])
        mean[tr[inf]] = kk.map(gm).values
        cnt[tr[inf]] = kk.map(g["count"]).fillna(0.0).values
    # frozen full-train for val/test
    for sidx in (va, te):
        kk = pd.Series(key_arr[sidx])
        mean[sidx] = kk.map(full_mean).values
        cnt[sidx] = kk.map(full["count"]).fillna(0.0).values
    cnt = np.nan_to_num(cnt, nan=0.0)
    mu = np.where(np.isnan(mean), gml, mean)
    eb = (cnt * mu + lam * gml) / (cnt + lam)
    return eb.astype(np.float32), np.log1p(cnt).astype(np.float32)

def s(col):  # string view of a column
    return df[col].astype(str).values

# ---------- build hierarchical keys (coarse-signature backoff) ----------
u, g, w, gp = s("user"), s("group"), s("workload"), s("gpu_type_spec")
cb, gb, mb, ib = s("cpu_b"), s("gpu_b"), s("mem_b"), s("inst_b")
ntask_b = np.minimum(df["num_tasks"].fillna(0).astype(int).values, 30).astype(str)
keys = {
    "sig8":   s("task_signature"),                 # finest (== existing signature)
    "ugw":    np.char.add(np.char.add(np.char.add(u, "|"), np.char.add(g, "|")), w),
    "gw":     np.char.add(np.char.add(g, "|"), w),
    "uw":     np.char.add(np.char.add(u, "|"), w),
    "wl":     w,
    "usr":    u,
    "grp":    g,
    "gpu":    gp,
    "wlgpu":  np.char.add(np.char.add(w, "|"), gp),
    "instb":  ib, "gpub": gb, "cpub": cb, "memb": mb,
    "ntaskb": ntask_b,
    "scaleb": np.char.add(np.char.add(ib, "-"), gb),
    # 2-way resource-shape priors: generalize to truly-new jobs that lack
    # any user/group/workload history (the hardest cold-start cases).
    "gpub_instb": np.char.add(np.char.add(gb, "-"), ib),
    "gpu_gpub":   np.char.add(np.char.add(gp, "-"), gb),
    "cpub_memb":  np.char.add(np.char.add(cb, "-"), mb),
    "wl_instb":   np.char.add(np.char.add(w, "-"), ib),
}

prior_cols = {}
for name, arr in keys.items():
    eb, lc = eb_prior(arr)
    prior_cols[f"pri_{name}_eb"] = eb
    prior_cols[f"pri_{name}_lcnt"] = lc
print(f"[priors] built {len(keys)} hierarchical OOF priors ({2*len(keys)} features)")

# ---------- direct (non-target) submit-time features ----------
def col(c): return df[c].values.astype(np.float32)
eps = 1e-6
direct = {
    "log_total_plan_cpu": col("log_total_plan_cpu"), "log_total_plan_gpu": col("log_total_plan_gpu"),
    "log_total_plan_mem": col("log_total_plan_mem"), "log_total_inst_num": col("log_total_inst_num"),
    "log_num_tasks": col("log_num_tasks"),
    "cpu_per_inst": col("cpu_per_inst"), "gpu_per_inst": col("gpu_per_inst"),
    "mem_per_inst": col("mem_per_inst"), "tasks_per_inst": col("tasks_per_inst"),
    "hour": col("hour"), "dow": col("dow"), "sin_hour": col("sin_hour"),
    "cos_hour": col("cos_hour"), "is_weekend": col("is_weekend"),
    # new submit-time shape features
    "inst_per_task": (df["total_inst_num"] / np.maximum(1.0, df["num_tasks"])).values.astype(np.float32),
    "log_cpu_per_gpu": np.log1p(df["total_plan_cpu"] / (df["total_plan_gpu"] + eps)).values.astype(np.float32),
    "gpu_intensity": (df["total_plan_gpu"] / (df["total_plan_cpu"] + eps)).values.astype(np.float32),
    "cpu_b": df["cpu_b"].values.astype(np.float32), "gpu_b": df["gpu_b"].values.astype(np.float32),
    "mem_b": df["mem_b"].values.astype(np.float32), "inst_b": df["inst_b"].values.astype(np.float32),
}
# causal shift(1) histories (I4) — reuse verified columns
for c in ["gro_hist_mean", "gro_hist_count", "gro_ewm", "gro_dt_prev",
          "use_hist_mean", "use_hist_count", "use_ewm", "use_dt_prev"]:
    direct[c] = col(c)

# train-only label encoders (unseen -> -1)
for c in ["user", "group", "workload", "gpu_type_spec"]:
    classes = pd.Index(df.iloc[tr][c].dropna().unique().tolist())
    m = {v: i for i, v in enumerate(classes)}
    direct[f"{c}_enc"] = df[c].map(m).fillna(-1).astype(np.int32).values.astype(np.float32)

# ---------- assemble matrix ----------
groups = {
    "direct": list(direct.keys()),
    "priors_mean": [c for c in prior_cols if c.endswith("_eb")],
    "priors_count": [c for c in prior_cols if c.endswith("_lcnt")],
}
feat_names = groups["direct"] + groups["priors_mean"] + groups["priors_count"]
allcols = {**direct, **prior_cols}
X = np.column_stack([np.nan_to_num(allcols[c], nan=0.0, posinf=0.0, neginf=0.0) for c in feat_names]).astype(np.float32)

# ---------- I2/I3 guard: target not in features ----------
assert "p_star" not in feat_names and "p_star_log" not in feat_names and "y_log" not in feat_names
# spot-check no feature equals the target on train rows (no accidental leakage)
for j, name in enumerate(feat_names):
    if np.array_equal(X[tr, j].astype(np.float64), y[tr]):
        raise AssertionError(f"I2/I3 FAILED: feature {name} equals target on train")
print(f"[I2/I3] OK  {len(feat_names)} features, target excluded, none equals y on train")

sig_count = df["sig_count"].values.astype(np.float64)
np.savez(CACHE / "cs_feats.npz",
         Xtr=X[tr], Xva=X[va], Xte=X[te],
         ytr_log=y[tr], yva_log=y[va],
          p_te=df["p_star"].values[te], p_va=df["p_star"].values[va],
         sig_count_va=sig_count[va], sig_count_te=sig_count[te],
         st_va=st[va], st_te=st[te], gml=gml)
json.dump({"feat_names": feat_names, "groups": groups},
          open(CACHE / "cs_meta.json", "w"), indent=1)
print(f"[done] cached cs_feats.npz  X={X.shape}  tr/va/te={len(tr)}/{len(va)}/{len(te)}")
print(f"       feature groups: direct={len(groups['direct'])}, "
      f"priors_mean={len(groups['priors_mean'])}, priors_count={len(groups['priors_count'])}")
# population sizes
print(f"[pops] VAL cold(==0)={int((sig_count[va]==0).sum()):,} seen(>=1)={int((sig_count[va]>=1).sum()):,}")
print(f"[pops] TEST cold(==0)={int((sig_count[te]==0).sum()):,} seen(>=1)={int((sig_count[te]>=1).sum()):,}")
