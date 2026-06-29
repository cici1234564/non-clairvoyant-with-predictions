# ============================================================
# PATH B - DATA + FEATURE BUILD (corrected, leakage-free)
# Ports cell 4 of Prediction+Scheduling.ipynb with the Part-0 fix.
#
# Part 0 fix (index-misalignment bug):
#   The original add_causal_histories_no_leak() did
#       df = df.sort_values("submit_time").reset_index(drop=True)
#   AFTER time_split() had computed idx_tr/va/te on the *pre-sort* df,
#   and then RETURNED that re-indexed df.  prepare_matrices()/scheduling
#   therefore did df.loc[idx_te] on a re-indexed frame -> scrambled set.
#   Fix: sort by r_j (submit_time) ONCE, up front, then compute the split
#   indices on the sorted df; the history function no longer re-sorts.
#
# Performance-only deviation (documented in REPORT.md):
#   The original val/test "frozen stats" step looped row-by-row with scalar
#   df.loc[i, col] assignments (~275k iters). That is replaced here with the
#   numerically-identical vectorized .map() form (the same approach cell 3
#   already uses). Train-side expanding mean/count are computed with the
#   cumsum identity (== expanding().mean().shift(1) / .count().shift(1));
#   ewm uses the same per-group ewm(span=10, adjust=False).mean().shift(1).
# ============================================================
import pathlib, warnings, gc, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)
RANDOM_STATE = 42

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO                      # CSVs extracted at repo root
CACHE = pathlib.Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATA LOADING  (load_tables pointed at DATA_DIR)
# ============================================================
def load_tables():
    job_cols   = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
    task_cols  = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
                  "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
    group_cols = ["inst_id", "user_from_group_table", "gpu_type_spec", "group", "workload"]

    job  = pd.read_csv(DATA_DIR / "pai_job_table.csv",  header=None, names=job_cols,  low_memory=False)
    task = pd.read_csv(DATA_DIR / "pai_task_table.csv", header=None, names=task_cols, low_memory=False)
    gtag = pd.read_csv(DATA_DIR / "pai_group_tag_table.csv", header=None, names=group_cols, low_memory=False)

    job  = job[job.status  == "Terminated"].copy()
    task = task[task.status == "Terminated"].copy()

    for df in (job, task):
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"]   = pd.to_numeric(df["end_time"],   errors="coerce")

    for c in ["plan_cpu", "plan_gpu", "plan_mem", "inst_num"]:
        task[c] = pd.to_numeric(task[c], errors="coerce")

    task["cpu_per_inst"] = task["plan_cpu"] / 100.0
    task["gpu_per_inst"] = task["plan_gpu"] / 100.0
    task["mem_per_inst"] = task["plan_mem"]

    task["cpu_total"] = task["cpu_per_inst"] * task["inst_num"]
    task["gpu_total"] = task["gpu_per_inst"] * task["inst_num"]
    task["mem_total"] = task["mem_per_inst"] * task["inst_num"]

    job_span = (task.groupby("job_name")
                    .agg(min_start=("start_time", "min"),
                         max_end=("end_time", "max"),
                         total_inst_num=("inst_num", "sum"),
                         total_plan_cpu=("cpu_total", "sum"),
                         total_plan_mem=("mem_total", "sum"),
                         total_plan_gpu=("gpu_total", "sum"),
                         num_tasks=("task_name", "nunique"))
                    .reset_index())
    # I7: target span = max_t e_t - min_t s_t (fork-join envelope), NOT max_t d_t
    job_span["p_star"] = (job_span["max_end"] - job_span["min_start"]).clip(lower=0)

    base = job[["job_name", "inst_id", "user", "start_time"]].merge(job_span, on="job_name", how="inner")
    base = base[base["p_star"] > 0].copy()

    base = base.merge(gtag[["inst_id", "group", "workload", "gpu_type_spec"]],
                      on="inst_id", how="left")

    for c in ["group", "workload", "gpu_type_spec", "user"]:
        base[c] = base[c].fillna("Unknown").astype(str)

    base.rename(columns={"start_time": "submit_time"}, inplace=True)
    base = base.dropna(subset=["submit_time"]).copy()

    del job, task, gtag, job_span
    gc.collect()
    return base


# ============================================================
# 2. TIME-BASED SPLIT (computed on the already-sorted df)
# ============================================================
def time_split(df, train_frac=0.70, val_frac=0.15):
    qt = df["submit_time"].quantile([train_frac, train_frac + val_frac]).values
    t_train, t_val = int(qt[0]), int(qt[1])
    idx_tr = df.index[df["submit_time"] < t_train]
    idx_va = df.index[(df["submit_time"] >= t_train) & (df["submit_time"] < t_val)]
    idx_te = df.index[df["submit_time"] >= t_val]
    return (idx_tr, idx_va, idx_te), (t_train, t_val)


# ============================================================
# 3. FEATURE ENGINEERING (LEAKAGE-FREE)
# ============================================================
def add_basic_features(df):
    df = df.copy()
    for c in ["total_plan_cpu", "total_plan_gpu", "total_plan_mem", "total_inst_num", "num_tasks"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    df["cpu_per_inst"]   = df["total_plan_cpu"] / np.maximum(1.0, df["total_inst_num"])
    df["gpu_per_inst"]   = df["total_plan_gpu"] / np.maximum(1.0, df["total_inst_num"])
    df["mem_per_inst"]   = df["total_plan_mem"] / np.maximum(1.0, df["total_inst_num"])
    df["tasks_per_inst"] = df["num_tasks"]      / np.maximum(1.0, df["total_inst_num"])
    hour = ((df["submit_time"] // 3600) % 24).astype(int)
    dow  = ((df["submit_time"] // 86400) % 7).astype(int)
    df["hour"] = hour
    df["dow"]  = dow
    df["sin_hour"]   = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"]   = np.cos(2 * np.pi * hour / 24.0)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def make_signatures(df, idx_tr, global_mean_log):
    df = df.copy()
    train_df = df.loc[idx_tr]

    def qbin(col, q=10):
        edges = np.quantile(train_df[col].clip(lower=1e-6), np.linspace(0, 1, q + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        return np.searchsorted(edges, df[col].values, side="right") - 1

    df["cpu_b"]  = qbin("total_plan_cpu", 10)
    df["gpu_b"]  = qbin("total_plan_gpu", 10)
    df["mem_b"]  = qbin("total_plan_mem", 10)
    df["inst_b"] = qbin("total_inst_num", 10)

    df["task_signature"] = (
        df["user"].astype(str) + "|" + df["group"].astype(str) + "|" +
        df["workload"].astype(str) + "|" + df["cpu_b"].astype(str) + "-" +
        df["gpu_b"].astype(str) + "-" + df["mem_b"].astype(str) + "-" +
        df["inst_b"].astype(str)
    )

    df["p_star_log"] = np.log1p(df["p_star"])
    sig_stats = (df.loc[idx_tr].groupby("task_signature")["p_star_log"]
                   .agg(sig_mean="mean",
                        sig_median="median",
                        sig_std="std",
                        sig_count="count",
                        sig_q25=lambda x: x.quantile(0.25),
                        sig_q75=lambda x: x.quantile(0.75))
                   .reset_index())

    n_s = sig_stats["sig_count"].astype(float)
    mu_s = sig_stats["sig_mean"].astype(float)
    lambda_shrink = 5.0
    sig_stats["sig_mean_shrink"] = (n_s * mu_s + lambda_shrink * global_mean_log) / (n_s + lambda_shrink)

    df = df.merge(sig_stats, on="task_signature", how="left")
    return df


def add_causal_histories_no_leak(df, idx_tr, idx_va, idx_te, global_mean_log):
    """
    Strictly causal histories, vectorized.  Numerically identical to the
    original cell-4 logic (expanding().mean().shift(1), expanding().count()
    .shift(1), ewm(span=10, adjust=False).mean().shift(1) within group;
    val/test use frozen full-train per-entity mean/count, ewm=mean).

    Part-0 fix: NO global re-sort/reset_index here. df is already sorted by
    submit_time up front, so idx_tr/va/te stay aligned to df throughout.
    """
    df = df.copy()

    # init with global prior
    for key, prefix in [("group", "gro"), ("user", "use")]:
        df[f"{prefix}_hist_mean"]  = global_mean_log
        df[f"{prefix}_hist_count"] = 0.0
        df[f"{prefix}_ewm"]        = global_mean_log
        df[f"{prefix}_dt_prev"]    = 0.0

    # train rows are the leading contiguous block (df sorted by submit_time)
    df_tr = df.loc[idx_tr].copy()
    df_tr["p_star_log"] = np.log1p(df_tr["p_star"])

    for key, prefix in [("group", "gro"), ("user", "use")]:
        g = df_tr.groupby(key, sort=False)
        y = df_tr["p_star_log"]

        # expanding mean / count, shifted by one (causal) -- cumsum identity
        csum = g["p_star_log"].cumsum() - y          # sum of strictly-prior rows
        cnt  = g.cumcount().astype(float)            # count of strictly-prior rows
        hist_mean = (csum / cnt.replace(0, np.nan)).fillna(global_mean_log)

        df.loc[df_tr.index, f"{prefix}_hist_mean"]  = hist_mean.values
        df.loc[df_tr.index, f"{prefix}_hist_count"] = cnt.values

        # ewm(span=10), shifted by one, per group (exact)
        ewm = g["p_star_log"].transform(
            lambda s: s.ewm(span=10, adjust=False).mean().shift(1)
        ).fillna(global_mean_log)
        df.loc[df_tr.index, f"{prefix}_ewm"] = ewm.values

        # time since previous submission within group
        dt = g["submit_time"].diff()
        df.loc[df_tr.index, f"{prefix}_dt_prev"] = np.log1p(dt).fillna(0).values

    # frozen full-train per-entity stats for val/test (vectorized .map)
    for key, prefix in [("group", "gro"), ("user", "use")]:
        gstats = df_tr.groupby(key)["p_star_log"].agg(["mean", "count"])
        mean_map, count_map = gstats["mean"], gstats["count"]
        for idx_split in [idx_va, idx_te]:
            ent = df.loc[idx_split, key]
            df.loc[idx_split, f"{prefix}_hist_mean"]  = ent.map(mean_map).fillna(global_mean_log).values
            df.loc[idx_split, f"{prefix}_hist_count"] = ent.map(count_map).fillna(0.0).values
            df.loc[idx_split, f"{prefix}_ewm"]        = ent.map(mean_map).fillna(global_mean_log).values
            # dt_prev stays 0 for val/test
    return df


def engineer_all_features(df, idx_tr, idx_va, idx_te):
    global_mean_log = np.log1p(df.loc[idx_tr, "p_star"]).mean()
    df = add_basic_features(df)
    df = make_signatures(df, idx_tr, global_mean_log)
    df = add_causal_histories_no_leak(df, idx_tr, idx_va, idx_te, global_mean_log)

    n   = df["gro_hist_count"].fillna(0.0)
    mu  = df["gro_hist_mean"].fillna(global_mean_log)
    lam = 5.0
    df["grp_mean_eb"] = (n * mu + lam * global_mean_log) / (n + lam)

    for c in ["sig_mean", "sig_median", "sig_q25", "sig_q75"]:
        if c in df.columns:
            df[c] = df[c].fillna(global_mean_log)
    df["sig_count"] = df["sig_count"].fillna(0)
    df["sig_std"]   = df["sig_std"].fillna(1.0)
    df["is_recurring"] = (df["sig_count"] > 0).astype(int)
    df["global_mean"]  = global_mean_log
    return df, global_mean_log


def build():
    t0 = time.time()
    print("[1] load_tables() <-", DATA_DIR)
    df = load_tables()
    print(f"    loaded rows: {len(df):,}")

    # ---- Part 0 / I1: sort by r_j FIRST (stable), then split on sorted df ----
    df = df.sort_values("submit_time", kind="mergesort").reset_index(drop=True)
    (idx_tr, idx_va, idx_te), (t_train, t_val) = time_split(df, 0.70, 0.15)
    print(f"    splits  train={len(idx_tr):,}  val={len(idx_va):,}  test={len(idx_te):,}")
    print(f"    cuts    t_train={t_train}  t_val={t_val}")

    print("[2] engineer_all_features() ...")
    df, gml = engineer_all_features(df, idx_tr, idx_va, idx_te)
    print(f"    columns: {df.shape[1]}  global_mean_log={gml:.6f}")

    # persist engineered frame + split indices (positional, since index is 0..N-1 sorted)
    df.to_pickle(CACHE / "feat_df.pkl")
    np.savez(CACHE / "splits.npz",
             idx_tr=np.asarray(idx_tr), idx_va=np.asarray(idx_va), idx_te=np.asarray(idx_te),
             t_train=t_train, t_val=t_val, global_mean_log=gml)
    print(f"[3] cached feat_df.parquet + splits.npz  ({time.time()-t0:.1f}s)")

    # ---- quick invariant prints (full detail done in verify.py) ----
    boundary = t_val
    frac = float((df.loc[idx_te, "submit_time"].values >= boundary).mean())
    print(f"[V2] frac(r_j[idx_te] >= t_val) = {frac:.6f}  (require 1.0)")
    print(f"[V1] r_j ranges: "
          f"train[{df.loc[idx_tr,'submit_time'].min():.0f},{df.loc[idx_tr,'submit_time'].max():.0f}] "
          f"val[{df.loc[idx_va,'submit_time'].min():.0f},{df.loc[idx_va,'submit_time'].max():.0f}] "
          f"test[{df.loc[idx_te,'submit_time'].min():.0f},{df.loc[idx_te,'submit_time'].max():.0f}]")
    sc = df.loc[idx_te, "sig_count"].values
    n_te = len(idx_te)
    print(f"[V4] n_test={n_te:,}  "
          f"Rec(>=5)={100*np.mean(sc>=5):.1f}%  New(==0)={100*np.mean(sc==0):.1f}%  "
          f"resid(1-4)={100*np.mean((sc>=1)&(sc<=4)):.1f}%")
    return df


if __name__ == "__main__":
    build()
