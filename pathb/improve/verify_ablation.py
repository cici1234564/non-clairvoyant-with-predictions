# Verify the feature-ablation + generalization prose numbers on the CORRECTED
# (I1-intact) split. Plain single LightGBM, all features from the leakage-safe
# feat_df. Read-only (writes nothing committed).
import pickle, numpy as np, pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold

PB = pathlib.Path("pathb")
df = pd.read_pickle(PB/"cache/feat_df.pkl"); sp = np.load(PB/"cache/splits.npz")
itr, iva, ite = sp["idx_tr"], sp["idx_va"], sp["idx_te"]
y = np.log1p(df["p_star"].values); ptrue = df["p_star"].values
for c in ["user", "group", "workload", "gpu_type_spec"]:
    cl = pd.Index(df.iloc[itr][c].dropna().unique().tolist()); mp = {v: i for i, v in enumerate(cl)}
    df[c+"_enc"] = df[c].map(mp).fillna(-1).astype(int)

G_res = ["log_total_plan_cpu","log_total_plan_gpu","log_total_plan_mem","log_total_inst_num","log_num_tasks",
         "cpu_per_inst","gpu_per_inst","mem_per_inst","tasks_per_inst"]
G_tmp = ["hour","dow","sin_hour","cos_hour","is_weekend"]
G_sig = ["sig_mean","sig_median","sig_q25","sig_q75","sig_count","sig_mean_shrink"]
G_grp = ["gro_hist_mean","gro_hist_count","gro_ewm","gro_dt_prev","grp_mean_eb","group_enc","workload_enc","gpu_type_spec_enc"]
G_usr = ["use_hist_mean","use_hist_count","use_ewm","use_dt_prev","user_enc"]

def Xof(cols): return df[cols].replace([np.inf,-np.inf],np.nan).fillna(0).values
def cov(yt, yh, p=0.25): yt = np.maximum(yt,1e-12); return 100*float(np.mean(np.abs(yh-yt)/yt <= p))
def mk(): return LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

print("I1 re-check:", bool(df['submit_time'].values[itr].max() < df['submit_time'].values[ite].min()))
Xall = Xof(G_res+G_tmp+G_sig+G_grp+G_usr)

# ---- feature ablation (plain LGBM, early-stop on val, eval test) ----
def fit_eval(cols):
    X = Xof(cols); m = mk()
    m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return cov(ptrue[ite], np.expm1(m.predict(X[ite])).clip(min=0)), m
s1 = G_res+G_tmp; s2 = s1+G_sig+G_grp; s3 = s2+G_usr
c1,_ = fit_eval(s1); c2,_ = fit_eval(s2); c3, mfull = fit_eval(s3)
print("\n=== FEATURE ABLATION (plain LGBM, test Cov@25) — prose 15.1/21.4/27.9 ===")
print(f"  resources+temporal       ({len(s1):2d} feats): {c1:.1f}")
print(f"  +signature+group         ({len(s2):2d} feats): {c2:.1f}   (+{c2-c1:.1f})")
print(f"  +per-user history (ALL)  ({len(s3):2d} feats): {c3:.1f}   (+{c3-c2:.1f})   <- plain all-feature base")

# ---- within-train 5-fold CV (prose-style: precomputed features) ----
kf = KFold(5, shuffle=True, random_state=42); covs = []
for trf, vaf in kf.split(itr):
    m = mk(); m.set_params(n_estimators=400)
    m.fit(Xall[itr[trf]], y[itr[trf]])
    covs.append(cov(ptrue[itr[vaf]], np.expm1(m.predict(Xall[itr[vaf]])).clip(min=0)))
print(f"\n=== WITHIN-TRAIN 5-fold CV Cov@25 — prose 70.5 ===")
print(f"  mean={np.mean(covs):.1f}  folds={['%.1f'%c for c in covs]}")

# ---- same-era control ----
n = len(itr); cut = int(6/7*n); early, late7 = itr[:cut], itr[cut:]   # itr sorted by time
m = mk(); m.set_params(n_estimators=400); m.fit(Xall[early], y[early])
cov_late = cov(ptrue[late7], np.expm1(m.predict(Xall[late7])).clip(min=0))
cov_test = cov(ptrue[ite],  np.expm1(m.predict(Xall[ite])).clip(min=0))
rng = np.random.RandomState(0); perm = rng.permutation(n); rho, rtr = perm[:n//7], perm[n//7:]
m2 = mk(); m2.set_params(n_estimators=400); m2.fit(Xall[itr[rtr]], y[itr[rtr]])
cov_rand = cov(ptrue[itr[rho]], np.expm1(m2.predict(Xall[itr[rho]])).clip(min=0))
print(f"\n=== SAME-ERA control (train on early train) — prose 62.1/60.1/27.1 ===")
print(f"  random same-era holdout : {cov_rand:.1f}")
print(f"  last-1/7 pseudo-test     : {cov_late:.1f}")
print(f"  real future test         : {cov_test:.1f}")
print(f"  within-era drop = {cov_rand-cov_late:.1f}pp ; far-future drop = {cov_rand-cov_test:.1f}pp")

# ---- seen/unseen-user with plain all-feature model ----
tu = set(df["user"].values[itr]); unseen = np.array([u not in tu for u in df["user"].values[ite]])
yh = np.expm1(mfull.predict(Xall[ite])).clip(min=0)
print(f"\n=== seen/unseen-user (plain LGBM) — prose 28.4/5.9 ===")
print(f"  seen-user Cov@25   = {cov(ptrue[ite][~unseen], yh[~unseen]):.1f}")
print(f"  unseen-user Cov@25 = {cov(ptrue[ite][unseen], yh[unseen]):.1f}  (n={int(unseen.sum())}, {100*unseen.mean():.1f}%)")
print(f"\nReference: my Meta base test Cov@25 = 29.1 ; Meta+Cal = 29.9")
