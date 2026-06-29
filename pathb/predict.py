# ============================================================
# PATH B - PART A : prediction methods, Meta base, determinism
# gate, persistence, and principled isotonic calibration.
#
# Loads the cached corrected feature frame (build_data.py).
# Methods are ported verbatim from cell 4 of the notebook; the only
# additions are (i) an `extra` dict threaded into every LightGBM model
# so the determinism gate can flip on deterministic settings, and
# (ii) method_meta_stack also returns the base's val/test LOG preds.
# ============================================================
import json, pathlib, time, pickle, warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")
np.random.seed(42)
RANDOM_STATE = 42

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# determinism knobs flipped on by the A.2 gate when needed
LGB_EXTRA = {}


# ---------------- metrics ----------------
def rmsle(y, yhat):
    y = np.maximum(y, 0); yhat = np.maximum(yhat, 0)
    return float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(yhat))))

def coverage_at(y, yhat, pct=0.25):
    y = np.maximum(y, 1e-12)
    rel = np.abs(yhat - y) / y
    return 100.0 * float(np.mean(rel <= pct))

def metrics(y, yhat):
    return dict(
        MAE=float(mean_absolute_error(y, yhat)),
        RMSLE=rmsle(y, yhat),
        Cov25=coverage_at(y, yhat, 0.25),
        Cov50=coverage_at(y, yhat, 0.50),
        Spearman=float(spearmanr(y, yhat).correlation) if np.std(y) > 0 and np.std(yhat) > 0 else float("nan"),
    )


# ---------------- lgbm helper ----------------
def fit_lgbm(params, Xtr, ytr_log, Xva=None, yva_log=None, n_estimators=600, es=50):
    p = dict(params); p.update(LGB_EXTRA)
    model = LGBMRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, **p)
    if Xva is not None and yva_log is not None:
        model.fit(Xtr, ytr_log, eval_set=[(Xva, yva_log)],
                  callbacks=[early_stopping(es, verbose=False), log_evaluation(0)])
    else:
        model.fit(Xtr, ytr_log)
    return model


# ---------------- matrices ----------------
def prepare_matrices(df, idx_tr, idx_va, idx_te):
    NUM_FEATS = [
        "log_total_plan_cpu", "log_total_plan_gpu", "log_total_plan_mem",
        "log_total_inst_num", "log_num_tasks",
        "cpu_per_inst", "gpu_per_inst", "mem_per_inst", "tasks_per_inst",
        "hour", "dow", "sin_hour", "cos_hour", "is_weekend",
        "sig_mean", "sig_median", "sig_q25", "sig_q75", "sig_count", "sig_mean_shrink",
        "gro_hist_mean", "gro_hist_count", "gro_ewm", "gro_dt_prev",
        "use_hist_mean", "use_hist_count", "use_ewm", "use_dt_prev",
        "grp_mean_eb",
    ]
    NUM_FEATS = [f for f in NUM_FEATS if f in df.columns]

    def fit_label_encoder_from_train(train_series):
        classes = pd.Index(train_series.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(classes)}
        return lambda s: s.map(mapping).fillna(-1).astype(int)

    for c in ["user", "group", "workload", "gpu_type_spec"]:
        enc = fit_label_encoder_from_train(df.loc[idx_tr, c])
        df[f"{c}_enc"] = enc(df[c])
        NUM_FEATS.append(f"{c}_enc")

    Xtr = df.loc[idx_tr, NUM_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).values
    Xva = df.loc[idx_va, NUM_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).values
    Xte = df.loc[idx_te, NUM_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).values
    ytr_log = np.log1p(df.loc[idx_tr, "p_star"].values)
    yva_log = np.log1p(df.loc[idx_va, "p_star"].values)
    yte     = df.loc[idx_te, "p_star"].values
    return NUM_FEATS, Xtr, Xva, Xte, ytr_log, yva_log, yte


# ---------------- methods (ported from cell 4) ----------------
def method_cqr(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te):
    base = fit_lgbm(dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8),
                    Xtr, ytr_log, Xva, yva_log)
    base_va, base_te = base.predict(Xva), base.predict(Xte)
    qL = fit_lgbm(dict(objective="quantile", alpha=0.1, learning_rate=0.05, num_leaves=63),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400, es=40)
    qM = fit_lgbm(dict(objective="quantile", alpha=0.5, learning_rate=0.05, num_leaves=63),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400, es=40)
    qU = fit_lgbm(dict(objective="quantile", alpha=0.9, learning_rate=0.05, num_leaves=63),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400, es=40)
    qL_va, qM_va, qU_va = qL.predict(Xva), qM.predict(Xva), qU.predict(Xva)
    qL_te, qM_te, qU_te = qL.predict(Xte), qM.predict(Xte), qU.predict(Xte)
    r = np.maximum(qL_va - yva_log, yva_log - qU_va)
    k = np.quantile(np.maximum(r, 0.0), 0.60)
    L_va, U_va = qL_va - k, qU_va + k
    L_te, U_te = qL_te - k, qU_te + k
    qM_clip_va = np.clip(qM_va, L_va, U_va)
    Z_va = np.c_[L_va, qM_va, U_va, qM_clip_va, base_va,
                 df.loc[idx_va, "grp_mean_eb"].values,
                 df.loc[idx_va, "sig_median"].fillna(df.loc[idx_va, "grp_mean_eb"]).values]
    ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_STATE)
    ridge.fit(Z_va, yva_log)
    qM_clip_te = np.clip(qM_te, L_te, U_te)
    Z_te = np.c_[L_te, qM_te, U_te, qM_clip_te, base_te,
                 df.loc[idx_te, "grp_mean_eb"].values,
                 df.loc[idx_te, "sig_median"].fillna(df.loc[idx_te, "grp_mean_eb"]).values]
    return np.expm1(ridge.predict(Z_te)).clip(min=0)


def method_hras(df, idx_te):
    pred_log = (df.loc[idx_te, "sig_mean_shrink"]
                .fillna(df.loc[idx_te, "grp_mean_eb"])
                .fillna(df.loc[idx_te, "global_mean"]).values)
    return np.expm1(pred_log).clip(min=0)


def method_isotonic(Xtr, Xva, Xte, ytr_log, yva_log):
    qmed = fit_lgbm(dict(objective="quantile", alpha=0.5, learning_rate=0.05, num_leaves=63),
                    Xtr, ytr_log, Xva, yva_log, n_estimators=800, es=60)
    pred_va, pred_te = qmed.predict(Xva), qmed.predict(Xte)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_va, yva_log)
    return np.expm1(iso.predict(pred_te)).clip(min=0)


def method_meta_stack(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te, return_logs=False):
    base_models = []
    m1 = fit_lgbm(dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8),
                  Xtr, ytr_log, Xva, yva_log); base_models.append(m1)
    m2 = fit_lgbm(dict(learning_rate=0.03, num_leaves=127, min_child_samples=50,
                       reg_alpha=0.5, reg_lambda=1.0, subsample=0.7),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400); base_models.append(m2)
    m3 = fit_lgbm(dict(objective="quantile", alpha=0.5, learning_rate=0.05, num_leaves=63),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400); base_models.append(m3)
    m4 = fit_lgbm(dict(objective="huber", alpha=0.9, learning_rate=0.05, num_leaves=63),
                  Xtr, ytr_log, Xva, yva_log, n_estimators=400); base_models.append(m4)

    va_preds = np.column_stack([m.predict(Xva) for m in base_models])
    te_preds = np.column_stack([m.predict(Xte) for m in base_models])
    va_std = np.std(va_preds, axis=1).reshape(-1, 1)
    te_std = np.std(te_preds, axis=1).reshape(-1, 1)
    va_ctx = np.column_stack([df.loc[idx_va, "grp_mean_eb"].values,
                              df.loc[idx_va, "sig_median"].fillna(df.loc[idx_va, "grp_mean_eb"]).values])
    te_ctx = np.column_stack([df.loc[idx_te, "grp_mean_eb"].values,
                              df.loc[idx_te, "sig_median"].fillna(df.loc[idx_te, "grp_mean_eb"]).values])
    Z_va = np.hstack([va_preds, va_std, va_ctx])
    Z_te = np.hstack([te_preds, te_std, te_ctx])
    meta = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         min_child_samples=30, random_state=RANDOM_STATE, **LGB_EXTRA)
    meta.fit(Z_va, yva_log)
    log_va = meta.predict(Z_va)
    log_te = meta.predict(Z_te)
    yhat = np.expm1(log_te).clip(min=0)
    if return_logs:
        return yhat, log_va, log_te
    return yhat


def method_two_stage(Xtr, Xva, Xte, ytr_log, yva_log):
    p30, p70 = np.percentile(ytr_log, [30, 70])
    y_class = np.zeros_like(ytr_log, dtype=int)
    y_class[ytr_log < p30] = 0
    y_class[(ytr_log >= p30) & (ytr_log < p70)] = 1
    y_class[ytr_log >= p70] = 2
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, random_state=RANDOM_STATE, **LGB_EXTRA)
    clf.fit(Xtr, y_class)
    probs_te = clf.predict_proba(Xte)
    experts = {}
    for c in range(3):
        mask = (y_class == c)
        if mask.sum() > 100:
            expert = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=20, random_state=RANDOM_STATE, **LGB_EXTRA)
            expert.fit(Xtr[mask], ytr_log[mask])
            experts[c] = expert
    fallback = fit_lgbm(dict(learning_rate=0.05, num_leaves=63), Xtr, ytr_log, n_estimators=400)
    yhat_log = np.zeros(len(Xte))
    for c in range(3):
        model = experts.get(c, fallback)
        yhat_log += probs_te[:, c] * model.predict(Xte)
    return np.expm1(yhat_log).clip(min=0)


def method_recency(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_tr):
    times_tr = df.loc[idx_tr, "submit_time"].values
    tmax, tmin = times_tr.max(), times_tr.min()
    models = []
    time_w = np.exp(-0.5 * (tmax - times_tr) / (tmax - tmin + 1.0))
    m1 = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                       min_child_samples=20, random_state=RANDOM_STATE, **LGB_EXTRA)
    m1.fit(Xtr, ytr_log, sample_weight=time_w); models.append(("full", m1))
    cut50 = np.percentile(times_tr, 50); mask50 = times_tr >= cut50
    if mask50.sum() > 1000:
        m2 = fit_lgbm(dict(learning_rate=0.06, num_leaves=63), Xtr[mask50], ytr_log[mask50], n_estimators=300)
        models.append(("recent_50", m2))
    cut20 = np.percentile(times_tr, 80); mask20 = times_tr >= cut20
    if mask20.sum() > 500:
        m3 = fit_lgbm(dict(learning_rate=0.08, num_leaves=31), Xtr[mask20], ytr_log[mask20], n_estimators=200)
        models.append(("recent_20", m3))
    errs = []
    for name, model in models:
        errs.append(np.mean(np.abs(model.predict(Xva) - yva_log)))
    errs = np.array(errs); w = 1.0 / (errs + 1e-2); w = w / w.sum()
    yhat_log = np.zeros(len(Xte))
    for i, (name, model) in enumerate(models):
        yhat_log += w[i] * model.predict(Xte)
    return np.expm1(yhat_log).clip(min=0)


def slice_metrics(yte, yhat, rec_mask, new_mask):
    return dict(All=metrics(yte, yhat),
                Rec=metrics(yte[rec_mask], yhat[rec_mask]),
                New=metrics(yte[new_mask], yhat[new_mask]))


def main():
    global LGB_EXTRA
    t0 = time.time()
    df = pd.read_pickle(CACHE / "feat_df.pkl")
    sp = np.load(CACHE / "splits.npz")
    idx_tr = pd.Index(sp["idx_tr"]); idx_va = pd.Index(sp["idx_va"]); idx_te = pd.Index(sp["idx_te"])
    gml = float(sp["global_mean_log"])
    print(f"[load] df={df.shape}  tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")

    NUM_FEATS, Xtr, Xva, Xte, ytr_log, yva_log, yte = prepare_matrices(df, idx_tr, idx_va, idx_te)
    print(f"[features] #={len(NUM_FEATS)}")

    # recurrence masks (TRAIN signature count): Rec=>=5, New==0
    sig_count_te = df.loc[idx_te, "sig_count"].values
    rec_mask = (sig_count_te >= 5)
    new_mask = (sig_count_te == 0)

    # ---------- A.2 determinism gate on the Meta base ----------
    print("\n[A.2] determinism gate on Meta base (existing config) ...")
    LGB_EXTRA = {}
    r1, lv1, lt1 = method_meta_stack(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te, return_logs=True)
    r2, lv2, lt2 = method_meta_stack(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te, return_logs=True)
    cov1 = coverage_at(yte, r1, 0.25); cov2 = coverage_at(yte, r2, 0.25)
    identical = bool(np.array_equal(r1, r2))
    print(f"      run1 Cov@25={cov1:.4f}  run2 Cov@25={cov2:.4f}  "
          f"cov_equal={cov1==cov2}  array_equal={identical}")

    gate = dict(existing_run1_cov25=cov1, existing_run2_cov25=cov2,
                existing_cov_equal=bool(cov1 == cov2), existing_array_equal=identical)

    if cov1 == cov2 and identical:
        decision = "deterministic_existing"
        LGB_EXTRA = {}
        base_yhat, base_log_va, base_log_te = r1, lv1, lt1
        base_cov25 = cov1
        print("      -> existing config is DETERMINISTIC (seeded). Base reproducible.")
    else:
        decision = "rebaselined_deterministic"
        print("      -> NON-deterministic. Setting deterministic seed/config and re-baselining.")
        np.random.seed(42)
        LGB_EXTRA = dict(deterministic=True, force_row_wise=True, num_threads=1, bagging_seed=42,
                         feature_fraction_seed=42, data_random_seed=42)
        d1, dlv1, dlt1 = method_meta_stack(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te, return_logs=True)
        d2, dlv2, dlt2 = method_meta_stack(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te, return_logs=True)
        dcov1 = coverage_at(yte, d1, 0.25); dcov2 = coverage_at(yte, d2, 0.25)
        det_identical = bool(np.array_equal(d1, d2))
        print(f"      det run1 Cov@25={dcov1:.4f}  det run2 Cov@25={dcov2:.4f}  array_equal={det_identical}")
        gate.update(det_run1_cov25=dcov1, det_run2_cov25=dcov2, det_array_equal=det_identical)
        assert det_identical and dcov1 == dcov2, "deterministic config still not reproducible!"
        base_yhat, base_log_va, base_log_te = d1, dlv1, dlt1
        base_cov25 = dcov1

    gate["decision"] = decision
    gate["base_cov25_X"] = float(base_cov25)
    gate["reproduces_33_4"] = bool(abs(base_cov25 - 33.4) <= 0.5)
    print(f"[A.1] Meta base test Cov@25 = X = {base_cov25:.4f}  (old anchor 33.4 -> "
          f"{'reproduced' if gate['reproduces_33_4'] else 'NOT reproduced'})")

    # ---------- run all predictor methods under the adopted config ----------
    print("\n[methods] running all under adopted config:", LGB_EXTRA if LGB_EXTRA else "(existing)")
    predictions = {}
    predictions["M1"] = method_cqr(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_va, idx_te)
    predictions["M3"] = method_hras(df, idx_te)
    predictions["M4"] = method_isotonic(Xtr, Xva, Xte, ytr_log, yva_log)
    predictions["M5"] = base_yhat                       # Meta base == adopted base
    predictions["M6"] = method_two_stage(Xtr, Xva, Xte, ytr_log, yva_log)
    predictions["M7"] = method_recency(Xtr, Xva, Xte, ytr_log, yva_log, df, idx_tr)

    name_map = {"M1": "CQR", "M3": "HRAS", "M4": "Isotonic", "M5": "Meta",
                "M6": "TwoStage", "M7": "Recency"}
    results = {}
    for k, yhat in predictions.items():
        results[k] = dict(name=name_map[k], **slice_metrics(yte, yhat, rec_mask, new_mask))
        m = results[k]["All"]
        print(f"  [{k} {name_map[k]:9s}] All Cov@25={m['Cov25']:5.2f}  Cov@50={m['Cov50']:5.2f}  "
              f"RMSLE={m['RMSLE']:.3f}  rho={m['Spearman']:.3f}")

    # ---------- A.4 principled isotonic calibration (val only) ----------
    print("\n[A.4] isotonic calibration (increasing, clip) fit on VALIDATION only ...")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(base_log_va, yva_log)               # predicted log-span -> true log-span, VAL only
    cal_log_te = iso.predict(base_log_te)
    metacal_raw = np.expm1(cal_log_te).clip(min=0)
    metacal = slice_metrics(yte, metacal_raw, rec_mask, new_mask)
    base_metrics = results["M5"]
    rho_rank = float(spearmanr(base_yhat, metacal_raw).correlation)   # V6 monotonicity
    print(f"  Meta+Cal  All Cov@25={metacal['All']['Cov25']:.2f} (base {base_metrics['All']['Cov25']:.2f}) "
          f"Cov@50={metacal['All']['Cov50']:.2f}  RMSLE={metacal['All']['RMSLE']:.3f}  "
          f"rho={metacal['All']['Spearman']:.3f}")
    print(f"  Rec Cov@25={metacal['Rec']['Cov25']:.2f}  New Cov@25={metacal['New']['Cov25']:.2f}  "
          f"Rec Cov@50={metacal['Rec']['Cov50']:.2f}")
    print(f"  Spearman(base, Meta+Cal) on test = {rho_rank:.6f}  (V6 monotone => ~1.0)")

    # ---------- persistence ----------
    with open(ART / "base_preds.pkl", "wb") as f:
        pickle.dump(dict(base_log_va=base_log_va, base_log_te=base_log_te,
                         yva_log=yva_log, yte=yte, base_raw_te=base_yhat,
                         adopted_extra=LGB_EXTRA, decision=decision), f)
    with open(ART / "predictions.pkl", "wb") as f:
        pickle.dump(dict(predictions=predictions, metacal_raw=metacal_raw,
                         sig_count_te=sig_count_te, yte=yte,
                         rec_mask=rec_mask, new_mask=new_mask), f)

    out = dict(
        n_test=int(len(idx_te)),
        gate=gate, adopted_extra=LGB_EXTRA, decision=decision,
        base_cov25_X=float(base_cov25),
        results={k: results[k] for k in predictions},
        metacal=metacal, base_all_cov25=float(base_metrics["All"]["Cov25"]),
        metacal_gain_pp=float(metacal["All"]["Cov25"] - base_metrics["All"]["Cov25"]),
        spearman_base_vs_metacal=rho_rank,
        calibration_spec=("IsotonicRegression(increasing=True, out_of_bounds='clip') of true "
                          "log-span on Meta base predicted log-span, fit on VALIDATION only, "
                          "applied to test."),
    )
    with open(ART / "results_pred.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] persisted base_preds.pkl, predictions.pkl, results_pred.json  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
