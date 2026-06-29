# ============================================================
# TASK 2 - finalize the VALIDATION-selected best config and evaluate
# on TEST exactly once. To keep val and test directly comparable (for
# the leakage check), the final model uses the SAME protocol as the
# search: train on TRAIN, early-stop + calibrate on val_es (earliest
# 60% of val), then report held-out val_sel (latest 40% of val) AND
# test side by side. Test is read ONLY here.
# ============================================================
import json, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; ART = HERE / "artifacts"
SEED = 42

d = np.load(CACHE / "cs_feats.npz")
meta = json.load(open(CACHE / "cs_meta.json"))
FEATS = meta["feat_names"]; GROUPS = meta["groups"]
best = json.load(open(ART / "best_config.json")); cfg = best["cfg"]
Xtr, ytr = d["Xtr"], d["ytr_log"]
Xva_all, yva_log = d["Xva"], d["yva_log"]
p_va = d["p_va"]; sc_va = d["sig_count_va"]
Xte = d["Xte"]; p_te = d["p_te"]; sc_te = d["sig_count_te"]
gml = float(d["gml"])

# I1 re-check from saved submit_times
st_va, st_te = d["st_va"], d["st_te"]
assert st_va.max() < st_te.min(), "I1 FAILED at finalize"

nva = len(Xva_all); cut = int(0.60 * nva)
es = slice(0, cut); sel = slice(cut, nva)
yva_es_log = yva_log[es]
p_sel = p_va[sel]; sc_sel = sc_va[sel]

col_idx = {c: i for i, c in enumerate(FEATS)}
cidx = [col_idx[c] for g in cfg["groups"] for c in GROUPS[g]]
Xt = Xtr[:, cidx]; Xe = Xva_all[es][:, cidx]; Xs = Xva_all[sel][:, cidx]; Xte_s = Xte[:, cidx]

def mk(cfg):
    p = dict(n_estimators=2000, random_state=SEED, n_jobs=-1, verbose=-1,
             learning_rate=cfg["lr"], num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child"],
             subsample=cfg["subsample"], subsample_freq=1 if cfg["subsample"] < 1.0 else 0,
             colsample_bytree=cfg["colsample"], reg_alpha=cfg["reg_alpha"],
             reg_lambda=cfg["reg_lambda"], max_depth=cfg["max_depth"])
    if cfg["objective"] == "quantile": p.update(objective="quantile", alpha=0.5)
    elif cfg["objective"] == "huber": p.update(objective="huber", alpha=0.9)
    return LGBMRegressor(**p)

m = mk(cfg)
m.fit(Xt, ytr, eval_set=[(Xe, yva_es_log)],
      callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
pe_log = m.predict(Xe); ps_log = m.predict(Xs); pt_log = m.predict(Xte_s)

def seg_mult(pred_es_lin, y_es_lin, sc_es, pred_t_lin, sc_t):
    grid = np.linspace(0.4, 2.5, 43); out = pred_t_lin.copy()
    for me, mt in [((sc_es == 0), (sc_t == 0)), ((sc_es >= 1), (sc_t >= 1))]:
        if me.sum() < 50: continue
        yt = np.maximum(y_es_lin[me], 1e-12); pr = pred_es_lin[me]
        bc, bv = 1.0, -1
        for c in grid:
            cv = np.mean(np.abs(c * pr - yt) / yt <= 0.25)
            if cv > bv: bv, bc = cv, c
        out[mt] = pred_t_lin[mt] * bc
    return out

def calibrate(cfg, pe_log, p_log, sc_t):
    if cfg["calib"] == "none":
        return np.expm1(p_log).clip(min=0)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    if cfg["calib"] in ("iso", "iso_segmult"):
        iso.fit(pe_log, yva_es_log)
        base = np.expm1(iso.predict(p_log)).clip(min=0)
        if cfg["calib"] == "iso":
            return base
        es_lin = np.expm1(iso.predict(pe_log)).clip(min=0)
        return seg_mult(es_lin, np.expm1(yva_es_log), sc_va[es], base, sc_t)
    if cfg["calib"] == "segmult":
        es_lin = np.expm1(pe_log).clip(min=0); base = np.expm1(p_log).clip(min=0)
        return seg_mult(es_lin, np.expm1(yva_es_log), sc_va[es], base, sc_t)

sel_lin = calibrate(cfg, pe_log, ps_log, sc_sel)
te_lin  = calibrate(cfg, pe_log, pt_log, sc_te)

def cov(yt, yh, pct):
    yt = np.maximum(yt, 1e-12); return 100.0 * float(np.mean(np.abs(yh - yt)/yt <= pct))
def rmsle(yt, yh):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(yt,0))-np.log1p(np.maximum(yh,0)))**2)))
def pops(p_true, yh, sc):
    o = {}
    for nm, mk_ in [("all", np.ones(len(sc), bool)), ("seen", sc >= 1), ("cold", sc == 0)]:
        yt, y = p_true[mk_], yh[mk_]
        o[nm] = dict(n=int(mk_.sum()), cov25=round(cov(yt,y,.25),2), cov50=round(cov(yt,y,.5),2),
                     rmsle=round(rmsle(yt,y),3),
                     rho=round(float(spearmanr(yt,y).correlation),3))
    return o

val_m = pops(p_sel, sel_lin, sc_sel)
test_m = pops(p_te, te_lin, sc_te)

leak_flag = (test_m["seen"]["cov25"] - val_m["seen"]["cov25"]) > 5.0
out = dict(cfg=cfg, best_iter=int(getattr(m, "best_iteration_", 0) or 0),
           val_sel=val_m, test=test_m,
           pathb_anchor=dict(base_all_cov25=29.11, base_new_cov25=12.74,
                             metacal_all_cov25=29.92, metacal_new_cov25=12.74),
           cold_delta_vs_pathb=round(test_m["cold"]["cov25"] - 12.74, 2),
           leakage_flag=bool(leak_flag))
json.dump(out, open(ART / "final_cs.json", "w"), indent=1)

print("="*72)
print("BEST CONFIG (validation-selected):", json.dumps(cfg))
print(f"best_iter={out['best_iter']}")
print("="*72)
hdr = f"{'pop':6s} | {'VAL_sel cov25':>13s} {'TEST cov25':>11s} | {'VAL cov50':>10s} {'TEST cov50':>11s} | {'VAL rmsle':>10s} {'TEST rmsle':>11s}"
print(hdr); print("-"*len(hdr))
for nm in ["all", "seen", "cold"]:
    v, t = val_m[nm], test_m[nm]
    print(f"{nm:6s} | {v['cov25']:13.2f} {t['cov25']:11.2f} | {v['cov50']:10.2f} {t['cov50']:11.2f} | {v['rmsle']:10.3f} {t['rmsle']:11.3f}  (n_val={v['n']}, n_test={t['n']})")
print("="*72)
print(f"Cold-start TEST Cov@25 = {test_m['cold']['cov25']:.2f}  vs Path B New 12.74  "
      f"=> delta {out['cold_delta_vs_pathb']:+.2f} pp")
print(f"Leakage check (test_seen - val_seen Cov@25) = "
      f"{test_m['seen']['cov25']-val_m['seen']['cov25']:+.2f} pp  -> "
      f"{'FLAG (>5pp, investigate)' if leak_flag else 'OK'}")
print(f"Ceiling check: cold TEST Cov@25 {'>30 LEAKAGE STOP' if test_m['cold']['cov25']>30 else 'within expected (<30)'}")
