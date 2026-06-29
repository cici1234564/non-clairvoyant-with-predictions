# ============================================================
# TASK 2 - validation-only iterative search for cold-start.
# STRICT: test is NEVER read here. Model/feature/hparam/calibration
# decisions use ONLY validation, and within validation we hold out a
# temporal slice for honest selection:
#   val_es  = earliest 60% of val  -> LGBM early stopping + calibrator fit
#   val_sel = latest   40% of val  -> selection metric (held out)
# Each trial logs metrics by population {all, seen(count>=1), cold(count==0)}
# on val_sel. Checkpoints every trial to trials.jsonl + best_config.json.
# ============================================================
import sys, json, time, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; ART = HERE / "artifacts"; ART.mkdir(parents=True, exist_ok=True)
SEED = 42

TIME_BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 5400.0   # seconds
MAX_TRIALS  = int(sys.argv[2]) if len(sys.argv) > 2 else 200

d = np.load(CACHE / "cs_feats.npz")
meta = json.load(open(CACHE / "cs_meta.json"))
FEATS = meta["feat_names"]; GROUPS = meta["groups"]
Xtr, ytr = d["Xtr"], d["ytr_log"]
Xva_all, yva_log = d["Xva"], d["yva_log"]
p_va = d["p_va"]; sc_va = d["sig_count_va"]
gml = float(d["gml"])

# val temporal sub-split (Xva is in submit_time order: idx_va is a contiguous block)
nva = len(Xva_all); cut = int(0.60 * nva)
es = slice(0, cut); sel = slice(cut, nva)
yva_es_log = yva_log[es]; yva_sel_log = yva_log[sel]
p_sel = p_va[sel]; sc_sel = sc_va[sel]
print(f"[val split] es={cut:,} (early-stop+calib)  sel={nva-cut:,} (held-out selection)")
print(f"[val_sel pops] cold(==0)={int((sc_sel==0).sum()):,} seen(>=1)={int((sc_sel>=1).sum()):,}")

col_idx = {c: i for i, c in enumerate(FEATS)}
def cols(group_set):
    names = []
    for gname in group_set:
        names += GROUPS[gname]
    return [col_idx[c] for c in names], names

def cov(yt, yh, pct):
    yt = np.maximum(yt, 1e-12)
    return 100.0 * float(np.mean(np.abs(yh - yt) / yt <= pct))
def rmsle(yt, yh):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(yt,0)) - np.log1p(np.maximum(yh,0)))**2)))
def by_pop(p_true, yhat_lin, sc):
    out = {}
    for nm, m in [("all", np.ones(len(sc), bool)), ("seen", sc >= 1), ("cold", sc == 0)]:
        yt, yh = p_true[m], yhat_lin[m]
        out[nm] = dict(cov25=cov(yt, yh, 0.25), cov50=cov(yt, yh, 0.50),
                       rmsle=rmsle(yt, yh),
                       rho=float(spearmanr(yt, yh).correlation) if len(yt) > 2 else float("nan"),
                       n=int(m.sum()))
    return out

def seg_mult(pred_es_lin, y_es_lin, sc_es, pred_target_lin, sc_target):
    """Per-segment (cold/seen) multiplicative scale chosen on val_es to max Cov@25."""
    grid = np.linspace(0.4, 2.5, 43)
    out = pred_target_lin.copy()
    for seg_mask_es, seg_mask_t in [((sc_es == 0), (sc_target == 0)), ((sc_es >= 1), (sc_target >= 1))]:
        if seg_mask_es.sum() < 50:
            continue
        yt = np.maximum(y_es_lin[seg_mask_es], 1e-12); pr = pred_es_lin[seg_mask_es]
        best_c, best_cov = 1.0, -1
        for c in grid:
            cv = np.mean(np.abs(c * pr - yt) / yt <= 0.25)
            if cv > best_cov:
                best_cov, best_c = cv, c
        out[seg_mask_t] = pred_target_lin[seg_mask_t] * best_c
    return out

def make_model(cfg):
    p = dict(n_estimators=2000, random_state=SEED, n_jobs=-1, verbose=-1,
             learning_rate=cfg["lr"], num_leaves=cfg["num_leaves"],
             min_child_samples=cfg["min_child"], subsample=cfg["subsample"],
             subsample_freq=1 if cfg["subsample"] < 1.0 else 0,
             colsample_bytree=cfg["colsample"], reg_alpha=cfg["reg_alpha"],
             reg_lambda=cfg["reg_lambda"], max_depth=cfg["max_depth"])
    if cfg["objective"] == "quantile":
        p.update(objective="quantile", alpha=0.5)
    elif cfg["objective"] == "huber":
        p.update(objective="huber", alpha=0.9)
    return LGBMRegressor(**p)

def run_trial(cfg):
    cidx, _ = cols(cfg["groups"])
    Xt = Xtr[:, cidx]; Xe = Xva_all[es][:, cidx]; Xs = Xva_all[sel][:, cidx]
    m = make_model(cfg)
    m.fit(Xt, ytr, eval_set=[(Xe, yva_es_log)],
          callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    pred_es_log = m.predict(Xe); pred_sel_log = m.predict(Xs)
    # calibration variants (fit on val_es only)
    if cfg["calib"] == "none":
        sel_lin = np.expm1(pred_sel_log).clip(min=0)
    elif cfg["calib"] == "iso":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(pred_es_log, yva_es_log)
        sel_lin = np.expm1(iso.predict(pred_sel_log)).clip(min=0)
    elif cfg["calib"] == "iso_segmult":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(pred_es_log, yva_es_log)
        es_lin = np.expm1(iso.predict(pred_es_log)).clip(min=0)
        sel_lin0 = np.expm1(iso.predict(pred_sel_log)).clip(min=0)
        sel_lin = seg_mult(es_lin, np.expm1(yva_es_log), sc_va[es], sel_lin0, sc_sel)
    elif cfg["calib"] == "segmult":
        es_lin = np.expm1(pred_es_log).clip(min=0); sel_lin0 = np.expm1(pred_sel_log).clip(min=0)
        sel_lin = seg_mult(es_lin, np.expm1(yva_es_log), sc_va[es], sel_lin0, sc_sel)
    metr = by_pop(p_sel, sel_lin, sc_sel)
    metr["best_iter"] = int(getattr(m, "best_iteration_", 0) or 0)
    return metr

def score(metr):
    # primary: cold Cov@25; guard against all-collapse (must stay near Path B base)
    cold = metr["cold"]["cov25"]; allc = metr["all"]["cov25"]
    return cold if allc >= 26.0 else cold - 100.0

# ---- search space ----
import random
rng = random.Random(SEED)
SP = dict(num_leaves=[31,63,127,255], min_child=[20,50,100,200,400], lr=[0.03,0.05,0.07],
          subsample=[0.7,0.8,1.0], colsample=[0.7,0.8,1.0], reg_alpha=[0,0.5,1.0,2.0],
          reg_lambda=[0,0.5,1.0,2.0], max_depth=[-1,8,12,16],
          objective=["l2","quantile","huber"], calib=["none","iso","iso_segmult","segmult"],
          groups=[["direct","priors_mean","priors_count"], ["direct","priors_mean"], ["direct"]])
def rand_cfg():
    return {k: rng.choice(v) for k, v in SP.items()}

# strong hand-seeded configs (feature-rich, calibration on) tried first
seeds = [
    dict(num_leaves=127,min_child=100,lr=0.05,subsample=0.8,colsample=0.8,reg_alpha=0.5,reg_lambda=1.0,
         max_depth=-1,objective="l2",calib="iso_segmult",groups=["direct","priors_mean","priors_count"]),
    dict(num_leaves=63,min_child=50,lr=0.05,subsample=0.8,colsample=0.8,reg_alpha=0.0,reg_lambda=1.0,
         max_depth=-1,objective="quantile",calib="iso_segmult",groups=["direct","priors_mean","priors_count"]),
    dict(num_leaves=255,min_child=200,lr=0.03,subsample=0.8,colsample=0.7,reg_alpha=1.0,reg_lambda=2.0,
         max_depth=12,objective="l2",calib="iso",groups=["direct","priors_mean","priors_count"]),
    dict(num_leaves=63,min_child=50,lr=0.05,subsample=1.0,colsample=1.0,reg_alpha=0.0,reg_lambda=0.0,
         max_depth=-1,objective="l2",calib="none",groups=["direct"]),  # ~baseline-ish (no priors)
]

t0 = time.time()
best = None; trials = []
log = open(ART / "trials.jsonl", "w")
i = 0
while i < MAX_TRIALS and (time.time() - t0) < TIME_BUDGET:
    cfg = seeds[i] if i < len(seeds) else rand_cfg()
    try:
        metr = run_trial(cfg)
    except Exception as e:
        log.write(json.dumps({"i": i, "cfg": cfg, "error": str(e)[:200]}) + "\n"); log.flush(); i += 1; continue
    sc = score(metr)
    rec = {"i": i, "score": sc, "cfg": cfg,
           "cold": metr["cold"], "seen": metr["seen"], "all": metr["all"], "best_iter": metr["best_iter"]}
    log.write(json.dumps(rec) + "\n"); log.flush()
    trials.append(rec)
    if best is None or sc > best["score"]:
        best = rec
        json.dump(best, open(ART / "best_config.json", "w"), indent=1)
    el = time.time() - t0
    print(f"[{i:3d}] {el:6.0f}s score={sc:6.2f}  cold25={metr['cold']['cov25']:5.2f} "
          f"seen25={metr['seen']['cov25']:5.2f} all25={metr['all']['cov25']:5.2f} "
          f"all_rmsle={metr['all']['rmsle']:.3f} | obj={cfg['objective']} cal={cfg['calib']} "
          f"nl={cfg['num_leaves']} grp={len(cfg['groups'])} | BEST cold25={best['cold']['cov25']:.2f}", flush=True)
    i += 1
log.close()
print(f"\n[search done] {i} trials in {time.time()-t0:.0f}s. "
      f"BEST val_sel cold Cov@25={best['cold']['cov25']:.2f} (all={best['all']['cov25']:.2f})")
print("BEST cfg:", json.dumps(best["cfg"]))
