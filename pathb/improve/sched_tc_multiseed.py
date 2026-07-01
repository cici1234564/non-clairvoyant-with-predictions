# Regenerate total-completion with the FIXED random-sampling eval, multi-seed,
# all methods. Reports mean +/- std. Read-only outputs to pathb/figures/.
import sys, json, pickle, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, "pathb")
import sched_tc_lib as tc

PB = pathlib.Path("pathb")
df = pd.read_pickle(PB/"cache/feat_df.pkl"); sp = np.load(PB/"cache/splits.npz")
idx_te = pd.Index(sp["idx_te"])
predictions = pickle.load(open(PB/"artifacts/predictions.pkl","rb"))["predictions"]
ORD = ["M1","M3","M4","M5","M6","M7"]; SEEDS = list(range(10)); N = 10000
t0 = time.time()
rows = {k: {"spjf": [], "prr": [], "fold": []} for k in ORD}
base = {b: [] for b in ["SJF","RR","FIFO"]}
for s in SEEDS:
    r = tc.evaluate_total_completion_all_methods(df=df, idx_te=idx_te, predictions=predictions,
                                                 sample_size=N, prr_lambda=0.7, seed=s, verbose=False)
    for b in ["SJF","RR","FIFO"]:
        base[b].append(r["baselines"][b]["ratio_vs_opt"])
    for k in ORD:
        m = r["methods"][k]
        rows[k]["spjf"].append(m["SPJF"]["ratio_vs_opt"])
        rows[k]["prr"].append(m["PRR"]["ratio_vs_opt"])
        rows[k]["fold"].append(1.0/m["PRR"]["ratio_vs_fifo"])   # FIFO/PRR = x lower than FIFO
    print(f"  seed={s}  M5 PRR/SRPT={rows['M5']['prr'][-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

def ms(x): x=np.array(x); return float(x.mean()), float(x.std())
out = {"n": N, "seeds": SEEDS, "baselines": {b: ms(v) for b,v in base.items()}, "methods": {}}
recs = []
for k in ORD:
    pm,ps = ms(rows[k]["prr"]); sm,ss = ms(rows[k]["spjf"]); fm,fs = ms(rows[k]["fold"])
    out["methods"][k] = dict(prr_mean=pm,prr_std=ps,spjf_mean=sm,spjf_std=ss,fold_mean=fm,fold_std=fs)
    recs.append(dict(method=k, prr_ratio_mean=round(pm,4), prr_ratio_std=round(ps,4),
                     spjf_ratio_mean=round(sm,4), spjf_ratio_std=round(ss,4),
                     prr_fold_fifo_mean=round(fm,4), prr_fold_fifo_std=round(fs,4)))
pd.DataFrame(recs).to_csv(PB/"figures/data_total_completion_multiseed.csv", index=False)
json.dump(out, open(PB/"improve/artifacts/tc_multiseed.json","w"), indent=1)

print("\n#### CORRECTED total completion (random multi-seed, n=10k, 10 seeds) ####")
print(f"  baselines /SRPT: SJF={base['SJF'][0]:.3f}.. mean SJF={ms(base['SJF'])[0]:.3f}, "
      f"RR={ms(base['RR'])[0]:.3f}±{ms(base['RR'])[1]:.3f}, FIFO={ms(base['FIFO'])[0]:.3f}±{ms(base['FIFO'])[1]:.3f}")
print(f"  {'method':7} {'SPJF/SRPT':>14} {'PRR/SRPT':>14} {'PRR x vs FIFO':>15} {'<RR?':>5}")
rrm = ms(base["RR"])[0]
for k in ORD:
    o = out["methods"][k]
    print(f"  {k:7} {o['spjf_mean']:>6.3f}±{o['spjf_std']:<5.3f} {o['prr_mean']:>6.3f}±{o['prr_std']:<5.3f} "
          f"{o['fold_mean']:>6.2f}±{o['fold_std']:<5.2f}   {'yes' if o['prr_mean']<rrm else 'no':>3}")
print(f"\n[done] {time.time()-t0:.0f}s -> data_total_completion_multiseed.csv, tc_multiseed.json")
