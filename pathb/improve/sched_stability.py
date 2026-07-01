# Read-only multi-seed stability of total-completion PRR-M5 on the corrected pipeline.
# Uses RANDOM seeded subsamples (the original eval used a deterministic first-N-by-time
# prefix, so its 'seed' was a no-op). Writes nothing committed.
import sys, json, pickle, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, "pathb")
from sched_tc_lib import Job, simulate_srpt, simulate_fifo, simulate_rr, simulate_prr

PB = pathlib.Path("pathb")
df = pd.read_pickle(PB/"cache/feat_df.pkl"); sp = np.load(PB/"cache/splits.npz")
ite = sp["idx_te"]; P = pickle.load(open(PB/"artifacts/predictions.pkl","rb"))
arr_full = df["submit_time"].values[ite].astype(float)
true_full = df["p_star"].values[ite].astype(float)
pred_full = np.asarray(P["predictions"]["M5"], float)
NFULL = len(true_full); SEEDS = list(range(10)); GRID = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def jobs(idx, pv):
    a = arr_full[idx]; a = a - a.min()
    t = true_full[idx]; p = pv[idx]
    return [Job(int(j), float(a[j]), float(t[j]), float(p[j])) for j in range(len(idx))]
def sample(n, seed):
    return np.sort(np.random.RandomState(1000+seed).choice(NFULL, n, replace=False))

def run_seed(n, seed, lams):
    idx = sample(n, seed)
    srpt,_ = simulate_srpt(jobs(idx, true_full))
    fifo,_ = simulate_fifo(jobs(idx, true_full))
    rr,_   = simulate_rr(jobs(idx, true_full))
    out = {"srpt": srpt, "fifo": fifo, "rr_rho": rr/srpt}
    for lam in lams:
        prr,_ = simulate_prr(jobs(idx, pred_full), lam=lam)
        out[f"rho_{lam}"] = prr/srpt
        out[f"imp_{lam}"] = 100*(1 - prr/fifo)
    return out

t0 = time.time(); res = {}
print(f"[stability] n_full={NFULL}, {len(SEEDS)} seeds; RANDOM subsamples", flush=True)

# Steps 1+2: n=10k, all lambdas, all seeds
res["n10000"] = []
for s in SEEDS:
    r = run_seed(10000, s, GRID); res["n10000"].append(r)
    print(f"  n=10000 seed={s}  RR={r['rr_rho']:.3f}  PRR(.7)rho={r['rho_0.7']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

# Step 3: n=5k and 20k at lambda=0.7 (10k reuses above)
for n in [5000, 20000]:
    res[f"n{n}"] = []
    for s in SEEDS:
        r = run_seed(n, s, [0.7]); res[f"n{n}"].append(r)
        print(f"  n={n} seed={s}  PRR(.7)rho={r['rho_0.7']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

# deterministic first-N-by-time prefixes (the ORIGINAL method) for reference, lambda=0.7
order = np.argsort(arr_full, kind="mergesort")
res["prefix"] = {}
for n in [5000, 10000, 20000]:
    idx = order[:n]
    srpt,_ = simulate_srpt(jobs(idx, true_full)); rr,_ = simulate_rr(jobs(idx, true_full))
    prr,_  = simulate_prr(jobs(idx, pred_full), lam=0.7)
    res["prefix"][n] = {"rho": prr/srpt, "rr": rr/srpt}

json.dump(res, open(PB/"improve/artifacts/sched_stability.json","w"), indent=1)

def ms(xs): xs=np.array(xs); return xs.mean(), xs.std(), xs.min(), xs.max()
print("\n#### TABLE 1: PRR-M5, lambda=0.7, n=10000, 10 RANDOM seeds ####")
r07=[r["rho_0.7"] for r in res["n10000"]]; rr=[r["rr_rho"] for r in res["n10000"]]
m,s_,lo,hi = ms(r07)
print(f"  PRR-M5 rho_TC : mean={m:.3f}  std={s_:.3f}  min={lo:.3f}  max={hi:.3f}")
print(f"  RR rho_TC     : mean={np.mean(rr):.3f}  std={np.std(rr):.3f}")
print(f"  (original deterministic first-10k-by-time prefix: rho={res['prefix'][10000]['rho']:.3f})")

print("\n#### TABLE 2: lambda sweep, n=10000, 10 seeds ####")
print(f"  {'lam':>4} {'rho_TC mean±std':>18} {'imp vs FIFO % mean±std':>24} {'< RR?':>6}")
rrm = np.mean(rr)
for lam in GRID:
    rs=[r[f"rho_{lam}"] for r in res["n10000"]]; im=[r[f"imp_{lam}"] for r in res["n10000"]]
    print(f"  {lam:>4} {np.mean(rs):>8.3f} ± {np.std(rs):<6.3f} {np.mean(im):>11.1f} ± {np.std(im):<6.1f}    {'yes' if np.mean(rs)<rrm else 'no':>4}")
print(f"  (RR baseline mean rho = {rrm:.3f})")

print("\n#### TABLE 3: sample-size effect, lambda=0.7, 10 seeds ####")
print(f"  {'n':>7} {'rho_TC mean±std':>18} {'min':>7} {'max':>7} {'prefix(orig)':>13}")
for n in [5000,10000,20000]:
    key=f"n{n}"; rs=[r["rho_0.7"] for r in res[key]]; m,s_,lo,hi=ms(rs)
    print(f"  {n:>7} {m:>8.3f} ± {s_:<6.3f} {lo:>7.3f} {hi:>7.3f} {res['prefix'][n]['rho']:>13.3f}")
print(f"\n[done] {time.time()-t0:.0f}s -> pathb/improve/artifacts/sched_stability.json")
