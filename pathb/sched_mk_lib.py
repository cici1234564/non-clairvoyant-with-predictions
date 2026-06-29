# =====================================================================
# MAKESPAN EVALUATION (P||C_max)
# =====================================================================


from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import time

@dataclass
class Job:
    job_id: int
    true_size: float
    pred_size: float

    def __post_init__(self):
        self.pred_size = max(self.pred_size, 1e-9)
        self.true_size = max(self.true_size, 1e-9)

# ======================== Core Algorithms ========================

def makespan_LPT(jobs: List[Job], m: int, use_predictions: bool = False) -> Tuple[float, List[List[int]]]:
    """Longest Processing Time First. LPPT when use_predictions=True."""
    if not jobs or m <= 0:
        return 0.0, []

    key = (lambda j: j.pred_size) if use_predictions else (lambda j: j.true_size)
    sorted_jobs = sorted(jobs, key=key, reverse=True)

    if use_predictions:
        perceived_loads = [0.0] * m
        actual_loads = [0.0] * m
        assignments = [[] for _ in range(m)]
        for job in sorted_jobs:
            min_idx = int(np.argmin(perceived_loads))
            perceived_loads[min_idx] += job.pred_size
            actual_loads[min_idx] += job.true_size
            assignments[min_idx].append(job.job_id)
        return max(actual_loads), assignments
    else:
        loads = [0.0] * m
        assignments = [[] for _ in range(m)]
        for job in sorted_jobs:
            min_idx = int(np.argmin(loads))
            loads[min_idx] += job.true_size
            assignments[min_idx].append(job.job_id)
        return max(loads), assignments

def makespan_SPT(jobs: List[Job], m: int, use_predictions: bool = False) -> Tuple[float, List[List[int]]]:
    """Shortest Processing Time First. SPPT when use_predictions=True."""
    if not jobs or m <= 0:
        return 0.0, []

    key = (lambda j: j.pred_size) if use_predictions else (lambda j: j.true_size)
    sorted_jobs = sorted(jobs, key=key)

    if use_predictions:
        perceived_loads = [0.0] * m
        actual_loads = [0.0] * m
        assignments = [[] for _ in range(m)]
        for job in sorted_jobs:
            min_idx = int(np.argmin(perceived_loads))
            perceived_loads[min_idx] += job.pred_size
            actual_loads[min_idx] += job.true_size
            assignments[min_idx].append(job.job_id)
        return max(actual_loads), assignments
    else:
        loads = [0.0] * m
        assignments = [[] for _ in range(m)]
        for job in sorted_jobs:
            min_idx = int(np.argmin(loads))
            loads[min_idx] += job.true_size
            assignments[min_idx].append(job.job_id)
        return max(loads), assignments

def makespan_random(jobs: List[Job], m: int, seed: int = 42) -> Tuple[float, List[List[int]]]:
    """Random assignment baseline."""
    if not jobs or m <= 0:
        return 0.0, []
    rng = np.random.RandomState(seed)
    loads = [0.0] * m
    assignments = [[] for _ in range(m)]
    for job in jobs:
        idx = rng.randint(m)
        loads[idx] += job.true_size
        assignments[idx].append(job.job_id)
    return max(loads), assignments

def compute_lower_bound(jobs: List[Job], m: int) -> float:
    """McNaughton's preemptive bound: OPT_pre = max(Σp/m, max_j p_j)."""
    if not jobs or m <= 0:
        return 0.0
    total = sum(j.true_size for j in jobs)
    mx = max(j.true_size for j in jobs)
    return max(total / m, mx)

# ======================== Data Preparation ========================

def _sample_jobs(df: pd.DataFrame, idx_te, y_pred: np.ndarray,
                 m: int, n_samples: Optional[int] = None,
                 seed: int = 42) -> Tuple[List[Job], np.ndarray]:
    """
    Sample jobs from test set. Returns (jobs_with_oracle_pred, true_sizes).
    Jobs are created with true_size only; predictions are set per-method later.
    """
    test_df = df.loc[idx_te].copy().reset_index(drop=True)

    if n_samples is None:
        n_samples = min(len(test_df), m * max(10, int(np.log2(m) * 10)))

    if len(test_df) > n_samples:
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(test_df), size=n_samples, replace=False)
        test_df = test_df.iloc[pick].copy().reset_index(drop=True)
    else:
        pick = np.arange(len(test_df))

    true_sizes = test_df['p_star'].values.astype(np.float64)
    return pick, true_sizes

# ======================== Main Evaluation ========================

def evaluate_makespan(
    df: pd.DataFrame,
    idx_te,
    predictions: Dict[str, np.ndarray],
    machine_counts: List[int] = [5, 10, 20, 50, 100],
    n_random_trials: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate makespan (P||C_max) for all prediction methods.

    Args:
        df: Full dataframe with column p_star
        idx_te: Test set indices
        predictions: Dict mapping method name -> prediction array
        machine_counts: List of m values to evaluate
        n_random_trials: Number of random assignment trials for averaging
        seed: Random seed for sampling
        verbose: Print results

    Returns:
        DataFrame with results for all methods and machine counts.
    """
    if verbose:
        print("=" * 80)
        print("MAKESPAN EVALUATION  P||C_max")
        print("=" * 80)

    all_rows = []

    for m in machine_counts:
        if verbose:
            print(f"\n--- m = {m} machines ---")

        # Sample ONCE per m (same jobs for all methods)
        pick, true_sizes = _sample_jobs(df, idx_te, None, m, seed=seed)
        n = len(true_sizes)

        # Oracle baselines (same for all prediction methods)
        oracle_jobs = [Job(i, true_sizes[i], true_sizes[i]) for i in range(n)]
        opt_pre = compute_lower_bound(oracle_jobs, m)
        lpt_ms, _ = makespan_LPT(oracle_jobs, m, use_predictions=False)
        spt_ms, _ = makespan_SPT(oracle_jobs, m, use_predictions=False)

        # Random baseline (averaged)
        random_ms = np.mean([makespan_random(oracle_jobs, m, seed=seed + i)[0]
                             for i in range(n_random_trials)])

        if verbose:
            print(f"  n={n}, OPT_pre={opt_pre:.1f}")
            print(f"  LPT={lpt_ms:.1f} (ρ={lpt_ms/opt_pre:.4f}), "
                  f"SPT={spt_ms:.1f} (ρ={spt_ms/opt_pre:.4f}), "
                  f"Random={random_ms:.1f} (ρ={random_ms/opt_pre:.4f})")

        # Per-method evaluation
        for method_name, y_pred_all in predictions.items():
            y_pred = y_pred_all[:len(df.loc[idx_te])][pick]

            # Build jobs with this method's predictions
            jobs = [Job(i, true_sizes[i], float(y_pred[i])) for i in range(n)]

            lppt_ms, _ = makespan_LPT(jobs, m, use_predictions=True)
            sppt_ms, _ = makespan_SPT(jobs, m, use_predictions=True)

            row = {
                'method': method_name,
                'm': m,
                'n_jobs': n,
                'OPT_pre': opt_pre,
                'LPT': lpt_ms,
                'LPT_ratio': lpt_ms / opt_pre,
                'SPT': spt_ms,
                'SPT_ratio': spt_ms / opt_pre,
                'LPPT': lppt_ms,
                'LPPT_ratio': lppt_ms / opt_pre,
                'SPPT': sppt_ms,
                'SPPT_ratio': sppt_ms / opt_pre,
                'Random': random_ms,
                'Random_ratio': random_ms / opt_pre,
            }
            all_rows.append(row)

            if verbose:
                print(f"  {method_name:8s}: LPPT ρ={lppt_ms/opt_pre:.4f}, "
                      f"SPPT ρ={sppt_ms/opt_pre:.4f}")

    results_df = pd.DataFrame(all_rows)

    # Print summary table
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        # Baselines (same across methods, print once per m)
        print(f"\n{'m':>5s}  {'OPT_pre':>10s}  {'LPT ρ':>8s}  {'SPT ρ':>8s}  {'Rand ρ':>8s}")
        print("-" * 50)
        for m in machine_counts:
            sub = results_df[results_df['m'] == m].iloc[0]
            print(f"{m:5d}  {sub['OPT_pre']:10.1f}  {sub['LPT_ratio']:8.4f}  "
                  f"{sub['SPT_ratio']:8.4f}  {sub['Random_ratio']:8.4f}")

        # Per-method prediction results
        print(f"\n{'Method':>8s} {'m':>5s}  {'LPPT ρ':>8s}  {'SPPT ρ':>8s}")
        print("-" * 40)
        for method_name in predictions.keys():
            for m in machine_counts:
                sub = results_df[(results_df['method'] == method_name) &
                                 (results_df['m'] == m)]
                if len(sub) > 0:
                    row = sub.iloc[0]
                    print(f"{method_name:>8s} {m:5d}  {row['LPPT_ratio']:8.4f}  "
                          f"{row['SPPT_ratio']:8.4f}")
        print("=" * 80)

    # Save
    results_df.to_csv('makespan_results.csv', index=False)
    if verbose:
        print(f"\n[INFO] Results saved to makespan_results.csv")

    return results_df

# ======================== Usage ========================
