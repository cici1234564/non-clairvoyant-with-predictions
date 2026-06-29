# =====================================================================
# TOTAL COMPLETION TIME SCHEDULING EVALUATION
# =====================================================================

import numpy as np
import pandas as pd
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr
import time

EPS = 1e-12


@dataclass
class Job:
    job_id: int
    arrival_time: float   # r_j
    true_size: float      # p*_j
    predicted_size: float # \hat p_j

def build_jobs_from_df(scheduling_df: pd.DataFrame) -> List[Job]:
    """Build job list from dataframe."""
    jobs: List[Job] = [
        Job(
            job_id=int(row.job_id),
            arrival_time=float(row.submit_time),
            true_size=float(row.p_star),
            predicted_size=float(row.y_pred),
        )
        for row in scheduling_df.itertuples(index=False)
    ]
    return jobs


def simulate_srpt(jobs: List[Job]) -> Tuple[float, pd.DataFrame]:
    """Optimal ΣC_j with release times (preemptive SRPT on true remaining time)."""
    J = sorted(jobs, key=lambda x: x.arrival_time)
    n, i, t, totalC = len(J), 0, 0.0, 0.0
    metrics = []
    ready: List[Tuple[float, float, int, Job]] = []

    while i < n or ready:
        if not ready and i < n and t < J[i].arrival_time:
            t = J[i].arrival_time
        while i < n and J[i].arrival_time <= t + EPS:
            heapq.heappush(ready, (J[i].true_size, J[i].arrival_time, J[i].job_id, J[i]))
            i += 1
        if not ready:
            continue
        rem, rj, jid, job = heapq.heappop(ready)
        next_arrival = J[i].arrival_time if i < n else float('inf')
        run = min(rem, next_arrival - t)
        if run <= EPS:
            heapq.heappush(ready, (rem, rj, jid, job))
            t = next_arrival
            continue
        t += run
        rem -= run
        if rem <= EPS:
            Cj = t
            totalC += Cj
            metrics.append({
                'job_id': jid, 'arrival_time': rj, 'completion_time': Cj,
                'flow_time': Cj - rj, 'wait_time': Cj - rj - job.true_size,
                'slowdown': (Cj - rj) / max(job.true_size, EPS),
            })
        else:
            heapq.heappush(ready, (rem, rj, jid, job))

    return totalC, pd.DataFrame(metrics)

def simulate_fifo(jobs: List[Job]) -> Tuple[float, pd.DataFrame]:
    """FIFO (non-preemptive)."""
    J = sorted(jobs, key=lambda x: x.arrival_time)
    t, totalC = 0.0, 0.0
    metrics = []

    for job in J:
        if t < job.arrival_time:
            t = job.arrival_time
        start = t
        t += job.true_size
        Cj = t
        totalC += Cj
        metrics.append({
            'job_id': job.job_id, 'arrival_time': job.arrival_time, 'completion_time': Cj,
            'flow_time': Cj - job.arrival_time, 'wait_time': start - job.arrival_time,
            'slowdown': (Cj - job.arrival_time) / max(job.true_size, EPS),
        })

    return totalC, pd.DataFrame(metrics)

def simulate_sjf_or_spjf(jobs: List[Job], use_predictions: bool) -> Tuple[float, pd.DataFrame]:
    """SJF (oracle true size) or SPJF (predicted size), non-preemptive."""
    J = sorted(jobs, key=lambda x: x.arrival_time)
    n, i, t, totalC = len(J), 0, 0.0, 0.0
    metrics = []
    ready: List[Tuple[float, float, int, Job]] = []

    while i < n or ready:
        if not ready and i < n and t < J[i].arrival_time:
            t = J[i].arrival_time
        while i < n and J[i].arrival_time <= t + EPS:
            key = J[i].predicted_size if use_predictions else J[i].true_size
            heapq.heappush(ready, (key, J[i].arrival_time, J[i].job_id, J[i]))
            i += 1
        if not ready:
            continue
        _, rj, jid, job = heapq.heappop(ready)
        start = t
        t += job.true_size
        Cj = t
        totalC += Cj
        metrics.append({
            'job_id': jid, 'arrival_time': rj, 'completion_time': Cj,
            'flow_time': Cj - rj, 'wait_time': start - rj,
            'slowdown': (Cj - rj) / max(job.true_size, EPS),
        })

    return totalC, pd.DataFrame(metrics)

def simulate_rr(jobs: List[Job]) -> Tuple[float, pd.DataFrame]:
    """Round Robin (equal sharing), preemptive."""
    J = sorted(jobs, key=lambda x: x.arrival_time)
    n, i, t, totalC = len(J), 0, 0.0, 0.0
    metrics = []
    active: List[Tuple[Job, float]] = []

    while i < n or active:
        if not active and i < n and t < J[i].arrival_time:
            t = J[i].arrival_time
        while i < n and J[i].arrival_time <= t + EPS:
            active.append((J[i], J[i].true_size))
            i += 1
        if not active:
            continue
        k = len(active)
        rate = 1.0 / k
        time_to_finish = min(rem / rate for (_, rem) in active)
        next_arrival = J[i].arrival_time if i < n else float('inf')
        dt = min(time_to_finish, next_arrival - t)
        t += dt
        finished_idx = []
        for idx, (job, rem) in enumerate(active):
            rem_new = rem - rate * dt
            active[idx] = (job, rem_new)
            if rem_new <= EPS and dt == time_to_finish:
                finished_idx.append(idx)
        for idx in reversed(finished_idx):
            job, _ = active.pop(idx)
            Cj = t
            totalC += Cj
            metrics.append({
                'job_id': job.job_id, 'arrival_time': job.arrival_time, 'completion_time': Cj,
                'flow_time': Cj - job.arrival_time,
                'wait_time': Cj - job.arrival_time - job.true_size,
                'slowdown': (Cj - job.arrival_time) / max(job.true_size, EPS),
            })

    return totalC, pd.DataFrame(metrics)

def simulate_prr(jobs: List[Job], lam: float = 0.7) -> Tuple[float, pd.DataFrame]:
    """Preferential Round Robin with λ parameter."""
    assert 0.0 < lam < 1.0
    J = sorted(jobs, key=lambda x: x.arrival_time)
    n, i, t, totalC = len(J), 0, 0.0, 0.0
    metrics = []
    active: List[Tuple[Job, float]] = []

    while i < n or active:
        if not active and i < n and t < J[i].arrival_time:
            t = J[i].arrival_time
        while i < n and J[i].arrival_time <= t + EPS:
            active.append((J[i], J[i].true_size))
            i += 1
        if not active:
            continue
        k = len(active)
        min_idx = min(range(k), key=lambda idx: active[idx][0].predicted_size)
        base = (1.0 - lam) / k
        rates = [base] * k
        rates[min_idx] += lam
        time_to_finish = min(rem / max(rates[idx], EPS) for idx, (_, rem) in enumerate(active))
        next_arrival = J[i].arrival_time if i < n else float('inf')
        dt = min(time_to_finish, next_arrival - t)
        t += dt
        finished_idx = []
        for idx, (job, rem) in enumerate(active):
            rem_new = rem - rates[idx] * dt
            active[idx] = (job, rem_new)
            if rem_new <= EPS and dt == time_to_finish:
                finished_idx.append(idx)
        for idx in reversed(finished_idx):
            job, _ = active.pop(idx)
            Cj = t
            totalC += Cj
            metrics.append({
                'job_id': job.job_id, 'arrival_time': job.arrival_time, 'completion_time': Cj,
                'flow_time': Cj - job.arrival_time,
                'wait_time': Cj - job.arrival_time - job.true_size,
                'slowdown': (Cj - job.arrival_time) / max(job.true_size, EPS),
            })

    return totalC, pd.DataFrame(metrics)

# ======================== MAIN EVALUATION ========================

def evaluate_total_completion_all_methods(
    df: pd.DataFrame,
    idx_te: pd.Index,
    predictions: Dict[str, np.ndarray],
    sample_size: Optional[int] = 10000,
    prr_lambda: float = 0.7,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Complete evaluation of total completion time for all prediction methods.

    Args:
        df: Full dataframe with columns [submit_time, p_star, ...]
        idx_te: Test set indices into df
        predictions: Dict mapping method name -> prediction array (aligned with idx_te)
        sample_size: Number of jobs to evaluate (None = all)
        prr_lambda: Lambda parameter for PRR
        seed: Random seed for consistent sampling
        verbose: Print detailed output

    Returns:
        Dictionary with complete results
    """

    if verbose:
        print("="*80)
        print("TOTAL COMPLETION TIME SCHEDULING EVALUATION")
        print(f"Sample size: {sample_size:,} | PRR λ: {prr_lambda:.2f} | Seed: {seed}")
        print("="*80)

    start_time = time.time()

    # ============ Data preparation ============
    test_df = df.loc[idx_te].copy().reset_index(drop=True)
    test_df['original_idx'] = np.arange(len(test_df))

    test_df = test_df.sort_values('submit_time').reset_index(drop=True)

    t0 = float(test_df['submit_time'].min())
    test_df['submit_time'] = test_df['submit_time'] - t0

    if sample_size is not None and sample_size < len(test_df):
        np.random.seed(seed)
        test_df = test_df.iloc[:sample_size].copy()

    selected_original_indices = test_df['original_idx'].values

    base_df = test_df[['submit_time', 'p_star']].copy()
    base_df['job_id'] = np.arange(len(base_df))

    if verbose:
        print(f"\nActual jobs being evaluated: {len(base_df):,}")
        print("Computing baselines...")

    # ============ Baselines ============
    baseline_df = base_df.copy()
    baseline_df['y_pred'] = baseline_df['p_star']
    baseline_jobs = build_jobs_from_df(baseline_df)

    srpt_total, srpt_metrics = simulate_srpt(baseline_jobs)
    fifo_total, fifo_metrics = simulate_fifo(baseline_jobs)
    sjf_total, sjf_metrics = simulate_sjf_or_spjf(baseline_jobs, use_predictions=False)
    rr_total, rr_metrics = simulate_rr(baseline_jobs)

    results = {
        'meta': {
            'n_jobs': len(base_df),
            'sample_size': sample_size,
            'prr_lambda': prr_lambda,
            'seed': seed
        },
        'baselines': {
            'SRPT': {
                'total_completion': srpt_total,
                'avg_flow_time': float(srpt_metrics['flow_time'].mean()),
                'ratio_vs_opt': 1.0,
                'ratio_vs_fifo': srpt_total / fifo_total
            },
            'SJF': {
                'total_completion': sjf_total,
                'avg_flow_time': float(sjf_metrics['flow_time'].mean()),
                'ratio_vs_opt': sjf_total / srpt_total,
                'ratio_vs_fifo': sjf_total / fifo_total
            },
            'RR': {
                'total_completion': rr_total,
                'avg_flow_time': float(rr_metrics['flow_time'].mean()),
                'ratio_vs_opt': rr_total / srpt_total,
                'ratio_vs_fifo': rr_total / fifo_total
            },
            'FIFO': {
                'total_completion': fifo_total,
                'avg_flow_time': float(fifo_metrics['flow_time'].mean()),
                'ratio_vs_opt': fifo_total / srpt_total,
                'ratio_vs_fifo': 1.0
            }
        },
        'methods': {}
    }

    # ============ Evaluate Each Prediction Method ============
    methods_to_eval = list(predictions.keys())

    if verbose:
        print(f"\nEvaluating {len(methods_to_eval)} prediction methods...")
        print("-"*80)

    for i, method_name in enumerate(methods_to_eval):
        if method_name not in predictions:
            continue

        method_start = time.time()

        all_predictions = predictions[method_name]
        y_pred = all_predictions[selected_original_indices]

        try:
            rho, _ = spearmanr(base_df['p_star'].values, y_pred)
            pred_quality = float(rho) if not np.isnan(rho) else 0.5
        except:
            pred_quality = 0.5

        sched_df = base_df.copy()
        sched_df['y_pred'] = y_pred
        jobs = build_jobs_from_df(sched_df)

        spjf_total, spjf_metrics = simulate_sjf_or_spjf(jobs, use_predictions=True)
        prr_total, prr_metrics = simulate_prr(jobs, lam=prr_lambda)

        results['methods'][method_name] = {
            'prediction_quality': pred_quality,
            'SPJF': {
                'total_completion': spjf_total,
                'avg_flow_time': float(spjf_metrics['flow_time'].mean()),
                'ratio_vs_opt': spjf_total / srpt_total,
                'ratio_vs_fifo': spjf_total / fifo_total
            },
            'PRR': {
                'total_completion': prr_total,
                'avg_flow_time': float(prr_metrics['flow_time'].mean()),
                'ratio_vs_opt': prr_total / srpt_total,
                'ratio_vs_fifo': prr_total / fifo_total,
                'lambda': prr_lambda
            }
        }

        if verbose:
            print(f"[{i+1}/{len(methods_to_eval)}] {method_name:8s}: "
                  f"ρ={pred_quality:.3f}, "
                  f"SPJF/OPT={spjf_total/srpt_total:.4f}, "
                  f"PRR/OPT={prr_total/srpt_total:.4f}, "
                  f"time={time.time() - method_start:.1f}s")

    total_time = time.time() - start_time

    # ============ Print Summary ============
    if verbose:
        print("\n" + "="*80)
        print(f"SUMMARY (n={len(base_df):,} jobs, runtime={total_time:.1f}s)")
        print("="*80)

        print("\nBASELINES:")
        print(f"{'Policy':12s} {'ΣC_j':>16s} {'Avg Flow':>12s} {'ρ_TC':>10s} {'ratio/FIFO':>11s}")
        print("-"*80)
        for name in ['SRPT', 'SJF', 'RR', 'FIFO']:
            r = results['baselines'][name]
            print(f"{name:12s} {r['total_completion']:16.3f} {r['avg_flow_time']:12.3f} "
                  f"{r['ratio_vs_opt']:10.4f} {r['ratio_vs_fifo']:11.4f}")

        print(f"\nPREDICTION METHODS (PRR with λ={prr_lambda:.2f}):")
        print(f"{'Method':8s} {'ρ':>6s} {'SPJF ρ_TC':>10s} {'PRR ρ_TC':>10s} {'Improvement':>12s}")
        print("-"*80)

        sorted_methods = sorted(results['methods'].items(),
                               key=lambda x: x[1]['PRR']['ratio_vs_opt'])

        for method_name, method_data in sorted_methods:
            spjf_ratio = method_data['SPJF']['ratio_vs_opt']
            prr_ratio = method_data['PRR']['ratio_vs_opt']
            improvement = (spjf_ratio - prr_ratio) / spjf_ratio * 100

            print(f"{method_name:8s} {method_data['prediction_quality']:6.3f} "
                  f"{spjf_ratio:10.4f} "
                  f"{prr_ratio:10.4f} "
                  f"{improvement:11.1f}%")

        if sorted_methods:
            best = sorted_methods[0]
            print("\n" + "="*80)
            print(f"*** BEST: {best[0]} achieves ρ_TC={best[1]['PRR']['ratio_vs_opt']:.4f} "
                  f"with PRR(λ={prr_lambda:.2f}) ***")
            print("="*80)

    # Save results
    summary_rows = []
    for method_name, method_data in results['methods'].items():
        summary_rows.append({
            'method': method_name,
            'spearman_rho': method_data['prediction_quality'],
            'spjf_total': method_data['SPJF']['total_completion'],
            'spjf_ratio_tc': method_data['SPJF']['ratio_vs_opt'],
            'prr_total': method_data['PRR']['total_completion'],
            'prr_ratio_tc': method_data['PRR']['ratio_vs_opt'],
            'prr_lambda': prr_lambda
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"total_completion_results_{sample_size}.csv", index=False)

    if verbose:
        print(f"\n[INFO] Results saved to: total_completion_results_{sample_size}.csv")

    return results

# ======================== LAMBDA SWEEP ========================

def sweep_prr_lambda_range(
    df: pd.DataFrame,
    idx_te: pd.Index,
    predictions: Dict[str, np.ndarray],
    sample_size: int = 10000,
    lambda_values: Optional[List[float]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Sweep PRR λ values to find optimal setting. Uses M5 predictions by default."""

    if lambda_values is None:
        lambda_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    print(f"\nRunning λ sweep: {lambda_values}")
    print("-"*50)

    sweep_results = []

    for lam in lambda_values:
        results = evaluate_total_completion_all_methods(
            df=df, idx_te=idx_te, predictions=predictions,
            sample_size=sample_size, prr_lambda=lam, seed=seed, verbose=False
        )

        if results and 'M5' in results['methods']:
            m5_prr = results['methods']['M5']['PRR']
            sweep_results.append({
                'lambda': lam,
                'ratio_vs_opt': m5_prr['ratio_vs_opt'],
                'total_completion': m5_prr['total_completion'],
                'avg_flow_time': m5_prr['avg_flow_time']
            })
            print(f"λ={lam:.2f}: ρ_TC={m5_prr['ratio_vs_opt']:.4f}")

    sweep_df = pd.DataFrame(sweep_results)
    best_lambda = sweep_df.loc[sweep_df['ratio_vs_opt'].idxmin(), 'lambda']

    print(f"\n*** Optimal λ={best_lambda:.2f} for M5 ***")

    return sweep_df

# ======================== MAIN EXECUTION ========================
