# =====================================================================
# MAX-STRETCH (1|r,pmtn|S_max)
# =====================================================================

from typing import Dict, Tuple, Optional, List
import numpy as np, heapq, time, math

EPS = 1e-9

# --------------------------- Preparation ----------------------------
def _prepare_jobs(
    df,
    idx_te,
    y_pred: np.ndarray,
    sample_size: Optional[int] = 5000,
    seed: int = 42,
    clip_ratio: Optional[Tuple[float, float]] = (0.1, 10.0),
):
    cols = ["submit_time", "p_star"]
    te = df.loc[idx_te, cols].reset_index(drop=True).copy()

    n_all = len(te)
    y_pred = np.asarray(y_pred[:n_all], dtype=np.float64)

    if sample_size and sample_size < n_all:
        rng = np.random.RandomState(seed)
        pick = np.sort(rng.choice(n_all, size=sample_size, replace=False))
        te = te.iloc[pick].copy()
        y_pred = y_pred[pick]

    sort_idx = np.argsort(te["submit_time"].values)
    te = te.iloc[sort_idx].reset_index(drop=True)
    y_pred = y_pred[sort_idx]

    r0 = float(te["submit_time"].iloc[0])
    r = te["submit_time"].to_numpy(dtype=np.float64) - r0
    p = np.maximum(te["p_star"].to_numpy(dtype=np.float64), 1.0)
    q = np.maximum(y_pred.astype(np.float64), 1.0)

    if clip_ratio:
        lo, hi = clip_ratio
        q = np.clip(q, lo * p, hi * p)

    return r, p, q

# ------------------------------ Metrics -----------------------------
def _stretch_array(r: np.ndarray, C: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.maximum((C - r) / np.maximum(p, EPS), 1.0)

def _compute_metrics(r: np.ndarray, C: np.ndarray, p: np.ndarray, normS: float) -> Dict:
    s = _stretch_array(r, C, p)
    tau = 10.0
    bsld = (C - r) / np.maximum(p, tau)
    return {
        "n": int(s.size),
        "max": float(np.max(s)),
        "p99": float(np.percentile(s, 99)),
        "p95": float(np.percentile(s, 95)),
        "p90": float(np.percentile(s, 90)),
        "median": float(np.median(s)),
        "mean": float(np.mean(s)),
        "max_over_OPT": float(np.max(s) / normS),
        "p99_over_OPT": float(np.percentile(s, 99) / normS),
        "p95_over_OPT": float(np.percentile(s, 95) / normS),
        "med_over_OPT": float(np.median(s) / normS),
        "bsld10_max": float(np.max(bsld)),
        "bsld10_p99": float(np.percentile(bsld, 99)),
        "bsld10_med": float(np.median(bsld)),
    }

# ---------------------- EDF feasibility & OPT(S*) --------------------
def _edf_feasible_fast(r: np.ndarray, p: np.ndarray, d: np.ndarray) -> bool:
    """EDF is feasibility-optimal on a preemptive uniprocessor."""
    n = len(r)
    rem = p.copy()
    heap = []  # (deadline, j)
    i, t = 0, 0.0

    while i < n or heap:
        if not heap and i < n:
            t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            heapq.heappush(heap, (d[i], i)); i += 1
        if not heap:
            continue
        dj, j = heapq.heappop(heap)
        if t > dj + EPS:
            return False
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t, dj - t)
        if dt <= 1e-12:
            t = min(next_arr, dj)
            if rem[j] > EPS:
                heapq.heappush(heap, (dj, j))
            continue
        rem[j] -= dt
        t += dt
        if rem[j] > EPS:
            if t >= dj - EPS:
                return False
            heapq.heappush(heap, (dj, j))
    return True

def _opt_max_stretch_fast(r: np.ndarray, p: np.ndarray, tol: float = 1e-3) -> float:
    """Exact S* via bisection on S with EDF-feasibility; O(log(1/tol) * n log n)."""
    def quick_feasible(S: float) -> bool:
        d = r + S * p
        if np.sum(p) > (np.max(d) - r[0]) + 1e-12:
            return False
        return _edf_feasible_fast(r, p, d)

    lo, hi = 1.0, 2.0
    while not quick_feasible(hi):
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket S*; check inputs.")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if quick_feasible(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return hi

def _edf_schedule_fast(r: np.ndarray, p: np.ndarray, S: float, slack: float = 0.0) -> np.ndarray:
    """Construct EDF schedule at S and return completion times."""
    d = r + (1.0 + slack) * S * p
    n = len(r)
    rem = p.copy()
    C = np.full(n, np.nan, dtype=np.float64)
    heap = []
    i, t = 0, 0.0

    while i < n or heap:
        if not heap and i < n:
            t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            heapq.heappush(heap, (d[i], i)); i += 1
        if not heap:
            continue
        dj, j = heapq.heappop(heap)
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t, dj - t)
        if dt <= 1e-12:
            t = min(next_arr, dj)
            if rem[j] > EPS:
                heapq.heappush(heap, (dj, j))
            continue
        rem[j] -= dt
        t += dt
        if rem[j] <= EPS:
            C[j] = t
        else:
            heapq.heappush(heap, (dj, j))
    return C

# -------------------- Scheduling Algorithms --------------------
def _srpt_fast(r: np.ndarray, p_true: np.ndarray, key: np.ndarray) -> np.ndarray:
    """SRPT/SPRPT: preemptive, event-driven. key=p for SRPT, key=q for SPRPT."""
    n = len(r)
    rem_t = p_true.copy()
    rem_k = key.copy()
    C = np.full(n, np.nan, dtype=np.float64)
    heap = []
    i, t = 0, 0.0
    active = -1

    while i < n or active >= 0 or heap:
        if active < 0 and not heap and i < n:
            t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            if active < 0 or rem_k[i] < rem_k[active]:
                if active >= 0:
                    heapq.heappush(heap, (rem_k[active], active))
                active = i
            else:
                heapq.heappush(heap, (rem_k[i], i))
            i += 1
        if active < 0:
            if not heap:
                continue
            _, active = heapq.heappop(heap)
        next_arr = r[i] if i < n else math.inf
        dt = min(rem_t[active], next_arr - t)
        if dt <= 1e-12:
            t = next_arr
            continue
        rem_t[active] -= dt
        rem_k[active] = max(0.0, rem_k[active] - dt)
        t += dt
        if rem_t[active] <= EPS:
            C[active] = t
            active = -1
    return C

def _fifo_fast(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """FIFO: non-preemptive, process in arrival order."""
    n = len(r)
    C = np.full(n, np.nan, dtype=np.float64)
    t = 0.0
    for j in range(n):
        if t < r[j]:
            t = r[j]
        t += p[j]
        C[j] = t
    return C

def _las_fast(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Least-Attained-Service / Foreground-Background baseline."""
    n = len(r)
    rem = p.copy()
    att = np.zeros(n, dtype=np.float64)
    C = np.full(n, np.nan, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    i, t = 0, 0.0
    done = 0

    while done < n:
        if not active.any() and i < n:
            t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            active[i] = True; i += 1
        if not active.any():
            continue
        idx = np.flatnonzero(active)
        j = idx[np.argmin(att[idx])]
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t)
        if dt <= 1e-12:
            t = next_arr; continue
        rem[j] -= dt; att[j] += dt; t += dt
        if rem[j] <= EPS:
            C[j] = t; active[j] = False; done += 1
    return C

def _edf_pred_deadlines(r: np.ndarray, p_true: np.ndarray, p_pred: np.ndarray,
                        factor: float) -> np.ndarray:
    """EDF-P: EDF with predicted deadlines d_j = r_j + factor * p_hat_j."""
    n = len(r)
    d = r + factor * p_pred
    rem = p_true.copy()
    C = np.full(n, np.nan, dtype=np.float64)
    heap = []; i, t = 0, 0.0
    while i < n or heap:
        if not heap and i < n:
            t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            heapq.heappush(heap, (d[i], i)); i += 1
        if not heap:
            continue
        dj, j = heapq.heappop(heap)
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t)
        if dt <= 1e-12:
            t = next_arr; heapq.heappush(heap, (dj, j)); continue
        rem[j] -= dt; t += dt
        if rem[j] <= EPS:
            C[j] = t
        else:
            heapq.heappush(heap, (dj, j))
    return C

# ------------------------- Main Runner ------------------------------
def run_max_stretch(
    df,
    idx_te,
    y_pred: np.ndarray,
    sample_size: Optional[int] = 5000,
    clip_ratio: Optional[Tuple[float, float]] = (0.1, 10.0),
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate max-stretch for all baselines and prediction-based algorithms.

    Args:
        df: Full dataframe with columns [submit_time, p_star, ...]
        idx_te: Test set indices into df
        y_pred: Prediction array aligned with idx_te
        sample_size: Number of jobs to evaluate
        clip_ratio: (lo, hi) to clip prediction/true ratio for robustness
        verbose: Print results
    """
    r, p, q = _prepare_jobs(df, idx_te, y_pred,
                            sample_size=sample_size, clip_ratio=clip_ratio)
    n = len(p)
    t0 = time.time()

    # 1) Bisection to find S*
    S_bisect = _opt_max_stretch_fast(r, p, tol=1e-3)

    # 2) Build EDF schedule at S* and get realized max stretch
    C_opt = _edf_schedule_fast(r, p, S_bisect, slack=0.0)
    s_opt = _stretch_array(r, C_opt, p)
    S_emp = float(np.max(s_opt))

    # 3) Normalization anchor so OPT's max/OPT = 1.000
    normS = S_emp

    results: Dict[str, Dict] = {}

    # OPT
    results["OPT (EDF at S*)"] = _compute_metrics(r, C_opt, p, normS)

    # Clairvoyant
    C_srpt = _srpt_fast(r, p, p.copy())
    results["SRPT (true)"] = _compute_metrics(r, C_srpt, p, normS)

    # Non-clairvoyant baselines
    C_fifo = _fifo_fast(r, p)
    results["FIFO"] = _compute_metrics(r, C_fifo, p, normS)

    C_las = _las_fast(r, p)
    results["LAS/FB"] = _compute_metrics(r, C_las, p, normS)

    # Prediction-based
    C_sprpt = _srpt_fast(r, p, q.copy())
    results["SPRPT (pred)"] = _compute_metrics(r, C_sprpt, p, normS)

    # FIX: use S* directly as factor, not clamped to [5, 20]
    C_edfp = _edf_pred_deadlines(r, p, q, factor=float(S_bisect))
    results["EDF-P (pred)"] = _compute_metrics(r, C_edfp, p, normS)

    elapsed = time.time() - t0
    results["_meta"] = {
        "n": n,
        "S_bisect": float(S_bisect),
        "S_emp": float(S_emp),
        "elapsed_sec": float(elapsed),
    }

    if verbose:
        print("\n" + "=" * 96)
        print(f"MAX-STRETCH  1|r_j,pmtn|S_max  (n={n:,}, time={elapsed:.2f}s)")
        print(f"S* (bisection) = {S_bisect:.4f}   |   S_emp (EDF realized) = {S_emp:.4f}")
        print("-" * 96)
        hdr = f"{'Algorithm':22s} {'S_max':>10s} {'S_99':>10s} {'S_med':>10s} {'ρ_max':>9s} {'ρ_99':>9s} {'ρ_med':>9s}"
        print(hdr)
        print("-" * 96)
        order = ["OPT (EDF at S*)", "SRPT (true)", "FIFO", "LAS/FB",
                 "SPRPT (pred)", "EDF-P (pred)"]
        for name in order:
            m = results[name]
            print(f"{name:22s} {m['max']:10.2f} {m['p99']:10.2f} {m['median']:10.2f} "
                  f"{m['max_over_OPT']:9.3f} {m['p99_over_OPT']:9.3f} {m['med_over_OPT']:9.3f}")

        print(f"\nBounded Slowdown (tau=10):  {'max':>10s} {'p99':>10s} {'median':>10s}")
        for name in order:
            m = results[name]
            print(f"  {name:22s} {m['bsld10_max']:10.2f} {m['bsld10_p99']:10.2f} {m['bsld10_med']:10.2f}")
        print("=" * 96)

    return results

# -------------------- Multi-method evaluation -----------------------
def run_max_stretch_all_methods(
    df,
    idx_te,
    predictions: Dict[str, np.ndarray],
    sample_size: Optional[int] = 5000,
    clip_ratio: Optional[Tuple[float, float]] = (0.1, 10.0),
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate max-stretch across all prediction methods.

    Args:
        df: Full dataframe
        idx_te: Test indices
        predictions: Dict mapping method name -> prediction array
        sample_size: Number of jobs
        clip_ratio: Prediction clipping bounds
        verbose: Print results
    """
    # Prepare shared job data (same sample for all methods)
    r, p, _ = _prepare_jobs(df, idx_te,
                            np.zeros(len(df.loc[idx_te])),  # dummy
                            sample_size=sample_size, clip_ratio=None)
    n = len(p)

    # Shared OPT computation (only depends on r, p)
    t0 = time.time()
    S_bisect = _opt_max_stretch_fast(r, p, tol=1e-3)
    C_opt = _edf_schedule_fast(r, p, S_bisect, slack=0.0)
    s_opt = _stretch_array(r, C_opt, p)
    S_emp = float(np.max(s_opt))
    normS = S_emp

    # Shared baselines
    C_srpt = _srpt_fast(r, p, p.copy())
    C_fifo = _fifo_fast(r, p)
    C_las = _las_fast(r, p)

    all_results = {
        "_meta": {
            "n": n,
            "S_bisect": float(S_bisect),
            "S_emp": float(S_emp),
            "sample_size": sample_size,
        },
        "baselines": {
            "OPT (EDF at S*)": _compute_metrics(r, C_opt, p, normS),
            "SRPT (true)": _compute_metrics(r, C_srpt, p, normS),
            "FIFO": _compute_metrics(r, C_fifo, p, normS),
            "LAS/FB": _compute_metrics(r, C_las, p, normS),
        },
        "methods": {},
    }

    if verbose:
        print("=" * 96)
        print(f"MAX-STRETCH  1|r_j,pmtn|S_max  (n={n:,})")
        print(f"S* = {S_bisect:.4f}   |   S_emp = {S_emp:.4f}")
        print("-" * 96)

    # Per-method evaluation
    for method_name, y_pred_raw in predictions.items():
        # Re-prepare with this method's predictions (same sample via same seed)
        _, _, q = _prepare_jobs(df, idx_te, y_pred_raw,
                                sample_size=sample_size, clip_ratio=clip_ratio)

        C_sprpt = _srpt_fast(r, p, q.copy())
        C_edfp = _edf_pred_deadlines(r, p, q, factor=float(S_bisect))

        all_results["methods"][method_name] = {
            "SPRPT": _compute_metrics(r, C_sprpt, p, normS),
            "EDF-P": _compute_metrics(r, C_edfp, p, normS),
        }

        if verbose:
            m_sprpt = all_results["methods"][method_name]["SPRPT"]
            m_edfp = all_results["methods"][method_name]["EDF-P"]
            print(f"{method_name:8s}  SPRPT ρ_max={m_sprpt['max_over_OPT']:.3f}  "
                  f"ρ_99={m_sprpt['p99_over_OPT']:.3f}  ρ_med={m_sprpt['med_over_OPT']:.3f}  |  "
                  f"EDF-P ρ_max={m_edfp['max_over_OPT']:.3f}  "
                  f"ρ_99={m_edfp['p99_over_OPT']:.3f}  ρ_med={m_edfp['med_over_OPT']:.3f}")

    elapsed = time.time() - t0
    all_results["_meta"]["elapsed_sec"] = float(elapsed)

    if verbose:
        print("-" * 96)
        print(f"\nBASELINES:")
        print(f"{'Algorithm':22s} {'ρ_max':>9s} {'ρ_99':>9s} {'ρ_med':>9s}")
        for name in ["OPT (EDF at S*)", "SRPT (true)", "FIFO", "LAS/FB"]:
            m = all_results["baselines"][name]
            print(f"{name:22s} {m['max_over_OPT']:9.3f} {m['p99_over_OPT']:9.3f} {m['med_over_OPT']:9.3f}")
        print(f"\nTotal time: {elapsed:.1f}s")
        print("=" * 96)

    return all_results

# -------------------------- Usage -----------------------------------
