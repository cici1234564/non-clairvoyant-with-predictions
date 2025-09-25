# Dataset and Benchmark Project

## Overview
The ATLAS dataset is built on the publicly available Alibaba PAI--2020 GPU-cluster trace, capturing two months of MLaaS activity on a large heterogeneous GPU cluster with over 6,500 GPUs across ~1,800 machines. The dataset provides a comprehensive view of job life-cycles, resource requests, utilization, and scheduling in production ML workloads.

## Dataset Download and Usage
To use the ATLAS dataset and benchmark:
1. Download the following three files.
2. Open and run the `Dataset_demonstration.ipynb` notebook to explore and analyze the dataset.

## Dataset Description
### Data Source and Formalization
  - Captures job submission, queuing, scheduling, and execution.
  - Hierarchical structure: jobs → tasks → instances.
  - Users submit ML jobs via frameworks (TensorFlow, PyTorch, Graph-Learn), which are scheduled and instantiated as Docker containers across machines.
  - Per-instance system metrics (CPU/GPU utilization, memory) are collected every 15 seconds.
  - Machine-level statistics include network throughput and hardware specs.
  - The dataset formalizes trace columns for clear time semantics, label construction, and reproducibility.

### Job Life-Cycle
- Jobs arrive and consist of multiple tasks, each with resource demands (GPU, CPU, memory), instance counts, and constraints (GPU type, gang scheduling, locality).
- The scheduler seeks feasible placements for tasks based on resource availability and constraints.
- Gang scheduling and locality are enforced for certain tasks.
- Job start time, queuing delay, and processing time are tracked.
- Resource metrics are aggregated at instance, task, job, and cluster levels.

### Revised Data Columns
- Submit-time features are extracted from job, task, and group-tag tables.
- Instance and sensor tables are excluded to focus on non-clairvoyant scheduling.
- Features include job start time, user ID, resource requirements, parallelism metrics, and semantic identifiers.

### Workload Characterization
- Only terminated jobs/tasks are kept; incomplete or invalid rows are dropped.
- Timestamps are standardized, and only complete weeks are used for time-series analysis.
- The workload is highly heterogeneous, with job sizes and resource requests spanning several orders of magnitude.
- Temporal patterns show diurnal and weekly regularity in job submissions and resource demand.

## Benchmark Tasks
The benchmark covers learning-augmented non-clairvoyant scheduling tasks. Evaluation metrics include job processing time, resource utilization, and scheduling efficiency. Baselines and reference results are provided in the demonstration notebook.

# Benchmark

## Preparation
**Data pre-processing:**
We use the ATLAS dataset, which constructs a submit-time, job-level table by joining the job, task, and group-tag relations from the ALIBABA PAI trace and retaining only terminated records. Timestamps are parsed as seconds and empty rows removed. For each task $t$ with instances, we set $s_t=\min_i s_{t,i}$ and $e_t=\max_i e_{t,i}$, and define the job processing time as $p_j^*=\max_t e_t - \min_t s_t$. Jobs with $p_j^*\le 0$ are discarded. The submission time $r_j$ anchors chronology and all causal features. Submit-time resource declarations are aggregated per job by summing the times of per-task plans multiplied by their multiplicity. We join the group tag, user identifier, workload tag, and requested GPU specification via the instance identifier; assigned hardware and any post-submission outcomes are excluded. To avoid leakage, we split by $r_j$: the earliest 70% for training, next 15% for validation, and final 15% for testing. Before training, we run simple checks and use $y_j=\log(1+p_j^*)$ as the prediction target to stabilize heavy tails.

**To run the benchmark tasks (job size prediction and scheduling), open and execute the `Prediction+Scheduling.ipynb` notebook. This notebook guides you through the process of running prediction models and scheduling algorithms on the ATLAS dataset.**

To run the benchmark tasks (job size prediction and scheduling), open and execute the `Prediction+Scheduling.ipynb` notebook. This notebook guides you through the process of running prediction models and scheduling algorithms on the ATLAS dataset.

## Prediction Task
**Feature engineering:**
We extract submit-time features in five categories—resources, temporal patterns, recurrence signatures, historical, and categorical encoding—with all thresholds, statistics, and encoders fit on training data only to prevent leakage.

**Resources / Logs / Ratios (9):**
log_total_plan_cpu, log_total_plan_gpu, log_total_plan_mem, log_total_inst_num, log_num_tasks, cpu_per_inst, gpu_per_inst, mem_per_inst, tasks_per_inst

**Temporal (5):**
hour, dow, sin_hour, cos_hour, is_weekend

**Signature statistics (6) (train-only stats merged back):**
sig_mean_log, sig_median_log, sig_q25_log, sig_q75_log, sig_count, sig_mean_shrink

**Causal histories (8) (within-group/user, with shift(1)):**
gro_hist_mean, gro_hist_count, gro_ewm, gro_dt_prev, use_hist_mean, use_hist_count, use_ewm, use_dt_prev

**Prior (1):**
grp_mean_eb

**Categoricals (4), label-encoded from train:**
user_enc, group_enc, workload_enc, gpu_type_spec_enc

These features are designed to capture resource planning, temporal patterns, user/group signatures, historical behaviors, prior statistics, and categorical identifiers for effective modeling and scheduling analysis.

**Prediction models:**
We model $y_j=\log(1+p_j^*)$, where $p_j^*$ is from earliest task start to latest task end, from submit-time features $\mathbf{x}_j$ using gradient boosting with validation-based calibration. Our methods include: (1) Conformal quantile regression (CQR) training quantile regressor at $\alpha$ with Ridge-blended final predictions; (2) Isotonic calibration ensuring monotonic probability mapping and adapted for regression/uncertainty calibration; (3) Meta-stacking combines diverse base models (L2, regularized, quantile, Huber) via gradient boosting on validation predictions; (4) Gated experts (two-stage): a mixture-of-experts design in which a classifier network routes examples to capacity-matched regressor and aggregates them by soft probabilities; (5) Weighted recency uses exponential time-decay for drift adaptation; (6) Historical Recency-Aware with Shrinkage uses per-signature means with EB shrinkage to stabilize predictions for rare user-group-resource patterns. All calibrators fit exclusively on validation data following honest prediction principles, with LightGBM as our primary regressor using early stopping and monotone constraints if applicable.


**Evaluation:**
We report Root Mean Squared Logarithmic Error (RMSLE), Coverage at $\tau$ (Cov@25%, Cov@50%), and Spearman’s rank correlation for prediction error. These metrics provide a comprehensive view of model performance, especially in heavy-tailed job distributions.

## Scheduling Task
**Implementation Setup:**
LASched evaluates objectives under the following settings: for total completion time $(\sum_j C_j)$, a single machine with online arrivals and preemptive scheduling; for makespan, $m$ parallel machines with batch release and non-preemptive; and for max-stretch, a single machine with online arrivals and preemption to capture fairness and prevent starvation of large jobs. We study job-level scheduling with imperfect predictions across all job types, scaling from single-machine to thousand-machine clusters.

**Scheduling Algorithms:**
We evaluate general-purpose and objective-specific policies across all three metrics. FIFO processes jobs in arrival order and serves as a natural online baseline. SRPT preemptively runs the job with the least remaining work and is optimal for minimizing total completion time with release dates. SJF and its predicted variant SPJF are the non-preemptive size-order analogues. RR (Round Robin) shares the processor equally among all active jobs. For total completion time, we include PRR (Preferential Round-Robin), which splits rate across active jobs and assigns an extra portion to the job with the minimum predicted size. For max-stretch, we compute the OPT by bisection on $S$ using EDF (Earliest-Deadline-First) feasibility on deadlines $d_j=r_j+S p_j$. We then benchmark SPRPT (preemptive SRPT using predictions), EDF with predicted deadlines, and LAS/FB (Least-Attained-Service / Foreground–Background). For makespan, we use LPT (Longest Processing Time)—descending-size list scheduling to the least-loaded machine—and its predicted variant LPPT; we also report SPT (Shortest Processing Time) and SPPT, plus a random baseline for comparison.

**Evaluation:**
We evaluate algorithms via empirical competitive ratios against optimal solutions or tight bounds. For total completion time, we normalize by SRPT. For makespan, we use McNaughton's preemptive bound as baseline. For max-stretch, we obtain $S^*$ by bisection on $S$ with EDF-feasibility, then run EDF at $S^*$ and normalize by the realized $S_{emp}$. We report competitive ratios for all metrics.

## Usage
1. Download the required dataset files.
2. Open `Dataset_demonstration.ipynb` in Jupyter or VS Code.
3. Follow the notebook instructions to analyze the dataset and run benchmark tasks.



## Contact
For questions or contributions, please contact the dataset authors or open an issue in the repository.
