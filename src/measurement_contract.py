from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


SERVING_FOOTPRINT_MODE = "serialized_artifacts_only"
SERVING_FOOTPRINT_MODE_WITH_VECTORS = "serialized_artifacts_plus_external_vector_payload"
SERVING_PAYLOAD_MODE = SERVING_FOOTPRINT_MODE_WITH_VECTORS
INDEX_OVERHEAD_MODE = "index_overhead_manifest"
QUERY_MEM_MODE_IN_PROCESS = "in_process_rss_delta"
QUERY_MEM_MODE_ISOLATED = "isolated_subprocess_rss_delta"
AVG_COMPUTATIONS_MODE_IVF = "avg_exact_scored_ivf_candidates_per_query"
AVG_COMPUTATIONS_MODE_HNSW = "avg_distance_computations_per_query"
AVG_COMPUTATIONS_MODE_ROUTER = "avg_exact_scored_router_candidates_per_query"
AVG_COMPUTATIONS_MODE_IVFPQ = "avg_full_adc_codes_per_query"
AVG_COMPUTATIONS_MODE_LIRA = "avg_unique_global_rerank_candidates_per_query"
WORK_EQ_MODE = "analytical_distance_scoring_equivalent"
CPU_TIME_MODE = "process_cpu_time_batch_256"
TIMING_STATISTIC = "median"
TIMING_TRIALS = 3
WARMUP_QUERIES = 32
BENCHMARK_BATCH_1 = 1
BENCHMARK_BATCH_256 = 256


@dataclass(frozen=True)
class TimingSummary:
    """Deterministic summary of repeated end-to-end query passes."""

    batch_size: int
    query_count: int
    warmup_queries: int
    wall_seconds_median: float
    wall_seconds_iqr: float
    process_cpu_seconds_median: float
    process_cpu_seconds_iqr: float
    qps_median: float
    qps_iqr: float
    selected_trial: int
    wall_seconds_trials: tuple[float, ...]
    process_cpu_seconds_trials: tuple[float, ...]

    def to_metrics(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_batch_size": int(self.batch_size),
            f"{prefix}_warmup_queries": int(self.warmup_queries),
            f"{prefix}_timing_trials": len(self.wall_seconds_trials),
            f"{prefix}_timing_statistic": TIMING_STATISTIC,
            f"{prefix}_wall_s_median": float(self.wall_seconds_median),
            f"{prefix}_wall_s_iqr": float(self.wall_seconds_iqr),
            f"{prefix}_process_cpu_s_median": float(self.process_cpu_seconds_median),
            f"{prefix}_process_cpu_s_iqr": float(self.process_cpu_seconds_iqr),
            f"{prefix}_qps_iqr": float(self.qps_iqr),
            f"{prefix}_selected_trial": int(self.selected_trial),
            f"{prefix}_wall_s_trials": ",".join(f"{value:.9g}" for value in self.wall_seconds_trials),
            f"{prefix}_process_cpu_s_trials": ",".join(
                f"{value:.9g}" for value in self.process_cpu_seconds_trials
            ),
        }


def summarize_timing_trials(
    *,
    batch_size: int,
    query_count: int,
    warmup_queries: int,
    wall_seconds: list[float] | tuple[float, ...],
    process_cpu_seconds: list[float] | tuple[float, ...],
) -> TimingSummary:
    """Summarize repeated full-path passes and identify the median-wall trial."""

    wall = np.asarray(wall_seconds, dtype=np.float64)
    cpu = np.asarray(process_cpu_seconds, dtype=np.float64)
    if wall.ndim != 1 or cpu.ndim != 1 or wall.size != cpu.size or wall.size != TIMING_TRIALS:
        raise ValueError(f"exactly {TIMING_TRIALS} paired timing trials are required")
    if int(batch_size) <= 0 or int(query_count) <= 0:
        raise ValueError("batch_size and query_count must be positive")
    if int(warmup_queries) < 0 or int(warmup_queries) > int(query_count):
        raise ValueError("warmup_queries must be between zero and query_count")
    if not np.all(np.isfinite(wall)) or not np.all(np.isfinite(cpu)):
        raise ValueError("timing trials must be finite")
    if np.any(wall <= 0.0) or np.any(cpu < 0.0):
        raise ValueError("wall times must be positive and CPU times must be non-negative")
    order = np.argsort(wall, kind="stable")
    selected = int(order[wall.size // 2])
    qps = float(query_count) / wall
    percentile = np.percentile
    return TimingSummary(
        batch_size=int(batch_size),
        query_count=int(query_count),
        warmup_queries=int(warmup_queries),
        wall_seconds_median=float(np.median(wall)),
        wall_seconds_iqr=float(percentile(wall, 75) - percentile(wall, 25)),
        process_cpu_seconds_median=float(np.median(cpu)),
        process_cpu_seconds_iqr=float(percentile(cpu, 75) - percentile(cpu, 25)),
        qps_median=float(np.median(qps)),
        qps_iqr=float(percentile(qps, 75) - percentile(qps, 25)),
        selected_trial=selected,
        wall_seconds_trials=tuple(float(value) for value in wall),
        process_cpu_seconds_trials=tuple(float(value) for value in cpu),
    )


def timing_contract_metrics(*, batch1: TimingSummary, batch256: TimingSummary) -> dict[str, object]:
    """Return the frozen serving-timing fields shared by every method family."""

    if batch1.query_count != batch256.query_count:
        raise ValueError("batch-1 and batch-256 passes must evaluate the same queries")
    if batch1.batch_size != BENCHMARK_BATCH_1 or batch256.batch_size != BENCHMARK_BATCH_256:
        raise ValueError("timing summaries must use the frozen batch-1 and batch-256 modes")
    if batch1.warmup_queries != batch256.warmup_queries:
        raise ValueError("batch-1 and batch-256 passes must use the same warm-up query count")
    query_count = int(batch256.query_count)
    return {
        "qps": float(batch256.qps_median),
        "qps_batch_1": float(batch1.qps_median),
        "qps_batch_256": float(batch256.qps_median),
        "latency_ms_batch_1_mean": float(1000.0 * batch1.wall_seconds_median / query_count),
        "cpu_time_ms_per_query_batch_256": float(
            1000.0 * batch256.process_cpu_seconds_median / query_count
        ),
        "cpu_time_mode": CPU_TIME_MODE,
        "timing_trials": TIMING_TRIALS,
        "timing_statistic": TIMING_STATISTIC,
        "warmup_queries": int(batch1.warmup_queries),
        "benchmark_batch_sizes": f"{BENCHMARK_BATCH_1},{BENCHMARK_BATCH_256}",
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
        "gomp_spincount": os.environ.get("GOMP_SPINCOUNT"),
        **batch1.to_metrics("batch1"),
        **batch256.to_metrics("batch256"),
    }


def serving_payload_metrics(
    *,
    serialized_artifacts_mb: float,
    external_vector_payload_mb: float = 0.0,
    canonical_vector_payload_mb: float = 0.0,
    requires_full_vectors: bool | None = None,
) -> dict[str, float | str | bool]:
    """Return explicit, non-overlapping serving-storage fields.

    ``serialized_artifacts_mb`` counts method-owned files loaded for serving.
    ``external_vector_payload_mb`` counts the raw vector store required for an
    exact scan or rerank when that store is not serialized in the method's
    experiment directory.  Their sum is a deterministic payload estimate,
    not process RSS; isolated serving RSS is measured separately.
    """

    serialized = float(serialized_artifacts_mb)
    external = float(external_vector_payload_mb)
    canonical = float(canonical_vector_payload_mb)
    if not all(np.isfinite(value) and value >= 0.0 for value in (serialized, external, canonical)):
        raise ValueError("serving payload inputs must be finite and non-negative")
    total = serialized + external
    requires_vectors = bool(external > 0.0) if requires_full_vectors is None else bool(requires_full_vectors)
    if external > 0.0 and not requires_vectors:
        raise ValueError("external vector payload requires requires_full_vectors=True")
    if requires_vectors and total + 1e-9 < canonical:
        raise ValueError("full-vector serving payload cannot be smaller than one canonical corpus")
    incremental = total - canonical if requires_vectors else total
    with_vectors = incremental + canonical
    return {
        "serialized_artifacts_mb": serialized,
        "external_vector_payload_mb": external,
        "total_serving_payload_mb": total,
        "canonical_vector_payload_mb": canonical,
        "incremental_structure_mb": incremental,
        "index_size_without_vectors_mb": incremental,
        "index_size_with_vectors_mb": with_vectors,
        "required_serving_payload_mb": total,
        "requires_full_vectors": requires_vectors,
        "requires_external_vectors": bool(external > 0.0),
        # Backward-compatible fields retained for older collectors.
        "serving_footprint_mb": total,
        "serving_footprint_mode": (
            SERVING_FOOTPRINT_MODE_WITH_VECTORS if external > 0.0 else SERVING_FOOTPRINT_MODE
        ),
    }


def distance_scoring_equivalent_work(
    *,
    dimension: int,
    exact_candidates: float = 0.0,
    candidates_before_prefix: float = 0.0,
    candidates_after_prefix: float = 0.0,
    pq_m: int = 0,
    prefix_m: int = 0,
) -> float:

    d = int(dimension)
    m = int(pq_m)
    prefix = int(prefix_m)
    if d <= 0:
        raise ValueError("dimension must be positive")
    if m < 0 or prefix < 0 or prefix > m:
        raise ValueError("require 0 <= prefix_m <= pq_m")
    exact = float(exact_candidates)
    before = float(candidates_before_prefix)
    after = float(candidates_after_prefix)
    if not all(np.isfinite(value) and value >= 0.0 for value in (exact, before, after)):
        raise ValueError("candidate counts must be finite and non-negative")
    if after > before and prefix > 0:
        raise ValueError("prefix survivors cannot exceed candidates scored by the prefix")
    return float(exact + before * (prefix / d) + after * ((m - prefix) / d))


@dataclass(frozen=True)
class CandidateStats:
    mean: float
    minimum: float
    maximum: float
    mode: str


def recall_at_k(retrieved: np.ndarray, neighbors: np.ndarray, k: int) -> float:
    if retrieved.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("retrieved and neighbors must be rank-2 arrays")
    if retrieved.shape[0] != neighbors.shape[0]:
        raise ValueError("retrieved and neighbors must have the same number of queries")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    recalls = np.zeros(retrieved.shape[0], dtype=np.float64)
    for idx in range(retrieved.shape[0]):
        hits = len(np.intersect1d(retrieved[idx, :k], neighbors[idx, :k]))
        recalls[idx] = hits / float(k)
    return float(np.mean(recalls))


def recall_profile(
    retrieved: np.ndarray,
    neighbors: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
) -> dict[int, float | None]:
    """Return Recall@k values without relabeling a shorter ranking.

    A metric is ``None`` when either the retrieved ranking or ground truth has
    fewer than ``k`` columns.  This prevents a historical ``k=10`` run from
    being presented as Recall@100 while still allowing old artifacts to be
    loaded explicitly.
    """

    if retrieved.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("retrieved and neighbors must be rank-2 arrays")
    return {
        int(k): (
            recall_at_k(retrieved, neighbors, int(k))
            if retrieved.shape[1] >= int(k) and neighbors.shape[1] >= int(k)
            else None
        )
        for k in ks
    }


def per_query_recall_profile(
    retrieved: np.ndarray,
    neighbors: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
) -> dict[int, np.ndarray | None]:
    """Per-query companion to :func:`recall_profile`."""

    return {
        int(k): (
            per_query_recalls(retrieved, neighbors, int(k))
            if retrieved.shape[1] >= int(k) and neighbors.shape[1] >= int(k)
            else None
        )
        for k in ks
    }


def per_query_recalls(retrieved: np.ndarray, neighbors: np.ndarray, k: int) -> np.ndarray:
    """Return per-query recall@k as a float64 array of shape (n_queries,)."""
    k = int(k)
    recalls = np.zeros(retrieved.shape[0], dtype=np.float64)
    for idx in range(retrieved.shape[0]):
        hits = len(np.intersect1d(retrieved[idx, :k], neighbors[idx, :k]))
        recalls[idx] = hits / float(k)
    return recalls


def ivf_candidate_stats(list_sizes: np.ndarray, probe_lists: np.ndarray) -> CandidateStats:
    if list_sizes.ndim != 1 or probe_lists.ndim != 2:
        raise ValueError("list_sizes must be rank 1 and probe_lists rank 2")
    valid = (probe_lists >= 0) & (probe_lists < list_sizes.shape[0])
    safe_ids = np.where(valid, probe_lists, 0)
    selected = list_sizes[safe_ids]
    counts = np.sum(np.where(valid, selected, 0), axis=1).astype(np.float64, copy=False)
    return CandidateStats(
        mean=float(np.mean(counts)),
        minimum=float(np.min(counts)),
        maximum=float(np.max(counts)),
        mode=AVG_COMPUTATIONS_MODE_IVF,
    )


def hnsw_candidate_stats(ndis_total: float, total_queries: int) -> CandidateStats:
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    mean = float(ndis_total) / float(total_queries)
    return CandidateStats(
        mean=mean,
        minimum=mean,
        maximum=mean,
        mode=AVG_COMPUTATIONS_MODE_HNSW,
    )
