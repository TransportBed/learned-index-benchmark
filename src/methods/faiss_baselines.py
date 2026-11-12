from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from artifacts import ensure_dir, write_csv, write_json
from console import banner, info, print_table
from index_manifest import write_index_manifest
from measurement_contract import (
    BENCHMARK_BATCH_1,
    BENCHMARK_BATCH_256,
    AVG_COMPUTATIONS_MODE_IVFPQ,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    QUERY_MEM_MODE_IN_PROCESS,
    TIMING_TRIALS,
    WARMUP_QUERIES,
    distance_scoring_equivalent_work,
    hnsw_candidate_stats,
    ivf_candidate_stats,
    per_query_recall_profile,
    recall_at_k,
    recall_profile,
    serving_payload_metrics,
    summarize_timing_trials,
    timing_contract_metrics,
)
from presets import DATASETS


def _rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def _isolated_query_memory(*, exp_dir: Path, dataset: str, data_root: Path) -> dict[str, float] | None:
    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "run_faiss_query_isolated",
            "--dataset",
            dataset,
            "--data-root",
            str(data_root),
            "--experiment-dir",
            str(exp_dir),
        ],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout.strip())
    if not isinstance(payload, dict):
        return None
    return {str(key): float(value) for key, value in payload.items() if value is not None}


def _normalize_if_cosine(arr: np.ndarray, metric: str) -> np.ndarray:
    if metric != "cosine":
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _pad_vectors(arr: np.ndarray, m: int) -> np.ndarray:
    if arr.shape[1] % m == 0:
        return arr
    new_dim = int(math.ceil(arr.shape[1] / float(m)) * m)
    output = np.zeros((arr.shape[0], new_dim), dtype=arr.dtype)
    output[:, : arr.shape[1]] = arr
    return output


def _exact_scores_for_ids(
    *,
    train_view: np.ndarray,
    train_sq_norms: np.ndarray | None,
    query_vec: np.ndarray,
    ids: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric in {"ip", "cosine"}:
        return train_view[ids] @ query_vec
    if metric == "l2":
        if train_sq_norms is None:
            raise ValueError("train_sq_norms required for l2 rerank")
        q_norm = float(np.dot(query_vec, query_vec))
        return -(train_sq_norms[ids] + q_norm - 2.0 * (train_view[ids] @ query_vec))
    raise ValueError(f"Unsupported metric: {metric}")



def _rerank_candidate_ids_prepared(
    *,
    train_view: np.ndarray,
    query_view: np.ndarray,
    train_sq_norms: np.ndarray | None,
    candidate_ids: np.ndarray,
    metric: str,
    k: int,
) -> np.ndarray:
    output = np.full((candidate_ids.shape[0], int(k)), -1, dtype=np.int32)
    for idx in range(candidate_ids.shape[0]):
        ids = np.asarray(candidate_ids[idx], dtype=np.int64)
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue
        scores = _exact_scores_for_ids(
            train_view=train_view,
            train_sq_norms=train_sq_norms,
            query_vec=query_view[idx],
            ids=ids,
            metric=metric,
        )
        top_n = min(int(k), int(ids.size))
        top_idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
        output[idx, :top_n] = ids[top_idx].astype(np.int32, copy=False)
    return output


@dataclass(frozen=True)
class FaissConfig:
    dataset: str
    k: int = 100
    nprobe: int | None = 10
    pq_m: int = 16
    pq_bits: int = 8
    train_samples: int = 200_000
    index_seed: int = 0
    ivf_niter: int = 25
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    return_candidates_k: int | None = None
    query_batch_size: int = 256
    warmup_queries: int = WARMUP_QUERIES

    @property
    def partitions(self) -> int:
        return DATASETS[self.dataset].partitions

    @property
    def metric(self) -> str:
        return DATASETS[self.dataset].metric


@dataclass
class FaissBuild:
    index: Any
    train_vecs: np.ndarray
    query_vecs: np.ndarray
    list_sizes: np.ndarray | None
    rerank_train: np.ndarray
    rerank_queries: np.ndarray
    rerank_train_sq_norms: np.ndarray | None
    build_train_s: float
    build_add_s: float

    @property
    def build_total_s(self) -> float:
        return self.build_train_s + self.build_add_s


class FaissBaseline:
    name: str = ""
    index_kind: str = ""
    backend: str = "faiss"

    def __init__(self, cfg: FaissConfig) -> None:
        self.cfg = cfg

    @property
    def exact_rerank_enabled(self) -> bool:
        return self.index_kind == "ivfpq" and int(self.cfg.return_candidates_k or 0) > int(self.cfg.k)

    def _write_config(self, exp_dir: Path, *, operating_points: Iterable[int] | None = None) -> None:
        write_json(
            exp_dir / "config.json",
            {
                "method": self.name,
                "index_kind": self.index_kind,
                "backend": self.backend,
                "metric": self.cfg.metric,
                **asdict(self.cfg),
                "operating_points": list(operating_points) if operating_points is not None else None,
            },
        )

    def _build(self, train: np.ndarray, queries: np.ndarray, exp_dir: Path) -> FaissBuild:
        try:
            import faiss
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss-cpu is required for FAISS baselines") from exc

        train_vecs = _normalize_if_cosine(train, self.cfg.metric)
        query_vecs = _normalize_if_cosine(queries, self.cfg.metric)
        if self.index_kind == "ivfpq":
            train_vecs = _pad_vectors(train_vecs, self.cfg.pq_m)
            query_vecs = _pad_vectors(query_vecs, self.cfg.pq_m)
        train_vecs = train_vecs.astype(np.float32, copy=False)
        query_vecs = query_vecs.astype(np.float32, copy=False)
        n, dimension = train_vecs.shape

        banner("method", f"{self.name} | dataset={self.cfg.dataset} | index={self.index_kind}")
        info(f"train={n}, queries={query_vecs.shape[0]}, dim={dimension}, metric={self.cfg.metric}")
        sample = train_vecs
        if n > self.cfg.train_samples:
            rng = np.random.default_rng(self.cfg.index_seed)
            sample = train_vecs[rng.choice(n, size=self.cfg.train_samples, replace=False)]

        t0 = time.perf_counter()
        if self.index_kind == "ivf":
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, self.cfg.partitions, faiss.METRIC_L2)
            index.cp.seed = int(self.cfg.index_seed)
            index.cp.niter = int(self.cfg.ivf_niter)
            index.train(sample)
            build_train_s = time.perf_counter() - t0
        elif self.index_kind == "ivfpq":
            index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(dimension),
                dimension,
                self.cfg.partitions,
                self.cfg.pq_m,
                self.cfg.pq_bits,
            )
            index.cp.seed = int(self.cfg.index_seed)
            index.cp.niter = int(self.cfg.ivf_niter)
            index.pq.cp.seed = int(self.cfg.index_seed)
            index.train(sample)
            build_train_s = time.perf_counter() - t0
        elif self.index_kind == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, int(self.cfg.hnsw_m), faiss.METRIC_L2)
            index.hnsw.efConstruction = int(self.cfg.hnsw_ef_construction)
            build_train_s = 0.0
        else:  # pragma: no cover
            raise ValueError(f"Unsupported index kind: {self.index_kind}")

        t1 = time.perf_counter()
        index.add(train_vecs)
        build_add_s = time.perf_counter() - t1
        staging = ensure_dir(exp_dir / "staging")
        faiss.write_index(index, str(staging / "faiss.index"))
        write_json(
            staging / "faiss_meta.json",
            {
                "method": self.name,
                "index_kind": self.index_kind,
                "build_train_s": build_train_s,
                "build_add_s": build_add_s,
                "build_total_s": build_train_s + build_add_s,
                "ntotal": int(index.ntotal),
                "dim": int(dimension),
            },
        )

        list_sizes = None
        if self.index_kind in {"ivf", "ivfpq"}:
            _, list_ids = index.quantizer.search(train_vecs, 1)
            list_ids = list_ids.reshape(-1).astype(np.int32, copy=False)
            np.save(staging / "ivf_list_ids_index.npy", list_ids)
            list_sizes = np.bincount(list_ids, minlength=self.cfg.partitions)
        rerank_train = _normalize_if_cosine(train, self.cfg.metric)
        rerank_queries = _normalize_if_cosine(queries, self.cfg.metric)
        rerank_train_sq_norms = (
            np.sum(rerank_train * rerank_train, axis=1).astype(np.float32, copy=False)
            if self.cfg.metric == "l2"
            else None
        )
        return FaissBuild(
            index,
            train_vecs,
            query_vecs,
            list_sizes,
            rerank_train,
            rerank_queries,
            rerank_train_sq_norms,
            build_train_s,
            build_add_s,
        )

    def _set_operating_point(self, build: FaissBuild, value: int) -> tuple[str, int, int]:
        requested = int(value)
        if self.index_kind == "hnsw":
            build.index.hnsw.efSearch = requested
            return "efSearch", requested, requested
        # FAISS tolerates nprobe > nlist, but its coarse-quantizer search may
        # pad list IDs with -1. Clamp the executed value while retaining the
        # requested frontier label in provenance (relevant to glove10k, whose
        # smoke index has 64 lists while the paper sweep includes nprobe=100).
        effective = min(requested, int(build.index.nlist))
        build.index.nprobe = effective
        return "nprobe", requested, effective

    def _query(
        self,
        *,
        build: FaissBuild,
        train: np.ndarray,
        queries: np.ndarray,
        batch_size: int,
        query_limit: int | None = None,
    ) -> tuple[np.ndarray, float, float, float | None, float | None]:
        try:
            import faiss
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss-cpu is required") from exc
        return_k = max(self.cfg.k, int(self.cfg.return_candidates_k or self.cfg.k))
        del train, queries  # the prepared views in ``build`` are authoritative
        query_count = int(build.query_vecs.shape[0])
        if query_limit is not None:
            query_count = min(query_count, max(0, int(query_limit)))
        query_vecs = build.query_vecs[:query_count]
        if self.index_kind == "hnsw":
            try:
                faiss.cvar.hnsw_stats.reset()
            except Exception:
                pass
        search_t0 = time.perf_counter()
        parts: list[np.ndarray] = []
        for start in range(0, query_count, int(batch_size)):
            end = min(start + int(batch_size), query_count)
            _, ids = build.index.search(query_vecs[start:end], return_k)
            parts.append(ids.astype(np.int32, copy=False))
        candidates = np.concatenate(parts, axis=0)
        search_s = time.perf_counter() - search_t0
        hnsw_ndis_total = None
        if self.index_kind == "hnsw":
            try:
                hnsw_ndis_total = float(faiss.cvar.hnsw_stats.ndis)
            except Exception:
                pass
        rerank_s = 0.0
        if self.exact_rerank_enabled:
            exact_rerank_count = float(np.mean(np.sum(candidates >= 0, axis=1)))
            rerank_t0 = time.perf_counter()
            retrieved = _rerank_candidate_ids_prepared(
                train_view=build.rerank_train,
                query_view=build.rerank_queries[:query_count],
                train_sq_norms=build.rerank_train_sq_norms,
                candidate_ids=candidates,
                metric=self.cfg.metric,
                k=self.cfg.k,
            )
            rerank_s = time.perf_counter() - rerank_t0
        else:
            retrieved = candidates[:, : self.cfg.k]
            exact_rerank_count = None
        return retrieved, search_s, rerank_s, hnsw_ndis_total, exact_rerank_count

    def _benchmark_query_mode(
        self,
        *,
        build: FaissBuild,
        train: np.ndarray,
        queries: np.ndarray,
        batch_size: int,
    ) -> tuple[tuple[np.ndarray, float, float, float | None, float | None], object]:
        """Warm once, execute three full-path trials, and select median wall time."""

        warmup_queries = min(int(self.cfg.warmup_queries), int(build.query_vecs.shape[0]))
        if warmup_queries > 0:
            self._query(
                build=build,
                train=train,
                queries=queries,
                batch_size=int(batch_size),
                query_limit=warmup_queries,
            )
        results: list[tuple[np.ndarray, float, float, float | None, float | None]] = []
        walls: list[float] = []
        cpu_times: list[float] = []
        for _ in range(TIMING_TRIALS):
            cpu_start = time.process_time()
            result = self._query(
                build=build,
                train=train,
                queries=queries,
                batch_size=int(batch_size),
            )
            cpu_times.append(float(time.process_time() - cpu_start))
            results.append(result)
            walls.append(float(result[1] + result[2]))
        summary = summarize_timing_trials(
            batch_size=int(batch_size),
            query_count=int(build.query_vecs.shape[0]),
            warmup_queries=warmup_queries,
            wall_seconds=walls,
            process_cpu_seconds=cpu_times,
        )
        return results[summary.selected_trial], summary

    def _evaluate(
        self,
        *,
        build: FaissBuild,
        train: np.ndarray,
        queries: np.ndarray,
        neighbors: np.ndarray,
        exp_dir: Path,
        data_root: Path,
        operating_point: int,
        isolated_mem: dict[str, float] | None,
    ) -> dict[str, float | int | str | bool | None]:
        op_name, op_requested, op_effective = self._set_operating_point(build, operating_point)
        mem_before = _rss_mb()
        batch256_result, batch256_timing = self._benchmark_query_mode(
            build=build,
            train=train,
            queries=queries,
            batch_size=BENCHMARK_BATCH_256,
        )
        retrieved, search_s, rerank_s, hnsw_ndis_total, exact_rerank_count = batch256_result
        total_s = float(batch256_timing.wall_seconds_median)
        _, batch1_timing = self._benchmark_query_mode(
            build=build,
            train=train,
            queries=queries,
            batch_size=BENCHMARK_BATCH_1,
        )
        mem_after = _rss_mb()

        recalls = recall_profile(retrieved, neighbors)
        if recalls[10] is None:
            available_k = min(10, retrieved.shape[1], neighbors.shape[1])
            recalls[10] = recall_at_k(retrieved, neighbors, available_k)
        candidate_counts_per_query: np.ndarray | None = None
        if self.index_kind == "hnsw" and hnsw_ndis_total is not None:
            candidate_stats = hnsw_candidate_stats(hnsw_ndis_total, retrieved.shape[0])
        elif build.list_sizes is not None:
            _, probe_lists = build.index.quantizer.search(build.query_vecs, int(op_effective))
            candidate_counts_per_query = np.sum(build.list_sizes[probe_lists], axis=1).astype(
                np.float64, copy=False
            )
            candidate_stats = ivf_candidate_stats(build.list_sizes, probe_lists)
            if self.index_kind == "ivfpq":
                candidate_stats = type(candidate_stats)(
                    mean=candidate_stats.mean,
                    minimum=candidate_stats.minimum,
                    maximum=candidate_stats.maximum,
                    mode=AVG_COMPUTATIONS_MODE_IVFPQ,
                )
        else:  # pragma: no cover
            raise RuntimeError("Candidate accounting unavailable")

        manifest = json.loads(write_index_manifest(exp_dir).read_text(encoding="utf-8"))
        canonical_vector_payload_mb = float(train.nbytes) / (1024.0 * 1024.0)
        # IVF and HNSW serialize their full vectors inside ``faiss.index``;
        # adding the corpus again as an external dependency would double-count
        # the same canonical payload. IVFPQ+R is the only FAISS baseline here
        # that needs a separate raw-vector store for terminal exact reranking.
        payload = serving_payload_metrics(
            serialized_artifacts_mb=float(manifest["serving_footprint_mb"]),
            external_vector_payload_mb=(
                canonical_vector_payload_mb if self.exact_rerank_enabled else 0.0
            ),
            canonical_vector_payload_mb=canonical_vector_payload_mb,
            requires_full_vectors=(self.index_kind != "ivfpq" or self.exact_rerank_enabled),
        )
        timing_metrics = timing_contract_metrics(batch1=batch1_timing, batch256=batch256_timing)
        exact_rerank_work = float(exact_rerank_count or 0.0)
        if self.index_kind == "ivfpq":
            work_equiv = distance_scoring_equivalent_work(
                dimension=int(train.shape[1]),
                exact_candidates=exact_rerank_work,
                candidates_after_prefix=candidate_stats.mean,
                pq_m=int(self.cfg.pq_m),
            )
        else:
            work_equiv = float(candidate_stats.mean)

        query_mem_delta_mb = max(0.0, float(mem_after) - float(mem_before))
        query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
        if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
            query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
            query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED
        operating_point_id = f"{op_name.lower()}-{op_requested}"
        metrics: dict[str, float | int | str | bool | None] = {
            "dataset": self.cfg.dataset,
            "method": self.name,
            "seed": int(self.cfg.index_seed),
            "index_kind": self.index_kind,
            "operating_point_id": operating_point_id,
            "recall_at_1": recalls[1],
            "recall_at_10": recalls[10],
            "recall_at_100": recalls[100],
            "avg_computations": candidate_stats.mean,
            "avg_computations_mode": candidate_stats.mode,
            "avg_candidates": candidate_stats.mean,
            "avg_candidates_mode": candidate_stats.mode,
            "computation_min": candidate_stats.minimum,
            "computation_max": candidate_stats.maximum,
            **timing_metrics,
            "qps_batch_1_contract": "faiss_end_to_end_batch_1",
            "qps_batch_256_contract": "faiss_end_to_end_batch_256",
            "qps_native_batched": float(batch256_timing.qps_median),
            "qps_native_batched_contract": "faiss_end_to_end_batch_256",
            "index_overhead_mb": float(manifest["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            **payload,
            "index_size_mb": float(manifest["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": query_mem_delta_mb,
            "query_mem_delta_mode": query_mem_delta_mode,
            "query_mem_measurement_operating_point": "shared_index_default_query_path",
            "build_train_s": build.build_train_s,
            "build_add_s": build.build_add_s,
            "build_total_s": build.build_total_s,
            "faiss_search_s": search_s,
            "faiss_rerank_s": rerank_s,
            "faiss_total_query_s": total_s,
            "query_batch_size": BENCHMARK_BATCH_256,
            "candidates_before_prefilter": candidate_stats.mean,
            "candidates_after_prefilter": candidate_stats.mean,
            "full_adc_candidates": candidate_stats.mean if self.index_kind == "ivfpq" else None,
            "exact_distance_candidates": (
                exact_rerank_work if self.exact_rerank_enabled else (candidate_stats.mean if self.index_kind != "ivfpq" else 0.0)
            ),
            "exact_rerank_count": exact_rerank_count,
            "exact_rerank_enabled": self.exact_rerank_enabled,
            "pq_m": int(self.cfg.pq_m) if self.index_kind == "ivfpq" else None,
            "pq_bits": int(self.cfg.pq_bits) if self.index_kind == "ivfpq" else None,
            "prefix_m": 0 if self.index_kind == "ivfpq" else None,
            "dimension": int(train.shape[1]),
            "distance_scoring_equivalent_work": work_equiv,
            "distance_scoring_equivalent_mode": "analytical_distance_scoring_equivalent",
            "queries_evaluated": int(retrieved.shape[0]),
            "k": int(self.cfg.k),
            "nprobe": int(op_effective) if op_name == "nprobe" else None,
            "nprobe_requested": int(op_requested) if op_name == "nprobe" else None,
            "nprobe_effective": int(op_effective) if op_name == "nprobe" else None,
            "hnsw_ef_search": int(op_effective) if op_name == "efSearch" else None,
            "hnsw_ndis_total": hnsw_ndis_total,
            "experiment_dir": str(exp_dir),
        }
        if isolated_mem is not None:
            metrics.update(isolated_mem)
        operating_dir = ensure_dir(exp_dir / "operating_points")
        write_json(operating_dir / f"{operating_point_id}.json", metrics)
        per_query = per_query_recall_profile(retrieved, neighbors)
        sidecar_rows: list[dict[str, object]] = []
        for query_id in range(retrieved.shape[0]):
            sidecar_rows.append(
                {
                    "query_id": query_id,
                    "recall_at_1": float(per_query[1][query_id]) if per_query[1] is not None else None,
                    "recall_at_10": float(per_query[10][query_id]) if per_query[10] is not None else None,
                    "recall_at_100": float(per_query[100][query_id]) if per_query[100] is not None else None,
                    "candidate_count": (
                        float(candidate_counts_per_query[query_id])
                        if candidate_counts_per_query is not None
                        else None
                    ),
                    "candidate_count_available": candidate_counts_per_query is not None,
                }
            )
        write_csv(operating_dir / f"{operating_point_id}_per_query.csv", sidecar_rows)
        return metrics

    def run_sweep(
        self,
        train: np.ndarray,
        queries: np.ndarray,
        neighbors: np.ndarray,
        exp_dir: Path,
        *,
        data_root: Path,
        operating_points: Iterable[int],
    ) -> list[dict[str, float | int | str | bool | None]]:
        points = tuple(int(value) for value in operating_points)
        if not points:
            raise ValueError("operating_points must not be empty")
        self._write_config(exp_dir, operating_points=points)
        build = self._build(train, queries, exp_dir)
        # Isolated RSS is invariant to nprobe/efSearch at this granularity and
        # is intentionally measured once from the immutable serialized index.
        isolated_mem = _isolated_query_memory(exp_dir=exp_dir, dataset=self.cfg.dataset, data_root=data_root)
        rows = [
            self._evaluate(
                build=build,
                train=train,
                queries=queries,
                neighbors=neighbors,
                exp_dir=exp_dir,
                data_root=data_root,
                operating_point=value,
                isolated_mem=isolated_mem,
            )
            for value in points
        ]
        write_json(
            exp_dir / "timing_breakdown.json",
            {
                "operating_points": [
                    {
                        key: row.get(key)
                        for key in row
                        if key.startswith("batch1_")
                        or key.startswith("batch256_")
                        or key
                        in {
                            "operating_point_id",
                            "qps_batch_1",
                            "qps_batch_256",
                            "qps_native_batched",
                            "qps_batch_1_contract",
                            "qps_batch_256_contract",
                            "qps_native_batched_contract",
                            "cpu_time_ms_per_query_batch_256",
                            "cpu_time_mode",
                            "timing_trials",
                            "timing_statistic",
                            "warmup_queries",
                            "omp_wait_policy",
                            "gomp_spincount",
                        }
                    }
                    for row in rows
                ]
            },
        )
        write_json(
            exp_dir / "candidate_budget_breakdown.json",
            {
                "operating_points": [
                    {
                        key: row.get(key)
                        for key in (
                            "operating_point_id",
                            "avg_computations",
                            "avg_computations_mode",
                            "candidates_before_prefilter",
                            "candidates_after_prefilter",
                            "exact_rerank_count",
                            "distance_scoring_equivalent_work",
                            "pq_m",
                            "prefix_m",
                        )
                    }
                    for row in rows
                ]
            },
        )
        write_json(exp_dir / "metrics.json", rows)
        return rows

    def run(
        self,
        train: np.ndarray,
        queries: np.ndarray,
        neighbors: np.ndarray,
        exp_dir: Path,
        *,
        data_root: Path,
    ) -> dict[str, float | int | str | bool | None]:
        default = int(self.cfg.hnsw_ef_search if self.index_kind == "hnsw" else (self.cfg.nprobe or 10))
        row = self.run_sweep(
            train,
            queries,
            neighbors,
            exp_dir,
            data_root=data_root,
            operating_points=(default,),
        )[0]
        write_json(exp_dir / "metrics.json", row)
        print_table(
            "Query summary",
            ["metric", "value"],
            [
                ["Recall@10", f"{float(row['recall_at_10']):.4f}"],
                ["Avg computations", f"{float(row['avg_computations']):.1f}"],
                ["QPS (batch 256)", f"{float(row['qps']):.1f}"],
                ["Index overhead (MB)", f"{float(row['index_overhead_mb']):.2f}"],
                ["Serving payload (MB)", f"{float(row['total_serving_payload_mb']):.2f}"],
            ],
        )
        return row


class Ivf(FaissBaseline):
    name = "IVF"
    index_kind = "ivf"


class IvfPQ(FaissBaseline):
    name = "IVFPQ"
    index_kind = "ivfpq"


class IvfPQR(FaissBaseline):
    name = "IVFPQ+R"
    index_kind = "ivfpq"


class Hnsw(FaissBaseline):
    name = "HNSW"
    index_kind = "hnsw"


def make_method(method: str, dataset: str, *, seed: int = 0) -> FaissBaseline:
    method_key = method.strip().upper()
    cfg = FaissConfig(dataset=dataset, index_seed=seed)
    if method_key == "IVF":
        return Ivf(cfg)
    if method_key == "IVFPQ":
        return IvfPQ(cfg)
    if method_key in {"IVFPQ+R", "IVFPQ_R"}:
        return IvfPQR(FaissConfig(dataset=dataset, index_seed=seed, return_candidates_k=256))
    if method_key == "HNSW":
        return Hnsw(cfg)
    raise ValueError(f"Unsupported FAISS method: {method}")
