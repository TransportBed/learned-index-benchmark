from __future__ import annotations

import json
import os
import random
import re
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ann_ops import distance_to_centroids, maybe_set_threads, normalize_rows, scaled_centroid_distances_with_scaler
from artifacts import ensure_dir, write_csv, write_json
from console import info
from datasets import LearnedDataset, _exact_query_knn, load_learned_dataset
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_LIRA,
    BENCHMARK_BATCH_1,
    BENCHMARK_BATCH_256,
    QUERY_MEM_MODE_IN_PROCESS,
    QUERY_MEM_MODE_ISOLATED,
    SERVING_FOOTPRINT_MODE,
    TIMING_TRIALS,
    WARMUP_QUERIES,
    summarize_timing_trials,
    timing_contract_metrics,
)
from presets import DATASETS

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required for LIRA.") from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch is required for LIRA.") from exc

try:
    from numba import njit, prange  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]
    _HAS_NUMBA = False


def _current_rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DeviceChoice:
    device: torch.device
    source: str
    detail: str
    cuda_index: int | None


def _pick_idle_cuda_index_via_nvidia_smi() -> tuple[int | None, str]:
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return None, f"nvidia-smi unavailable ({exc})"
    best_idx = None
    best_mem = None
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        idx = None
        used = None
        if len(parts) == 2:
            try:
                idx = int(parts[0])
                used = int(parts[1])
            except Exception:
                idx = None
                used = None
        if idx is None or used is None:
            match = re.search(r"^\s*(\d+)\s*,\s*(\d+)(?:\s*Mi?B)?\s*$", raw)
            if match is not None:
                idx = int(match.group(1))
                used = int(match.group(2))
        if idx is None or used is None:
            continue
        if best_mem is None or used < best_mem:
            best_idx = idx
            best_mem = used
    if best_idx is None:
        return None, "no parseable GPU rows from nvidia-smi"
    return best_idx, f"lowest nvidia-smi memory.used={best_mem} MB"


def choose_device_with_fallback(prefer_idle_gpu: bool = True) -> DeviceChoice:
    if not prefer_idle_gpu:
        return DeviceChoice(torch.device("cpu"), "cpu_policy", "GPU use disabled by campaign policy", None)
    if not torch.cuda.is_available():
        return DeviceChoice(torch.device("cpu"), "cpu", "cuda not available", None)
    idle_idx, detail = _pick_idle_cuda_index_via_nvidia_smi()
    if idle_idx is not None:
        try:
            torch.cuda.set_device(idle_idx)
            return DeviceChoice(torch.device(f"cuda:{idle_idx}"), "cuda_idle", detail, idle_idx)
        except Exception as exc:
            detail = f"{detail}; set_device failed ({exc})"
    try:
        current_idx = int(torch.cuda.current_device())
        return DeviceChoice(torch.device(f"cuda:{current_idx}"), "cuda_default", detail, current_idx)
    except Exception as exc:
        return DeviceChoice(torch.device("cpu"), "cpu_fallback", f"cuda fallback failed ({exc})", None)


class LiraProbeMLP(nn.Module):
    def __init__(self, input_dim_dist: int, input_dim_vec: int, output_dim: int) -> None:
        super().__init__()
        self.distance_net = nn.Sequential(
            nn.Linear(input_dim_dist, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim_vec, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_dist: torch.Tensor, x_vec: torch.Tensor) -> torch.Tensor:
        out_dist = self.distance_net(x_dist)
        out_vec = self.vector_net(x_vec)
        return self.fc(torch.cat((out_dist, out_vec), dim=1))


@dataclass(frozen=True)
class ProbeEvalResult:
    targets: np.ndarray
    predicts: np.ndarray
    outputs: np.ndarray
    loss: float


def train_epoch(
    *,
    model: LiraProbeMLP,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x_dist: np.ndarray,
    x_vec: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, x_dist.shape[0], batch_size):
        end = min(start + batch_size, x_dist.shape[0])
        batch_dist = torch.from_numpy(x_dist[start:end]).to(device=device, dtype=torch.float32)
        batch_vec = torch.from_numpy(x_vec[start:end]).to(device=device, dtype=torch.float32)
        batch_targets = torch.from_numpy(targets[start:end]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(batch_dist, batch_vec)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_probe(
    *,
    model: LiraProbeMLP,
    criterion: nn.Module,
    x_dist: np.ndarray,
    x_vec: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
    sigma: float = 0.5,
) -> ProbeEvalResult:
    model.eval()
    outputs_parts: list[np.ndarray] = []
    predicts_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, x_dist.shape[0], batch_size):
            end = min(start + batch_size, x_dist.shape[0])
            batch_dist = torch.from_numpy(x_dist[start:end]).to(device=device, dtype=torch.float32)
            batch_vec = torch.from_numpy(x_vec[start:end]).to(device=device, dtype=torch.float32)
            batch_targets = torch.from_numpy(targets[start:end]).to(device=device, dtype=torch.float32)
            outputs = model(batch_dist, batch_vec)
            loss = criterion(outputs, batch_targets)
            out_cpu = outputs.detach().cpu().numpy()
            outputs_parts.append(out_cpu)
            predicts_parts.append(out_cpu > sigma)
            targets_parts.append(batch_targets.detach().cpu().numpy() > 0.5)
            total_loss += float(loss.item())
            n_batches += 1
    return ProbeEvalResult(
        targets=np.concatenate(targets_parts, axis=0),
        predicts=np.concatenate(predicts_parts, axis=0),
        outputs=np.concatenate(outputs_parts, axis=0),
        loss=(total_loss / max(1, n_batches)),
    )


def infer_probe(
    *,
    model: LiraProbeMLP,
    x_dist: np.ndarray,
    x_vec: np.ndarray,
    batch_size: int,
    device: torch.device,
    sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs_parts: list[np.ndarray] = []
    predicts_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_dist.shape[0], batch_size):
            end = min(start + batch_size, x_dist.shape[0])
            batch_dist = torch.from_numpy(x_dist[start:end]).to(device=device, dtype=torch.float32)
            batch_vec = torch.from_numpy(x_vec[start:end]).to(device=device, dtype=torch.float32)
            outputs = model(batch_dist, batch_vec)
            out_cpu = outputs.detach().cpu().numpy()
            outputs_parts.append(out_cpu)
            predicts_parts.append(out_cpu > sigma)
    return np.concatenate(predicts_parts, axis=0), np.concatenate(outputs_parts, axis=0)


@dataclass(frozen=True)
class LiraConfig:
    dataset: str
    n_bkt: int
    k: int = 100
    index_full_dataset: bool = True
    batch_size: int = 512
    n_epoch: int = 30
    n_mul: int = 2
    # Official LIRA applies redundancy over repa_step incremental steps
    # (1/repa_step of the flagged points per step) and tunes how many steps to
    # keep. The earlier port used repa_step=1, which dumps all model-flagged
    # redundancy in a single step and inflates LIRA's index overhead beyond
    # what its own tuning would select; 10 matches the reference default.
    repa_step: int = 10
    duplicate_type: str = "model"
    inner_index_type: str = "HNSW"
    ef_fixed: int = 128
    hnsw_M: int = 32
    learning_rate: float = 1e-4
    sigma: float = 0.5
    threshold_start: float = 0.1
    threshold_stop: float = 1.0
    threshold_step: float = 0.02
    seed: int = 0
    prefer_idle_gpu: bool = True
    num_threads: int = 0
    selected_part_policy: str = "latest"
    query_limit: int | None = None
    eval_every_repartition: bool = True
    compare_part: int | None = None
    probe_depth: int = 10
    prepare_max_samples: int = 1_000_000
    prepare_k: int = 100

    @property
    def metric(self) -> str:
        metric = DATASETS[self.dataset].metric
        return "inner_product" if metric == "cosine" else "l2"


@dataclass(frozen=True)
class LiraRunResult:
    output_dir: Path
    selected_part: int
    selected_rows: list[dict[str, float]]
    model_metrics: list[dict[str, float]]
    metadata: dict[str, Any]


def _build_kmeans_assignments(
    base_vecs: np.ndarray,
    n_bkt: int,
    seed: int,
) -> tuple[faiss.Kmeans, np.ndarray, list[list[int]]]:
    kmeans = faiss.Kmeans(base_vecs.shape[1], n_bkt, niter=20, verbose=False, seed=seed)
    kmeans.train(base_vecs.astype(np.float32, copy=False))
    _, assign = kmeans.index.search(base_vecs.astype(np.float32, copy=False), 1)
    assign = assign.reshape(-1).astype(np.int32, copy=False)
    data_to_buckets = np.full((base_vecs.shape[0], 2), -1, dtype=np.int32)
    data_to_buckets[:, 0] = assign
    cluster_ids: list[list[int]] = [[] for _ in range(n_bkt)]
    for idx, cid in enumerate(assign.tolist()):
        cluster_ids[int(cid)].append(int(idx))
    return kmeans, data_to_buckets, cluster_ids


def _knn_bucket_labels(knn_ids: np.ndarray, data_to_buckets: np.ndarray, n_bkt: int) -> np.ndarray:
    if _HAS_NUMBA:
        return _knn_bucket_labels_numba(knn_ids.astype(np.int32), data_to_buckets.astype(np.int32), n_bkt)

    n = knn_ids.shape[0]
    out = np.zeros((n, n_bkt), dtype=np.uint8)
    for i in range(n):
        buckets = data_to_buckets[knn_ids[i]].reshape(-1)
        buckets = buckets[buckets >= 0]
        if buckets.size:
            out[i, np.unique(buckets)] = 1
    return out


if _HAS_NUMBA:  # pragma: no branch

    @njit(parallel=True, cache=True, fastmath=False)  # type: ignore[misc]
    def _knn_bucket_labels_numba(knn_ids: np.ndarray, data_to_buckets: np.ndarray, n_bkt: int) -> np.ndarray:
        n = knn_ids.shape[0]
        k = knn_ids.shape[1]
        n_mul = data_to_buckets.shape[1]
        out = np.zeros((n, n_bkt), dtype=np.uint8)
        for i in prange(n):
            for j in range(k):
                nid = knn_ids[i, j]
                for m in range(n_mul):
                    b = data_to_buckets[nid, m]
                    if b >= 0:
                        out[i, b] = 1
        return out


def _query_bucket_membership(
    knn_ids: np.ndarray,
    data_to_buckets: np.ndarray,
    n_bkt: int,
) -> tuple[np.ndarray, list[dict[int, np.ndarray]]]:
    counts = np.zeros((knn_ids.shape[0], n_bkt), dtype=np.int32)
    bucket_ids_per_query: list[dict[int, np.ndarray]] = []
    n_mul = data_to_buckets.shape[1]
    for qid in range(knn_ids.shape[0]):
        ids = knn_ids[qid].astype(np.int32, copy=False)
        buckets = data_to_buckets[ids].reshape(-1)
        rep_ids = np.repeat(ids, n_mul)
        valid = buckets >= 0
        buckets = buckets[valid]
        rep_ids = rep_ids[valid]
        q_map: dict[int, np.ndarray] = {}
        if buckets.size:
            uniq, cnt = np.unique(buckets, return_counts=True)
            counts[qid, uniq] = cnt.astype(np.int32, copy=False)
            for b in uniq.tolist():
                q_map[int(b)] = rep_ids[buckets == b].astype(np.int32, copy=False)
        bucket_ids_per_query.append(q_map)
    return counts, bucket_ids_per_query


def _build_inner_indexes(
    base_vecs: np.ndarray,
    cluster_ids: list[list[int]],
    cfg: LiraConfig,
) -> tuple[list[Any | None], list[np.ndarray]]:
    indexes: list[Any | None] = []
    cluster_arrays: list[np.ndarray] = []
    for ids in cluster_ids:
        id_arr = np.asarray(ids, dtype=np.int32)
        cluster_arrays.append(id_arr)
        if id_arr.size == 0:
            indexes.append(None)
            continue
        xb = base_vecs[id_arr].astype(np.float32, copy=False)
        if cfg.inner_index_type.upper() == "HNSW":
            idx = faiss.IndexHNSWFlat(xb.shape[1], cfg.hnsw_M)
            if cfg.metric == "inner_product":
                idx.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            idx = faiss.IndexFlatIP(xb.shape[1]) if cfg.metric == "inner_product" else faiss.IndexFlatL2(xb.shape[1])
        idx.add(xb)
        indexes.append(idx)
    return indexes, cluster_arrays


def _clone_cluster_ids(cluster_ids: list[list[int]]) -> list[list[int]]:
    return [list(ids) for ids in cluster_ids]


def _replay_repartition_state(
    initial_data_to_buckets: np.ndarray,
    initial_cluster_ids: list[list[int]],
    data_scores: np.ndarray,
    data_predicts: np.ndarray,
    cfg: LiraConfig,
    *,
    selected_part: int,
) -> tuple[np.ndarray, list[list[int]]]:
    if selected_part <= 0:
        return initial_data_to_buckets.copy(), _clone_cluster_ids(initial_cluster_ids)

    data_to_buckets = initial_data_to_buckets.copy()
    cluster_ids = _clone_cluster_ids(initial_cluster_ids)
    nprobe_predicts = np.sum(data_predicts, axis=1)
    sorted_ids = np.argsort(-nprobe_predicts)
    n_t = int(np.count_nonzero(nprobe_predicts))
    batch_t = max(1, n_t // max(1, cfg.repa_step))

    current_part = 0
    for step in range(cfg.repa_step):
        begin = step * batch_t
        end = min((step + 1) * batch_t, n_t)
        if begin >= end:
            break
        _apply_redundancy_by_model(
            data_scores,
            data_predicts,
            sorted_ids,
            data_to_buckets,
            cluster_ids,
            n_mul=cfg.n_mul,
            begin=begin,
            end=end,
        )
        current_part += 1
        if current_part >= selected_part:
            break
    return data_to_buckets, cluster_ids


def _apply_redundancy_by_model(
    data_scores: np.ndarray,
    data_predicts: np.ndarray,
    sorted_ids: np.ndarray,
    data_to_buckets: np.ndarray,
    cluster_ids: list[list[int]],
    *,
    n_mul: int,
    begin: int,
    end: int,
) -> None:
    for t_id in sorted_ids[begin:end]:
        t_id = int(t_id)
        cur_c_id = int(data_to_buckets[t_id, 0])
        sorted_cands = np.argsort(-data_scores[t_id])
        n_effective = int(np.count_nonzero(data_predicts[t_id]))
        n_actual = int(min(n_mul - 1, n_effective))
        if n_actual <= 0:
            continue
        loc = np.where(sorted_cands == cur_c_id)[0]
        if loc.size == 0 or int(loc[0]) >= n_actual:
            actual = sorted_cands[:n_actual]
            data_to_buckets[t_id, 1 : n_actual + 1] = actual
        elif n_effective == n_actual:
            actual = sorted_cands[:n_actual]
            data_to_buckets[t_id, 0:n_actual] = actual
        else:
            actual = sorted_cands[: n_actual + 1]
            data_to_buckets[t_id, 0 : n_actual + 1] = actual
        for c_id in actual.tolist():
            c_id = int(c_id)
            if c_id != cur_c_id:
                cluster_ids[c_id].append(t_id)


def _model_metrics_row(
    epoch: int,
    loss: float,
    outputs: np.ndarray,
    predicts: np.ndarray,
    targets: np.ndarray,
    query_bucket_ids: list[dict[int, np.ndarray]],
    cfg: LiraConfig,
) -> dict[str, float]:
    preds_bool = predicts.astype(bool, copy=False)
    targets_bool = targets.astype(bool, copy=False)
    accuracy = float(np.mean(preds_bool == targets_bool))
    target_hits = np.sum(np.logical_and(preds_bool, targets_bool), axis=1).astype(np.float64)
    target_totals = np.sum(targets_bool, axis=1).astype(np.float64)
    hit_rates = np.divide(target_hits, target_totals, out=np.full_like(target_hits, np.nan), where=target_totals > 0)
    recall_proxy = np.zeros(preds_bool.shape[0], dtype=np.float64)
    for qid in range(preds_bool.shape[0]):
        top_buckets = np.where(preds_bool[qid])[0]
        found: set[int] = set()
        q_map = query_bucket_ids[qid]
        for b in top_buckets.tolist():
            ids = q_map.get(int(b))
            if ids is not None and ids.size:
                found.update(int(v) for v in ids.tolist())
        recall_proxy[qid] = len(found) / float(cfg.k)
    return {
        "epoch": float(epoch),
        "loss": float(loss),
        "accuracy": accuracy,
        "hit_rate": float(np.nanmean(hit_rates)),
        "nprobe_predict": float(np.mean(np.sum(preds_bool, axis=1))),
        "nprobe_target": float(np.mean(np.sum(targets_bool, axis=1))),
        "knn_recall_proxy": float(np.mean(recall_proxy)),
    }


def _query_tuning_curve(
    outputs: np.ndarray,
    query_bucket_ids_k: list[dict[int, np.ndarray]],
    query_bucket_ids_10: list[dict[int, np.ndarray]],
    found_aknn_id: np.ndarray,
    cmp_distr_all: np.ndarray,
    cfg: LiraConfig,
) -> list[dict[str, float]]:
    n_q = outputs.shape[0]
    thresholds = np.arange(cfg.threshold_start, cfg.threshold_stop, cfg.threshold_step, dtype=np.float32)
    if thresholds.size == 0:
        thresholds = np.asarray([float(cfg.threshold_start)], dtype=np.float32)
    rows: list[dict[str, float]] = []
    recall10_den = float(max(1, min(10, int(cfg.k))))
    for threshold in thresholds:
        t_query = time.perf_counter()
        recall_k_sum = 0.0
        recall10_sum = 0.0
        comp_sum = 0.0
        comp_per_query: list[float] = []
        cmp_sum = 0.0
        nprobe_sum = 0.0
        for qid in range(n_q):
            top_buckets = np.where(outputs[qid] > threshold)[0]
            nprobe_sum += float(top_buckets.size)
            unique_cands: set[int] = set()
            q_map_k = query_bucket_ids_k[qid]
            q_map_10 = query_bucket_ids_10[qid]
            found_knn_k: set[int] = set()
            found_knn_10: set[int] = set()
            if top_buckets.size > 0:
                cmp_sum += float(np.sum(cmp_distr_all[qid, top_buckets]))
            for b in top_buckets.tolist():
                cand = found_aknn_id[qid, int(b)]
                cand = cand[cand >= 0]
                if cand.size == 0:
                    continue
                unique_cands.update(int(v) for v in cand.tolist())
                gt_k = q_map_k.get(int(b))
                if gt_k is not None and gt_k.size:
                    inter_k = np.intersect1d(gt_k, cand, assume_unique=False)
                    if inter_k.size:
                        found_knn_k.update(int(v) for v in inter_k.tolist())
                gt_10 = q_map_10.get(int(b))
                if gt_10 is not None and gt_10.size:
                    inter_10 = np.intersect1d(gt_10, cand, assume_unique=False)
                    if inter_10.size:
                        found_knn_10.update(int(v) for v in inter_10.tolist())
            comp_sum += float(len(unique_cands))
            comp_per_query.append(float(len(unique_cands)))
            recall_k_sum += len(found_knn_k) / float(cfg.k)
            recall10_sum += len(found_knn_10) / recall10_den
        query_total_s = float(time.perf_counter() - t_query)
        comp_arr = np.asarray(comp_per_query, dtype=np.float64)
        rows.append(
            {
                "threshold": float(threshold),
                "nprobe": nprobe_sum / float(n_q),
                "recall_at_k": recall_k_sum / float(n_q),
                "recall10_at_10": recall10_sum / float(n_q),
                "avg_computations": comp_sum / float(n_q),
                "computation_min": float(comp_arr.min()) if comp_arr.size > 0 else float("nan"),
                "computation_max": float(comp_arr.max()) if comp_arr.size > 0 else float("nan"),
                "avg_computations_mode": AVG_COMPUTATIONS_MODE_LIRA,
                "avg_unique_candidate_vectors_per_query": comp_sum / float(n_q),
                "dist_computations": cmp_sum / float(n_q),
                "avg_candidates": comp_sum / float(n_q),
                "query_total_s": query_total_s,
                "query_s_per_query": float(query_total_s / float(n_q)) if n_q > 0 else float("nan"),
                "qps": float(float(n_q) / query_total_s) if query_total_s > 0.0 else float("nan"),
            }
        )
    return rows


def _write_selected_serving_snapshot(
    output_dir: Path,
    model: LiraProbeMLP,
    centroids: np.ndarray,
    data_to_buckets: np.ndarray,
    cluster_ids: list[list[int]],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    indexes: list[Any | None],
) -> None:
    models_dir = ensure_dir(output_dir / "models")
    staging_dir = ensure_dir(output_dir / "staging")
    lookups_dir = ensure_dir(staging_dir / "lookups")
    inner_dir = ensure_dir(staging_dir / "lira_inner_indexes")
    torch.save({"state_dict": model.state_dict(), "n_bkt": int(centroids.shape[0]), "dim": int(centroids.shape[1])}, models_dir / "lira_probe.pt")
    np.save(staging_dir / "lira_centroids.npy", centroids.astype(np.float32, copy=False))
    np.savez(staging_dir / "lira_distance_scaler.npz", mean=np.asarray(scaler_mean, dtype=np.float32), scale=np.asarray(scaler_scale, dtype=np.float32))
    np.save(lookups_dir / "point_to_buckets.npy", data_to_buckets.astype(np.int32, copy=False))
    cluster_arrays = [np.asarray(ids, dtype=np.int32) for ids in cluster_ids]
    offsets = np.zeros(len(cluster_arrays) + 1, dtype=np.int64)
    for idx, ids in enumerate(cluster_arrays):
        offsets[idx + 1] = offsets[idx] + int(ids.size)
    flat_ids = np.concatenate([ids for ids in cluster_arrays if ids.size > 0]).astype(np.int32, copy=False) if offsets[-1] > 0 else np.empty(0, dtype=np.int32)
    np.save(staging_dir / "lira_selected_offsets.npy", offsets)
    np.save(staging_dir / "lira_selected_ids.npy", flat_ids)
    meta_rows: list[dict[str, Any]] = []
    for b_id, idx in enumerate(indexes):
        if idx is None:
            continue
        index_path = inner_dir / f"bucket_{b_id:05d}.index"
        faiss.write_index(idx, str(index_path))
        meta_rows.append({"bucket": int(b_id), "path": str(index_path.name)})
    (inner_dir / "manifest.json").write_text(json.dumps({"indexes": meta_rows}, indent=2), encoding="utf-8")


def _load_selected_inner_indexes(output_dir: Path, cfg: LiraConfig) -> tuple[list[Any | None], list[np.ndarray]]:
    staging_dir = output_dir / "staging"
    inner_dir = staging_dir / "lira_inner_indexes"
    ids = np.load(staging_dir / "lira_selected_ids.npy").astype(np.int32, copy=False)
    offsets = np.load(staging_dir / "lira_selected_offsets.npy").astype(np.int64, copy=False)
    cluster_arrays: list[np.ndarray] = []
    indexes: list[Any | None] = []
    for b_id in range(cfg.n_bkt):
        begin = int(offsets[b_id])
        end = int(offsets[b_id + 1])
        cluster_arrays.append(ids[begin:end].astype(np.int32, copy=False))
        index_path = inner_dir / f"bucket_{b_id:05d}.index"
        indexes.append(faiss.read_index(str(index_path)) if index_path.exists() else None)
    return indexes, cluster_arrays


def _search_buckets(
    indexes: list[Any | None],
    cluster_arrays: list[np.ndarray],
    query_vecs: np.ndarray,
    cfg: LiraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_q = query_vecs.shape[0]
    found = np.full((n_q, cfg.n_bkt, cfg.k), -1, dtype=np.int32)
    cmp_counts = np.zeros((n_q, cfg.n_bkt), dtype=np.int32)
    for b_id in range(cfg.n_bkt):
        idx = indexes[b_id]
        ids = cluster_arrays[b_id]
        if idx is None or ids.size == 0:
            continue
        if cfg.inner_index_type.upper() == "HNSW":
            for q_id in range(n_q):
                idx.hnsw.efSearch = cfg.ef_fixed
                faiss.cvar.hnsw_stats.reset()
                _, local = idx.search(query_vecs[q_id : q_id + 1], cfg.k)
                local = local.reshape(-1)
                valid = local >= 0
                if np.any(valid):
                    found[q_id, b_id, valid] = ids[local[valid]]
                cmp_counts[q_id, b_id] = int(faiss.cvar.hnsw_stats.ndis)
        else:
            _, local = idx.search(query_vecs.astype(np.float32, copy=False), cfg.k)
            valid = local >= 0
            global_ids = np.full_like(local, -1, dtype=np.int32)
            if np.any(valid):
                global_ids[valid] = ids[local[valid]]
            found[:, b_id, :] = global_ids
            cmp_counts[:, b_id] = int(idx.ntotal)
    return cmp_counts, found


def _load_bundle(cfg: LiraConfig, data_root: Path, force_prepare: bool) -> LearnedDataset:
    return load_learned_dataset(
        cfg.dataset,
        data_root,
        max_samples=cfg.prepare_max_samples,
        k=cfg.prepare_k,
        seed=0,
        force_prepare=force_prepare,
    )


def _online_selected_query_summary(
    cfg: LiraConfig,
    output_dir: Path,
    *,
    data_root: Path,
    threshold: float,
) -> dict[str, Any]:
    bundle = _load_bundle(cfg, data_root, force_prepare=False)
    queries_raw = bundle.queries
    eval_neighbors = bundle.eval_neighbors
    if cfg.query_limit is not None:
        q_cap = min(int(cfg.query_limit), queries_raw.shape[0], eval_neighbors.shape[0])
        queries_raw = queries_raw[:q_cap].astype(np.float32, copy=False)
        eval_neighbors = eval_neighbors[:q_cap].astype(np.int32, copy=False)
    base_raw = bundle.index_train if cfg.index_full_dataset else bundle.router_train
    base_vecs = normalize_rows(base_raw) if cfg.metric == "inner_product" else base_raw.astype(np.float32, copy=False)
    queries = normalize_rows(queries_raw) if cfg.metric == "inner_product" else queries_raw.astype(np.float32, copy=False)
    model_blob = torch.load(output_dir / "models" / "lira_probe.pt", map_location="cpu")
    centroids = np.load(output_dir / "staging" / "lira_centroids.npy").astype(np.float32, copy=False)
    scaler_blob = np.load(output_dir / "staging" / "lira_distance_scaler.npz")
    scaler_mean = np.asarray(scaler_blob["mean"], dtype=np.float32)
    scaler_scale = np.asarray(scaler_blob["scale"], dtype=np.float32)
    scaler_scale = np.where(np.abs(scaler_scale) < 1e-12, 1.0, scaler_scale).astype(np.float32, copy=False)
    device = choose_device_with_fallback(prefer_idle_gpu=cfg.prefer_idle_gpu).device
    model = LiraProbeMLP(int(centroids.shape[0]), int(queries.shape[1]), int(centroids.shape[0])).to(device)
    model.load_state_dict(model_blob["state_dict"])
    indexes, cluster_arrays = _load_selected_inner_indexes(output_dir, cfg)
    def execute(*, routing_batch_size: int, query_count: int, diagnostic: bool) -> dict[str, Any]:
        ranked_recall_sums = {1: 0.0, 10: 0.0, 100: 0.0}
        candidate_recall_sums = {1: 0.0, 10: 0.0, 100: 0.0}
        comp_sum = 0.0
        comp_per_query: list[float] = []
        per_query_rows: list[dict[str, object]] = []
        cmp_sum = 0.0
        nprobe_sum = 0.0
        wall_start = time.perf_counter()
        for chunk_start in range(0, query_count, int(routing_batch_size)):
            chunk_end = min(chunk_start + int(routing_batch_size), query_count)
            chunk_queries = queries[chunk_start:chunk_end]
            query_dist_raw = distance_to_centroids(chunk_queries, centroids)
            query_dist = ((query_dist_raw - scaler_mean) / scaler_scale).astype(np.float32, copy=False)
            _, chunk_outputs = infer_probe(
                model=model,
                x_dist=query_dist,
                x_vec=chunk_queries,
                batch_size=int(routing_batch_size),
                device=device,
                sigma=cfg.sigma,
            )
            for local_qid in range(chunk_outputs.shape[0]):
                qid = chunk_start + local_qid
                top_buckets = np.where(chunk_outputs[local_qid] > threshold)[0]
                nprobe_sum += float(top_buckets.size)
                unique_cands: set[int] = set()
                q_vec = queries[qid : qid + 1].astype(np.float32, copy=False)
                for b_id in top_buckets.tolist():
                    idx = indexes[int(b_id)]
                    ids = cluster_arrays[int(b_id)]
                    if idx is None or ids.size == 0:
                        continue
                    if cfg.inner_index_type.upper() == "HNSW":
                        faiss.cvar.hnsw_stats.reset()
                        idx.hnsw.efSearch = cfg.ef_fixed
                        _, local = idx.search(q_vec, cfg.k)
                        cmp_sum += float(faiss.cvar.hnsw_stats.ndis)
                    else:
                        _, local = idx.search(q_vec, cfg.k)
                        cmp_sum += float(idx.ntotal)
                    local = local.reshape(-1)
                    valid = local >= 0
                    if np.any(valid):
                        unique_cands.update(int(v) for v in ids[local[valid]].tolist())
                comp_sum += float(len(unique_cands))
                comp_per_query.append(float(len(unique_cands)))
                ranked_recalls = {1: 0.0, 10: 0.0, 100: 0.0}
                candidate_recalls = {1: 0.0, 10: 0.0, 100: 0.0}
                if unique_cands:
                    cand_arr = np.fromiter(unique_cands, dtype=np.int32)
                    if cfg.metric == "inner_product":
                        scores = base_vecs[cand_arr] @ queries[qid]
                    else:
                        delta = base_vecs[cand_arr] - queries[qid]
                        scores = -np.sum(delta * delta, axis=1)
                    top_n = min(int(cfg.k), int(cand_arr.size))
                    top_idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
                    top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
                    ranked = cand_arr[top_idx]
                    if diagnostic:
                        for k_eval in (1, 10, 100):
                            if cfg.k < k_eval or eval_neighbors.shape[1] < k_eval:
                                continue
                            candidate_recalls[k_eval] = (
                                float(np.intersect1d(eval_neighbors[qid, :k_eval], cand_arr).size)
                                / float(k_eval)
                            )
                            ranked_recalls[k_eval] = (
                                float(np.intersect1d(eval_neighbors[qid, :k_eval], ranked[:k_eval]).size)
                                / float(k_eval)
                            )
                            candidate_recall_sums[k_eval] += candidate_recalls[k_eval]
                            ranked_recall_sums[k_eval] += ranked_recalls[k_eval]
                if diagnostic:
                    per_query_rows.append(
                        {
                            "query_id": qid,
                            "probe_count": int(top_buckets.size),
                            "candidate_count": int(len(unique_cands)),
                            "recall_at_1": ranked_recalls[1],
                            "recall_at_10": ranked_recalls[10],
                            "recall_at_100": ranked_recalls[100],
                            "candidate_recall_at_1": candidate_recalls[1],
                            "candidate_recall_at_10": candidate_recalls[10],
                            "candidate_recall_at_100": candidate_recalls[100],
                        }
                    )
        wall_seconds = float(time.perf_counter() - wall_start)
        comp_arr = np.asarray(comp_per_query, dtype=np.float64)
        denominator = float(query_count)
        return {
            "threshold": float(threshold),
            "nprobe": nprobe_sum / denominator,
            "recall_at_1": ranked_recall_sums[1] / denominator,
            "recall_at_10": ranked_recall_sums[10] / denominator,
            "recall_at_100": ranked_recall_sums[100] / denominator,
            "candidate_recall_at_1": candidate_recall_sums[1] / denominator,
            "candidate_recall_at_10": candidate_recall_sums[10] / denominator,
            "candidate_recall_at_100": candidate_recall_sums[100] / denominator,
            "avg_computations": comp_sum / denominator,
            "computation_min": float(comp_arr.min()),
            "computation_max": float(comp_arr.max()),
            "avg_computations_mode": AVG_COMPUTATIONS_MODE_LIRA,
            "dist_computations": cmp_sum / denominator,
            "exact_rerank_count": comp_sum / denominator,
            "distance_scoring_equivalent_work": (cmp_sum + comp_sum) / denominator,
            "query_total_s": wall_seconds,
            "per_query_rows": per_query_rows,
        }

    def benchmark(routing_batch_size: int):
        warmup_queries = min(WARMUP_QUERIES, int(queries.shape[0]))
        if warmup_queries > 0:
            execute(routing_batch_size=routing_batch_size, query_count=warmup_queries, diagnostic=False)
        results: list[dict[str, Any]] = []
        walls: list[float] = []
        cpu_times: list[float] = []
        for _ in range(TIMING_TRIALS):
            cpu_start = time.process_time()
            result = execute(
                routing_batch_size=routing_batch_size,
                query_count=int(queries.shape[0]),
                diagnostic=False,
            )
            cpu_times.append(float(time.process_time() - cpu_start))
            walls.append(float(result["query_total_s"]))
            results.append(result)
        timing = summarize_timing_trials(
            batch_size=routing_batch_size,
            query_count=int(queries.shape[0]),
            warmup_queries=warmup_queries,
            wall_seconds=walls,
            process_cpu_seconds=cpu_times,
        )
        return results[timing.selected_trial], timing

    rss_before = _current_rss_mb()
    _, batch1_timing = benchmark(BENCHMARK_BATCH_1)
    _, batch256_timing = benchmark(BENCHMARK_BATCH_256)
    if int(cfg.batch_size) == BENCHMARK_BATCH_256:
        native_timing = batch256_timing
    else:
        _, native_timing = benchmark(int(cfg.batch_size))
    diagnostic = execute(
        routing_batch_size=BENCHMARK_BATCH_256,
        query_count=int(queries.shape[0]),
        diagnostic=True,
    )
    rss_after = _current_rss_mb()
    try:
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        peak_rss = float("nan")
    query_mem_delta_mb = max(0.0, rss_after - rss_before) if np.isfinite(rss_after) and np.isfinite(rss_before) else float("nan")
    write_csv(output_dir / "per_query_debug.csv", diagnostic["per_query_rows"])
    timing_metrics = timing_contract_metrics(batch1=batch1_timing, batch256=batch256_timing)
    return {
        "threshold": float(threshold),
        "nprobe": diagnostic["nprobe"],
        "recall_at_1": diagnostic["recall_at_1"],
        "recall_at_10": diagnostic["recall_at_10"],
        "recall_at_100": diagnostic["recall_at_100"],
        "recall_at_k": diagnostic.get(
            f"recall_at_{int(cfg.k)}",
            diagnostic["recall_at_10"],
        ),
        "recall10_at_10": diagnostic["recall_at_10"],
        "candidate_recall_at_1": diagnostic["candidate_recall_at_1"],
        "candidate_recall_at_10": diagnostic["candidate_recall_at_10"],
        "candidate_recall_at_100": diagnostic["candidate_recall_at_100"],
        "avg_computations": diagnostic["avg_computations"],
        "computation_min": diagnostic["computation_min"],
        "computation_max": diagnostic["computation_max"],
        "avg_computations_mode": AVG_COMPUTATIONS_MODE_LIRA,
        "avg_unique_candidate_vectors_per_query": diagnostic["avg_computations"],
        "dist_computations": diagnostic["dist_computations"],
        "exact_rerank_count": diagnostic["exact_rerank_count"],
        "distance_scoring_equivalent_work": diagnostic["distance_scoring_equivalent_work"],
        "query_total_s": batch256_timing.wall_seconds_median,
        "query_s_per_query": batch256_timing.wall_seconds_median / float(queries.shape[0]),
        **timing_metrics,
        "qps_native_batched": float(native_timing.qps_median),
        "qps_native_batched_contract": "batched_routing_serial_local_search",
        "native_batch_size": int(cfg.batch_size),
        "query_rss_mb": float(peak_rss if np.isfinite(peak_rss) else rss_after),
        "query_mem_delta_mb": float(query_mem_delta_mb),
    }


def run_lira_selected_online_eval(
    cfg: LiraConfig,
    output_dir: Path,
    *,
    data_root: Path,
    threshold: float,
) -> dict[str, float | str]:
    env = os.environ.copy()
    serving_threads = env.get("LISBENCH_SERVING_THREADS")
    if serving_threads:
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[key] = serving_threads
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "-m",
        "run_lira_query_isolated",
        "--dataset",
        cfg.dataset,
        "--data-root",
        str(data_root),
        "--experiment-dir",
        str(output_dir),
        "--threshold",
        str(float(threshold)),
    ]
    proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout.strip())


def run_lira_smallscale(
    cfg: LiraConfig,
    output_dir: Path,
    *,
    data_root: Path,
    force_prepare: bool = False,
) -> LiraRunResult:
    _set_global_seed(cfg.seed)
    maybe_set_threads(cfg.num_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ensure_dir(output_dir / "data")
    t0_total = time.perf_counter()
    timings_s: dict[str, float] = {}
    t0 = time.perf_counter()
    bundle = _load_bundle(cfg, data_root, force_prepare)
    timings_s["data_load"] = float(time.perf_counter() - t0)
    info(
        f"router_train={bundle.router_train.shape[0]}, index_train={bundle.index_train.shape[0]}, "
        f"queries={bundle.queries.shape[0]}, dim={bundle.router_train.shape[1]}, metric={cfg.metric}, n_bkt={cfg.n_bkt}"
    )
    info(
        f"epochs={cfg.n_epoch}, batch_size={cfg.batch_size}, n_mul={cfg.n_mul}, repa_step={cfg.repa_step}, "
        f"inner_index={cfg.inner_index_type}, selected_part_policy={cfg.selected_part_policy}"
    )
    info(f"data_load={timings_s['data_load']:.2f}s")

    queries_raw = bundle.queries
    eval_neighbors = bundle.eval_neighbors
    if cfg.query_limit is not None:
        q_cap = min(int(cfg.query_limit), queries_raw.shape[0], eval_neighbors.shape[0])
        queries_raw = queries_raw[:q_cap].astype(np.float32, copy=False)
        eval_neighbors = eval_neighbors[:q_cap].astype(np.int32, copy=False)

    train_raw = bundle.router_train
    if cfg.index_full_dataset:
        base_raw = bundle.index_train
        if bundle.sample_ids is not None:
            train_knn_base = bundle.sample_ids[bundle.router_neighbors].astype(np.int32, copy=False)
        else:
            train_knn_base = bundle.router_neighbors.astype(np.int32, copy=False)
    else:
        base_raw = bundle.router_train
        train_knn_base = bundle.router_neighbors.astype(np.int32, copy=False)
        eval_neighbors = _exact_query_knn(base_raw, queries_raw, k=cfg.k, metric=DATASETS[cfg.dataset].metric)

    train_vecs = normalize_rows(train_raw) if cfg.metric == "inner_product" else train_raw.astype(np.float32, copy=False)
    base_vecs = normalize_rows(base_raw) if cfg.metric == "inner_product" else base_raw.astype(np.float32, copy=False)
    queries = normalize_rows(queries_raw) if cfg.metric == "inner_product" else queries_raw.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    kmeans, data_to_buckets, cluster_ids = _build_kmeans_assignments(base_vecs, cfg.n_bkt, cfg.seed)
    centroids = kmeans.centroids.astype(np.float32, copy=False)
    initial_data_to_buckets = data_to_buckets.copy()
    initial_cluster_ids = _clone_cluster_ids(cluster_ids)
    labels_data = _knn_bucket_labels(train_knn_base, data_to_buckets, cfg.n_bkt).astype(np.float32, copy=False)
    query_counts_k, query_bucket_ids_k = _query_bucket_membership(eval_neighbors[:, : cfg.k], data_to_buckets, cfg.n_bkt)
    _ = query_counts_k
    query_counts_10, query_bucket_ids_10 = _query_bucket_membership(eval_neighbors[:, : min(10, cfg.k)], data_to_buckets, cfg.n_bkt)
    _ = query_counts_10
    labels_query = (query_counts_k != 0).astype(np.float32, copy=False)
    timings_s["partition_and_labels"] = float(time.perf_counter() - t0)
    info(f"partition_and_labels={timings_s['partition_and_labels']:.2f}s")

    t0 = time.perf_counter()
    train_dist, query_dist, infer_dist, scaler_mean, scaler_scale = scaled_centroid_distances_with_scaler(train_vecs, queries, base_vecs, centroids)
    timings_s["distance_scaling"] = float(time.perf_counter() - t0)
    info(f"distance_scaling={timings_s['distance_scaling']:.2f}s")
    device_choice = choose_device_with_fallback(prefer_idle_gpu=cfg.prefer_idle_gpu)
    device = device_choice.device
    info(f"device={device} ({device_choice.source}: {device_choice.detail})")
    model = LiraProbeMLP(cfg.n_bkt, train_vecs.shape[1], cfg.n_bkt).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model_metrics: list[dict[str, float]] = []
    t0 = time.perf_counter()
    eval0 = evaluate_probe(
        model=model,
        criterion=criterion,
        x_dist=query_dist,
        x_vec=queries,
        targets=labels_query,
        batch_size=cfg.batch_size,
        device=device,
        sigma=cfg.sigma,
    )
    model_metrics.append(_model_metrics_row(-1, float(eval0.loss), eval0.outputs, eval0.predicts, eval0.targets, query_bucket_ids_k, cfg))
    query_outputs = eval0.outputs
    info(
        f"eval epoch=-1 loss={model_metrics[-1]['loss']:.4f} "
        f"nprobe_predict={model_metrics[-1]['nprobe_predict']:.2f} "
        f"nprobe_target={model_metrics[-1]['nprobe_target']:.2f} "
        f"recall_proxy={model_metrics[-1]['knn_recall_proxy']:.4f}"
    )
    epoch_log_interval = max(1, cfg.n_epoch // 5)
    for epoch in range(cfg.n_epoch):
        train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            x_dist=train_dist,
            x_vec=train_vecs,
            targets=labels_data,
            batch_size=cfg.batch_size,
            device=device,
        )
        eval_ep = evaluate_probe(
            model=model,
            criterion=criterion,
            x_dist=query_dist,
            x_vec=queries,
            targets=labels_query,
            batch_size=cfg.batch_size,
            device=device,
            sigma=cfg.sigma,
        )
        model_metrics.append(_model_metrics_row(epoch, float(eval_ep.loss), eval_ep.outputs, eval_ep.predicts, eval_ep.targets, query_bucket_ids_k, cfg))
        query_outputs = eval_ep.outputs
        if epoch == 0 or epoch == cfg.n_epoch - 1 or (epoch + 1) % epoch_log_interval == 0:
            info(
                f"eval epoch={epoch} loss={model_metrics[-1]['loss']:.4f} "
                f"nprobe_predict={model_metrics[-1]['nprobe_predict']:.2f} "
                f"nprobe_target={model_metrics[-1]['nprobe_target']:.2f} "
                f"recall_proxy={model_metrics[-1]['knn_recall_proxy']:.4f}"
            )
    timings_s["probe_training"] = float(time.perf_counter() - t0)
    info(f"probe_training={timings_s['probe_training']:.2f}s")
    part_curves: dict[int, list[dict[str, float]]] = {}
    part_eval_seconds: dict[int, float] = {}
    part_bucket_sizes: dict[int, np.ndarray] = {}
    part_memory_stats: dict[int, dict[str, float]] = {}

    def _evaluate_part(part_idx: int) -> None:
        t_part = time.perf_counter()
        _, query_bucket_ids_k_part = _query_bucket_membership(eval_neighbors[:, : cfg.k], data_to_buckets, cfg.n_bkt)
        _, query_bucket_ids_10_part = _query_bucket_membership(
            eval_neighbors[:, : min(10, cfg.k)],
            data_to_buckets,
            cfg.n_bkt,
        )
        rss_before = _current_rss_mb()
        inner, cluster_arrays = _build_inner_indexes(base_vecs, cluster_ids, cfg)
        cmp_distr_all, found_aknn_id = _search_buckets(inner, cluster_arrays, queries, cfg)
        rss_after = _current_rss_mb()
        try:
            peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:
            peak_rss = float("nan")
        query_mem_delta_mb = max(0.0, rss_after - rss_before) if np.isfinite(rss_before) and np.isfinite(rss_after) else float("nan")
        curve = _query_tuning_curve(
            query_outputs,
            query_bucket_ids_k_part,
            query_bucket_ids_10_part,
            found_aknn_id,
            cmp_distr_all,
            cfg,
        )
        for row in curve:
            row["query_rss_mb"] = float(peak_rss if np.isfinite(peak_rss) else rss_after)
            row["query_mem_delta_mb"] = float(query_mem_delta_mb)
        part_curves[int(part_idx)] = curve
        part_eval_seconds[int(part_idx)] = float(time.perf_counter() - t_part)
        bucket_sizes = np.asarray([int(ids.size) for ids in cluster_arrays], dtype=np.int64)
        part_bucket_sizes[int(part_idx)] = bucket_sizes
        part_memory_stats[int(part_idx)] = {
            "query_rss_mb": float(peak_rss if np.isfinite(peak_rss) else rss_after),
            "query_mem_delta_mb": float(query_mem_delta_mb),
        }
        if curve:
            best = max(curve, key=lambda row: float(row["recall10_at_10"]))
            cheapest = min(curve, key=lambda row: float(row["avg_computations"]))
            info(
                f"part={part_idx} thresholds={len(curve)} best_recall10={float(best['recall10_at_10']):.4f} "
                f"best_nprobe={float(best['nprobe']):.2f} min_avg_comp={float(cheapest['avg_computations']):.1f} "
                f"eval_s={part_eval_seconds[int(part_idx)]:.2f}s"
            )
        pd.DataFrame(curve).to_csv(data_dir / f"lira_threshold_part{part_idx}.csv", index=False)
        np.save(data_dir / f"lira_bucket_sizes_part{part_idx}.npy", bucket_sizes)

    t0 = time.perf_counter()
    _evaluate_part(0)
    data_predicts: np.ndarray | None = None
    data_scores: np.ndarray | None = None
    if cfg.duplicate_type.lower() == "model":
        data_predicts, data_scores = infer_probe(
            model=model,
            x_dist=infer_dist,
            x_vec=base_vecs,
            batch_size=cfg.batch_size,
            device=device,
            sigma=cfg.sigma,
        )
        nprobe_predicts = np.sum(data_predicts, axis=1)
        sorted_ids = np.argsort(-nprobe_predicts)
        n_t = int(np.count_nonzero(nprobe_predicts))
        batch_t = max(1, n_t // max(1, cfg.repa_step))
        for step in range(cfg.repa_step):
            begin = step * batch_t
            end = min((step + 1) * batch_t, n_t)
            if begin >= end:
                break
            _apply_redundancy_by_model(
                data_scores,
                data_predicts,
                sorted_ids,
                data_to_buckets,
                cluster_ids,
                n_mul=cfg.n_mul,
                begin=begin,
                end=end,
            )
            if cfg.eval_every_repartition:
                _evaluate_part(step + 1)
        if not cfg.eval_every_repartition:
            _evaluate_part(1)
    timings_s["redundancy_and_eval"] = float(time.perf_counter() - t0)
    info(f"redundancy_and_eval={timings_s['redundancy_and_eval']:.2f}s")

    non_empty_parts = [int(p) for p, rows in part_curves.items() if rows]
    if not non_empty_parts:
        raise RuntimeError("LIRA produced no threshold-evaluation rows for any part.")
    if cfg.compare_part is not None:
        selected_part = int(cfg.compare_part)
    elif cfg.selected_part_policy == "paper":
        selected_part = min(non_empty_parts)
    elif cfg.selected_part_policy == "latest":
        selected_part = max(non_empty_parts)
    elif cfg.selected_part_policy == "best_recall":
        selected_part = max(non_empty_parts, key=lambda p: max(float(r["recall_at_k"]) for r in part_curves[p]))
    else:
        selected_part = max(non_empty_parts)
    if selected_part not in part_curves or not part_curves[selected_part]:
        selected_part = min(non_empty_parts) if cfg.selected_part_policy == "paper" else max(non_empty_parts)
    selected_rows = part_curves[selected_part]
    info(f"selected_part={selected_part} policy={cfg.selected_part_policy} available_parts={non_empty_parts}")
    if selected_part in part_bucket_sizes:
        np.save(data_dir / "lira_bucket_sizes_selected.npy", part_bucket_sizes[selected_part])
    latest_part = max(part_curves.keys())
    if selected_part != latest_part and data_predicts is not None and data_scores is not None:
        selected_data_to_buckets, selected_cluster_ids = _replay_repartition_state(
            initial_data_to_buckets,
            initial_cluster_ids,
            data_scores,
            data_predicts,
            cfg,
            selected_part=selected_part,
        )
    elif selected_part == 0:
        selected_data_to_buckets = initial_data_to_buckets
        selected_cluster_ids = initial_cluster_ids
    else:
        selected_data_to_buckets = data_to_buckets
        selected_cluster_ids = cluster_ids
    selected_indexes, _ = _build_inner_indexes(base_vecs, selected_cluster_ids, cfg)
    _write_selected_serving_snapshot(output_dir, model, centroids, selected_data_to_buckets, selected_cluster_ids, scaler_mean, scaler_scale, selected_indexes)
    pd.DataFrame(model_metrics).to_csv(data_dir / "lira_model_training_metrics.csv", index=False)
    part_stats: list[dict[str, float | int]] = []
    for p in sorted(part_curves.keys()):
        rows = part_curves[p]
        if rows:
            best = max(rows, key=lambda r: float(r["recall_at_k"]))
            low = min(rows, key=lambda r: float(r["dist_computations"]))
            best_recall_at_k = float(best["recall_at_k"])
            best_recall10_at_10 = float(best["recall10_at_10"])
            best_threshold = float(best["threshold"])
            best_dist_computations = float(best["dist_computations"])
            min_dist_computations = float(low["dist_computations"])
            min_dist_recall_at_k = float(low["recall_at_k"])
            min_dist_recall10_at_10 = float(low["recall10_at_10"])
        else:
            best_recall_at_k = float("nan")
            best_recall10_at_10 = float("nan")
            best_threshold = float("nan")
            best_dist_computations = float("nan")
            min_dist_computations = float("nan")
            min_dist_recall_at_k = float("nan")
            min_dist_recall10_at_10 = float("nan")
        part_stats.append(
            {
                "part": int(p),
                "points": int(len(rows)),
                "best_recall_at_k": best_recall_at_k,
                "best_recall10_at_10": best_recall10_at_10,
                "best_threshold": best_threshold,
                "best_dist_computations": best_dist_computations,
                "min_dist_computations": min_dist_computations,
                "min_dist_recall_at_k": min_dist_recall_at_k,
                "min_dist_recall10_at_10": min_dist_recall10_at_10,
                "eval_seconds": float(part_eval_seconds.get(int(p), float("nan"))),
                "query_rss_mb": float(part_memory_stats.get(int(p), {}).get("query_rss_mb", float("nan"))),
                "query_mem_delta_mb": float(part_memory_stats.get(int(p), {}).get("query_mem_delta_mb", float("nan"))),
            }
        )
    timings_s["total"] = float(time.perf_counter() - t0_total)
    info(
        "timings total="
        f"{timings_s['total']:.2f}s "
        f"(data_load={timings_s['data_load']:.2f}s, "
        f"partition_and_labels={timings_s['partition_and_labels']:.2f}s, "
        f"distance_scaling={timings_s['distance_scaling']:.2f}s, "
        f"probe_training={timings_s['probe_training']:.2f}s, "
        f"redundancy_and_eval={timings_s['redundancy_and_eval']:.2f}s)"
    )
    metadata = {
        "dataset": cfg.dataset,
        "method": "LIRA",
        "selected_part": int(selected_part),
        "selected_part_policy": str(cfg.selected_part_policy),
        "parts": sorted(int(p) for p in part_curves.keys()),
        "query_size": int(queries.shape[0]),
        "base_size": int(base_vecs.shape[0]),
        "index_full_dataset": bool(cfg.index_full_dataset),
        "query_limit": cfg.query_limit,
        "eval_every_repartition": bool(cfg.eval_every_repartition),
        "compare_part": cfg.compare_part,
        "duplicate_type": str(cfg.duplicate_type),
        "device": str(device),
        "device_source": str(device_choice.source),
        "device_detail": str(device_choice.detail),
        "device_cuda_index": (int(device_choice.cuda_index) if device_choice.cuda_index is not None else None),
        "timings_s": timings_s,
        "part_stats": part_stats,
    }
    write_json(output_dir / "config.json", {"method": "LIRA", "backend": "torch+faiss", **asdict(cfg)})
    write_json(output_dir / "metadata.json", metadata)
    write_index_manifest(output_dir)
    return LiraRunResult(
        output_dir=output_dir,
        selected_part=int(selected_part),
        selected_rows=selected_rows,
        model_metrics=model_metrics,
        metadata=metadata,
    )
