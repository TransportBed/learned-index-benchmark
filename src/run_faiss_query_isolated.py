from __future__ import annotations

import argparse
import json
import math
import resource
from pathlib import Path

import numpy as np

from datasets import load_queries_only, load_search_dataset
from methods.faiss_baselines import (
    _normalize_if_cosine,
    _pad_vectors,
    _rerank_candidate_ids_prepared,
)


def _rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure FAISS query RSS in an isolated subprocess.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()

    try:
        import faiss
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss-cpu is required for isolated query measurement") from exc

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))

    data_root = Path(args.data_root)
    queries = load_queries_only(args.dataset, data_root)
    metric = str(cfg.get("metric", "l2"))
    queries = _normalize_if_cosine(queries, metric)
    if str(cfg.get("index_kind")) == "ivfpq":
        queries = _pad_vectors(queries, int(cfg.get("pq_m", 16)))
    query_batch = queries.astype(np.float32, copy=False)
    rerank_train = None
    rerank_queries = None
    rerank_train_sq_norms = None
    if cfg.get("return_candidates_k"):
        train_raw, query_raw, _ = load_search_dataset(args.dataset, data_root)
        rerank_train = _normalize_if_cosine(train_raw, metric).astype(np.float32, copy=False)
        rerank_queries = _normalize_if_cosine(query_raw, metric).astype(np.float32, copy=False)
        if metric == "l2":
            rerank_train_sq_norms = np.sum(rerank_train * rerank_train, axis=1).astype(
                np.float32, copy=False
            )

    rss_before = _rss_mb()

    index = faiss.read_index(str(exp_dir / "staging" / "faiss.index"))
    index_kind = str(cfg.get("index_kind"))
    if index_kind in {"ivf", "ivfpq"} and cfg.get("nprobe") is not None:
        index.nprobe = int(cfg["nprobe"])
    if index_kind == "hnsw" and cfg.get("hnsw_ef_search") is not None:
        index.hnsw.efSearch = int(cfg["hnsw_ef_search"])

    k = max(int(cfg.get("k", 10)), int(cfg.get("return_candidates_k") or cfg.get("k", 10)))
    _, candidate_ids = index.search(query_batch, k)
    if rerank_train is not None and rerank_queries is not None:
        _rerank_candidate_ids_prepared(
            train_view=rerank_train,
            query_view=rerank_queries,
            train_sq_norms=rerank_train_sq_norms,
            candidate_ids=candidate_ids,
            metric=metric,
            k=int(cfg.get("k", 100)),
        )

    rss_after = _rss_mb()
    try:
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        peak_rss = float("nan")

    payload = {
        "query_mem_rss_before_mb": float(rss_before),
        "query_mem_rss_after_mb": float(rss_after),
        "query_mem_peak_rss_mb": float(peak_rss),
        "query_mem_delta_mb_isolated": max(0.0, float(rss_after) - float(rss_before)),
    }
    if math.isfinite(peak_rss) and math.isfinite(rss_before):
        payload["query_mem_peak_delta_mb"] = max(0.0, float(peak_rss) - float(rss_before))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
