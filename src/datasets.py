from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from presets import DATASETS, PAPER_DATASETS

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency in dev envs
    faiss = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LearnedDataset:
    dataset: str
    router_train: np.ndarray
    router_neighbors: np.ndarray
    index_train: np.ndarray
    queries: np.ndarray
    eval_neighbors: np.ndarray
    sample_ids: np.ndarray | None


def _normalize_if_cosine(arr: np.ndarray, metric: str) -> np.ndarray:
    if str(metric).lower() != "cosine":
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_hdf5_train_queries(dataset: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[dataset]
    h5_path = data_root / spec.key / spec.hdf5_filename
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        train = np.array(handle["train"], dtype=np.float32)
        queries = np.array(handle["test"], dtype=np.float32)
        neighbors = np.array(handle["neighbors"], dtype=np.int32)
    return train, queries, neighbors


def _sorted_topk_ids(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k column ids per row, sorted by descending score.

    ``np.argpartition`` alone returns the top-k set in arbitrary order, which is
    not a valid ground-truth ranking: recall@k' for k' < k compares against the
    leading k' entries, so they must be the true nearest neighbors in order.
    """
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)


def _filter_self_neighbors(row: np.ndarray, self_id: int, k: int) -> np.ndarray:
    filtered = row[row != self_id]
    if filtered.shape[0] >= k:
        return filtered[:k].astype(np.int32, copy=False)
    out = np.empty(k, dtype=np.int32)
    if filtered.shape[0] > 0:
        out[: filtered.shape[0]] = filtered
        out[filtered.shape[0] :] = filtered[-1]
        return out
    out.fill(self_id)
    return out


def _exact_self_knn(train: np.ndarray, *, k: int, metric: str, batch_size: int = 2048) -> np.ndarray:
    n, d = train.shape
    if n <= 1:
        return np.zeros((n, max(1, k)), dtype=np.int32)
    k = max(1, min(int(k), int(n - 1)))
    data = _normalize_if_cosine(train, metric)

    if faiss is None:
        if n > 50_000:
            raise RuntimeError("faiss is required to build exact train-train labels at this scale.")
        output = np.empty((n, k), dtype=np.int32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = data[start:end]
            if str(metric).lower() == "l2":
                x_norm = np.sum(batch * batch, axis=1, keepdims=True)
                y_norm = np.sum(data * data, axis=1, keepdims=True).T
                scores = -(x_norm + y_norm - 2.0 * (batch @ data.T))
            else:
                scores = batch @ data.T
            for i_local in range(end - start):
                scores[i_local, start + i_local] = -np.inf
            output[start:end] = _sorted_topk_ids(scores, k)
        return output

    if str(metric).lower() == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(data)
    search_k = min(k + 1, n)
    output = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, ids = index.search(data[start:end], search_k)
        for i_local, row in enumerate(ids):
            output[start + i_local] = _filter_self_neighbors(row, start + i_local, k)
    return output


def _exact_query_knn(
    train: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    metric: str,
    batch_size: int = 2048,
) -> np.ndarray:
    n, d = train.shape
    if n <= 0:
        raise ValueError("train must be non-empty")
    k = max(1, min(int(k), int(n)))
    train_used = _normalize_if_cosine(train, metric)
    query_used = _normalize_if_cosine(queries, metric)

    if faiss is None:
        if n > 50_000:
            raise RuntimeError("faiss is required to build exact query-train labels at this scale.")
        output = np.empty((query_used.shape[0], k), dtype=np.int32)
        y_norm = np.sum(train_used * train_used, axis=1, keepdims=True).T
        for start in range(0, query_used.shape[0], batch_size):
            end = min(start + batch_size, query_used.shape[0])
            batch = query_used[start:end]
            if str(metric).lower() == "l2":
                x_norm = np.sum(batch * batch, axis=1, keepdims=True)
                scores = -(x_norm + y_norm - 2.0 * (batch @ train_used.T))
            else:
                scores = batch @ train_used.T
            output[start:end] = _sorted_topk_ids(scores, k)
        return output

    if str(metric).lower() == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(train_used)
    output = np.empty((query_used.shape[0], k), dtype=np.int32)
    for start in range(0, query_used.shape[0], batch_size):
        end = min(start + batch_size, query_used.shape[0])
        _, ids = index.search(query_used[start:end], k)
        output[start:end] = ids.astype(np.int32, copy=False)
    return output


def parse_datasets(values: list[str] | None) -> list[str]:
    if not values:
        return list(PAPER_DATASETS)
    out: list[str] = []
    for raw in values:
        for token in raw.split(","):
            key = token.strip().lower()
            if not key:
                continue
            if key not in DATASETS:
                raise SystemExit(f"Unknown dataset: {key}")
            out.append(key)
    return out


def download_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            )
        },
    )
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    tmp.replace(dst)
    return dst


def download_ann_dataset(dataset: str, data_root: Path) -> Path:
    spec = DATASETS[dataset]
    if dataset == "glove10k":
        raise ValueError("glove10k is generated from glove")
    dst = data_root / spec.key / spec.hdf5_filename
    return download_file(spec.public_url, dst)


def prepare_glove10k(data_root: Path, *, force: bool = False) -> Path:
    out_dir = data_root / "glove10k"
    train_path = out_dir / "train.npy"
    query_path = out_dir / "queries.npy"
    gt_path = out_dir / "groundTruth.npy"
    neigh_path = out_dir / "neighbors100.npy"
    norms_path = out_dir / "norms.npy"
    memmap_path = out_dir / "fulldata.dat"

    if (
        not force
        and train_path.exists()
        and query_path.exists()
        and gt_path.exists()
        and neigh_path.exists()
        and norms_path.exists()
        and memmap_path.exists()
    ):
        return out_dir

    glove_h5 = data_root / "glove" / DATASETS["glove"].hdf5_filename
    if not glove_h5.exists():
        raise FileNotFoundError(f"Missing {glove_h5}. Download glove first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(glove_h5, "r") as handle:
        full_train = np.array(handle["train"], dtype=np.float32)
        full_test = np.array(handle["test"], dtype=np.float32)

    n_train = DATASETS["glove10k"].indexed_size
    n_queries = DATASETS["glove10k"].query_count
    k = 100
    seed = 0

    train = full_train[:n_train]
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(full_test.shape[0], size=n_queries, replace=False)
    q_idx.sort()
    queries = full_test[q_idx]

    norms = np.linalg.norm(train, axis=1)
    train_normed = train / norms[:, None]

    gt = np.zeros((n_train, k), dtype=np.int32)
    batch_size = 1000
    kth = k - 1
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        sims = train_normed[start:end] @ train_normed.T
        for row_idx in range(start, end):
            sims[row_idx - start, row_idx] = -1.0
        gt[start:end] = _sorted_topk_ids(sims, k)

    q_normed = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    q_sims = q_normed @ train_normed.T
    neighbors100 = _sorted_topk_ids(q_sims, k)

    np.save(train_path, train)
    np.save(query_path, queries)
    np.save(gt_path, gt)
    np.save(neigh_path, neighbors100.astype(np.int32))
    np.save(norms_path, norms)

    mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=train.shape)
    mm[:] = train
    mm.flush()
    return out_dir


def prepare_learned_dataset(
    dataset: str,
    data_root: Path,
    *,
    max_samples: int = 1_000_000,
    k: int = 100,
    seed: int = 0,
    force: bool = False,
) -> Path:
    spec = DATASETS[dataset]
    data_root = data_root.expanduser().resolve()
    base = data_root / spec.key

    if dataset == "glove10k":
        return prepare_glove10k(data_root, force=force)

    router_train_path = base / "router_train.npy"
    router_gt_path = base / "router_groundtruth.npy"
    router_pick_path = base / "router_pick.npy"
    query_path = base / "queries.npy"
    neigh_path = base / "neighbors100.npy"
    meta_path = base / "router_meta.json"

    expected_meta = {
        "dataset": dataset,
        "metric": spec.metric,
        "max_samples": int(max_samples),
        "k": int(k),
        "seed": int(seed),
        # Any dataset larger than the supervision cap is subsampled uniformly
        # at random (seeded). The former prefix mode silently assumed the
        # on-disk vector order is unbiased, which ann-benchmarks does not
        # guarantee; deep1m (9.99M -> 1M) was the affected case.
        "sampling_mode": "random_without_replacement",
    }

    ready = (
        router_train_path.exists()
        and router_gt_path.exists()
        and query_path.exists()
        and neigh_path.exists()
        and meta_path.exists()
    )
    if ready and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if meta == expected_meta:
            return base

    base.mkdir(parents=True, exist_ok=True)
    full_train, queries, query_neighbors = _load_hdf5_train_queries(dataset, data_root)

    if full_train.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(full_train.shape[0], size=max_samples, replace=False).astype(np.int32, copy=False)
        router_train = full_train[pick]
    else:
        pick = None
        router_train = full_train

    router_gt = _exact_self_knn(router_train, k=k, metric=spec.metric)

    np.save(router_train_path, router_train.astype(np.float32, copy=False))
    np.save(router_gt_path, router_gt.astype(np.int32, copy=False))
    np.save(query_path, queries.astype(np.float32, copy=False))
    np.save(neigh_path, query_neighbors.astype(np.int32, copy=False))
    if pick is not None:
        np.save(router_pick_path, pick.astype(np.int32, copy=False))
    elif router_pick_path.exists():
        router_pick_path.unlink()
    _write_json(meta_path, expected_meta)
    return base


def load_learned_dataset(
    dataset: str,
    data_root: Path,
    *,
    max_samples: int = 1_000_000,
    k: int = 100,
    seed: int = 0,
    force_prepare: bool = False,
) -> LearnedDataset:
    spec = DATASETS[dataset]
    data_root = data_root.expanduser().resolve()

    if dataset == "glove10k":
        base = prepare_glove10k(data_root, force=force_prepare)
        router_train = np.load(base / "train.npy").astype(np.float32, copy=False)
        router_neighbors = np.load(base / "groundTruth.npy").astype(np.int32, copy=False)
        queries = np.load(base / "queries.npy").astype(np.float32, copy=False)
        eval_neighbors = np.load(base / "neighbors100.npy").astype(np.int32, copy=False)
        q_cap = min(spec.eval_queries, queries.shape[0], eval_neighbors.shape[0])
        return LearnedDataset(
            dataset=dataset,
            router_train=router_train,
            router_neighbors=router_neighbors,
            index_train=router_train,
            queries=queries[:q_cap].astype(np.float32, copy=False),
            eval_neighbors=eval_neighbors[:q_cap].astype(np.int32, copy=False),
            sample_ids=None,
        )

    base = prepare_learned_dataset(
        dataset,
        data_root,
        max_samples=max_samples,
        k=k,
        seed=seed,
        force=force_prepare,
    )
    full_train, _, _ = _load_hdf5_train_queries(dataset, data_root)
    router_train = np.load(base / "router_train.npy").astype(np.float32, copy=False)
    router_neighbors = np.load(base / "router_groundtruth.npy").astype(np.int32, copy=False)
    queries = np.load(base / "queries.npy").astype(np.float32, copy=False)
    eval_neighbors = np.load(base / "neighbors100.npy").astype(np.int32, copy=False)
    sample_ids = None
    router_pick_path = base / "router_pick.npy"
    if router_pick_path.exists():
        sample_ids = np.load(router_pick_path).astype(np.int32, copy=False)
    q_cap = min(spec.eval_queries, queries.shape[0], eval_neighbors.shape[0])
    return LearnedDataset(
        dataset=dataset,
        router_train=router_train,
        router_neighbors=router_neighbors,
        index_train=full_train.astype(np.float32, copy=False),
        queries=queries[:q_cap].astype(np.float32, copy=False),
        eval_neighbors=eval_neighbors[:q_cap].astype(np.int32, copy=False),
        sample_ids=sample_ids,
    )


def load_search_dataset(dataset: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[dataset]
    data_root = data_root.expanduser().resolve()

    if dataset == "glove10k":
        base = data_root / "glove10k"
        train = np.load(base / "train.npy")
        queries = np.load(base / "queries.npy")
        neighbors = np.load(base / "neighbors100.npy")
    else:
        h5_path = data_root / spec.key / spec.hdf5_filename
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {h5_path}")
        with h5py.File(h5_path, "r") as handle:
            train = np.array(handle["train"], dtype=np.float32)
            queries = np.array(handle["test"], dtype=np.float32)
            neighbors = np.array(handle["neighbors"], dtype=np.int32)

    q_cap = min(spec.eval_queries, queries.shape[0], neighbors.shape[0])
    return (
        train.astype(np.float32, copy=False),
        queries[:q_cap].astype(np.float32, copy=False),
        neighbors[:q_cap].astype(np.int32, copy=False),
    )


def load_queries_only(dataset: str, data_root: Path) -> np.ndarray:
    spec = DATASETS[dataset]
    data_root = data_root.expanduser().resolve()

    if dataset == "glove10k":
        queries = np.load(data_root / "glove10k" / "queries.npy")
    else:
        h5_path = data_root / spec.key / spec.hdf5_filename
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {h5_path}")
        with h5py.File(h5_path, "r") as handle:
            queries = np.array(handle["test"], dtype=np.float32)

    q_cap = min(spec.eval_queries, queries.shape[0])
    return queries[:q_cap].astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset metadata and download helper.")
    parser.add_argument("datasets", nargs="*", help="Datasets to list or download.")
    parser.add_argument("--data-root", default="data", help="Dataset root directory.")
    parser.add_argument("--download", action="store_true", help="Download or prepare datasets.")
    parser.add_argument("--force", action="store_true", help="Force regeneration where supported.")
    parser.add_argument("--list", action="store_true", help="List supported datasets.")
    args = parser.parse_args()

    if args.list:
        for key in ("glove", "sift", "gist", "deep1m", "glove10k"):
            spec = DATASETS[key]
            print(
                f"{spec.key:8s} metric={spec.metric:6s} dim={spec.dim:4d} "
                f"N={spec.indexed_size:>9d} Q={spec.query_count:>5d} B={spec.partitions:>4d}"
            )
        return

    datasets = parse_datasets(args.datasets)
    data_root = Path(args.data_root).expanduser().resolve()

    if not args.download:
        for key in datasets:
            spec = DATASETS[key]
            print(f"{key}: {spec.public_url if key != 'glove10k' else 'generated from glove'}")
        return

    for key in datasets:
        if key == "glove10k":
            out_dir = prepare_glove10k(data_root, force=args.force)
            print(f"[datasets] ready: {out_dir}")
            continue
        dst = download_ann_dataset(key, data_root)
        print(f"[datasets] downloaded: {dst}")


if __name__ == "__main__":
    main()
