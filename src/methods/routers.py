from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import murmurhash3_32

from artifacts import ensure_dir, write_csv, write_json
from console import banner, info, print_table
from datasets import LearnedDataset, load_learned_dataset
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_ROUTER,
    BENCHMARK_BATCH_1,
    BENCHMARK_BATCH_256,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    QUERY_MEM_MODE_IN_PROCESS,
    TIMING_TRIALS,
    WARMUP_QUERIES,
    per_query_recall_profile,
    per_query_recalls,
    recall_at_k,
    recall_profile,
    serving_payload_metrics,
    summarize_timing_trials,
    timing_contract_metrics,
)
from presets import DATASETS


_CPP_IMPORT_ERROR: Exception | None = None
_MISSING_CPP_SYMBOLS: tuple[str, ...] = ()

try:
    import lisbench_native as _cpp_ops

    _search_exact_batch_native = getattr(_cpp_ops, "search_exact_batch_cpp", None)
    _MISSING_CPP_SYMBOLS = tuple(
        name
        for name, value in {
            "search_exact_batch_cpp": _search_exact_batch_native,
        }.items()
        if value is None
    )
    _HAS_NATIVE_OPS = not _MISSING_CPP_SYMBOLS
    _NATIVE_BACKEND = "cpp" if _HAS_NATIVE_OPS else "incomplete_cpp"
except ImportError as exc:  # pragma: no cover
    _search_exact_batch_native = None
    _CPP_IMPORT_ERROR = exc
    _HAS_NATIVE_OPS = False
    _NATIVE_BACKEND = "missing_cpp"

try:
    import numba  # type: ignore
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    numba = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    _HAS_NUMBA = False


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            rss_pages = int(handle.readline().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _base_vector_payload_mb(bundle: LearnedDataset) -> float:
    """Size of one canonical float32 corpus-vector store."""
    n, d = bundle.index_train.shape
    return float(n) * float(d) * 4.0 / (1024.0 * 1024.0)

def _requires_full_vectors(cfg: RouterConfig) -> bool:
    """Whether the serving path dereferences raw corpus vectors."""

    return True



def _require_cpp_backend(feature: str) -> None:
    if _HAS_NATIVE_OPS:
        return
    if _CPP_IMPORT_ERROR is not None:
        detail = f" import failed: {_CPP_IMPORT_ERROR}"
    elif _MISSING_CPP_SYMBOLS:
        detail = f" missing symbols: {', '.join(_MISSING_CPP_SYMBOLS)}"
    else:
        detail = ""
    raise RuntimeError(
        f"{feature} requires the C++ backend; no Python/Cython fallback is allowed."
        f"{detail} Build it with `pip install -e .` (or `cmake --build build`)."
    )


def _ensure_tf_graph():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for learned router training.") from exc
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    return tf


@dataclass(frozen=True)
class RouterConfig:
    dataset: str
    method: str
    init_mode: str
    repetitions: int = 4
    partitions: int | None = None
    probe_depth: int = 10
    min_reps: int = 2
    # One deterministic top-100 ranking supports Recall@1/10/100.  Tiny unit
    # fixtures may explicitly lower k, but paper runs must leave this at 100.
    k: int = 100
    hidden: int = 512
    epochs: int = 20
    batch_size: int = 1024
    reassign_batch_size: int = 1024
    query_batch_size: int = 256
    learning_rate: float = 1e-3
    reassign_interval: int = 5
    top_k_reassign: int = 2
    drop_tail: bool = True
    weight_init: str = "truncated_normal"
    clamp_targets: bool = False
    label_smoothing: float = 0.0
    prepare_max_samples: int = 1_000_000
    prepare_k: int = 100
    ivf_permute_reps: bool = True
    exact_rerank_enabled: bool = True
    seed: int = 0

    @property
    def metric(self) -> str:
        return DATASETS[self.dataset].metric

    @property
    def index_partitions(self) -> int:
        return int(self.partitions or DATASETS[self.dataset].partitions)


@dataclass
class RouterWeights:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class InitState:
    assignments: np.ndarray


@dataclass
class FitState:
    exp_dir: Path
    weights: list[RouterWeights]
    lookups: list[tuple[np.ndarray, np.ndarray]]
    point_to_buckets: np.ndarray
    train_seconds: float
    index_seconds: float


@dataclass
class QueryResult:
    probe_depth: int
    recall_at_10: float
    avg_computations: float
    computation_min: float
    computation_max: float
    qps: float | None
    query_mem_delta_mb: float
    # Diagnostic fields (populated when evaluate(..., diagnostic=True))
    timing_breakdown: dict | None = None
    exact_rerank_count: float | None = None
    avg_probe_count: float | None = None
    per_query: list[dict] | None = None
    recall_at_1: float | None = None
    recall_at_100: float | None = None
    # Process CPU consumed by the same serving interval used for ``qps``.
    # Ground-truth recall and diagnostic construction are intentionally
    # excluded so CPU work remains comparable with FAISS and LIRA.
    query_cpu_s: float | None = None


def _prepare_batches(n: int, batch_size: int, drop_tail: bool) -> tuple[int, list[int]]:
    train_limit = n - (n % batch_size) if drop_tail else n
    return train_limit, list(range(0, train_limit, batch_size))


def _topk_sorted(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.argmax(logits, axis=1, keepdims=True).astype(np.int32, copy=False)
    part = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(logits, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)


def _router_forward(x: np.ndarray, weights: RouterWeights) -> np.ndarray:
    hidden = np.maximum(x @ weights.W1 + weights.b1, 0.0).astype(np.float32, copy=False)
    return (hidden @ weights.W2 + weights.b2).astype(np.float32, copy=False)


def _normalize_if_cosine(arr: np.ndarray, metric: str) -> np.ndarray:
    if str(metric).lower() != "cosine":
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _query_buffer(
    *,
    index_full_dataset: bool,
    n_points: int,
    repetitions: int,
    probe_depth: int,
    partitions: int,
    min_reps: int,
    k: int,
) -> tuple[int, int]:
    raw = int(2 * repetitions * n_points * probe_depth / (partitions * max(min_reps, 1)))
    if index_full_dataset:
        aligned = 1024 * (raw // 1024)
        return raw, max(aligned, k)
    return raw, max(1024, int(raw * 2), k)


def _build_inverted(assign: np.ndarray, partitions: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(assign, minlength=partitions).astype(np.int64, copy=False)
    offsets = np.zeros(partitions + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    ids = np.argsort(assign, kind="stable").astype(np.int32, copy=False)
    return offsets, ids


def _write_training_artifacts(
    exp_dir: Path,
    cfg: RouterConfig,
    weights: list[RouterWeights],
    lookups: list[tuple[np.ndarray, np.ndarray]],
    point_to_buckets: np.ndarray,
    train_assignments: np.ndarray,
    train_seconds: float,
    index_seconds: float,
) -> None:
    models_dir = ensure_dir(exp_dir / "models")
    lookups_dir = ensure_dir(exp_dir / "staging" / "lookups")
    staging_dir = ensure_dir(exp_dir / "staging")
    write_json(
        exp_dir / "config.json",
        {
            "method": cfg.method,
            "backend": "tensorflow",
            **asdict(cfg),
        },
    )
    np.savez(
        models_dir / "router_weights.npz",
        W1=np.stack([w.W1 for w in weights], axis=0),
        b1=np.stack([w.b1 for w in weights], axis=0),
        W2=np.stack([w.W2 for w in weights], axis=0),
        b2=np.stack([w.b2 for w in weights], axis=0),
    )
    np.save(staging_dir / "assignments.npy", train_assignments.astype(np.int32, copy=False))
    np.save(staging_dir / "point_to_buckets.npy", point_to_buckets.astype(np.int32, copy=False))
    for rep, (offsets, ids) in enumerate(lookups):
        np.save(lookups_dir / f"rep{rep}_offsets.npy", offsets.astype(np.int64, copy=False))
        np.save(lookups_dir / f"rep{rep}_ids.npy", ids.astype(np.int32, copy=False))
    write_json(
        exp_dir / "train_metrics.json",
        {
            "dataset": cfg.dataset,
            "method": cfg.method,
            "train_s": train_seconds,
            "full_index_lookup_s": index_seconds,
        },
    )
    write_index_manifest(exp_dir)


def _load_fit_state(exp_dir: Path, cfg: RouterConfig) -> FitState:
    payload = np.load(exp_dir / "models" / "router_weights.npz")
    weights: list[RouterWeights] = []
    for rep in range(payload["W1"].shape[0]):
        weights.append(
            RouterWeights(
                W1=payload["W1"][rep].astype(np.float32, copy=False),
                b1=payload["b1"][rep].astype(np.float32, copy=False),
                W2=payload["W2"][rep].astype(np.float32, copy=False),
                b2=payload["b2"][rep].astype(np.float32, copy=False),
            )
        )
    lookups: list[tuple[np.ndarray, np.ndarray]] = []
    lookups_dir = exp_dir / "staging" / "lookups"
    for rep in range(cfg.repetitions):
        offsets = np.load(lookups_dir / f"rep{rep}_offsets.npy").astype(np.int64, copy=False)
        ids = np.load(lookups_dir / f"rep{rep}_ids.npy").astype(np.int32, copy=False)
        lookups.append((offsets, ids))
    point_to_buckets = np.load(exp_dir / "staging" / "point_to_buckets.npy").astype(np.int32, copy=False)
    metrics_path = exp_dir / "train_metrics.json"
    train_seconds = 0.0
    index_seconds = 0.0
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            train_seconds = float(payload.get("train_s", 0.0))
            index_seconds = float(payload.get("full_index_lookup_s", 0.0))
        except Exception:
            train_seconds = 0.0
            index_seconds = 0.0
    return FitState(
        exp_dir=exp_dir,
        weights=weights,
        lookups=lookups,
        point_to_buckets=point_to_buckets,
        train_seconds=train_seconds,
        index_seconds=index_seconds,
    )


def _load_router_checkpoint(
    checkpoint_dir: Path,
    cfg: RouterConfig,
) -> tuple[list[RouterWeights], np.ndarray, InitState, dict[str, float], dict[str, object]]:

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "models" / "router_weights.npz"
    assignments_path = checkpoint_dir / "staging" / "assignments.npy"
    if not config_path.exists() or not weights_path.exists() or not assignments_path.exists():
        raise FileNotFoundError(f"Incomplete router checkpoint: {checkpoint_dir}")
    source_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": cfg.dataset,
        "seed": int(cfg.seed),
        "repetitions": int(cfg.repetitions),
        "hidden": int(cfg.hidden),
    }
    for key, expected in checks.items():
        actual = source_cfg.get(key)
        if actual != expected:
            raise ValueError(
                f"Router checkpoint mismatch for {key}: expected {expected!r}, got {actual!r} "
                f"from {checkpoint_dir}"
            )

    blob = np.load(weights_path)
    weights = [
        RouterWeights(
            W1=blob["W1"][rep].astype(np.float32, copy=False),
            b1=blob["b1"][rep].astype(np.float32, copy=False),
            W2=blob["W2"][rep].astype(np.float32, copy=False),
            b2=blob["b2"][rep].astype(np.float32, copy=False),
        )
        for rep in range(blob["W1"].shape[0])
    ]
    train_assignments = np.load(assignments_path).astype(np.int32, copy=False)
    init_state = InitState(
        assignments=train_assignments.copy(),
    )
    train_metrics_path = checkpoint_dir / "train_metrics.json"
    source_timings: dict[str, float] = {}
    if train_metrics_path.exists():
        payload = json.loads(train_metrics_path.read_text(encoding="utf-8"))
        for key in (
            "initialization_s",
            "router_train_s",
            "train_s",
            "index_build_compute_s",
            "artifact_write_s",
            "build_total_s",
        ):
            if payload.get(key) is not None:
                source_timings[key] = float(payload[key])
    return weights, train_assignments, init_state, source_timings, source_cfg


def _build_timing_summary(exp_dir: Path) -> dict[str, float]:
    path = exp_dir / "train_metrics.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: float(payload[key])
        for key in (
            "initialization_s",
            "router_train_s",
            "train_s",
            "index_build_compute_s",
            "artifact_write_s",
            "build_total_s",
            "build_wall_s",
        )
        if payload.get(key) is not None
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isolated_query_memory(
    *,
    exp_dir: Path,
    dataset: str,
    data_root: Path,
    probe_depth: int,
) -> dict[str, float] | None:
    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "-m",
        "run_router_query_isolated",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--experiment-dir",
        str(exp_dir),
        "--probe-depth",
        str(probe_depth),
        "--max-queries",
        "1024",
    ]
    proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
    payload = json.loads(proc.stdout.strip())
    if not isinstance(payload, dict):
        return None
    return {str(k): float(v) for k, v in payload.items() if v is not None}


def _kmeans_cache_paths(data_root: Path, cfg: RouterConfig) -> tuple[Path, Path]:
    dataset_dir = data_root / cfg.dataset
    stem = f"kmeans_labels_B{cfg.index_partitions}_seed{cfg.seed}"
    labels_path = dataset_dir / f"{stem}.npy"
    meta_path = dataset_dir / f"{stem}.meta.json"
    return labels_path, meta_path


def _hash_assignments(train: np.ndarray, cfg: RouterConfig) -> InitState:
    n = train.shape[0]
    assignments = np.empty((cfg.repetitions, n), dtype=np.int32)
    for rep in range(cfg.repetitions):
        seed = int(cfg.seed + rep)
        assignments[rep] = np.fromiter(
            (murmurhash3_32(int(i), seed=seed) % cfg.index_partitions for i in range(n)),
            dtype=np.int32,
            count=n,
        )
    return InitState(assignments=assignments)


def _kmeans_assignments(train: np.ndarray, cfg: RouterConfig, data_root: Path) -> InitState:
    labels_path, meta_path = _kmeans_cache_paths(data_root, cfg)
    expected_meta = {
        "dataset": cfg.dataset,
        "metric": cfg.metric,
        "train_rows": int(train.shape[0]),
        "dim": int(train.shape[1]),
        "B": int(cfg.index_partitions),
        "seed": int(cfg.seed),
    }

    def _fit_labels() -> np.ndarray:
        batch_size = min(10_000, max(1_000, train.shape[0]))
        kmeans = MiniBatchKMeans(
            n_clusters=cfg.index_partitions,
            batch_size=batch_size,
            max_iter=100,
            random_state=cfg.seed,
            verbose=0,
        )
        labels = kmeans.fit_predict(train).astype(np.int32, copy=False)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, labels)
        write_json(meta_path, expected_meta)
        return labels

    rebuild = True
    if labels_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            labels = np.load(labels_path).astype(np.int32, copy=False)
            rebuild = (
                meta != expected_meta
                or labels.shape[0] != train.shape[0]
                or labels.min(initial=0) < 0
                or int(labels.max(initial=0)) + 1 > cfg.index_partitions
            )
        except Exception:
            rebuild = True
    if rebuild:
        labels = _fit_labels()

    assignments = np.empty((cfg.repetitions, train.shape[0]), dtype=np.int32)
    rng = np.random.default_rng(cfg.seed)
    for rep in range(cfg.repetitions):
        if cfg.ivf_permute_reps:
            perm = rng.permutation(cfg.index_partitions).astype(np.int32, copy=False)
            assignments[rep] = perm[labels]
        else:
            assignments[rep] = labels
    return InitState(assignments=assignments)


def _router_weights(tf, d: int, hidden: int, partitions: int, weight_init: str):
    if weight_init == "truncated_normal":
        W1 = tf.compat.v1.get_variable(
            "W1",
            initializer=tf.compat.v1.truncated_normal([d, hidden], stddev=0.05, dtype=tf.float32),
        )
        b1 = tf.compat.v1.get_variable(
            "b1",
            initializer=tf.compat.v1.truncated_normal([hidden], stddev=0.05, dtype=tf.float32),
        )
        W2 = tf.compat.v1.get_variable(
            "W2",
            initializer=tf.compat.v1.truncated_normal([hidden, partitions], stddev=0.05, dtype=tf.float32),
        )
        b2 = tf.compat.v1.get_variable(
            "b2",
            initializer=tf.compat.v1.truncated_normal([partitions], stddev=0.05, dtype=tf.float32),
        )
    else:
        W1 = tf.compat.v1.get_variable(
            "W1", shape=[d, hidden], initializer=tf.compat.v1.initializers.glorot_uniform()
        )
        b1 = tf.compat.v1.get_variable("b1", shape=[hidden], initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(
            "W2", shape=[hidden, partitions], initializer=tf.compat.v1.initializers.glorot_uniform()
        )
        b2 = tf.compat.v1.get_variable("b2", shape=[partitions], initializer=tf.compat.v1.zeros_initializer())
    return W1, b1, W2, b2


def _build_graph(
    tf,
    *,
    d: int,
    hidden: int,
    partitions: int,
    assign: np.ndarray,
    neighbors_per_item: int,
    weight_init: str,
    clamp_targets: bool,
    label_smoothing: float,
    learning_rate: float,
    top_k: int,
):
    graph = tf.Graph()
    with graph.as_default():
        x_pl = tf.compat.v1.placeholder(tf.float32, shape=(None, d), name="x")
        neighbors_pl = tf.compat.v1.placeholder(tf.int64, shape=(None,), name="neighbors")
        lookup_init = tf.constant(assign, dtype=tf.int64)
        lookup_var = tf.compat.v1.get_variable("lookup_assignments", initializer=lookup_init, trainable=False)
        lookup_update_pl = tf.compat.v1.placeholder(tf.int64, shape=(assign.shape[0],), name="lookup_update")
        lookup_assign_op = tf.compat.v1.assign(lookup_var, lookup_update_pl)

        W1, b1, W2, b2 = _router_weights(tf, d, hidden, partitions, weight_init)
        hidden_act = tf.nn.relu(tf.matmul(x_pl, W1) + b1)
        logits = tf.matmul(hidden_act, W2) + b2

        batch_rows = tf.shape(neighbors_pl, out_type=tf.int64)[0] // neighbors_per_item
        row_ids = tf.reshape(tf.tile(tf.range(batch_rows, dtype=tf.int64)[:, None], [1, neighbors_per_item]), [-1])
        bucket_ids = tf.gather(lookup_var, neighbors_pl)
        scatter_indices = tf.stack([row_ids, bucket_ids], axis=1)
        sparse = tf.SparseTensor(
            indices=scatter_indices,
            values=tf.ones_like(row_ids, dtype=tf.float32),
            dense_shape=[batch_rows, partitions],
        )
        targets = tf.compat.v1.sparse_tensor_to_dense(sparse, validate_indices=False)
        if clamp_targets:
            targets = tf.minimum(targets, 1.0)
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        topk = tf.nn.top_k(logits, k=top_k, sorted=True)

    return {
        "graph": graph,
        "x_pl": x_pl,
        "neighbors_pl": neighbors_pl,
        "lookup_update_pl": lookup_update_pl,
        "lookup_assign_op": lookup_assign_op,
        "logits": logits,
        "loss": loss,
        "optimiser": optimiser,
        "topk_indices": topk.indices,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }


def _greedy_reassign_python(top_candidates: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n, k = top_candidates.shape
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        best = 0
        best_count = counts[top_candidates[i, 0]]
        for j in range(1, k):
            bucket = top_candidates[i, j]
            bucket_count = counts[bucket]
            if bucket_count < best_count:
                best = j
                best_count = bucket_count
        chosen = top_candidates[i, best]
        out[i] = chosen
        counts[chosen] += 1
    return out


if _HAS_NUMBA:
    _greedy_reassign_numba = njit(cache=True)(_greedy_reassign_python)
else:  # pragma: no cover
    _greedy_reassign_numba = None


def _greedy_reassign(top_candidates: np.ndarray, partitions: int) -> np.ndarray:
    counts = np.zeros(partitions, dtype=np.int64)
    if _greedy_reassign_numba is not None:
        return _greedy_reassign_numba(top_candidates.astype(np.int32, copy=False), counts)
    return _greedy_reassign_python(top_candidates.astype(np.int32, copy=False), counts)


def _greedy_reassign_with_counts(top_candidates: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if _greedy_reassign_numba is not None:
        return _greedy_reassign_numba(top_candidates.astype(np.int32, copy=False), counts)
    return _greedy_reassign_python(top_candidates.astype(np.int32, copy=False), counts)


def _fit_single_rep(
    rep: int,
    train: np.ndarray,
    neighbors: np.ndarray,
    initial_assign: np.ndarray,
    cfg: RouterConfig,
) -> tuple[np.ndarray, RouterWeights]:
    tf = _ensure_tf_graph()
    n, d = train.shape
    train_limit, batch_starts = _prepare_batches(n, cfg.batch_size, cfg.drop_tail)
    assign = initial_assign.astype(np.int64, copy=True)
    tf.compat.v1.set_random_seed(cfg.seed + rep)
    np.random.seed(cfg.seed + rep)
    ops = _build_graph(
        tf,
        d=d,
        hidden=cfg.hidden,
        partitions=cfg.index_partitions,
        assign=assign,
        neighbors_per_item=neighbors.shape[1],
        weight_init=cfg.weight_init,
        clamp_targets=cfg.clamp_targets,
        label_smoothing=cfg.label_smoothing,
        learning_rate=cfg.learning_rate,
        top_k=cfg.top_k_reassign,
    )
    cfg_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    cfg_tf.gpu_options.allow_growth = True  # type: ignore[attr-defined]
    intra = os.environ.get("TF_NUM_INTRAOP_THREADS")
    inter = os.environ.get("TF_NUM_INTEROP_THREADS")
    if intra:
        cfg_tf.intra_op_parallelism_threads = int(intra)
    if inter:
        cfg_tf.inter_op_parallelism_threads = int(inter)
    with tf.compat.v1.Session(graph=ops["graph"], config=cfg_tf) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(1, cfg.epochs + 1):
            loss_last = 0.0
            t0 = time.perf_counter()
            for start in batch_starts:
                end = start + cfg.batch_size
                _, loss_last = sess.run(
                    [ops["optimiser"], ops["loss"]],
                    feed_dict={
                        ops["x_pl"]: train[start:end],
                        ops["neighbors_pl"]: neighbors[start:end].reshape(-1),
                    },
                )
            train_s = time.perf_counter() - t0

            moved = 0.0
            reassign_s = 0.0
            if cfg.reassign_interval > 0 and epoch % cfg.reassign_interval == 0:
                t0 = time.perf_counter()
                top_candidates = np.empty((train_limit, cfg.top_k_reassign), dtype=np.int32)
                fill = 0
                for start in range(0, train_limit, cfg.reassign_batch_size):
                    end = min(start + cfg.reassign_batch_size, train_limit)
                    top_batch = sess.run(
                        ops["topk_indices"],
                        feed_dict={ops["x_pl"]: train[start:end]},
                    )
                    top_candidates[fill : fill + top_batch.shape[0]] = top_batch.astype(np.int32, copy=False)
                    fill += top_batch.shape[0]
                new_assign = assign.copy()
                new_assign[:train_limit] = _greedy_reassign(top_candidates, cfg.index_partitions)
                moved = float(np.mean(new_assign != assign))
                assign = new_assign
                sess.run(ops["lookup_assign_op"], feed_dict={ops["lookup_update_pl"]: assign})
                reassign_s = time.perf_counter() - t0
            info(
                f"[{cfg.method} rep {rep + 1}/{cfg.repetitions}] "
                f"epoch={epoch:02d} loss={loss_last:.4f} train_s={train_s:.1f} "
                f"reassign_s={reassign_s:.1f} moved={moved:.3f}"
            )

        W1, b1, W2, b2 = sess.run([ops["W1"], ops["b1"], ops["W2"], ops["b2"]])
    return (
        assign.astype(np.int32, copy=False),
        RouterWeights(
            W1=W1.astype(np.float32, copy=False),
            b1=b1.astype(np.float32, copy=False),
            W2=W2.astype(np.float32, copy=False),
            b2=b2.astype(np.float32, copy=False),
        ),
    )


def _full_index_lookups(
    index_train: np.ndarray,
    weights: list[RouterWeights],
    cfg: RouterConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    lookups: list[tuple[np.ndarray, np.ndarray]] = []
    point_to_buckets = np.empty((index_train.shape[0], cfg.repetitions), dtype=np.int32)
    for rep, rep_weights in enumerate(weights):
            counts = np.zeros(cfg.index_partitions, dtype=np.int64)
            assign = np.empty(index_train.shape[0], dtype=np.int32)
            fill = 0
            for start in range(0, index_train.shape[0], cfg.reassign_batch_size):
                end = min(start + cfg.reassign_batch_size, index_train.shape[0])
                logits = _router_forward(index_train[start:end], rep_weights)
                top_batch = _topk_sorted(logits, cfg.top_k_reassign)
                assign_batch = _greedy_reassign_with_counts(top_batch, counts)
                assign[fill : fill + assign_batch.shape[0]] = assign_batch
                fill += top_batch.shape[0]
            point_to_buckets[:, rep] = assign
            lookups.append(_build_inverted(assign, cfg.index_partitions))
    return lookups, point_to_buckets


def _search_exact_batch(
    *,
    queries: np.ndarray,
    topm: np.ndarray,
    probe_counts: np.ndarray,
    state: FitState,
    train_view: np.ndarray,
    train_sq_norms: np.ndarray | None,
    metric: str,
    cfg: RouterConfig,
    buffer: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Serve an uncompressed batch through the exact-scoring C++ hot path.

    Collects each query's probed union (with the method's min_reps/buffer
    contract) and ranks it by exact similarity, parallel over queries.
    """
    _require_cpp_backend("exact batched query serving")
    metric_kind = 1 if metric == "l2" else 0
    sq_norms = (
        np.ascontiguousarray(train_sq_norms, dtype=np.float32)
        if train_sq_norms is not None
        else np.empty(0, dtype=np.float32)
    )
    retrieved, candidate_counts, timing = _search_exact_batch_native(
        np.ascontiguousarray(queries, dtype=np.float32),
        np.ascontiguousarray(topm, dtype=np.int32),
        np.ascontiguousarray(probe_counts, dtype=np.int32),
        state.lookups,
        np.ascontiguousarray(train_view, dtype=np.float32),
        sq_norms,
        metric_kind,
        int(cfg.k),
        int(cfg.min_reps),
        int(buffer),
    )
    return (
        np.asarray(retrieved, dtype=np.int32),
        np.asarray(candidate_counts, dtype=np.int32),
        {str(k): float(v) for k, v in dict(timing).items()},
    )


class RouterIndex:
    name: str = ""
    init_mode: str = ""

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg

    def _initial_assignments(self, train: np.ndarray, data_root: Path) -> InitState:
        if self.cfg.init_mode == "hash":
            return _hash_assignments(train, self.cfg)
        if self.cfg.init_mode == "kmeans":
            return _kmeans_assignments(train, self.cfg, data_root)
        raise ValueError(f"Unsupported init_mode: {self.cfg.init_mode}")

    def fit(
        self,
        bundle: LearnedDataset,
        exp_dir: Path,
        *,
        data_root: Path,
        router_checkpoint_dir: Path | None = None,
    ) -> FitState:
        fit_wall_start = time.perf_counter()
        banner("method", f"{self.cfg.method} | dataset={self.cfg.dataset} | init={self.cfg.init_mode}")
        info(
            f"router_train={bundle.router_train.shape[0]}, index_train={bundle.index_train.shape[0]}, "
            f"queries={bundle.queries.shape[0]}, dim={bundle.router_train.shape[1]}"
        )
        source_cfg: dict[str, object] | None = None
        if router_checkpoint_dir is None:
            init_t0 = time.perf_counter()
            init_state = self._initial_assignments(bundle.router_train, data_root)
            initialization_seconds = time.perf_counter() - init_t0
            train_assignments = np.empty_like(init_state.assignments)
            weights: list[RouterWeights] = []
            t0 = time.perf_counter()
            for rep in range(self.cfg.repetitions):
                learned_assign, rep_weights = _fit_single_rep(
                    rep,
                    bundle.router_train,
                    bundle.router_neighbors,
                    init_state.assignments[rep],
                    self.cfg,
                )
                train_assignments[rep] = learned_assign
                weights.append(rep_weights)
            train_seconds = time.perf_counter() - t0
        else:
            weights, train_assignments, init_state, source_timings, source_cfg = _load_router_checkpoint(
                router_checkpoint_dir,
                self.cfg,
            )
            initialization_seconds = float(source_timings.get("initialization_s", 0.0))
            train_seconds = float(source_timings.get("router_train_s", source_timings.get("train_s", 0.0)))
            info(f"  - reusing router checkpoint: {Path(router_checkpoint_dir).resolve()}")

        index_build_t0 = time.perf_counter()
        t0 = time.perf_counter()
        lookups, point_to_buckets = _full_index_lookups(bundle.index_train, weights, self.cfg)
        lookup_seconds = time.perf_counter() - t0
        index_seconds = time.perf_counter() - index_build_t0
        artifact_t0 = time.perf_counter()
        _write_training_artifacts(
            exp_dir,
            self.cfg,
            weights,
            lookups,
            point_to_buckets,
            train_assignments,
            train_seconds,
            lookup_seconds,
        )
        if router_checkpoint_dir is not None:
            checkpoint_dir = Path(router_checkpoint_dir).expanduser().resolve()
            shared_components = ["router_weights", "training_assignments"]
            write_json(
                exp_dir / "router_checkpoint_lineage.json",
                {
                    "source": str(checkpoint_dir),
                    "source_method": source_cfg.get("method") if source_cfg is not None else None,
                    "source_router_weights_sha256": _sha256_file(
                        checkpoint_dir / "models" / "router_weights.npz"
                    ),
                    "source_assignments_sha256": _sha256_file(
                        checkpoint_dir / "staging" / "assignments.npy"
                    ),
                    "shared_components": shared_components,
                },
            )
        write_index_manifest(exp_dir)
        artifact_write_seconds = time.perf_counter() - artifact_t0
        build_total_seconds = (
            float(initialization_seconds)
            + float(train_seconds)
            + float(index_seconds)
            + float(artifact_write_seconds)
        )
        build_wall_seconds = time.perf_counter() - fit_wall_start
        write_json(
            exp_dir / "train_metrics.json",
            {
                "dataset": self.cfg.dataset,
                "method": self.cfg.method,
                "initialization_s": initialization_seconds,
                "router_train_s": train_seconds,
                "train_s": train_seconds,
                "full_index_lookup_s": lookup_seconds,
                "index_build_compute_s": index_seconds,
                "artifact_write_s": artifact_write_seconds,
                "build_total_s": build_total_seconds,
                "build_wall_s": build_wall_seconds,
                "reused_router_checkpoint": router_checkpoint_dir is not None,
            },
        )
        return FitState(
            exp_dir=exp_dir,
            weights=weights,
            lookups=lookups,
            point_to_buckets=point_to_buckets,
            train_seconds=train_seconds,
            index_seconds=index_seconds,
        )

    def evaluate(
        self,
        bundle: LearnedDataset,
        state: FitState,
        *,
        probe_depth: int,
        diagnostic: bool = False,
        query_limit: int | None = None,
        compute_quality: bool = True,
    ) -> QueryResult:
        if diagnostic and not compute_quality:
            raise ValueError("diagnostic evaluation requires quality computation")
        if query_limit is not None:
            limit = min(max(0, int(query_limit)), int(bundle.queries.shape[0]))
            bundle = replace(
                bundle,
                queries=bundle.queries[:limit],
                eval_neighbors=bundle.eval_neighbors[:limit],
            )
        metric = self.cfg.metric
        m_cap = int(probe_depth)
        _, buffer = _query_buffer(
            index_full_dataset=True,
            n_points=bundle.index_train.shape[0],
            repetitions=self.cfg.repetitions,
            probe_depth=m_cap,
            partitions=self.cfg.index_partitions,
            min_reps=self.cfg.min_reps,
            k=self.cfg.k,
        )
        train_view = (
            bundle.index_train.astype(np.float32, copy=False)
            if self.cfg.exact_rerank_enabled
            else np.empty((0, bundle.index_train.shape[1]), dtype=np.float32)
        )
        query_view = bundle.queries.astype(np.float32, copy=False)
        train_sq_norms = None
        if metric == "cosine":
            query_view = _normalize_if_cosine(bundle.queries, metric)
            if self.cfg.exact_rerank_enabled:
                train_view = _normalize_if_cosine(bundle.index_train, metric)
        elif metric == "l2" and self.cfg.exact_rerank_enabled:
            train_sq_norms = np.sum(bundle.index_train * bundle.index_train, axis=1).astype(np.float32, copy=False)

        n_queries = bundle.queries.shape[0]
        mem_before = _rss_mb()
        t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        retrieved = np.full((n_queries, self.cfg.k), -1, dtype=np.int32)
        candidate_counts = np.zeros(n_queries, dtype=np.int32)
        rerank_counts = np.zeros(n_queries, dtype=np.int32) if diagnostic else None
        probe_count_trace = np.zeros(n_queries, dtype=np.int32) if diagnostic else None
        t_probe_total = 0.0
        t_collect_total = 0.0
        t_rerank_total = 0.0
        for start in range(0, n_queries, self.cfg.query_batch_size):
            end = min(start + self.cfg.query_batch_size, n_queries)
            batch = bundle.queries[start:end]
            batch_query_view = query_view[start:end]
            t_probe_start = time.perf_counter()
            per_rep_topm: list[np.ndarray] = []
            for rep_weights in state.weights:
                logits = _router_forward(batch, rep_weights)
                per_rep_topm.append(_topk_sorted(logits, m_cap))
            stacked = np.stack(per_rep_topm, axis=1)  # (batch, R, m)
            probe_counts = np.full(end - start, int(probe_depth), dtype=np.int32)
            if diagnostic:
                t_probe_total += time.perf_counter() - t_probe_start
                if probe_count_trace is not None:
                    probe_count_trace[start:end] = probe_counts
            batch_retrieved, batch_candidate_counts, batch_timing = _search_exact_batch(
                queries=batch_query_view,
                topm=stacked,
                probe_counts=probe_counts,
                state=state,
                train_view=train_view,
                train_sq_norms=train_sq_norms,
                metric=metric,
                cfg=self.cfg,
                buffer=buffer,
            )
            retrieved[start:end] = batch_retrieved
            candidate_counts[start:end] = batch_candidate_counts
            if diagnostic:
                if rerank_counts is not None:
                    rerank_counts[start:end] = batch_candidate_counts
                t_collect_total += float(batch_timing.get("collect_s", 0.0))
                t_rerank_total += float(batch_timing.get("rerank_s", 0.0))
        total_s = time.perf_counter() - t0
        query_cpu_s = time.process_time() - cpu_t0
        mem_after = _rss_mb()

        if compute_quality:
            recalls = recall_profile(retrieved, bundle.eval_neighbors)
            recall = recalls[10]
            if recall is None:
                available_k = min(10, retrieved.shape[1], bundle.eval_neighbors.shape[1])
                recall = recall_at_k(retrieved, bundle.eval_neighbors, available_k)
        else:
            recalls = {1: None, 10: None, 100: None}
            recall = float("nan")
        qps = float(n_queries / total_s) if total_s > 0 else None

        timing_breakdown = None
        avg_rerank = None
        avg_probe_count = None
        per_query_list = None
        if diagnostic:
            timing_breakdown = {
                "probe_s": round(t_probe_total, 6),
                "collect_s": round(t_collect_total, 6),
                "rerank_s": round(t_rerank_total, 6),
                "total_s": round(total_s, 6),
            }
            if rerank_counts is not None:
                avg_rerank = float(np.mean(rerank_counts))
            if probe_count_trace is not None:
                avg_probe_count = float(np.mean(probe_count_trace))
            per_query_profiles = per_query_recall_profile(retrieved, bundle.eval_neighbors)
            legacy_recalls = per_query_profiles[10]
            if legacy_recalls is None:
                available_k = min(10, retrieved.shape[1], bundle.eval_neighbors.shape[1])
                legacy_recalls = per_query_recalls(retrieved, bundle.eval_neighbors, available_k)
            per_query_list = []
            for qi in range(n_queries):
                row: dict = {
                    "recall": float(legacy_recalls[qi]),
                    "recall_at_1": (
                        float(per_query_profiles[1][qi]) if per_query_profiles[1] is not None else None
                    ),
                    "recall_at_10": (
                        float(per_query_profiles[10][qi]) if per_query_profiles[10] is not None else None
                    ),
                    "recall_at_100": (
                        float(per_query_profiles[100][qi]) if per_query_profiles[100] is not None else None
                    ),
                    "candidates_after": int(candidate_counts[qi]),
                }
                if probe_count_trace is not None:
                    row["probe_count"] = int(probe_count_trace[qi])
                if rerank_counts is not None:
                    row["rerank_count"] = int(rerank_counts[qi])
                per_query_list.append(row)

        return QueryResult(
            probe_depth=probe_depth,
            recall_at_10=recall,
            avg_computations=float(np.mean(candidate_counts)),
            computation_min=float(np.min(candidate_counts)),
            computation_max=float(np.max(candidate_counts)),
            qps=qps,
            query_mem_delta_mb=max(0.0, float(mem_after) - float(mem_before)),
            timing_breakdown=timing_breakdown,
            exact_rerank_count=avg_rerank,
            avg_probe_count=avg_probe_count,
            per_query=per_query_list,
            recall_at_1=recalls[1],
            recall_at_100=recalls[100],
            query_cpu_s=float(query_cpu_s),
        )

    def _benchmark_query_mode(
        self,
        *,
        bundle: LearnedDataset,
        state: FitState,
        probe_depth: int,
        batch_size: int,
    ) -> tuple[QueryResult, object]:
        """Measure one explicit serving mode after a full-path warm-up."""

        original_cfg = self.cfg
        try:
            self.cfg = replace(self.cfg, query_batch_size=int(batch_size))
            warmup_queries = min(WARMUP_QUERIES, int(bundle.queries.shape[0]))
            if warmup_queries > 0:
                self.evaluate(
                    bundle,
                    state,
                    probe_depth=int(probe_depth),
                    diagnostic=False,
                    query_limit=warmup_queries,
                    compute_quality=False,
                )
            results: list[QueryResult] = []
            walls: list[float] = []
            cpu_times: list[float] = []
            for _ in range(TIMING_TRIALS):
                result = self.evaluate(
                    bundle,
                    state,
                    probe_depth=int(probe_depth),
                    diagnostic=False,
                    compute_quality=False,
                )
                if result.query_cpu_s is None or float(result.query_cpu_s) < 0.0:
                    raise RuntimeError("query benchmark did not report serving-only process CPU time")
                cpu_times.append(float(result.query_cpu_s))
                if result.qps is None or float(result.qps) <= 0.0:
                    raise RuntimeError("query benchmark produced unavailable throughput")
                walls.append(float(bundle.queries.shape[0]) / float(result.qps))
                results.append(result)
            summary = summarize_timing_trials(
                batch_size=int(batch_size),
                query_count=int(bundle.queries.shape[0]),
                warmup_queries=warmup_queries,
                wall_seconds=walls,
                process_cpu_seconds=cpu_times,
            )
            return results[summary.selected_trial], summary
        finally:
            self.cfg = original_cfg

    def _competitiveness_row(
        self,
        *,
        bundle: LearnedDataset,
        state: FitState,
        router_checkpoint_dir: Path | None = None,
        operating_point_id: str = "default",
        isolated_mem: dict[str, float] | None = None,
    ) -> dict[str, object]:
        _, batch256_timing = self._benchmark_query_mode(
            bundle=bundle,
            state=state,
            probe_depth=int(self.cfg.probe_depth),
            batch_size=BENCHMARK_BATCH_256,
        )
        _, batch1_timing = self._benchmark_query_mode(
            bundle=bundle,
            state=state,
            probe_depth=int(self.cfg.probe_depth),
            batch_size=BENCHMARK_BATCH_1,
        )
        # Trace construction is deliberately outside the timed serving passes.
        query_result = self.evaluate(bundle, state, probe_depth=int(self.cfg.probe_depth), diagnostic=True)
        manifest = json.loads((state.exp_dir / "index_manifest.json").read_text(encoding="utf-8"))
        requires_vectors = _requires_full_vectors(self.cfg)
        serving_payload = serving_payload_metrics(
            serialized_artifacts_mb=float(manifest["serving_footprint_mb"]),
            external_vector_payload_mb=(
                _base_vector_payload_mb(bundle) if requires_vectors else 0.0
            ),
            canonical_vector_payload_mb=_base_vector_payload_mb(bundle),
            requires_full_vectors=requires_vectors,
        )
        timing_metrics = timing_contract_metrics(batch1=batch1_timing, batch256=batch256_timing)
        work_equiv = float(query_result.avg_computations)
        query_mem_delta_mb = query_result.query_mem_delta_mb
        query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
        if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
            query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
            query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED
        row: dict[str, object] = {
            "dataset": self.cfg.dataset,
            "method": self.cfg.method,
            "seed": int(self.cfg.seed),
            "operating_point_id": operating_point_id,
            "probe_depth": int(self.cfg.probe_depth),
            "recall_at_1": query_result.recall_at_1,
            "recall_at_10": query_result.recall_at_10,
            "recall_at_100": query_result.recall_at_100,
            "avg_computations": query_result.avg_computations,
            "computation_min": query_result.computation_min,
            "computation_max": query_result.computation_max,
            "avg_computations_mode": (
                AVG_COMPUTATIONS_MODE_ROUTER
            ),
            "avg_candidates": query_result.avg_computations,
            "avg_candidates_mode": (
                AVG_COMPUTATIONS_MODE_ROUTER
            ),
            **timing_metrics,
            "qps_batch_1_contract": "router_end_to_end_batch_1",
            "qps_batch_256_contract": "router_end_to_end_batch_256",
            "qps_native_batched": float(batch256_timing.qps_median),
            "qps_native_batched_contract": "router_end_to_end_batch_256",
            "index_overhead_mb": float(manifest["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            **serving_payload,
            "index_size_mb": float(manifest["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": query_mem_delta_mb,
            "query_mem_delta_mode": query_mem_delta_mode,
            "query_mem_measurement_operating_point": "shared_index_factory_query_path",
            **_build_timing_summary(state.exp_dir),
            "dimension": int(bundle.index_train.shape[1]),
            "distance_scoring_equivalent_work": work_equiv,
            "distance_scoring_equivalent_mode": "analytical_distance_scoring_equivalent",
            "avg_probe_count": query_result.avg_probe_count,
            "native_backend": _NATIVE_BACKEND,
            "query_batch_size": BENCHMARK_BATCH_256,
            "router_checkpoint_dir": (
                str(Path(router_checkpoint_dir).expanduser().resolve())
                if router_checkpoint_dir is not None
                else None
            ),
            "experiment_dir": str(state.exp_dir),
        }
        canonical_point = operating_point_id in {"default", "probe-10"}
        point_dir = (
            state.exp_dir
            if canonical_point
            else ensure_dir(state.exp_dir / "operating_points" / operating_point_id)
        )
        if query_result.timing_breakdown is not None:
            row.update({f"timing_{key}": value for key, value in query_result.timing_breakdown.items()})
            write_json(
                point_dir / "timing_breakdown.json",
                {
                    "operating_point_id": operating_point_id,
                    "stage_breakdown": query_result.timing_breakdown,
                    "serving_contract": {
                        **timing_metrics,
                        "qps_batch_1_contract": row["qps_batch_1_contract"],
                        "qps_batch_256_contract": row["qps_batch_256_contract"],
                        "qps_native_batched_contract": row["qps_native_batched_contract"],
                    },
                },
            )
        candidate_budget = {
            "avg_computations": query_result.avg_computations,
            "computation_min": query_result.computation_min,
            "computation_max": query_result.computation_max,
            "exact_rerank_count": query_result.exact_rerank_count,
            "dimension": int(bundle.index_train.shape[1]),
            "distance_scoring_equivalent_work": work_equiv,
            "avg_probe_count": query_result.avg_probe_count,
            "operating_point_id": operating_point_id,
        }
        write_json(point_dir / "candidate_budget_breakdown.json", candidate_budget)
        row.update(
            {
                "exact_rerank_count": query_result.exact_rerank_count,
                "avg_probe_count": query_result.avg_probe_count,
            }
        )
        if query_result.per_query is not None:
            per_query_rows = [
                {"query_id": idx, **per_query}
                for idx, per_query in enumerate(query_result.per_query)
            ]
            write_csv(point_dir / "per_query_debug.csv", per_query_rows)
        if isolated_mem is not None:
            row.update(isolated_mem)
        print_table(
            f"{self.cfg.method} | competitiveness",
            ["metric", "value"],
            [
                ["Recall@10", f"{query_result.recall_at_10:.4f}"],
                ["Avg computations", f"{query_result.avg_computations:.1f}"],
                ["QPS (batch 256 median)", f"{float(timing_metrics['qps_batch_256']):.1f}"],
                ["Index overhead (MB)", f"{float(manifest['index_overhead_mb']):.3f}"],
                ["Serving payload (MB)", f"{float(serving_payload['total_serving_payload_mb']):.3f}"],
                ["Native backend", _NATIVE_BACKEND],
                [
                    "Rerank count",
                    f"{query_result.exact_rerank_count:.1f}" if query_result.exact_rerank_count is not None else "n/a",
                ],
            ],
        )
        return row

    def _load_and_fit_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool,
        router_checkpoint_dir: Path | None,
    ) -> tuple[LearnedDataset, FitState, dict[str, float] | None]:
        bundle = load_learned_dataset(
            self.cfg.dataset,
            data_root,
            max_samples=self.cfg.prepare_max_samples,
            k=self.cfg.prepare_k,
            seed=self.cfg.seed,
            force_prepare=force_prepare,
        )
        state = self.fit(
            bundle,
            exp_dir,
            data_root=data_root,
            **({"router_checkpoint_dir": router_checkpoint_dir} if router_checkpoint_dir is not None else {}),
        )
        isolated_mem = _isolated_query_memory(
            exp_dir=state.exp_dir,
            dataset=self.cfg.dataset,
            data_root=data_root,
            probe_depth=int(self.cfg.probe_depth),
        )
        return bundle, state, isolated_mem

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
        router_checkpoint_dir: Path | None = None,
    ) -> dict[str, object]:
        bundle, state, isolated_mem = self._load_and_fit_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
            router_checkpoint_dir=router_checkpoint_dir,
        )
        row = self._competitiveness_row(
            bundle=bundle,
            state=state,
            router_checkpoint_dir=router_checkpoint_dir,
            operating_point_id="default",
            isolated_mem=isolated_mem,
        )
        write_json(state.exp_dir / "competitiveness_summary.json", row)
        return row

    def run_competitiveness_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        operating_points: list[dict[str, object]],
        force_prepare: bool = False,
        router_checkpoint_dir: Path | None = None,
    ) -> list[dict[str, object]]:
        """Serve multiple query policies on one immutable trained index."""

        if not operating_points:
            raise ValueError("operating_points must not be empty")
        bundle, state, isolated_mem = self._load_and_fit_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
            router_checkpoint_dir=router_checkpoint_dir,
        )
        original_cfg = self.cfg
        rows: list[dict[str, object]] = []
        allowed = {"probe_depth"}
        try:
            for point in operating_points:
                point_id = str(point["operating_point_id"])
                overrides = {key: point[key] for key in allowed if key in point}
                unknown = set(point) - allowed - {"operating_point_id"}
                if unknown:
                    raise ValueError(f"Unsupported query-only overrides for {point_id}: {sorted(unknown)}")
                self.cfg = replace(original_cfg, **overrides)
                rows.append(
                    self._competitiveness_row(
                        bundle=bundle,
                        state=state,
                        router_checkpoint_dir=router_checkpoint_dir,
                        operating_point_id=point_id,
                        isolated_mem=isolated_mem,
                    )
                )
        finally:
            self.cfg = original_cfg
        write_json(
            state.exp_dir / "serving_operating_points.json",
            {"operating_points": operating_points, "shared_built_index": True},
        )
        write_json(state.exp_dir / "competitiveness_summary.json", rows)
        return rows

    def run_initialization_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        probes: list[int],
        force_prepare: bool = False,
    ) -> list[dict[str, object]]:
        bundle = load_learned_dataset(
            self.cfg.dataset,
            data_root,
            max_samples=self.cfg.prepare_max_samples,
            k=self.cfg.prepare_k,
            seed=self.cfg.seed,
            force_prepare=force_prepare,
        )
        state = self.fit(bundle, exp_dir, data_root=data_root)
        rows: list[dict[str, object]] = []
        sweep_payload: list[dict[str, object]] = []
        for probe_depth in probes:
            query_result = self.evaluate(bundle, state, probe_depth=probe_depth)
            manifest = json.loads((state.exp_dir / "index_manifest.json").read_text(encoding="utf-8"))
            requires_vectors = _requires_full_vectors(self.cfg)
            serving_payload = serving_payload_metrics(
                serialized_artifacts_mb=float(manifest["serving_footprint_mb"]),
                external_vector_payload_mb=(
                    _base_vector_payload_mb(bundle) if requires_vectors else 0.0
                ),
                canonical_vector_payload_mb=_base_vector_payload_mb(bundle),
                requires_full_vectors=requires_vectors,
            )
            isolated_mem = _isolated_query_memory(
                exp_dir=state.exp_dir,
                dataset=self.cfg.dataset,
                data_root=data_root,
                probe_depth=int(probe_depth),
            )
            query_mem_delta_mb = query_result.query_mem_delta_mb
            query_mem_delta_mode = QUERY_MEM_MODE_IN_PROCESS
            if isolated_mem is not None and "query_mem_delta_mb_isolated" in isolated_mem:
                query_mem_delta_mb = float(isolated_mem["query_mem_delta_mb_isolated"])
                query_mem_delta_mode = QUERY_MEM_MODE_ISOLATED
            row = {
                "dataset": self.cfg.dataset,
                "method": self.cfg.method,
                "seed": int(self.cfg.seed),
                "probe_depth": int(probe_depth),
                "recall_at_1": query_result.recall_at_1,
                "recall_at_10": query_result.recall_at_10,
                "recall_at_100": query_result.recall_at_100,
                "avg_computations": query_result.avg_computations,
                "avg_computations_mode": AVG_COMPUTATIONS_MODE_ROUTER,
                "avg_candidates": query_result.avg_computations,
                "avg_candidates_mode": AVG_COMPUTATIONS_MODE_ROUTER,
                "qps": query_result.qps,
                "index_overhead_mb": float(manifest["index_overhead_mb"]),
                "index_overhead_mode": INDEX_OVERHEAD_MODE,
                **serving_payload,
                "index_size_mb": float(manifest["index_overhead_mb"]),
                "index_size_mode": INDEX_OVERHEAD_MODE,
                "query_mem_delta_mb": query_mem_delta_mb,
                "query_mem_delta_mode": query_mem_delta_mode,
                **_build_timing_summary(state.exp_dir),
                "experiment_dir": str(state.exp_dir),
            }
            if isolated_mem is not None:
                row.update(isolated_mem)
            rows.append(row)
            sweep_payload.append(row)
            print_table(
                f"{self.cfg.method} @ m={probe_depth}",
                ["metric", "value"],
                [
                    ["Recall@10", f"{query_result.recall_at_10:.4f}"],
                    ["Avg computations", f"{query_result.avg_computations:.1f}"],
                    ["QPS", f"{query_result.qps:.1f}" if query_result.qps is not None else "nan"],
                    ["Index overhead (MB)", f"{float(manifest['index_overhead_mb']):.3f}"],
                    ["Serving payload (MB)", f"{float(serving_payload['total_serving_payload_mb']):.3f}"],
                ],
            )
        write_json(state.exp_dir / "query_sweep.json", {"rows": sweep_payload})
        return rows


def make_router_method(method: str, dataset: str, *, seed: int = 0) -> RouterIndex:
    """Build a router from a named, decision-complete method variant.

    Experiment runners should select behavior through explicit method names
    rather than generic runtime overrides. This keeps comparisons auditable and
    makes obsolete variants removable from one factory surface.
    """
    key = method.strip().upper()
    if key == "BLISS":
        cfg = RouterConfig(dataset=dataset, method="BLISS", init_mode="hash", seed=seed)
        return RouterIndex(cfg)
    if key in {"BLISS-KMEANS", "BLISS_KMEANS"}:
        cfg = RouterConfig(dataset=dataset, method="BLISS-KMeans", init_mode="kmeans", seed=seed)
        return RouterIndex(cfg)
    raise ValueError(f"Unsupported router method: {method}")
