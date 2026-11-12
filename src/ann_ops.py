from __future__ import annotations


import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def maybe_set_threads(num_threads: int) -> None:
    # Deliberately does not touch os.environ["OMP_NUM_THREADS"]. OpenMP reads
    # that variable when the runtime initialises, which has already happened by
    # the time any caller reaches here, so assigning it cannot change the thread
    # count of this process -- only the calls below do. It is also actively
    # harmful: the variable is part of the frozen runtime environment contract
    # (run_contract.runtime_environment_identity), and because
    # cli_run_competitiveness processes several datasets in one process, a
    # mutation here leaks into the next dataset's build_run_request() and aborts
    # the campaign on a fingerprint mismatch.
    if num_threads <= 0:
        return
    if faiss is not None:
        try:
            faiss.omp_set_num_threads(int(num_threads))
        except Exception:
            pass
    if torch is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass


def distance_to_centroids(vecs: np.ndarray, centroids: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    n = vecs.shape[0]
    c_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T
    out = np.empty((n, centroids.shape[0]), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = vecs[start:end]
        x_norm = np.sum(x * x, axis=1, keepdims=True)
        d2 = x_norm + c_norm - 2.0 * (x @ centroids.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:end] = np.sqrt(d2, dtype=np.float32)
    return out


def scaled_centroid_distances_with_scaler(
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
    infer_vecs: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dist = distance_to_centroids(train_vecs, centroids)
    query_dist = distance_to_centroids(query_vecs, centroids)
    infer_dist = distance_to_centroids(infer_vecs, centroids)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_dist).astype(np.float32, copy=False)
    query_scaled = scaler.transform(query_dist).astype(np.float32, copy=False)
    infer_scaled = scaler.transform(infer_dist).astype(np.float32, copy=False)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    return train_scaled, query_scaled, infer_scaled, mean, scale
