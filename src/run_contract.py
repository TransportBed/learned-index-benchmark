from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
from typing import Any

from artifacts import ensure_dir, write_json


RUN_SCHEMA_VERSION = 3
COMPETITIVENESS_REQUIRED_ARTIFACTS = (
    "config.json",
    "index_manifest.json",
    "competitiveness_summary.json",
    "timing_breakdown.json",
    "candidate_budget_breakdown.json",
)
INITIALIZATION_REQUIRED_ARTIFACTS = (
    "config.json",
    "train_metrics.json",
    "index_manifest.json",
    "query_sweep.json",
)

_RUNTIME_SOURCE_FILES = (
    "CMakeLists.txt",
    "cpp/CMakeLists.txt",
    "pyproject.toml",
    "bindings/module.cpp",
    "src/ann_ops.py",
    "src/artifacts.py",
    "src/cli_run_competitiveness.py",
    "src/cli_run_router_methods.py",
    "src/datasets.py",
    "src/index_manifest.py",
    "src/measurement_contract.py",
    "src/method_registry.py",
    "src/metrics.py",
    "src/presets.py",
    "src/run_contract.py",
    "src/run_faiss_query_isolated.py",
    "src/run_lira_query_isolated.py",
    "src/run_router_query_isolated.py",
)
_RUNTIME_SOURCE_DIRS = (
    "cpp/include",
    "cpp/src",
    "src/methods",
)
_RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".cpp", ".hpp", ".h", ".cc", ".cxx"})
FROZEN_RUNTIME_FINGERPRINT_ENV = "LISBENCH_RUNTIME_FINGERPRINT"
FROZEN_RUNTIME_ENVIRONMENT_ENV = "LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_file_identity(path: Path, repo_root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def runtime_source_manifest(repo_root: Path) -> dict[str, object]:
    """Fingerprint only source that can change numerical run outputs.

    Presentation code, documentation, logs, exports, and unrelated untracked
    files are deliberately outside this contract. Runtime changes remain
    fail-closed, while plot/table edits cannot invalidate completed cells.
    """

    root = repo_root.expanduser().resolve()
    paths: set[Path] = set()
    missing: list[str] = []
    for relative in _RUNTIME_SOURCE_FILES:
        path = root / relative
        if path.is_file():
            paths.add(path)
        else:
            missing.append(relative)
    for relative in _RUNTIME_SOURCE_DIRS:
        directory = root / relative
        if not directory.is_dir():
            missing.append(relative)
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in _RUNTIME_SOURCE_SUFFIXES:
                paths.add(path)
    if missing:
        raise FileNotFoundError(f"Runtime source contract is incomplete: {', '.join(missing)}")
    files = [_source_file_identity(path, root) for path in sorted(paths)]
    identity = {"schema_version": 1, "files": files}
    return {**identity, "fingerprint": stable_hash(identity)}


def _cpu_model() -> str | None:
    """Return the stable CPU model string used by the timing contract."""

    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        return None
    return None


def runtime_environment_identity() -> dict[str, object]:
    """Identity of numerical libraries, hardware, and execution policy."""

    packages = ("numpy", "faiss-cpu", "tensorflow", "torch", "scikit-learn", "numba")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    execution_keys = (
        "LISBENCH_SERVING_THREADS",
        "LISBENCH_TRAINING_THREADS",
        "LISBENCH_NATIVE_ARCH",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_WAIT_POLICY",
        "GOMP_SPINCOUNT",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "CUDA_VISIBLE_DEVICES",
    )
    execution = {key: os.environ.get(key) for key in execution_keys}
    identity = {
        "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": versions,
        "execution_contract": execution,
        "hardware": {
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
        },
    }
    return {**identity, "fingerprint": stable_hash(identity)}


def assert_frozen_runtime_source(manifest: dict[str, object]) -> None:
    frozen = os.environ.get(FROZEN_RUNTIME_FINGERPRINT_ENV)
    current = str(manifest["fingerprint"])
    if frozen is not None and frozen != current:
        raise RuntimeError(
            "Numerical runtime source changed after campaign launch: "
            f"frozen={frozen}, current={current}. Presentation-only changes do not affect this check."
        )


def assert_frozen_runtime_environment(identity: dict[str, object]) -> None:
    frozen = os.environ.get(FROZEN_RUNTIME_ENVIRONMENT_ENV)
    current = str(identity["fingerprint"])
    if frozen is not None and frozen != current:
        raise RuntimeError(
            "Numerical runtime environment changed after campaign launch: "
            f"frozen={frozen}, current={current}."
        )


def git_state(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()

    try:
        sha = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        status = run("status", "--porcelain")
        diff_payload = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repo_root
        )
        diff_hash = hashlib.sha256(diff_payload).hexdigest() if status else None
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_branch": None, "git_dirty": None, "git_diff_sha256": None}
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_diff_sha256": diff_hash,
        "git_status": status.splitlines(),
    }


def _sampled_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": digest.hexdigest(),
    }


def dataset_fingerprint(dataset: str, data_root: Path) -> dict[str, object]:
    """Describe prepared inputs and return a content-stable identity.

    Modification times remain in the manifest for provenance, but they are
    intentionally excluded from the identity. Dataset preparation may rewrite
    byte-identical query and ground-truth arrays when changing router-training
    seeds; treating those metadata-only rewrites as data drift makes valid
    checkpoints impossible to reuse across a multi-seed campaign.
    """
    base = data_root.expanduser().resolve() / dataset
    candidates = (
        base / "train.npy",
        base / "neighbors100.npy",
        base / "queries.npy",
    )
    missing = [path.name for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared dataset {dataset} is incomplete under {base}; missing: {', '.join(missing)}"
        )
    files = [_sampled_file_fingerprint(path) for path in candidates]
    identity = {
        "dataset": dataset,
        "files": [
            {
                "name": Path(str(file_info["path"])).name,
                "size_bytes": file_info["size_bytes"],
                "sample_sha256": file_info["sample_sha256"],
            }
            for file_info in files
        ],
    }
    return {
        "dataset": dataset,
        "files": files,
        "identity": identity,
        "fingerprint": stable_hash(identity),
    }


def config_payload(method: object) -> dict[str, object]:
    cfg = getattr(method, "cfg", None)
    if cfg is None and hasattr(method, "_impl"):
        cfg = getattr(method._impl, "cfg", None)  # type: ignore[attr-defined]
    if cfg is None and hasattr(method, "_variant"):
        variant = getattr(method, "_variant")
        cfg = getattr(getattr(variant, "_impl", None), "cfg", None)
    if cfg is None:
        raise TypeError(f"Cannot inspect configuration for {type(method).__name__}")
    value = _jsonable(cfg)
    if not isinstance(value, dict):
        raise TypeError("Method configuration must serialize to an object")
    return value


def build_run_request(
    *,
    repo_root: Path,
    data_root: Path,
    dataset: str,
    method_id: str,
    requested_method: str,
    seed: int,
    config: dict[str, object],
    study: str,
    operating_point_id: str = "default",
) -> dict[str, object]:
    git = git_state(repo_root)
    source_manifest = runtime_source_manifest(repo_root)
    assert_frozen_runtime_source(source_manifest)
    environment = runtime_environment_identity()
    assert_frozen_runtime_environment(environment)
    dataset_info = dataset_fingerprint(dataset, data_root)
    core = {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_fingerprint": dataset_info["fingerprint"],
        "method_id": method_id,
        "requested_method": requested_method,
        "seed": int(seed),
        "study": study,
        "operating_point_id": operating_point_id,
        "config": config,
        "runtime_source_fingerprint": source_manifest["fingerprint"],
        "runtime_environment_fingerprint": environment["fingerprint"],
    }
    return {
        **core,
        "request_hash": stable_hash(core),
        "created_at": utc_now(),
        "git": git,
        "dataset_manifest": dataset_info,
        "runtime_source_manifest": source_manifest,
        "runtime_environment": environment,
    }


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def experiment_cell_dir(root: Path, request: dict[str, object]) -> Path:
    return ensure_dir(
        root
        / str(request["dataset"])
        / f"{_token(str(request['method_id']))}__seed{int(request['seed'])}"
        / f"{_token(str(request['operating_point_id']))}__{str(request['request_hash'])[:12]}"
    )


def write_run_request(exp_dir: Path, request: dict[str, object]) -> None:
    existing_path = exp_dir / "run_request.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("request_hash") != request.get("request_hash"):
            raise RuntimeError(f"Refusing mixed configurations in {exp_dir}")
    write_json(existing_path, request)


def validate_completed_cell(
    exp_dir: Path,
    request: dict[str, object],
    *,
    required_artifacts: tuple[str, ...] = COMPETITIVENESS_REQUIRED_ARTIFACTS,
) -> tuple[bool, str]:
    marker_path = exp_dir / "RUN_DONE.json"
    request_path = exp_dir / "run_request.json"
    if not marker_path.exists() or not request_path.exists():
        return False, "completion marker or request missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        stored = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid marker: {exc}"
    expected_hash = request.get("request_hash")
    if marker.get("request_hash") != expected_hash or stored.get("request_hash") != expected_hash:
        return False, "request hash mismatch"
    marker_required = tuple(str(name) for name in marker.get("required_artifacts", ()))
    if marker_required and marker_required != required_artifacts:
        return False, "required-artifact contract mismatch"
    missing = [name for name in required_artifacts if not (exp_dir / name).exists()]
    if missing:
        return False, f"missing required artifacts: {', '.join(missing)}"
    return True, "validated"


def mark_completed_cell(
    exp_dir: Path,
    request: dict[str, object],
    result: object,
    *,
    required_artifacts: tuple[str, ...] = COMPETITIVENESS_REQUIRED_ARTIFACTS,
) -> None:
    missing = [name for name in required_artifacts if not (exp_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Cannot complete {exp_dir}; missing: {', '.join(missing)}")
    write_json(
        exp_dir / "RUN_DONE.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "request_hash": request["request_hash"],
            "completed_at": utc_now(),
            "result_hash": stable_hash(result),
            "required_artifacts": list(required_artifacts),
        },
    )


def package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "faiss-cpu", "tensorflow", "torch", "scikit-learn")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def machine_provenance(repo_root: Path) -> dict[str, object]:
    thread_keys = (
        "LISBENCH_SERVING_THREADS",
        "LISBENCH_TRAINING_THREADS",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "MALLOC_ARENA_MAX",
        "OMP_WAIT_POLICY",
        "GOMP_SPINCOUNT",
        "CUDA_VISIBLE_DEVICES",
    )
    cpu_model = None
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    memory_mb = None
    mem_info = Path("/proc/meminfo")
    if mem_info.exists():
        for line in mem_info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_mb = float(line.split()[1]) / 1024.0
                break
    return {
        "created_at": utc_now(),
        "git": git_state(repo_root),
        "runtime_source": runtime_source_manifest(repo_root),
        "runtime_environment": runtime_environment_identity(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "memory_total_mb": memory_mb,
        "process_address_space_limit": resource.getrlimit(resource.RLIMIT_AS),
        "hardware_policy": "cpu_only",
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
        "package_versions": package_versions(),
        "command": sys.argv,
    }


def assert_cpu_only_hardware() -> dict[str, object]:
    """Fail closed when the final CPU-only campaign can see an accelerator."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"", "-1"}:
        raise RuntimeError(
            "CPU-only campaign requires CUDA_VISIBLE_DEVICES='' (or -1); "
            f"found {visible!r}"
        )
    if os.environ.get("OMP_WAIT_POLICY") != "PASSIVE" or os.environ.get("GOMP_SPINCOUNT") != "0":
        raise RuntimeError(
            "Final measurement policy requires OMP_WAIT_POLICY=PASSIVE and GOMP_SPINCOUNT=0"
        )
    torch_cuda = False
    tensorflow_gpus: list[str] = []
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tensorflow_gpus = [str(device) for device in tf.config.list_physical_devices("GPU")]
    except ImportError:
        pass
    if torch_cuda or tensorflow_gpus:
        raise RuntimeError(
            "CPU-only campaign detected an accelerator: "
            f"torch_cuda={torch_cuda}, tensorflow_gpus={tensorflow_gpus}"
        )
    return {
        "hardware_policy": "cpu_only",
        "cuda_visible_devices": visible,
        "torch_cuda_available": torch_cuda,
        "tensorflow_gpus": tensorflow_gpus,
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
        "gomp_spincount": os.environ.get("GOMP_SPINCOUNT"),
    }
