from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple


_TOTAL_STAGING_INDEX_FILES = {
    "faiss.index",
    "faiss_meta.json",
    "ivf_list_ids_index.npy",
    "ivf_list_ids.npy",
    "point_to_buckets.npy",
    "assignments.npy",
    "lira_selected_ids.npy",
    "lira_selected_offsets.npy",
    "lira_centroids.npy",
    "train.npy",
    "index.npy",
}

_SERVING_STAGING_FILES = {
    "faiss.index",
    "faiss_meta.json",
    "point_to_buckets.npy",
    "lira_selected_ids.npy",
    "lira_selected_offsets.npy",
    "lira_centroids.npy",
    "train.npy",
    "index.npy",
}

_SHARED_VECTOR_FILES = {
    "train.npy",
    "index.npy",
}

_COMPONENT_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("models",), "models"),
    (("staging", "lookups"), "lookups"),
    (("staging", "lira_inner_indexes"), "lira_inner_indexes"),
    (("data",), "lira_data"),
)


def _classify_path(rel_parts: Sequence[str]) -> Tuple[str | None, str | None]:
    if not rel_parts:
        return None, None
    total_component: str | None = None
    serving_component: str | None = None
    for prefix, name in _COMPONENT_PREFIXES:
        if len(rel_parts) >= len(prefix) and tuple(rel_parts[: len(prefix)]) == prefix:
            total_component = name
            if name != "lira_data":
                serving_component = name
            break
    if rel_parts[0] == "staging" and rel_parts[-1] in _TOTAL_STAGING_INDEX_FILES:
        total_component = total_component or "staging_index"
    if rel_parts[0] == "staging" and rel_parts[-1] in _SERVING_STAGING_FILES:
        serving_component = serving_component or "serving_staging"
    return total_component, serving_component


def _iter_index_files(experiment_dir: Path) -> Iterable[tuple[str | None, str | None, str, int, bool]]:
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        return
    for path in experiment_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(experiment_dir)
        total_component, serving_component = _classify_path(rel.parts)
        if total_component is None and serving_component is None:
            continue
        size = int(path.stat().st_size)
        is_shared_vector = rel.parts[0] == "staging" and rel.parts[-1] in _SHARED_VECTOR_FILES
        yield total_component, serving_component, str(rel), size, is_shared_vector


def build_index_manifest(experiment_dir: Path) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    serving_components: dict[str, dict[str, Any]] = {}
    index_overhead_components: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    total_bytes = 0
    serving_bytes = 0
    shared_vector_bytes = 0
    index_overhead_bytes = 0
    for total_component, serving_component, rel_path, size, is_shared_vector in _iter_index_files(experiment_dir):
        if total_component is not None:
            total_bytes += size
            bucket = components.setdefault(total_component, {"bytes": 0, "files": 0})
            bucket["bytes"] = int(bucket["bytes"]) + size
            bucket["files"] = int(bucket["files"]) + 1
        if serving_component is not None:
            serving_bytes += size
            bucket = serving_components.setdefault(serving_component, {"bytes": 0, "files": 0})
            bucket["bytes"] = int(bucket["bytes"]) + size
            bucket["files"] = int(bucket["files"]) + 1
            if not is_shared_vector:
                index_overhead_bytes += size
                overhead_bucket = index_overhead_components.setdefault(serving_component, {"bytes": 0, "files": 0})
                overhead_bucket["bytes"] = int(overhead_bucket["bytes"]) + size
                overhead_bucket["files"] = int(overhead_bucket["files"]) + 1
        if is_shared_vector:
            shared_vector_bytes += size
        files.append(
            {
                "path": rel_path,
                "component": total_component,
                "serving_component": serving_component,
                "is_shared_vector_payload": bool(is_shared_vector),
                "bytes": size,
                "size_mb": size / (1024.0 * 1024.0),
            }
        )
    component_rows = [
        {
            "component": name,
            "bytes": int(meta["bytes"]),
            "size_mb": float(meta["bytes"]) / (1024.0 * 1024.0),
            "files": int(meta["files"]),
        }
        for name, meta in sorted(components.items())
    ]
    serving_component_rows = [
        {
            "component": name,
            "bytes": int(meta["bytes"]),
            "size_mb": float(meta["bytes"]) / (1024.0 * 1024.0),
            "files": int(meta["files"]),
        }
        for name, meta in sorted(serving_components.items())
    ]
    index_overhead_component_rows = [
        {
            "component": name,
            "bytes": int(meta["bytes"]),
            "size_mb": float(meta["bytes"]) / (1024.0 * 1024.0),
            "files": int(meta["files"]),
        }
        for name, meta in sorted(index_overhead_components.items())
    ]
    return {
        "schema_version": "1.1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment_dir),
        "total_index_size_bytes": total_bytes,
        "total_index_size_mb": total_bytes / (1024.0 * 1024.0),
        "serving_footprint_bytes": serving_bytes,
        "serving_footprint_mb": serving_bytes / (1024.0 * 1024.0),
        "shared_vector_payload_bytes": shared_vector_bytes,
        "shared_vector_payload_mb": shared_vector_bytes / (1024.0 * 1024.0),
        "index_overhead_bytes": index_overhead_bytes,
        "index_overhead_mb": index_overhead_bytes / (1024.0 * 1024.0),
        "components": component_rows,
        "serving_components": serving_component_rows,
        "index_overhead_components": index_overhead_component_rows,
        "files": sorted(files, key=lambda row: str(row["path"])),
    }


def write_index_manifest(experiment_dir: Path) -> Path:
    manifest = build_index_manifest(experiment_dir)
    path = experiment_dir / "index_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
