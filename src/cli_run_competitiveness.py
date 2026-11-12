from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import ensure_dir, write_csv, write_json
from console import banner, info
from datasets import parse_datasets
from method_registry import (
    COMPETITIVENESS_METHODS,
    METHODS,
    canonical_method_id,
    decorate_method_row,
    method_sort_key,
)
from methods.bliss import Bliss
from methods.faiss_baselines import FaissConfig, Hnsw, Ivf, IvfPQ, make_method as make_faiss_method
from methods.lira import Lira
from run_contract import (
    build_run_request,
    config_payload,
    experiment_cell_dir,
    mark_completed_cell,
    validate_completed_cell,
    write_run_request,
)


_TRAINED_ROUTER_SOURCES = {
    "BLISS": "BLISS",
}
_CLASSICAL_SWEEPS = {
    "HNSW": (32, 64, 128, 256),
    "IVF": (10, 20, 50, 100),
    "IVFPQ+R": (10, 20, 50, 100),
}
_ROUTER_PROBE_SWEEP = tuple(
    {"operating_point_id": f"probe-{probe}", "probe_depth": probe}
    for probe in (5, 10, 20, 40)
)
_QUERY_SWEEP_METHODS = {"BLISS", "LIRA"}


def _query_operating_points(method_id: str) -> tuple[dict[str, object], ...]:
    if method_id == "BLISS":
        return _ROUTER_PROBE_SWEEP
    if method_id == "LIRA":
        return (
            {"operating_point_id": "threshold-economy"},
            {"operating_point_id": "threshold-selected"},
            {"operating_point_id": "threshold-recall"},
        )
    return ()


def _make_method(method: str, dataset: str, *, seed: int):
    method_id = canonical_method_id(method)
    if method_id == "HNSW":
        return Hnsw(FaissConfig(dataset=dataset, index_seed=seed))
    if method_id == "IVF":
        return Ivf(FaissConfig(dataset=dataset, index_seed=seed))
    if method_id == "IVFPQ+R":
        return make_faiss_method(method_id, dataset, seed=seed)
    if method_id == "BLISS":
        return Bliss(dataset, seed=seed)
    if method_id == "LIRA":
        return Lira(dataset, seed=seed)
    raise ValueError(f"Unsupported competitiveness method: {method}")


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _norm_int(value: object) -> int:
    if value is None or not str(value).strip():
        return 0
    return int(float(str(value).strip()))


def _merge_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    replacement_scopes = {
        (
            _norm_text(row.get("dataset")),
            _norm_text(row.get("method_id") or row.get("method")),
            _norm_int(row.get("seed")),
        )
        for row in new_rows
    }
    retained_existing = [
        row
        for row in existing
        if (
            _norm_text(row.get("dataset")),
            _norm_text(row.get("method_id") or row.get("method")),
            _norm_int(row.get("seed")),
        )
        not in replacement_scopes
    ]
    for row in retained_existing + new_rows:
        method_raw = _norm_text(row.get("method_id") or row.get("method"))
        try:
            method_id = canonical_method_id(method_raw)
        except ValueError:
            method_id = method_raw
        operating_point_id = _norm_text(row.get("operating_point_id")) or "default"
        batch_size = _norm_int(row.get("query_batch_size") or row.get("batch_size"))
        key = (
            _norm_text(row.get("dataset")),
            method_id,
            _norm_int(row.get("seed")),
            operating_point_id,
            batch_size,
        )
        normalized = dict(row)
        normalized["dataset"] = key[0]
        normalized["method"] = method_id
        normalized["method_id"] = method_id
        normalized["seed"] = key[2]
        normalized["operating_point_id"] = operating_point_id
        merged[key] = normalized
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("dataset", "")),
            method_sort_key(str(row.get("method_id") or row.get("method") or "")),
            int(row.get("seed", 0) or 0),
            str(row.get("operating_point_id", "default")),
            int(row.get("query_batch_size", 0) or 0),
        ),
    )


def _checkpoint_from_rows(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    seed: int,
    method_id: str,
) -> Path | None:
    for row in rows:
        if (
            _norm_text(row.get("dataset")) == dataset
            and _norm_text(row.get("method_id") or row.get("method")) == method_id
            and _norm_int(row.get("seed")) == seed
        ):
            path = Path(_norm_text(row.get("experiment_dir"))).expanduser()
            if path.exists() and (path / "RUN_DONE.json").exists():
                return path.resolve()
    return None


def _validate_checkpoint_compatibility(
    checkpoint: Path,
    target_request: dict[str, object],
) -> None:
    """Reject a structurally valid checkpoint from another code/data state."""

    source_request_path = checkpoint / "run_request.json"
    marker_path = checkpoint / "RUN_DONE.json"
    if not source_request_path.exists() or not marker_path.exists():
        raise RuntimeError(f"Checkpoint is incomplete: {checkpoint}")
    source = json.loads(source_request_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("request_hash") != source.get("request_hash"):
        raise RuntimeError(f"Checkpoint completion marker does not match its request: {checkpoint}")
    comparisons = {
        "dataset": (source.get("dataset"), target_request.get("dataset")),
        "seed": (source.get("seed"), target_request.get("seed")),
        "dataset_fingerprint": (
            source.get("dataset_fingerprint"),
            target_request.get("dataset_fingerprint"),
        ),
        "runtime_source_fingerprint": (
            source.get("runtime_source_fingerprint"),
            target_request.get("runtime_source_fingerprint"),
        ),
        "runtime_environment_fingerprint": (
            source.get("runtime_environment_fingerprint"),
            target_request.get("runtime_environment_fingerprint"),
        ),
    }
    mismatches = [key for key, (left, right) in comparisons.items() if left != right]
    if mismatches:
        raise RuntimeError(
            f"Checkpoint {checkpoint} is incompatible with the target request: {mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible LisBench paper competitiveness matrix.")
    parser.add_argument("--datasets", default="glove10k")
    parser.add_argument("--methods", default=",".join(COMPETITIVENESS_METHODS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exports-root", default="paper_exports")
    parser.add_argument("--export-subdir", default="competitiveness")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-validated completed cells.")
    parser.add_argument(
        "--router-checkpoint-summary",
        help=(
            "Optional all_datasets_summary.json containing completed router checkpoints. "
            "Use an initialization "
            "summary for learned controls."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    datasets = parse_datasets([args.datasets])
    requested_methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    requested_methods.sort(key=method_sort_key)
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    exp_root = Path(args.experiments_root).expanduser().resolve()
    export_dir = ensure_dir(Path(args.exports_root).expanduser().resolve() / args.export_subdir)
    checkpoint_rows: list[dict[str, object]] = []
    if args.router_checkpoint_summary:
        checkpoint_summary = Path(args.router_checkpoint_summary).expanduser().resolve()
        checkpoint_rows = _load_rows(checkpoint_summary)
        if not checkpoint_rows:
            raise RuntimeError(f"No checkpoint rows found in {checkpoint_summary}")

    all_new_rows: list[dict[str, object]] = []
    for dataset in datasets:
        banner("dataset", f"{dataset} | {args.export_subdir}")
        dataset_json = export_dir / f"{dataset}_summary.json"
        existing_rows = _load_rows(dataset_json)
        rows: list[dict[str, object]] = []
        train = queries = neighbors = None
        for seed in seeds:
            for requested_method in requested_methods:
                method_id = canonical_method_id(requested_method)
                method = _make_method(requested_method, dataset, seed=seed)
                requested_config = config_payload(method)
                if method_id in _CLASSICAL_SWEEPS:
                    requested_config["operating_points"] = list(_CLASSICAL_SWEEPS[method_id])
                elif method_id in _QUERY_SWEEP_METHODS:
                    requested_config["query_operating_points"] = list(_query_operating_points(method_id))
                request = build_run_request(
                    repo_root=repo_root,
                    data_root=data_root,
                    dataset=dataset,
                    method_id=method_id,
                    requested_method=requested_method,
                    seed=seed,
                    config=requested_config,
                    study=args.export_subdir,
                    operating_point_id=(
                        "frontier"
                        if method_id in _CLASSICAL_SWEEPS or method_id in _QUERY_SWEEP_METHODS
                        else "default"
                    ),
                )
                exp_dir = experiment_cell_dir(exp_root, request)
                valid, reason = validate_completed_cell(exp_dir, request)
                if args.resume and valid:
                    stored = json.loads((exp_dir / "competitiveness_summary.json").read_text(encoding="utf-8"))
                    resumed_rows = stored if isinstance(stored, list) else [stored]
                    info(f"resume: {dataset}/{method_id}/seed{seed} -> {exp_dir.name} ({reason})")
                    rows.extend(resumed_rows)
                    all_new_rows.extend(resumed_rows)
                    continue
                if (exp_dir / "RUN_DONE.json").exists() and not valid:
                    raise RuntimeError(f"Stale or corrupt completed cell {exp_dir}: {reason}")
                write_run_request(exp_dir, request)
                info(f"running {method_id} seed={seed} -> {exp_dir}")
                if hasattr(method, "run"):
                    if train is None or queries is None or neighbors is None:
                        from datasets import load_search_dataset

                        train, queries, neighbors = load_search_dataset(dataset, data_root)
                    if method_id in _CLASSICAL_SWEEPS:
                        raw_rows = method.run_sweep(  # type: ignore[attr-defined]
                            train,
                            queries,
                            neighbors,
                            exp_dir,
                            data_root=data_root,
                            operating_points=_CLASSICAL_SWEEPS[method_id],
                        )
                    else:
                        raw_rows = [method.run(train, queries, neighbors, exp_dir, data_root=data_root)]  # type: ignore[attr-defined]
                else:
                    kwargs: dict[str, object] = {
                        "data_root": data_root,
                        "exp_dir": exp_dir,
                        "force_prepare": args.force_prepare,
                    }
                    if method_id in _TRAINED_ROUTER_SOURCES and checkpoint_rows:
                        source_method = _TRAINED_ROUTER_SOURCES[method_id]
                        checkpoint = _checkpoint_from_rows(
                            checkpoint_rows,
                            dataset=dataset,
                            seed=seed,
                            method_id=source_method,
                        )
                        if checkpoint is None:
                            raise RuntimeError(
                                f"{method_id} requires the completed {source_method} initialization "
                                f"checkpoint for {dataset}/seed{seed}."
                            )
                        _validate_checkpoint_compatibility(checkpoint, request)
                        kwargs["router_checkpoint_dir"] = checkpoint
                    if method_id == "LIRA":
                        raw_rows = method.run_competitiveness_sweep(**kwargs)  # type: ignore[attr-defined]
                    elif method_id == "BLISS":
                        raw_rows = method.run_competitiveness_sweep(  # type: ignore[attr-defined]
                            **kwargs,
                            operating_points=list(_query_operating_points(method_id)),
                        )
                    else:
                        raw_rows = [method.run_competitiveness(**kwargs)]  # type: ignore[attr-defined]
                completed_rows: list[dict[str, object]] = []
                for raw_row in raw_rows:
                    row = decorate_method_row(raw_row, requested_method=requested_method)
                    row["operating_point_id"] = (
                        str(row.get("operating_point_id") or "default")
                    )
                    row["request_hash"] = request["request_hash"]
                    row["experiment_dir"] = str(exp_dir)
                    completed_rows.append(row)
                write_json(
                    exp_dir / "competitiveness_summary.json",
                    completed_rows if len(completed_rows) > 1 else completed_rows[0],
                )
                mark_completed_cell(exp_dir, request, completed_rows)
                rows.extend(completed_rows)
                all_new_rows.extend(completed_rows)

        merged_rows = _merge_rows(existing_rows, rows)
        write_csv(export_dir / f"{dataset}_summary.csv", merged_rows)
        write_json(dataset_json, merged_rows)
        info(f"dataset summary -> {export_dir / f'{dataset}_summary.csv'}")

    all_json = export_dir / "all_datasets_summary.json"
    merged_all = _merge_rows(_load_rows(all_json), all_new_rows)
    write_csv(export_dir / "all_datasets_summary.csv", merged_all)
    write_json(all_json, merged_all)
    info(f"all-datasets summary -> {export_dir / 'all_datasets_summary.csv'}")


if __name__ == "__main__":
    main()
