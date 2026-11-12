from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import ensure_dir, write_csv, write_json
from console import banner, info
from datasets import parse_datasets
from method_registry import INITIALIZATION_METHODS, canonical_method_id, decorate_method_row, method_sort_key
from methods.bliss import Bliss, BlissKMeans
from run_contract import (
    INITIALIZATION_REQUIRED_ARTIFACTS,
    build_run_request,
    config_payload,
    experiment_cell_dir,
    mark_completed_cell,
    validate_completed_cell,
    write_run_request,
)


def _make_method(method: str, dataset: str, seed: int):
    method_id = canonical_method_id(method)
    if method_id == "BLISS":
        return Bliss(dataset, seed=seed)
    if method_id == "BLISS-KMeans":
        return BlissKMeans(dataset, seed=seed)
    raise ValueError(f"Unsupported initialization method: {method}")


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _merge_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, int, int], dict[str, object]] = {}
    replacement_scopes = {
        (
            str(row.get("dataset") or "").strip(),
            canonical_method_id(str(row.get("method_id") or row.get("method"))),
            int(float(row.get("seed") or 0)),
        )
        for row in new_rows
    }
    retained_existing = [
        row
        for row in existing
        if (
            str(row.get("dataset") or "").strip(),
            canonical_method_id(str(row.get("method_id") or row.get("method"))),
            int(float(row.get("seed") or 0)),
        )
        not in replacement_scopes
    ]
    for row in retained_existing + new_rows:
        method_id = canonical_method_id(str(row.get("method_id") or row.get("method")))
        key = (
            str(row.get("dataset") or "").strip(),
            method_id,
            int(float(row.get("probe_depth") or 0)),
            int(float(row.get("seed") or 0)),
        )
        merged[key] = {**row, "dataset": key[0], "method": method_id, "method_id": method_id,
                       "probe_depth": key[2], "seed": key[3]}
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row["dataset"]), method_sort_key(str(row["method_id"])),
            int(row["probe_depth"]), int(row["seed"]),
        ),
    )


def _read_sweep(exp_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((exp_dir / "query_sweep.json").read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise RuntimeError(f"Malformed query sweep: {exp_dir / 'query_sweep.json'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible initialization study.")
    parser.add_argument("--datasets", default="glove10k")
    parser.add_argument("--methods", default=",".join(INITIALIZATION_METHODS))
    parser.add_argument("--probes", default="5,10,20,40")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exports-root", default="exports/final")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-validated cells.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    datasets = parse_datasets([args.datasets])
    methods = [canonical_method_id(token) for token in args.methods.split(",") if token.strip()]
    methods.sort(key=method_sort_key)
    probes = [int(token) for token in args.probes.split(",") if token.strip()]
    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    data_root = Path(args.data_root).expanduser().resolve()
    exp_root = Path(args.experiments_root).expanduser().resolve()
    export_dir = ensure_dir(Path(args.exports_root).expanduser().resolve() / "initialization")

    all_new_rows: list[dict[str, object]] = []
    for dataset in datasets:
        banner("dataset", f"{dataset} | initialization")
        dataset_json = export_dir / f"{dataset}_summary.json"
        rows: list[dict[str, object]] = []
        for seed in seeds:
            for method_id in methods:
                method = _make_method(method_id, dataset, seed)
                requested_config = config_payload(method)
                requested_config["initialization_probe_depths"] = probes
                request = build_run_request(
                    repo_root=repo_root,
                    data_root=data_root,
                    dataset=dataset,
                    method_id=method_id,
                    requested_method=method_id,
                    seed=seed,
                    config=requested_config,
                    study="initialization",
                    operating_point_id="probe-sweep",
                )
                exp_dir = experiment_cell_dir(exp_root, request)
                valid, reason = validate_completed_cell(
                    exp_dir, request, required_artifacts=INITIALIZATION_REQUIRED_ARTIFACTS
                )
                resumed = bool(args.resume and valid)
                if resumed:
                    info(f"resume: {dataset}/{method_id}/seed{seed} -> {exp_dir.name} ({reason})")
                    raw_rows = _read_sweep(exp_dir)
                else:
                    if (exp_dir / "RUN_DONE.json").exists() and not valid:
                        raise RuntimeError(f"Stale or corrupt completed cell {exp_dir}: {reason}")
                    write_run_request(exp_dir, request)
                    info(f"running {method_id} seed={seed} -> {exp_dir}")
                    raw_rows = method.run_initialization_sweep(
                        data_root=data_root,
                        exp_dir=exp_dir,
                        probes=probes,
                        force_prepare=args.force_prepare,
                    )
                completed_rows: list[dict[str, object]] = []
                for raw in raw_rows:
                    row = decorate_method_row(raw, requested_method=method_id)
                    row["request_hash"] = request["request_hash"]
                    row["experiment_dir"] = str(exp_dir)
                    completed_rows.append(row)
                if not resumed:
                    write_json(exp_dir / "query_sweep.json", {"rows": completed_rows})
                    mark_completed_cell(
                        exp_dir,
                        request,
                        completed_rows,
                        required_artifacts=INITIALIZATION_REQUIRED_ARTIFACTS,
                    )
                rows.extend(completed_rows)
                all_new_rows.extend(completed_rows)
        merged_rows = _merge_rows(_load_rows(dataset_json), rows)
        write_csv(dataset_json.with_suffix(".csv"), merged_rows)
        write_json(dataset_json, merged_rows)

    all_json = export_dir / "all_datasets_summary.json"
    merged_all = _merge_rows(_load_rows(all_json), all_new_rows)
    write_csv(all_json.with_suffix(".csv"), merged_all)
    write_json(all_json, merged_all)
    info(f"all-datasets summary -> {all_json.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
