from __future__ import annotations

import argparse
import json
import math
import resource
from dataclasses import fields, replace
from pathlib import Path

from datasets import load_learned_dataset
from methods.routers import RouterConfig, RouterIndex, _load_fit_state


def _rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    return float("nan")


def _load_config(exp_dir: Path) -> RouterConfig:
    payload = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    keys = {f.name for f in fields(RouterConfig)}
    kwargs = {key: payload[key] for key in keys if key in payload}
    return RouterConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure router-method query RSS in an isolated subprocess.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--probe-depth", required=True, type=int)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Cap the number of evaluated queries (0 = all). The RSS delta plateaus "
        "after the first few hundred queries, so a cap keeps the isolated memory "
        "measurement from re-paying the full query sweep.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    cfg = _load_config(exp_dir)

    bundle = load_learned_dataset(
        args.dataset,
        data_root,
        max_samples=cfg.prepare_max_samples,
        k=cfg.prepare_k,
        seed=cfg.seed,
        force_prepare=False,
    )
    if args.max_queries > 0 and bundle.queries.shape[0] > args.max_queries:
        bundle = replace(
            bundle,
            queries=bundle.queries[: args.max_queries],
            eval_neighbors=bundle.eval_neighbors[: args.max_queries],
        )
    family = RouterIndex(cfg)
    state = _load_fit_state(exp_dir, cfg)

    rss_before = _rss_mb()
    result = family.evaluate(bundle, state, probe_depth=int(args.probe_depth))
    rss_after = _rss_mb()
    try:
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        peak_rss = float("nan")

    payload = {
        "isolated_avg_computations": float(result.avg_computations),
        "isolated_qps": float(result.qps) if result.qps is not None else None,
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
