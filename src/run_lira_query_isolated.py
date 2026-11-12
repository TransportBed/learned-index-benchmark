from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from methods.lira_runtime import LiraConfig, _online_selected_query_summary


def _load_config(exp_dir: Path) -> LiraConfig:
    payload = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    keys = {f.name for f in fields(LiraConfig)}
    kwargs = {key: payload[key] for key in keys if key in payload}
    return LiraConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LIRA query metrics in an isolated subprocess.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    cfg = _load_config(exp_dir)
    payload = _online_selected_query_summary(
        cfg,
        exp_dir,
        data_root=Path(args.data_root).expanduser().resolve(),
        threshold=float(args.threshold),
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
