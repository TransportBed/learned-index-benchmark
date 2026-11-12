from __future__ import annotations

import argparse
from pathlib import Path

from artifacts import write_json
from run_contract import assert_cpu_only_hardware, machine_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Write machine, package, Git, and thread provenance.")
    parser.add_argument("--out-root", default="exports/final")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    path = Path(args.out_root).expanduser().resolve() / "provenance" / "environment.json"
    hardware_validation = assert_cpu_only_hardware()
    write_json(path, {**machine_provenance(repo_root), "hardware_validation": hardware_validation})
    print(path)


if __name__ == "__main__":
    main()
