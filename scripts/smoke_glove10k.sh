#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

exec bash "${REPO_ROOT}/scripts/run_paper.sh" \
  --smoke \
  --no-tmux \
  --skip-preflight-smoke \
  --data-root "${DATA_ROOT:-${SHARED_ROOT}/data}" \
  --experiments-root "${EXPERIMENTS_ROOT:-${SHARED_ROOT}/experiments/lisbench_smoke}" \
  --out-root "${OUT_ROOT:-${REPO_ROOT}/exports/final_smoke}" \
  "$@"
