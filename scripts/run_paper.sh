#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${SHARED_ROOT}/experiments/lisbench_final}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/exports/final}"
DATASETS="glove,sift,gist,deep1m"
CONTROL_SEEDS="0"
LEARNED_CONTROL_SEEDS="0,1,2"
# Native distance kernels scale best to physical cores on SMT hosts.
PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | awk -F, '!/^#/ {seen[$1 FS $2]=1} END {print length(seen)}')"
if [[ ! "${PHYSICAL_CORES}" =~ ^[1-9][0-9]*$ ]]; then
  PHYSICAL_CORES="$(nproc 2>/dev/null || echo 8)"
fi
THREADS="${THREADS:-${PHYSICAL_CORES}}"
TRAINING_THREADS="${TRAINING_THREADS:-8}"
SESSION="${SESSION:-lisbench_final}"
NATIVE_ARCH="${NATIVE_ARCH:-native}"
NO_TMUX=0
INSIDE_TMUX=0
SMOKE=0
SKIP_TESTS=0
SKIP_PREFLIGHT_SMOKE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper.sh [options]


  --datasets LIST          Default: glove,sift,gist,deep1m
  --data-root PATH         Prepared datasets
  --experiments-root PATH  Hash-addressed run cells
  --out-root PATH          Self-contained manuscript export bundle
  --threads N              Standardized native serving thread budget
  --training-threads N     Learned training/build thread budget (default: 8)
  --session NAME           tmux session and latest-log name
  --native-arch VALUE      portable, native, or an explicit compiler target
  --no-tmux                Run in the current terminal
  --smoke                  One-seed glove10k end-to-end campaign
  --skip-tests             Skip pytest/native preflight (not recommended)
  --skip-preflight-smoke   Skip the disposable glove10k campaign
  -h, --help               Show this help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --experiments-root) EXPERIMENTS_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --training-threads) TRAINING_THREADS="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --native-arch) NATIVE_ARCH="$2"; shift 2 ;;
    --no-tmux) NO_TMUX=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-preflight-smoke) SKIP_PREFLIGHT_SMOKE=1; shift ;;
    --_inside-tmux) INSIDE_TMUX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done


PYTHON_RESOLVED="$(command -v -- "${PYTHON_BIN}" || true)"
if [[ -z "${PYTHON_RESOLVED}" || ! -x "${PYTHON_RESOLVED}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_RESOLVED}"
export PYTHON_BIN
if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--threads must be a positive integer, received: ${THREADS}" >&2
  exit 2
fi
if [[ ! "${TRAINING_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--training-threads must be a positive integer, received: ${TRAINING_THREADS}" >&2
  exit 2
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  DATASETS="glove10k"
  CONTROL_SEEDS="0"
  LEARNED_CONTROL_SEEDS="0"
  SKIP_PREFLIGHT_SMOKE=1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src"
export LISBENCH_SERVING_THREADS="${THREADS}"
export LISBENCH_TRAINING_THREADS="${TRAINING_THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_DYNAMIC="FALSE"
export OMP_WAIT_POLICY="PASSIVE"
export GOMP_SPINCOUNT="0"
export MKL_NUM_THREADS="${THREADS}"
export MKL_DYNAMIC="FALSE"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMBA_NUM_THREADS="${THREADS}"
export TF_NUM_INTRAOP_THREADS="${TRAINING_THREADS}"
export TF_NUM_INTEROP_THREADS=2
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export CUDA_VISIBLE_DEVICES=""
export LISBENCH_NATIVE_ARCH="${NATIVE_ARCH}"

mapfile -t RUNTIME_IDENTITIES < <(
  "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; from run_contract import runtime_environment_identity, runtime_source_manifest; root = Path(sys.argv[1]); print(runtime_source_manifest(root)["fingerprint"]); print(runtime_environment_identity()["fingerprint"])' "${REPO_ROOT}"
)
if [[ "${#RUNTIME_IDENTITIES[@]}" -ne 2 ]]; then
  echo "Failed to resolve frozen runtime identities." >&2
  exit 1
fi
CURRENT_RUNTIME_FINGERPRINT="${RUNTIME_IDENTITIES[0]}"
CURRENT_RUNTIME_ENVIRONMENT_FINGERPRINT="${RUNTIME_IDENTITIES[1]}"
if [[ -n "${LISBENCH_RUNTIME_FINGERPRINT:-}" && "${LISBENCH_RUNTIME_FINGERPRINT}" != "${CURRENT_RUNTIME_FINGERPRINT}" ]]; then
  echo "Numerical runtime source changed after campaign launch." >&2
  exit 1
fi
if [[ -n "${LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT:-}" && "${LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT}" != "${CURRENT_RUNTIME_ENVIRONMENT_FINGERPRINT}" ]]; then
  echo "Numerical runtime environment changed after campaign launch." >&2
  exit 1
fi
export LISBENCH_RUNTIME_FINGERPRINT="${CURRENT_RUNTIME_FINGERPRINT}"
export LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT="${CURRENT_RUNTIME_ENVIRONMENT_FINGERPRINT}"

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}" "${EXPERIMENTS_ROOT}"

if [[ "${NO_TMUX}" -eq 0 && "${INSIDE_TMUX}" -eq 0 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required unless --no-tmux is supplied." >&2
    exit 1
  fi
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
  fi
  # tmux may already be hosted by a server created from another environment.
  # Inject the resolved interpreter and active environment explicitly instead
  # of relying on tmux's limited update-environment allow-list.
  inner=(env "PYTHON_BIN=${PYTHON_BIN}" "PATH=${PATH}"
    "LISBENCH_RUNTIME_FINGERPRINT=${LISBENCH_RUNTIME_FINGERPRINT}"
    "LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT=${LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT}")
  [[ -n "${VIRTUAL_ENV:-}" ]] && inner+=("VIRTUAL_ENV=${VIRTUAL_ENV}")
  [[ -n "${CONDA_PREFIX:-}" ]] && inner+=("CONDA_PREFIX=${CONDA_PREFIX}")
  [[ -n "${LD_LIBRARY_PATH:-}" ]] && inner+=("LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")
  inner+=(bash scripts/run_paper.sh --datasets "${DATASETS}" --data-root "${DATA_ROOT}"
    --experiments-root "${EXPERIMENTS_ROOT}" --out-root "${OUT_ROOT}"
    --threads "${THREADS}" --training-threads "${TRAINING_THREADS}"
    --session "${SESSION}" --native-arch "${NATIVE_ARCH}"
    --no-tmux --_inside-tmux)
  [[ "${SMOKE}" -eq 1 ]] && inner+=(--smoke)
  [[ "${SKIP_TESTS}" -eq 1 ]] && inner+=(--skip-tests)
  [[ "${SKIP_PREFLIGHT_SMOKE}" -eq 1 ]] && inner+=(--skip-preflight-smoke)
  printf -v quoted ' %q' "${inner[@]}"
  tmux new-session -d -s "${SESSION}" "cd $(printf %q "${REPO_ROOT}") && exec${quoted}"
  echo "Started tmux session: ${SESSION}"
  echo "Attach: tmux attach -t ${SESSION}"
  echo "Latest log: ${REPO_ROOT}/logs/${SESSION}_latest/session.log"
  exit 0
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_ROOT="${REPO_ROOT}/logs/${SESSION}_${STAMP}"
mkdir -p "${LOG_ROOT}"
ln -sfn "${LOG_ROOT}" "${REPO_ROOT}/logs/${SESSION}_latest"
exec > >(tee -a "${LOG_ROOT}/session.log") 2>&1

echo "[campaign] started=$(date -u --iso-8601=seconds)"
echo "[campaign] python=${PYTHON_BIN} ($(command -v "${PYTHON_BIN}"))"
echo "[campaign] datasets=${DATASETS} controls=${CONTROL_SEEDS} learned_controls=${LEARNED_CONTROL_SEEDS} serving_threads=${THREADS} training_threads=${TRAINING_THREADS}"
echo "[campaign] experiments=${EXPERIMENTS_ROOT}"
echo "[campaign] exports=${OUT_ROOT}"
echo "[campaign] runtime_source_fingerprint=${LISBENCH_RUNTIME_FINGERPRINT}"
echo "[campaign] runtime_environment_fingerprint=${LISBENCH_RUNTIME_ENVIRONMENT_FINGERPRINT}"


run_logged() {
  local name="$1"
  shift
  echo "[stage:${name}] start=$(date -u --iso-8601=seconds)"
  "$@" > >(tee -a "${LOG_ROOT}/${name}.log") 2>&1
  mkdir -p "${OUT_ROOT}/provenance/stages"
  "${PYTHON_BIN}" - "${OUT_ROOT}/provenance/stages/${name}.done.json" "${name}" <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps({"stage": sys.argv[2], "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
os.replace(tmp, path)
PY
  echo "[stage:${name}] complete=$(date -u --iso-8601=seconds)"
}

  "${PYTHON_BIN}" -m cli_write_provenance --out-root "${OUT_ROOT}"


if [[ "${SKIP_TESTS}" -eq 0 ]]; then
  run_logged native_build bash -c \
    "cd $(printf %q "${REPO_ROOT}") && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLISBENCH_NATIVE_ARCH=$(printf %q "${NATIVE_ARCH}") -DPython_EXECUTABLE=$(printf %q "${PYTHON_BIN}") -Dpybind11_DIR=\$($(printf %q "${PYTHON_BIN}") -m pybind11 --cmakedir) && cmake --build build -j && cp build/lisbench_native*.so src/"
  "${PYTHON_BIN}" - <<'PY'
import methods.routers as routers
if routers._NATIVE_BACKEND != "cpp":
    raise SystemExit(f"C++ serving backend unavailable: {routers._NATIVE_BACKEND}")
PY
fi

if [[ "${SKIP_PREFLIGHT_SMOKE}" -eq 0 ]]; then
  SMOKE_ROOT="$(mktemp -d -p "${SHARED_ROOT}" lisbench-final-smoke.XXXXXX)"
  cleanup_smoke() { rm -rf -- "${SMOKE_ROOT}"; }
  trap cleanup_smoke EXIT
  run_logged smoke bash "${REPO_ROOT}/scripts/run_paper.sh" --smoke --no-tmux \
    --skip-tests --skip-preflight-smoke --threads "${THREADS}" \
    --training-threads "${TRAINING_THREADS}" \
    --native-arch "${NATIVE_ARCH}" --data-root "${DATA_ROOT}" \
    --experiments-root "${SMOKE_ROOT}/experiments" --out-root "${SMOKE_ROOT}/exports" \
    --session "${SESSION}_smoke"
  cleanup_smoke
  trap - EXIT
fi

CLASSICAL_METHODS="$("${PYTHON_BIN}" -m method_registry --group classical)"
MULTISEED_LEARNED_METHODS="$("${PYTHON_BIN}" -m method_registry --group learned-multiseed)"
EXPENSIVE_LEARNED_METHODS="$("${PYTHON_BIN}" -m method_registry --group learned-expensive)"

run_logged initialization_kmeans "${PYTHON_BIN}" -m cli_run_router_methods \
  --datasets "${DATASETS}" --methods "BLISS-KMeans" --seeds "${CONTROL_SEEDS}" --resume \
  --data-root "${DATA_ROOT}" --experiments-root "${EXPERIMENTS_ROOT}" --exports-root "${OUT_ROOT}"

run_logged initialization_multiseed "${PYTHON_BIN}" -m cli_run_router_methods \
  --datasets "${DATASETS}" --methods "${MULTISEED_LEARNED_METHODS}" \
  --seeds "${LEARNED_CONTROL_SEEDS}" --resume \
  --data-root "${DATA_ROOT}" --experiments-root "${EXPERIMENTS_ROOT}" --exports-root "${OUT_ROOT}"

run_logged classical "${PYTHON_BIN}" -m cli_run_competitiveness \
  --datasets "${DATASETS}" --methods "${CLASSICAL_METHODS}" --seeds "${CONTROL_SEEDS}" --resume \
  --data-root "${DATA_ROOT}" --experiments-root "${EXPERIMENTS_ROOT}" --exports-root "${OUT_ROOT}"

# One dataset per process, as defence in depth. The root cause -- an
# os.environ["OMP_NUM_THREADS"] assignment inside ann_ops.maybe_set_threads()
# that leaked across datasets and aborted a campaign on a runtime environment
# fingerprint mismatch -- is fixed at source. This isolation is kept anyway:
# the stage is the longest in the campaign, and confining any future in-process
# environment mutation to a process that is about to exit costs nothing.
IFS=',' read -r -a EXPENSIVE_DATASETS <<< "${DATASETS}"
for expensive_dataset in "${EXPENSIVE_DATASETS[@]}"; do
  run_logged "learned_controls_expensive_${expensive_dataset}" "${PYTHON_BIN}" -m cli_run_competitiveness \
    --datasets "${expensive_dataset}" --methods "${EXPENSIVE_LEARNED_METHODS}" --seeds "${CONTROL_SEEDS}" --resume \
    --data-root "${DATA_ROOT}" --experiments-root "${EXPERIMENTS_ROOT}" --exports-root "${OUT_ROOT}"
done

run_logged learned_controls_multiseed "${PYTHON_BIN}" -m cli_run_competitiveness \
  --datasets "${DATASETS}" --methods "${MULTISEED_LEARNED_METHODS}" \
  --seeds "${LEARNED_CONTROL_SEEDS}" --resume \
  --router-checkpoint-summary "${OUT_ROOT}/initialization/all_datasets_summary.json" \
  --data-root "${DATA_ROOT}" --experiments-root "${EXPERIMENTS_ROOT}" --exports-root "${OUT_ROOT}"

echo "[campaign] complete=$(date -u --iso-8601=seconds)"
echo "[campaign] bundle=${OUT_ROOT}"
echo "[campaign] logs=${LOG_ROOT}"
