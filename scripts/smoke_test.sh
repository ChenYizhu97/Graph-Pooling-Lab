#!/usr/bin/env bash
set -u

# GPLab smoke test runner.
# Runs a minimal 1-epoch experiment across built-in pools and TU datasets.
# Results are written to a JSON file for quick inspection.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_POOLS=(
  nopool
  topkpool
  sagpool
  asapool
  sparsepool
  mincutpool
  diffpool
  densepool
)

DEFAULT_DATASETS=(
  MUTAG
  PROTEINS
  ENZYMES
  FRANKENSTEIN
  Mutagenicity
  AIDS
  DD
  NCI1
  COX2
)

RESULTS_PATH="${RESULTS_PATH:-/tmp/gplab_smoke_result.json}"
LOG_FILE="${LOG_FILE:-}"
TAG_PREFIX="${TAG_PREFIX:-smoke}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
POOL_RATIO="${POOL_RATIO:-0.5}"
MODEL_VARIANT="${MODEL_VARIANT:-sum}"
RUNS="${RUNS:-1}"
LR="${LR:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-0}"
EPOCHS="${EPOCHS:-1}"
SEED_BASE="${SEED_BASE:-20260320}"
SPLIT_TRAIN="${SPLIT_TRAIN:-0.8}"
SPLIT_VAL="${SPLIT_VAL:-0.1}"

if [ -n "${POOLS:-}" ]; then
  # shellcheck disable=SC2206
  POOL_LIST=(${POOLS})
else
  POOL_LIST=("${DEFAULT_POOLS[@]}")
fi

if [ -n "${DATASETS:-}" ]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

if ! sh -c "$PYTHON_CMD --version" >/dev/null 2>&1; then
  echo "PYTHON_CMD is not runnable: $PYTHON_CMD" >&2
  echo "Set PYTHON_CMD to a working command, for example:" >&2
  echo "PYTHON_CMD='conda run -n torch_env python3' bash scripts/smoke_test.sh" >&2
  exit 1
fi

join_by_comma() {
  local IFS=,
  echo "$*"
}

POOLS_CSV="$(join_by_comma "${POOL_LIST[@]}")"
DATASETS_CSV="$(join_by_comma "${DATASET_LIST[@]}")"

cmd="$PYTHON_CMD -m gplab.cli.validate --pools \"$POOLS_CSV\" --datasets \"$DATASETS_CSV\" --model-variant \"$MODEL_VARIANT\" --pool-ratio \"$POOL_RATIO\" --runs \"$RUNS\" --epochs \"$EPOCHS\" --patience \"$PATIENCE\" --lr \"$LR\" --batch-size \"$BATCH_SIZE\" --split-train \"$SPLIT_TRAIN\" --split-val \"$SPLIT_VAL\" --seed-mode auto --seed-base \"$SEED_BASE\" --tag-prefix \"$TAG_PREFIX\" --output-format json"
if [ -n "$LOG_FILE" ]; then
  cmd="$cmd --log-file \"$LOG_FILE\""
fi

start="$(date +%s)"
sh -c "$cmd" >"$RESULTS_PATH" 2>/tmp/gplab_smoke_stderr.log
exit_code=$?
end="$(date +%s)"
elapsed="$((end - start))"

if [ "$exit_code" -eq 0 ]; then
  printf 'Smoke validation ok (%ss)\n' "$elapsed"
else
  status="$(tail -n 1 /tmp/gplab_smoke_stderr.log | tr '\t' ' ' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  if [ -z "$status" ]; then
    status="failed"
  fi
  printf 'Smoke validation failed (%ss): %s\n' "$elapsed" "$status" >&2
fi

printf 'Saved smoke validation JSON to %s\n' "$RESULTS_PATH"
exit "$exit_code"
