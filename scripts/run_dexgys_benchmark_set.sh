#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ALLOWED_GPUS="${ALLOWED_GPUS:-0,1,2,4,5,6}"
REQUIRED_GPUS="${REQUIRED_GPUS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_ID="${RUN_ID:-dexgys_benchmark_$(date '+%Y%m%d_%H%M%S')}"

CHECKPOINT_LABELS=(30000 40000 80000 90000)
CHECKPOINT_DIRS=(
  "$REPO_ROOT/checkpoints/eval_checkpoints/dexgys_6gpu_bs3_20260717_1048/checkpoint-30000"
  "$REPO_ROOT/checkpoints/eval_checkpoints/dexgys_6gpu_bs3_20260717_1048/checkpoint-40000"
  "$REPO_ROOT/checkpoints/eval_checkpoints/dexgys_6gpu_bs3_20260717_1048/checkpoint-80000"
  "$REPO_ROOT/checkpoints/dexgys_6gpu_bs3_20260717_1048/checkpoint-90000"
)

contains_gpu() {
  local needle="$1"
  local item
  IFS=',' read -ra items <<< "$ALLOWED_GPUS"
  for item in "${items[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

free_allowed_gpus() {
  local busy_uuids
  busy_uuids=" $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u | tr '\n' ' ') "

  while IFS=',' read -r idx uuid; do
    idx="${idx// /}"
    uuid="${uuid// /}"
    if contains_gpu "$idx" && [[ "$busy_uuids" != *" $uuid "* ]]; then
      echo "$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)
}

mkdir -p "$REPO_ROOT/logs"

echo "[$(date '+%F %T')] waiting for $REQUIRED_GPUS free GPUs from {$ALLOWED_GPUS}"
while true; do
  mapfile -t FREE_GPUS < <(free_allowed_gpus)
  echo "[$(date '+%F %T')] free GPUs: ${FREE_GPUS[*]:-none}"
  if (( ${#FREE_GPUS[@]} >= REQUIRED_GPUS )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

SELECTED_GPUS=("${FREE_GPUS[@]:0:$REQUIRED_GPUS}")
echo "[$(date '+%F %T')] selected GPUs: ${SELECTED_GPUS[*]}"

pids=()
for i in "${!CHECKPOINT_DIRS[@]}"; do
  label="${CHECKPOINT_LABELS[$i]}"
  ckpt="${CHECKPOINT_DIRS[$i]}"
  gpu="${SELECTED_GPUS[$i]}"
  log_path="$REPO_ROOT/logs/benchmark_checkpoint-${label}_${RUN_ID}.log"
  latest_log_path="$REPO_ROOT/logs/benchmark_checkpoint-${label}.log"

  echo "[$(date '+%F %T')] start checkpoint-$label on GPU $gpu"
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}" \
  "$REPO_ROOT/scripts/run_dexgys_benchmark_one.sh" "$ckpt" "$gpu" > "$log_path" 2>&1 &
  pids+=("$!")
  ln -sfn "$(basename "$log_path")" "$latest_log_path"
done

status=0
for i in "${!pids[@]}"; do
  label="${CHECKPOINT_LABELS[$i]}"
  if wait "${pids[$i]}"; then
    echo "[$(date '+%F %T')] checkpoint-$label done"
  else
    echo "[$(date '+%F %T')] checkpoint-$label failed"
    status=1
  fi
done

summary_log="$REPO_ROOT/logs/benchmark_${RUN_ID}_summary.log"
{
  echo "run_id: $RUN_ID"
  echo "selected_gpus: ${SELECTED_GPUS[*]}"
  for label in "${CHECKPOINT_LABELS[@]}"; do
    pred_dir="$REPO_ROOT/test_output/dexgys_6gpu_bs3_20260717_1048_checkpoint-${label}_test"
    echo
    echo "checkpoint-$label"
    if [[ -f "$pred_dir/summary.json" ]]; then
      cat "$pred_dir/summary.json"
    else
      echo "summary missing: $pred_dir/summary.json"
    fi
  done
} | tee "$summary_log"

exit "$status"
