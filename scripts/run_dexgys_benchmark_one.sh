#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_dir> <gpu_id>" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
GPU_ID="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/dexgys_final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/test_output}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRECISION="${PRECISION:-bfloat16}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"

if [[ ! -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "Missing checkpoint model: $CHECKPOINT_DIR/model.safetensors" >&2
  exit 1
fi

CKPT_PARENT="$(basename "$(dirname "$CHECKPOINT_DIR")")"
CKPT_NAME="$(basename "$CHECKPOINT_DIR")"
PRED_DIR="$OUTPUT_ROOT/${CKPT_PARENT}_${CKPT_NAME}_test"
PRED_PATH="$PRED_DIR/predictions.json"

echo "[$(date '+%F %T')] checkpoint=$CHECKPOINT_DIR gpu=$GPU_ID"
echo "[$(date '+%F %T')] output=$PRED_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" scripts/test.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --precision "$PRECISION" \
  --save-pred \
  --output-dir "$OUTPUT_ROOT"

if [[ ! -f "$PRED_PATH" ]]; then
  echo "Prediction file was not created: $PRED_PATH" >&2
  exit 1
fi

"$PYTHON" -m benchmark.dexgys.chamfer \
  --pred-path "$PRED_PATH" \
  --data-path "$DATA_DIR" \
  --use-dashboard=False

"$PYTHON" -m benchmark.dexgys.fid \
  --pred-path "$PRED_PATH" \
  --data-path "$DATA_DIR" \
  --batch-size "$FID_BATCH_SIZE"

"$PYTHON" - "$PRED_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

pred_dir = Path(sys.argv[1])
summary = {}

csv_path = pred_dir / "benchmark.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    for key in [
        "hand_chamfer_loss",
        "cmap_loss",
        "obj_penetration_loss",
        "self_penetration_loss",
    ]:
        if key in df:
            summary[key] = float(df[key].mean())

fid_path = pred_dir / "fid.txt"
if fid_path.exists():
    text = fid_path.read_text().strip()
    if ":" in text:
        summary["fid"] = float(text.split(":", 1)[1].strip())

(pred_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[$(date '+%F %T')] done checkpoint=$CHECKPOINT_DIR"
