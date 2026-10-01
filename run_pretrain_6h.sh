#!/bin/bash
set -e
# 6-hourly pretraining on the TVA headwater nodes (TVA rain + merged CAMELSH/TVA flow + StreamCat statics).
#
#   bash run_pretrain_6h.sh --tva_dir /path/to/TVA_headwater --camelsh_raw /ocean/projects/ees250003p/mtalha1/camelsh_raw
#   bash run_pretrain_6h.sh ... --resume        # continue after a time-limit stop (reuses the built dataset)
#
# Options (defaults): --window 120 --pred_len 40 --epochs 100 --patience 10 --batch_size 64 --device cuda
#                     --out_dir results/pretrain_6h ; any other flag is passed to pretrain_6h.py
TVA_DIR=""; CAM_RAW=""; WINDOW=120; PRED=40; EPOCHS=100; PATIENCE=10; BATCH=64; DEVICE="cuda"; OUT="results/pretrain_6h"; RESUME=""; EXTRA=()
while [[ $# -gt 0 ]]; do case $1 in
  --tva_dir) TVA_DIR="$2"; shift 2;; --camelsh_raw) CAM_RAW="$2"; shift 2;;
  --window) WINDOW="$2"; shift 2;; --pred_len) PRED="$2"; shift 2;; --epochs) EPOCHS="$2"; shift 2;;
  --patience) PATIENCE="$2"; shift 2;; --batch_size) BATCH="$2"; shift 2;; --device) DEVICE="$2"; shift 2;;
  --out_dir) OUT="$2"; shift 2;; --resume) RESUME="--resume"; shift;; *) EXTRA+=("$1"); shift;; esac; done
[[ -z "$TVA_DIR" ]] && { echo "need --tva_dir"; exit 1; }
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"; NPZ="$HERE/data_processing/data/dataset_6h.npz"
if [[ -n "$RESUME" && -f "$NPZ" ]]; then echo "[1/2] reusing $NPZ"; else
  echo "[1/2] building dataset..."
  CAMARG=(); [[ -n "$CAM_RAW" ]] && CAMARG=(--camelsh_raw "$CAM_RAW")
  (cd "$HERE/data_processing" && python3 build_6h_dataset.py --tva_dir "$TVA_DIR" "${CAMARG[@]}" --out "$NPZ"); fi
echo "[2/2] training..."
cd "$HERE/futuretst" && PYTHONPATH=./ python3 src/experiments/pretrain_6h.py --npz "$NPZ" --device "$DEVICE" \
  --window "$WINDOW" --pred_len "$PRED" --epochs "$EPOCHS" --patience "$PATIENCE" --batch_size "$BATCH" --out_dir "$OUT" $RESUME "${EXTRA[@]}"
