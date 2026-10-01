#!/bin/bash
set -e
# FutureTST pretraining on the TVA headwater basins (CAMELSH observed data).
#
#   bash run_pretrain_headwater.sh --parquet /path/to/camelsh_tennessee.parquet
#   bash run_pretrain_headwater.sh --parquet ... --epochs 100 --batch_size 64
#   bash run_pretrain_headwater.sh --parquet ... --resume        # continue after a time-limit stop
#
# Options (defaults):
#   --parquet      raw parquet with all needed basins (default ./data/camelsh_headwater.parquet, made by build_headwater_parquet.py)
#   --basins_file  basin IDs to use (default ./data/headwater_basin_ids.txt)
#   --window 720   --pred_len 240   --epochs 100   --patience 10   --batch_size 64
#   --device cuda  --seed 1         --out_dir results/pretrain_headwater
#   --resume       resume from results/pretrain_headwater/last.pt
# Any other flag is passed through to pretrain_headwater.py (see --help there).

PARQUET="./data/camelsh_headwater.parquet"
BASINS_FILE="./data/headwater_basin_ids.txt"
WINDOW=720; PRED_LEN=240; EPOCHS=100; PATIENCE=10; BATCH=64; DEVICE="cuda"; SEED=1
OUT_DIR="results/pretrain_headwater"; RESUME=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --parquet) PARQUET="$2"; shift 2;;
    --basins_file) BASINS_FILE="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --pred_len) PRED_LEN="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --batch_size) BATCH="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --resume) RESUME="--resume"; shift;;
    *) EXTRA+=("$1"); shift;;
  esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
[[ "$PARQUET" != /* ]] && PARQUET="$SCRIPT_DIR/${PARQUET#./}"
[[ "$BASINS_FILE" != /* ]] && BASINS_FILE="$SCRIPT_DIR/${BASINS_FILE#./}"
NPZ="$SCRIPT_DIR/data_processing/data/pretrain_headwater.npz"

echo "[1/2] Preprocessing..."
if [[ -n "$RESUME" && -f "$NPZ" ]]; then
  echo "  --resume: reusing existing $NPZ"
else
  (cd "$SCRIPT_DIR/data_processing" && python3 preprocess_headwater_pretrain.py \
      --parquet "$PARQUET" --basins_file "$BASINS_FILE" --out "$NPZ")
fi

echo "[2/2] Training..."
cd "$SCRIPT_DIR/futuretst"
PYTHONPATH=./ python3 src/experiments/pretrain_headwater.py \
    --npz "$NPZ" --device "$DEVICE" --window "$WINDOW" --pred_len "$PRED_LEN" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --batch_size "$BATCH" --seed "$SEED" \
    --out_dir "$OUT_DIR" $RESUME "${EXTRA[@]}"
