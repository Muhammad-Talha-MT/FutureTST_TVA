#!/bin/bash
#SBATCH -J camelsh_dl
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/ees250003p/mtalha1/camelsh_download_%j.log

source /ocean/projects/ees250003p/mtalha1/miniforge3/etc/profile.d/conda.sh
conda activate futuretst

echo "internet check:"; curl -sI -m 20 https://zenodo.org | head -1

cd /ocean/projects/ees250003p/mtalha1/FutureTST_TVA/data_processing
python download_camelsh.py \
    --out /ocean/projects/ees250003p/mtalha1/camelsh_raw \
    --basins_file ../data/headwater_basin_ids.txt
