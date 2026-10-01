#!/bin/bash
#SBATCH -J ftst_6h
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH -c 5
#SBATCH -o pretrain_6h_%j.out
# Edit the TVA_headwater path, then: sbatch slurm_pretrain_6h.sh   (add --resume to the last line to continue)
source /ocean/projects/ees250003p/mtalha1/miniforge3/etc/profile.d/conda.sh
conda activate futuretst
cd /ocean/projects/ees250003p/mtalha1/FutureTST_TVA
bash run_pretrain_6h.sh --tva_dir /ocean/projects/ees250003p/mtalha1/TVA_headwater \
     --camelsh_raw /ocean/projects/ees250003p/mtalha1/camelsh_raw
