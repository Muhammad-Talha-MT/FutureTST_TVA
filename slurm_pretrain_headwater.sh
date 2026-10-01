#!/bin/bash
#SBATCH -J ftst_pretrain
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 24:00:00
#SBATCH --mem=48G
#SBATCH -c 5
#SBATCH -o pretrain_%j.out

# Edit the two paths below, then: sbatch slurm_pretrain_headwater.sh
# If the job hits the time limit, resubmit with --resume added to the last line.
source /ocean/projects/ees250003p/mtalha1/miniforge3/etc/profile.d/conda.sh
conda activate futuretst
cd /ocean/projects/ees250003p/mtalha1/FutureTST_TVA
bash run_pretrain_headwater.sh --parquet /path/to/camelsh_tennessee.parquet
