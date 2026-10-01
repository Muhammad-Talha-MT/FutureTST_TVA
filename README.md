# FutureTST-TVA: Headwater Pretraining and TVA Forecast Evaluation

This branch extends the original **FutureTST TVA hourly streamflow forecasting repository** with a new pretraining and transfer workflow designed around TVA headwater basins, CAMELS-H observations, and operational TVA precipitation/flow forecasts.

The original repository provides an hourly FutureTST forecasting pipeline using CAMELS-H data. The `tva_hourly` branch adds:

- headwater-basin selection and metadata;
- CAMELS-H download and preprocessing utilities;
- hourly headwater pretraining;
- a 6-hourly TVA/CAMELS-H merged training dataset;
- Stage-1 FutureTST pretraining using observed precipitation, observed streamflow, static catchment attributes, and data-availability masks;
- Stage-2 evaluation/fine-tuning using **actual TVA forecast precipitation at TVA issue times**;
- direct comparison against TVA operational flow forecasts and a persistence baseline;
- SLURM scripts for running the workflow on PSC Bridges-2; and
- support for distributing large processed datasets, trained models, and evaluation outputs through GitHub Releases rather than Git history.

---

## 1. Motivation

The main goal of this branch is to test whether a FutureTST model can first learn transferable hydrologic behavior from historical headwater-basin observations and then be evaluated or fine-tuned under the same information structure available during an operational TVA forecast.

The workflow is organized into two stages.

### Stage 1 — Hydrologic pretraining

FutureTST is pretrained using historical observations from headwater basins. Two related workflows are provided:

1. **Hourly CAMELS-H headwater pretraining**
   - observed hourly precipitation;
   - static basin attributes;
   - historical streamflow and an observed-flow mask;
   - long historical context and multi-step streamflow prediction.

2. **6-hourly TVA/CAMELS-H pretraining**
   - TVA observed precipitation;
   - CAMELS-H streamflow aggregated to 6-hour intervals where available;
   - TVA observed flow used to fill remaining valid gaps when the two sources are sufficiently consistent;
   - StreamCat static attributes;
   - explicit precipitation-availability and streamflow-observation masks.

### Stage 2 — TVA forecast evaluation / fine-tuning

The pretrained model is then driven by **TVA forecast precipitation** at TVA's actual historical forecast issue times.

Two Stage-2 modes are supported:

- `eval`: evaluate the Stage-1 checkpoint directly, without fine-tuning;
- `finetune`: fine-tune the Stage-1 checkpoint on the historical TVA forecast archive, select using validation data, and evaluate on a later test period.

The resulting model can be compared on the same forecast windows against:

- the Stage-1 model driven by TVA forecast precipitation;
- the Stage-1 model driven by observed precipitation as an upper-bound diagnostic;
- TVA's own operational streamflow forecast; and
- a persistence forecast that holds the most recent observed streamflow constant.

---

## 2. Repository structure

```text
.
├── README.md
├── run_forecast.sh
├── run_pretrain_6h.sh
├── run_pretrain_headwater.sh
│
├── slurm_download_camelsh.sh
├── slurm_pretrain_6h.sh
├── slurm_pretrain_headwater.sh
│
├── data/
│   ├── headwater_basin_ids.txt
│   └── headwater_basins.csv
│
├── data_processing/
│   ├── build_6h_dataset.py
│   ├── build_headwater_parquet.py
│   ├── download_camelsh.py
│   ├── find_camelsh_candidates.py
│   ├── preprocess_headwater_pretrain.py
│   └── data/                         # generated data; ignored by Git
│
├── TVA/
│   ├── Observed_1H/
│   ├── Observed_6H/
│   ├── Forecast_1H/
│   ├── Forecast_6H/
│   ├── gauges.csv
│   ├── tva_nodes.csv
│   ├── variable_inventory.csv
│   └── README.md
│
└── futuretst/
    ├── requirements.txt
    ├── src/
    │   └── experiments/
    │       ├── pretrain_6h.py
    │       ├── pretrain_headwater.py
    │       └── stage2_tva_forecast.py
    ├── output/                       # generated; ignored by Git
    └── results/                      # checkpoints/results; ignored by Git
```

The original FutureTST forecasting code remains available in the repository. The files listed above are the main additions in the `tva_hourly` branch.

---

## 3. Environment

Install the original FutureTST requirements:

```bash
pip install -r futuretst/requirements.txt
```

The workflow is designed for GPU execution. CUDA is used by default when available.

For the data-processing utilities, the environment should also include packages used by the new scripts, including:

```text
numpy
pandas
torch
xarray
pyarrow
```

Some CAMELS-H download paths may additionally require utilities such as `7z` and the optional Python package `remotezip`.

---

## 4. TVA data included in this branch

The `TVA/` directory contains TVA observed and forecast NetCDF files associated with selected nodes in the Upper French Broad region.

The data include combinations of:

- observed river flow;
- observed river stage;
- observed basin-mean precipitation;
- estimated local inflow;
- forecast river flow; and
- forecast basin-mean precipitation.

See:

```text
TVA/README.md
TVA/variable_inventory.csv
TVA/gauges.csv
TVA/tva_nodes.csv
```

for details about node-to-gauge matching, variables, units, lead times, and data caveats.

---

## 5. CAMELS-H headwater data preparation

### 5.1 Basin list

The selected headwater basins are defined by:

```text
data/headwater_basin_ids.txt
data/headwater_basins.csv
```

### 5.2 Download CAMELS-H data

Use:

```bash
python data_processing/download_camelsh.py \
    --out /path/to/camelsh_raw \
    --basins_file data/headwater_basin_ids.txt
```

On Bridges-2, the provided batch script can be used:

```bash
sbatch slurm_download_camelsh.sh
```

The downloader retrieves the required CAMELS-H forcing, attribute, and streamflow files for the requested basin IDs.

### 5.3 Optional gauge-candidate search

Some TVA nodes do not have directly matched observed flow. Candidate CAMELS-H gauges can be screened spatially using:

```bash
python data_processing/find_camelsh_candidates.py \
    --raw /path/to/camelsh_raw
```

The script reports nearby gauges, drainage-area similarity, and available observation information. Candidate matches are not added automatically.

### 5.4 Build an hourly headwater parquet

Convert the downloaded CAMELS-H files into the tabular format used by the FutureTST preprocessing pipeline:

```bash
python data_processing/build_headwater_parquet.py \
    --raw /path/to/camelsh_raw \
    --basins_csv data/headwater_basins.csv \
    --out data/camelsh_headwater.parquet
```

The generated parquet contains one row per basin-hour, including:

- `Time`;
- `basin_id`;
- hourly precipitation (`Rainf`);
- static catchment attributes;
- normalized streamflow (`Q_camelsh_obs_norm`);
- streamflow in m3/s;
- latitude; and
- longitude.

---

## 6. Hourly headwater pretraining

The hourly pretraining workflow uses historical CAMELS-H observations and is intentionally simpler than the original operational forecasting input configuration.

Preprocessing uses:

- observed precipitation;
- static basin attributes;
- historical streamflow;
- an observed-streamflow mask.

Short streamflow gaps can be interpolated for **input history only**. Interpolated values are never treated as observed training or evaluation targets.

Run:

```bash
bash run_pretrain_headwater.sh \
    --parquet data/camelsh_headwater.parquet
```

Default settings include:

```text
history window : 720 hours
prediction     : 240 hours
epochs         : 100
patience       : 10
batch size     : 64
device         : cuda
```

Resume an interrupted run with:

```bash
bash run_pretrain_headwater.sh \
    --parquet data/camelsh_headwater.parquet \
    --resume
```

For Bridges-2:

```bash
sbatch slurm_pretrain_headwater.sh
```

---

## 7. Build the 6-hourly TVA/CAMELS-H dataset

The 6-hour workflow is the main path used by the Stage-1/Stage-2 TVA experiment.

Build the dataset with:

```bash
python data_processing/build_6h_dataset.py \
    --tva_dir /path/to/TVA_headwater \
    --camelsh_raw /path/to/camelsh_raw \
    --out data_processing/data/dataset_6h.npz
```

### Data construction

For each selected TVA headwater node:

**Precipitation**

TVA observed precipitation is placed on a continuous 6-hour axis.

**Streamflow**

CAMELS-H hourly observed streamflow is aggregated to 6-hour means when all six hourly values are available.

Where CAMELS-H observations are unavailable, TVA observed streamflow may be used to fill the record if the overlapping CAMELS-H and TVA series pass consistency checks.

The preprocessing script reports overlap correlation and mean-flow ratios before using TVA flow as a gap-filling source.

**Static attributes**

A common set of StreamCat attributes plus log drainage area is standardized across basins.

**Missing observations**

Short streamflow gaps may be interpolated for model history, while the observation mask ensures interpolated values are not treated as training/evaluation targets.

**Scaling**

Precipitation and streamflow scaling statistics are estimated using the training period only.

---

## 8. Stage 1 — 6-hourly FutureTST pretraining

The Stage-1 model is trained using:

- historical precipitation;
- precipitation-availability indicator;
- static catchment attributes;
- historical flow-observation mask;
- historical streamflow.

The default setup uses:

```text
history window     : 120 x 6 h = 30 days
forecast horizon   : 40 x 6 h  = 10 days
future rain reach  : 20 x 6 h  = 120 hours
```

Future precipitation beyond the configured rain horizon is replaced with zero precipitation and marked unavailable. This reproduces the information-loss pattern that occurs when the operational precipitation forecast does not span the full streamflow prediction horizon.

Run the complete dataset-build + training workflow with:

```bash
bash run_pretrain_6h.sh \
    --tva_dir /path/to/TVA_headwater \
    --camelsh_raw /path/to/camelsh_raw
```

Resume using:

```bash
bash run_pretrain_6h.sh \
    --tva_dir /path/to/TVA_headwater \
    --camelsh_raw /path/to/camelsh_raw \
    --resume
```

On Bridges-2:

```bash
sbatch slurm_pretrain_6h.sh
```

Stage-1 evaluation reports streamflow skill at selected lead times using metrics including NSE, KGE, and RMSE, together with a persistence baseline.

---

## 9. Stage 2 — TVA operational forecast evaluation

Stage 2 evaluates the Stage-1 representation using the actual historical TVA forecast archive.

The important difference from Stage 1 is that future precipitation is no longer taken from observations. It is replaced with **TVA forecast precipitation issued at the real TVA forecast issue time**.

Run from the `futuretst/` directory.

### Direct Stage-1 evaluation

```bash
PYTHONPATH=./ python3 src/experiments/stage2_tva_forecast.py \
    --mode eval \
    --npz ../data_processing/data/dataset_6h.npz \
    --tva_dir /path/to/TVA_headwater \
    --stage1_dir results/pretrain_6h
```

### Fine-tune on the TVA forecast archive

```bash
PYTHONPATH=./ python3 src/experiments/stage2_tva_forecast.py \
    --mode finetune \
    --npz ../data_processing/data/dataset_6h.npz \
    --tva_dir /path/to/TVA_headwater \
    --stage1_dir results/pretrain_6h \
    --out_dir results/stage2
```

Default temporal partitions in the Stage-2 script are:

```text
fine-tuning : Oct 2016 – Dec 2018
validation  : Jan 2019 – Dec 2019
test        : Jan 2020 – Dec 2025
```

These dates are configurable through command-line arguments.

### Evaluation baselines

Stage 2 evaluates forecasts on matched windows and can report:

- `NSE_stage1` or `NSE_finetuned`: FutureTST driven by TVA forecast precipitation;
- `NSE_stage1_obsrain`: the same model driven by observed future precipitation, used as an upper-bound diagnostic;
- `NSE_TVA`: TVA's archived operational flow forecast; and
- `NSE_persistence`: the most recent observed flow held constant.

RMSE is also reported for corresponding forecast sources.

---

## 10. Original FutureTST forecasting workflow

The original repository's end-to-end hourly forecasting workflow is still available:

```bash
bash run_forecast.sh
```

That path uses the original CAMELS-H forecasting preprocessing and FutureTST training/evaluation workflow.

The new headwater pretraining and TVA transfer experiments are additional workflows rather than replacements for the original implementation.

---

## 11. Generated outputs and large files

Generated datasets, checkpoints, predictions, and result directories are intentionally excluded from normal Git tracking.

Examples include:

```text
data_processing/data/
futuretst/output/
futuretst/results/
```

Large reproducibility artifacts should be distributed using **GitHub Releases** instead of committing them directly to the repository.

Recommended release assets include:

```text
tva_data.tar.gz
tva_trained_model.tar.gz
tva_results.tar.gz
```

This keeps the Git repository focused on source code while still allowing processed data, final model checkpoints, and evaluation products to be archived alongside a tagged release.

---

## 12. Bridges-2 workflow

Example sequence:

```bash
# 1. Download CAMELS-H data
sbatch slurm_download_camelsh.sh

# 2. Pretrain at hourly resolution if desired
sbatch slurm_pretrain_headwater.sh

# 3. Build the 6-hour dataset and train Stage 1
sbatch slurm_pretrain_6h.sh

# 4. Evaluate or fine-tune Stage 2
cd futuretst
PYTHONPATH=./ python3 src/experiments/stage2_tva_forecast.py \
    --mode eval \
    --npz ../data_processing/data/dataset_6h.npz \
    --tva_dir /path/to/TVA_headwater \
    --stage1_dir results/pretrain_6h
```

The provided SLURM files contain project-specific Bridges-2 paths and should be edited if the repository or Conda environment is located elsewhere.

---

## 13. Main differences from the original branch

| Component | Original repository | `tva_hourly` branch |
|---|---|---|
| Primary workflow | Hourly FutureTST forecasting | Adds headwater pretraining + operational TVA transfer |
| Training data | CAMELS-H forecasting dataset | CAMELS-H + TVA + StreamCat |
| Temporal resolution | Primarily hourly | Hourly pretraining and dedicated 6-hour Stage-1/Stage-2 workflow |
| Basin selection | TVA regional / auxiliary basins | Adds explicit TVA headwater basin lists |
| CAMELS-H acquisition | External/full datasets | Adds targeted CAMELS-H downloader |
| Missing-flow treatment | Original preprocessing | Adds masks and history-only short-gap interpolation |
| Operational precipitation | Forecasting workflow inputs | Explicit Stage-2 use of archived TVA forecast precipitation |
| Operational benchmark | Standard model metrics | Adds matched comparison to TVA flow forecast and persistence |
| HPC execution | General scripts | Adds Bridges-2 SLURM scripts |
| Large artifacts | Previously external / local | Designed for GitHub Release assets |
| Repository hygiene | Python cache files were tracked | Cache/output/result directories are ignored |

---

## 14. Reproducibility notes

- Check the time-zone convention before combining external hourly datasets. CAMELS-H timestamps are treated as UTC in the new preprocessing workflow.
- The 6-hourly Stage-2 script performs alignment diagnostics between TVA forecast issue times, forecast lead times, precipitation, and observed flow.
- Interpolated streamflow values are for input history only and are excluded from target masks.
- Scaling uses the configured training period to avoid information leakage.
- TVA and CAMELS-H flow are compared before TVA flow is used as a gap-filling source.
- Large generated files are intentionally excluded from Git and should be obtained from the corresponding GitHub Release when available.

---

## 15. Acknowledgement

This repository builds on the original FutureTST implementation for TVA streamflow forecasting.

The project supports development and evaluation of machine-learning approaches for hydrologic and hydropower inflow forecasting using historical observations, basin attributes, and operational forecast information.

The original repository acknowledges support from the U.S. Department of Energy's Hydropower and Hydrokinetic Office (H2O).
