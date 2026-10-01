"""
Preprocessing for FutureTST pretraining on the TVA headwater basins.

parquet (one row per basin-hour)  ->  pretrain_headwater.npz

Differences from preprocess_camelsh_forecast.py
  * Inputs are only observed precipitation (Rainf) + static basin attributes
    (+ a flow-observed mask that is built at training time). No temperature,
    radiation, cumulative features, ...
  * Data are stored as ONE continuous time axis per basin (not split into
    separate train/val/test arrays). The training script builds windows by
    target time, so a 720 h history can reach back across a split boundary.
  * Short flow gaps (<= --max_gap hours) are linearly interpolated for use as
    *input history only*. Interpolated values are marked as "not observed" in
    the mask and are never used as training/evaluation targets.
  * Long gaps are filled with 0 (the training mean after z-scoring) and
    marked "not observed".
  * All scaling statistics use the training period only (times <= --train_end).

Usage
    python preprocess_headwater_pretrain.py --parquet ../data/camelsh_tennessee.parquet \
        --basins_file ../data/headwater_basin_ids.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_camelsh_forecast import (  # noqa: E402
    load_parquet_streaming, STATIC_VARS, TARGET_VAR,
)


def fill_short_gaps(y, max_gap):
    """Linearly interpolate NaN runs of length <= max_gap that have observed
    values on both sides. Returns a new array."""
    out = y.copy()
    n = len(y)
    isn = np.isnan(y).astype(np.int8)
    d = np.diff(np.concatenate([[0], isn, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap and s > 0 and e < n:
            out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [y[s - 1], y[e]])
    return out


def read_ids(args):
    ids = []
    if args.basins_file:
        with open(args.basins_file) as f:
            ids += f.read().split()
    ids += list(args.basins)
    ids = [str(i).strip() for i in ids if str(i).strip()]
    return list(dict.fromkeys(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--basins_file", default=None,
                    help="text file with whitespace-separated basin IDs (e.g. 03451000)")
    ap.add_argument("--basins", nargs="*", default=[], help="basin IDs on the command line")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data", "pretrain_headwater.npz"))
    ap.add_argument("--train_end", default="2018-12-31 23:00:00",
                    help="last hour used for scaling statistics (training period end)")
    ap.add_argument("--max_gap", type=int, default=48,
                    help="interpolate flow gaps up to this many hours (history input only)")
    args = ap.parse_args()

    ids = read_ids(args)
    if not ids:
        sys.exit("No basin IDs given (use --basins_file or --basins).")
    print(f"Requested {len(ids)} basins")

    print("Loading parquet...")
    dynamic, target, static, times, basin_names = load_parquet_streaming(
        args.parquet, ["Rainf"], STATIC_VARS, TARGET_VAR, select_basins=ids)
    basin_names = [str(b) for b in basin_names]
    missing = [i for i in ids if i not in basin_names]
    if missing:
        print(f"WARNING: {len(missing)} requested basins are NOT in the parquet: {missing}")
    if not basin_names:
        sys.exit("None of the requested basins were found in the parquet.")
    n_times, n_basins = len(times), len(basin_names)
    print(f"Using {n_basins} basins, {n_times} hourly steps "
          f"[{pd.Timestamp(times[0])} .. {pd.Timestamp(times[-1])}]")

    times_pd = pd.to_datetime(times)
    train_mask_t = np.asarray(times_pd <= pd.Timestamp(args.train_end))

    # ---- rainfall: (n_basins, n_times), z-scored with training-period stats
    rain = dynamic["Rainf"].T.astype(np.float32)          # (B, T)
    n_rain_nan = int(np.isnan(rain).sum())
    if n_rain_nan:
        print(f"WARNING: {n_rain_nan} NaN Rainf values -> set to 0 before scaling")
        rain = np.nan_to_num(rain, nan=0.0)
    r_tr = rain[:, train_mask_t].astype(np.float64)
    rain_mean, rain_std = float(r_tr.mean()), float(r_tr.std())
    rain_std = rain_std if rain_std > 1e-8 else 1.0
    rain = ((rain - rain_mean) / rain_std).astype(np.float32)

    # ---- statics: (n_basins, n_static), z-scored across basins
    S = np.stack([static[v] for v in STATIC_VARS], axis=1).astype(np.float64)  # (B, S)
    s_mean = np.nanmean(S, axis=0)
    s_std = np.nanstd(S, axis=0)
    s_mean = np.where(np.isnan(s_mean), 0.0, s_mean)
    s_std = np.where((s_std < 1e-8) | np.isnan(s_std), 1.0, s_std)
    n_nan_static = int(np.isnan(S).sum())
    if n_nan_static:
        print(f"WARNING: {n_nan_static} NaN static values -> filled with the across-basin mean")
    S = np.where(np.isnan(S), s_mean[None, :], S)
    S = ((S - s_mean) / s_std).astype(np.float32)

    # ---- streamflow: per-basin z-score (training period, observed values only)
    y_raw = target.T.astype(np.float64)                   # (B, T), NaN = missing
    obs_mask = (~np.isnan(y_raw)).astype(np.uint8)
    y_mean = np.zeros(n_basins, dtype=np.float64)
    y_std = np.ones(n_basins, dtype=np.float64)
    for b in range(n_basins):
        v = y_raw[b, train_mask_t]
        v = v[~np.isnan(v)]
        if v.size > 1:
            y_mean[b] = v.mean()
            y_std[b] = v.std() if v.std() > 1e-8 else 1.0
        else:
            print(f"WARNING: basin {basin_names[b]} has no observed flow in the training period")

    y = np.zeros((n_basins, n_times), dtype=np.float32)
    n_filled = 0
    for b in range(n_basins):
        filled = fill_short_gaps(y_raw[b], args.max_gap)
        n_filled += int(np.isnan(y_raw[b]).sum() - np.isnan(filled).sum())
        z = (filled - y_mean[b]) / y_std[b]
        y[b] = np.nan_to_num(z, nan=0.0).astype(np.float32)
    print(f"Interpolated {n_filled} flow hours (gaps <= {args.max_gap} h); "
          f"observed fraction overall: {obs_mask.mean():.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        rain=rain, static=S, y=y, y_obs_mask=obs_mask,
        times=times.astype("datetime64[ns]"),
        basin_names=np.array(basin_names),
        static_vars=np.array(STATIC_VARS),
        y_mean=y_mean.astype(np.float32), y_std=y_std.astype(np.float32),
        rain_mean=np.float64(rain_mean), rain_std=np.float64(rain_std),
        static_mean=s_mean, static_std=s_std,
        scale_train_end=np.array(args.train_end),
        max_gap=np.int64(args.max_gap),
    )
    print(f"Saved {args.out}")
    print(f"  rain {rain.shape}, static {S.shape}, y {y.shape}")
    print("  observed-flow fraction per basin (whole record):")
    for b in range(n_basins):
        print(f"    {basin_names[b]}: {obs_mask[b].mean():.3f}")


if __name__ == "__main__":
    main()
