#!/usr/bin/env python3
"""
Build the 6-hourly training dataset for the TVA headwater nodes.

Inputs
  * TVA_headwater/observed/<NODE>.nc      6-hourly precipitation (mm/6h) and river_flow (m3/s), UTC, label = window START
  * TVA_headwater/TVA_headwater_basin_CAMELSH_StreamCat_attributes.csv   node list, CAMELSH IDs, StreamCat statics
  * <camelsh_raw>/flow/**/<ID>_hourly.nc  CAMELSH hourly observed flow (variable `streamflow`, m3/s), UTC

What it does
  * rain   : TVA observed precipitation (same product family as the TVA forecast precipitation)
  * flow   : CAMELSH hourly flow aggregated to 6 h (mean of the six hours starting at the label, all six required,
             exactly TVA's convention) where available; TVA observed flow fills the rest.
             On the overlap the two sources are compared (correlation, mean ratio) and reported.
  * gaps   : flow gaps up to --max_gap_steps (default 8 = 48 h) are interpolated for INPUT history only;
             interpolated steps are never targets (the observed mask stays 0 there)
  * statics: a set of StreamCat catchment attributes that exist for all 43 nodes (+ log area), z-scored across basins
  * scaling: rain and flow statistics use the training period only (times <= --train_end); flow is z-scored per basin

Output: one npz with a single continuous 6-hourly axis (windows are cut at training time).

    python build_6h_dataset.py --tva_dir /path/TVA_headwater --camelsh_raw /ocean/.../camelsh_raw
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

SQMI_TO_SQKM = 2.589988
DEFAULT_STATICS = [
    "elevcat_aw", "precip8110cat_aw", "tmean8110cat_aw", "tmax8110cat_aw", "tmin8110cat_aw", "runoffcat_aw",
    "wetindexcat_aw", "claycat_aw", "sandcat_aw", "omcat_aw", "permcat_aw", "wtdepcat_aw", "rckdepcat_aw",
    "kffactcat_aw", "bficat_aw", "hydrlcondcat_aw",
    "pctdecid2019cat_aw", "pctconif2019cat_aw", "pctmxfst2019cat_aw", "pctcrop2019cat_aw", "pcthay2019cat_aw",
    "pctimp2019cat_aw", "pctwdwet2019cat_aw", "damdenscat_aw", "nabd_denscat_aw", "rddenscat_aw",
    "popden2010cat_aw", "canaldenscat_aw", "pctcarbresidcat_aw", "pctalluvcoastcat_aw",
]


def fill_short_gaps(y, max_gap):
    out = y.copy()
    n = len(y)
    isn = np.isnan(y).astype(np.int8)
    d = np.diff(np.concatenate([[0], isn, [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (e - s) <= max_gap and s > 0 and e < n:
            out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [y[s - 1], y[e]])
    return out


def read_tva(path):
    with xr.open_dataset(path) as ds:
        t = pd.DatetimeIndex(ds.time.values)
        res = {}
        for v in ("precipitation", "river_flow"):
            if v in ds:
                s = pd.Series(np.asarray(ds[v].values, dtype="float64").reshape(-1), index=t)
                res[v] = s[~s.index.duplicated(keep="first")]
            else:
                res[v] = None
    return res


def read_camelsh_6h(path):
    with xr.open_dataset(path) as ds:
        v = "streamflow" if "streamflow" in ds else [x for x in ds.data_vars if "flow" in x.lower()][0]
        tn = "time" if "time" in ds.coords else [c for c in ds.coords if "time" in c.lower()][0]
        s = pd.Series(np.asarray(ds[v].values, dtype="float64").reshape(-1), index=pd.DatetimeIndex(ds[tn].values))
    r = s.resample("6h")  # label = start of the 6 h window, left-closed
    return r.mean().where(r.count() == 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tva_dir", required=True)
    ap.add_argument("--camelsh_raw", default=None, help="download_camelsh.py output dir (needs flow/ inside); optional")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_6h.npz"))
    ap.add_argument("--start", default="1985-01-01 00:00:00")
    ap.add_argument("--end", default=None, help="default: last TVA observed time")
    ap.add_argument("--train_end", default="2018-12-31 18:00:00")
    ap.add_argument("--max_gap_steps", type=int, default=8)
    ap.add_argument("--static_cols", nargs="*", default=DEFAULT_STATICS)
    ap.add_argument("--min_corr", type=float, default=0.95)
    ap.add_argument("--ratio_range", type=float, nargs=2, default=(0.85, 1.15))
    args = ap.parse_args()

    tva = os.path.abspath(args.tva_dir)
    attrs = pd.read_csv(os.path.join(tva, "TVA_headwater_basin_CAMELSH_StreamCat_attributes.csv"),
                        dtype={"camelsh_usgs_id": str}).set_index("tva_nwsid")
    nodes_all = sorted(attrs.index)

    obs = {n: read_tva(os.path.join(tva, "observed", f"{n}.nc")) for n in nodes_all}
    end = pd.Timestamp(args.end) if args.end else max(o["precipitation"].index.max() for o in obs.values())
    axis = pd.date_range(args.start, end, freq="6h")
    cam_files = {}
    if args.camelsh_raw:
        for p in glob.glob(os.path.join(args.camelsh_raw, "flow", "**", "*_hourly.nc"), recursive=True):
            cam_files[os.path.basename(p).split("_")[0]] = p
    print(f"axis: {axis[0]} .. {axis[-1]} ({len(axis)} steps); CAMELSH flow files found: {len(cam_files)}")

    rain, q, src, rows = {}, {}, {}, []
    for n in nodes_all:
        r = obs[n]["precipitation"].reindex(axis)
        tv = obs[n]["river_flow"].reindex(axis) if obs[n]["river_flow"] is not None else pd.Series(np.nan, index=axis)
        cid = attrs.loc[n, "camelsh_usgs_id"]
        cm = pd.Series(np.nan, index=axis)
        if isinstance(cid, str) and cid in cam_files:
            cm = read_camelsh_6h(cam_files[cid]).reindex(axis)
        both = cm.notna() & tv.notna()
        corr = float(np.corrcoef(cm[both], tv[both])[0, 1]) if both.sum() > 200 else np.nan
        ratio = float(tv[both].mean() / cm[both].mean()) if both.sum() > 200 else np.nan
        use_tva = True
        if both.sum() > 200 and (corr < args.min_corr or not (args.ratio_range[0] <= ratio <= args.ratio_range[1])):
            use_tva = False
            print(f"WARNING {n}: CAMELSH vs TVA flow disagree (corr {corr:.3f}, ratio {ratio:.3f}) -> TVA flow NOT used to fill gaps")
        merged = cm.copy()
        if use_tva:
            merged = cm.where(cm.notna(), tv)
        s = np.where(cm.notna(), 1, np.where(merged.notna(), 2, 0)).astype(np.uint8)
        rain[n], q[n], src[n] = r, merged, s
        rows.append(dict(node=n, camelsh_id=cid if isinstance(cid, str) else "", n_camelsh=int(cm.notna().sum()), n_tva=int(tv.notna().sum()),
                         n_overlap=int(both.sum()), corr=round(corr, 4), tva_over_camelsh=round(ratio, 3),
                         n_merged=int(merged.notna().sum()), tva_filled=use_tva))
    rep = pd.DataFrame(rows).set_index("node")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print(rep.to_string())

    names = [n for n in nodes_all if rep.loc[n, "n_merged"] > 0]
    print(f"\n{len(names)} nodes with flow; dropped (no flow anywhere): {[n for n in nodes_all if n not in names]}")

    times = axis.values
    train_t = np.asarray(axis <= pd.Timestamp(args.train_end))
    B, T = len(names), len(axis)
    rain_raw = np.stack([rain[n].values for n in names])                       # (B,T) mm/6h, NaN = missing
    rain_ok = np.isfinite(rain_raw).astype(np.uint8)
    rain_mean = float(np.nanmean(rain_raw[:, train_t])); rain_std = float(np.nanstd(rain_raw[:, train_t]))
    rain_z = np.nan_to_num((rain_raw - rain_mean) / rain_std, nan=(0 - rain_mean) / rain_std).astype(np.float32)

    y_raw = np.stack([q[n].values for n in names])                             # (B,T) m3/s
    obs_mask = np.isfinite(y_raw).astype(np.uint8)
    y_src = np.stack([src[n] for n in names])
    y_mean = np.zeros(B); y_std = np.ones(B)
    y = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        v = y_raw[b, train_t]; v = v[np.isfinite(v)]
        if v.size > 1 and v.std() > 1e-8:
            y_mean[b], y_std[b] = v.mean(), v.std()
        else:
            print(f"NOTE {names[b]}: no observed flow in the training period -> scaled with its full record")
            v = y_raw[b][np.isfinite(y_raw[b])]; y_mean[b], y_std[b] = v.mean(), max(v.std(), 1e-6)
        z = (fill_short_gaps(y_raw[b], args.max_gap_steps) - y_mean[b]) / y_std[b]
        y[b] = np.nan_to_num(z, nan=0.0).astype(np.float32)

    cols = [c for c in args.static_cols if c in attrs.columns]
    lost = [c for c in args.static_cols if c not in attrs.columns]
    if lost:
        print("WARNING static columns not found:", lost)
    S = attrs.loc[names, cols].astype("float64")
    area = attrs.loc[names, "camelsh_drainage_area_sqkm"].fillna(attrs.loc[names, "tva_area_sqmi"] * SQMI_TO_SQKM)
    S["log_area_km2"] = np.log(area.values)
    if S.isna().any().any():
        print("WARNING NaN statics filled with the across-basin mean:", list(S.columns[S.isna().any()]))
    S = S.fillna(S.mean())
    sd = S.std(ddof=0).replace(0, 1.0)
    keep = S.std(ddof=0) > 1e-9
    S = ((S - S.mean()) / sd).loc[:, keep]
    print(f"statics: {S.shape[1]} columns (dropped constant: {list(keep.index[~keep])})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, rain=rain_z, rain_ok=rain_ok, y=y, y_obs_mask=obs_mask, y_src=y_src,
             static=S.values.astype(np.float32), static_names=np.array([str(c) for c in S.columns]), times=times.astype("datetime64[ns]"),
             basin_names=np.array([str(n) for n in names]), y_mean=y_mean.astype(np.float64), y_std=y_std.astype(np.float64),
             rain_mean=rain_mean, rain_std=rain_std, rain_zero_std=(0 - rain_mean) / rain_std,
             scale_train_end=np.array(args.train_end), step_hours=6)
    rep.to_csv(os.path.splitext(args.out)[0] + "_merge_report.csv")
    print(f"saved {args.out}: rain {rain_z.shape}, y {y.shape}, static {S.shape}; merge report next to it")
    tr = obs_mask[:, train_t].mean(axis=1)
    print("observed-flow share in training period per node:", {n: round(float(v), 2) for n, v in zip(names, tr)})


if __name__ == "__main__":
    main()
