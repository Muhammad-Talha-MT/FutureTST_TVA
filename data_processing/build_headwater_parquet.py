#!/usr/bin/env python3
"""
Build a parquet in the format the FutureTST pipeline reads (one row per basin-hour) from
raw CAMELSH files that download_camelsh.py extracted.

Columns written: Time, basin_id, Rainf, the 24 STATIC_VARS, Q_camelsh_obs_norm,
streamflow(m3/s), latitude, longitude.

  * Rainf                = NLDAS-2 precipitation from the basin's forcing NetCDF (mm/h)
  * streamflow(m3/s)     = observed flow. Taken from the flow-only file (<out>/flow/**/<ID>.nc) if it
                           exists, otherwise from the "Streamflow" variable of the forcing file.
  * Q_camelsh_obs_norm   = streamflow * 86.4 / area_sqkm  (mm/day). This is the relation found in the
                           authors' demo parquet; negative flows are set to NaN.
  * statics              = CAMELSH climate + HydroATLAS attribute CSVs (matched case-insensitively,
                           with a few name aliases); area/lat/lon from headwater_basins.csv.

All timestamps are UTC (as in CAMELSH) and every basin is placed on the same hourly axis.

Usage
  python build_headwater_parquet.py --raw /path/to/camelsh_raw \
      --basins_csv ../data/headwater_basins.csv --out ../data/camelsh_headwater.parquet
  python build_headwater_parquet.py --raw ... --inspect        # print the structure of one NetCDF and exit
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_camelsh_forecast import STATIC_VARS  # noqa: E402

TIME_NAMES = ["time", "Time", "DateTime", "datetime", "date", "Date"]
RAIN_NAMES = ["Rainf", "rainf", "Precipitation", "precipitation", "Precip", "precip", "prcp"]
FLOW_NAMES = ["Streamflow", "streamflow", "Q", "discharge", "flow"]
ALIASES = {
    "aridity": ["aridity", "aridity_index"],
    "frac_snow": ["frac_snow", "snow_fraction", "frac_snow_daily"],
    "p_seasonality": ["p_seasonality"],
    "p_mean": ["p_mean"], "pet_mean": ["pet_mean"],
    "high_prec_freq": ["high_prec_freq"], "high_prec_dur": ["high_prec_dur"],
    "low_prec_freq": ["low_prec_freq"], "low_prec_dur": ["low_prec_dur"],
}


def norm_id(s):
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) <= 8:
        return s.zfill(8)
    if len(s) == 9:
        return s.zfill(10)
    return s


def find_files(root, ids):
    found = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".nc") and fn[:-3] in ids:
                found[fn[:-3]] = os.path.join(dp, fn)
    return found


def time_index(ds):
    for n in TIME_NAMES:
        if n in ds.coords or n in ds.variables:
            return pd.DatetimeIndex(pd.to_datetime(ds[n].values)).tz_localize(None), n
    if len(ds.dims) == 1:
        n = list(ds.dims)[0]
        return pd.DatetimeIndex(pd.to_datetime(ds[n].values)).tz_localize(None), n
    raise KeyError(f"cannot find the time coordinate; variables: {list(ds.variables)}")


def pick_var(ds, names, what, contains=None):
    low = {v.lower(): v for v in ds.variables}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if contains:
        for v in ds.variables:
            if contains in v.lower():
                return v
    raise KeyError(f"no {what} variable found; variables: {list(ds.variables)}")


def read_series(path, names, what, contains=None):
    with xr.open_dataset(path) as ds:
        tidx, tname = time_index(ds)
        v = pick_var(ds, names, what, contains)
        vals = np.asarray(ds[v].values, dtype="float64").reshape(-1)
        units = ds[v].attrs.get("units", "?")
    s = pd.Series(vals, index=tidx, name=what)
    return s[~s.index.duplicated(keep="first")], v, units


def load_attr_tables(raw):
    tabs = {}
    for name in ("attributes_nldas2_climate.csv", "attributes_hydroATLAS.csv"):
        hits = [p for p in glob.glob(os.path.join(raw, "**", "*"), recursive=True)
                if os.path.basename(p).lower() == name.lower()]
        if not hits:
            sys.exit(f"Could not find {name} under {raw}. Did `download_camelsh.py --which attributes` finish?")
        df = pd.read_csv(hits[0], dtype={0: str})
        df.index = [norm_id(x) for x in df.iloc[:, 0]]
        tabs[name] = df
    return tabs


def get_static(tabs, sid, var, allow_missing):
    cands = [c.lower() for c in ALIASES.get(var, [var])]
    for df in tabs.values():
        cols = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in cols:
                return float(df.loc[sid, cols[c]]) if sid in df.index else np.nan
    if allow_missing:
        return np.nan
    raise KeyError(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="the --out directory of download_camelsh.py")
    ap.add_argument("--basins_csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "..", "data", "headwater_basins.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "data", "camelsh_headwater.parquet"))
    ap.add_argument("--start", default="1980-01-01 00:00:00")
    ap.add_argument("--end", default="2024-12-31 23:00:00")
    ap.add_argument("--allow_missing_static", action="store_true",
                    help="fill unmatched static attributes with NaN instead of stopping")
    ap.add_argument("--inspect", action="store_true", help="print the structure of one forcing file and exit")
    args = ap.parse_args()

    meta = pd.read_csv(args.basins_csv, dtype={"camelsh_usgs_id": str})
    meta = meta[meta.has_camelsh_observed_flow.astype(bool)].copy()
    meta["sid"] = meta.camelsh_usgs_id.map(norm_id)
    ids = list(meta.sid)

    ts_files = find_files(os.path.join(args.raw, "timeseries"), set(ids))
    fl_files = find_files(os.path.join(args.raw, "flow"), set(ids))
    if args.inspect:
        p = next(iter(ts_files.values()), None) or next(iter(fl_files.values()), None)
        if not p:
            sys.exit("no NetCDF files found under --raw/timeseries or --raw/flow")
        print(p)
        print(xr.open_dataset(p))
        return
    miss = [i for i in ids if i not in ts_files]
    if miss:
        print(f"WARNING: no forcing file for {len(miss)} basins (dropped): {miss}")
    ids = [i for i in ids if i in ts_files]
    if not ids:
        sys.exit("no forcing files found")

    tabs = load_attr_tables(args.raw)
    axis = pd.date_range(args.start, args.end, freq="h")
    schema_cols = ["Time", "basin_id", "Rainf"] + STATIC_VARS + \
                  ["Q_camelsh_obs_norm", "streamflow(m3/s)", "latitude", "longitude"]
    schema = pa.schema([("Time", pa.timestamp("ns")), ("basin_id", pa.string())] +
                       [(c, pa.float32()) for c in schema_cols[2:]])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema)
    report = []
    try:
        for sid in ids:
            m = meta[meta.sid == sid].iloc[0]
            rain, rvar, runit = read_series(ts_files[sid], RAIN_NAMES, "Rainf", contains="rain")
            if sid in fl_files:
                q, qvar, _ = read_series(fl_files[sid], FLOW_NAMES, "Q", contains="flow")
                qsrc = "flow file"
            else:
                q, qvar, _ = read_series(ts_files[sid], FLOW_NAMES, "Q", contains="flow")
                qsrc = "forcing file"
            rain = rain.reindex(axis)
            q = q.reindex(axis)
            q = q.where(q >= 0)  # negative discharge -> missing
            area = float(m.camelsh_area_sqkm)
            df = pd.DataFrame({"Time": axis, "basin_id": sid, "Rainf": rain.values})
            for v in STATIC_VARS:
                if v == "area_sqkm":
                    df[v] = area
                else:
                    try:
                        df[v] = get_static(tabs, sid, v, args.allow_missing_static)
                    except KeyError:
                        cols = sorted({c for t in tabs.values() for c in t.columns})
                        sys.exit(f"Static attribute '{v}' not found in the CAMELSH attribute CSVs.\n"
                                 f"Rerun with --allow_missing_static or tell me the right column name.\n"
                                 f"Columns available: {cols[:300]}")
            df["Q_camelsh_obs_norm"] = q.values * 86.4 / area
            df["streamflow(m3/s)"] = q.values
            df["latitude"] = float(m.camelsh_lat)
            df["longitude"] = float(m.camelsh_lon)
            df = df[schema_cols].astype({c: "float32" for c in schema_cols[2:]})
            writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
            report.append(dict(basin=sid, tva=m.tva_nwsid, rain_var=f"{rvar} [{runit}]", flow_src=f"{qsrc}:{qvar}",
                               obs_hours=int(q.notna().sum()),
                               csv_hours=m.get("camelsh_hours_1980_2024", np.nan),
                               rain_nan=int(rain.isna().sum()), rain_mean_mm_per_h=round(float(rain.mean()), 4)))
    finally:
        writer.close()
    rep = pd.DataFrame(report)
    pd.set_option("display.width", 220)
    print(rep.to_string(index=False))
    print(f"\nWrote {args.out}: {len(rep)} basins x {len(axis)} hours")
    print("Checks: obs_hours should be close to csv_hours (the CAMELSH availability counts in your CSV);\n"
          "        rain_mean_mm_per_h should be about 0.1-0.2 for this region (~900-1700 mm/yr).")


if __name__ == "__main__":
    main()
