#!/usr/bin/env python3
"""
Look for CAMELSH gauges that could supply observed flow for the four TVA nodes that have none
(BRRT1, FTCT1, PORT1S, RLRV2). Run on Bridges-2 after `download_camelsh.py --which attributes`.

TVA's files give no USGS ID for BRRT1, FTCT1 and PORT1S, so the search is spatial: every CAMELSH gauge
within --radius_km of the node's coordinates is listed with its distance, drainage area (compared with
the TVA area) and, when the tables contain it, the number of hours of observed flow.
RLRV2 already maps to USGS 03521500 (Clinch River at Richlands, VA), but CAMELSH listed 0 hours for it.

Nothing is added automatically: look at the table, pick the gauges that are really the same river reach
(distance of a few km AND a similar area), write their IDs to a text file and fetch them with

    python download_camelsh.py --out <raw> --basins_file extra_ids.txt --which flow

Usage
    python find_camelsh_candidates.py --raw /ocean/projects/ees250003p/mtalha1/camelsh_raw
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

NODES = {  # from the TVA NetCDF attributes / attribute CSV
    "BRRT1": dict(lat=36.641350, lon=-86.985924, area_sqmi=542.3, name="Red blw Highway 61"),
    "FTCT1": dict(lat=34.992222, lon=-84.381111, area_sqmi=71.4, name="Fightingtown Creek at Copperhill"),
    "PORT1S": dict(lat=36.516998, lon=-87.056999, area_sqmi=185.6, name="Sulphur Fork of the Red nr Adams"),
    "RLRV2": dict(lat=37.083332, lon=-81.773888, area_sqmi=133.8, name="Clinch at Richlands"),
}
SQMI_TO_SQKM = 2.589988


def norm_id(s):
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        return s
    return s.zfill(8) if len(s) <= 8 else (s.zfill(10) if len(s) == 9 else s)


def pick(cols, exact, contains=()):
    low = {c.lower(): c for c in cols}
    for e in exact:
        if e in low:
            return low[e]
    for c in cols:
        if any(k in c.lower() for k in contains):
            return c
    return None


def load_gauges(raw):
    files = glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True)
    tabs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, nrows=None, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        idc = pick(df.columns, ["staid", "gauge_id", "site_no", "id", "station_id"])
        idc = idc or df.columns[0]
        lat = pick(df.columns, ["lat_gage", "lat", "latitude", "gauge_lat"], ["lat"])
        lon = pick(df.columns, ["lng_gage", "lon", "longitude", "lng", "gauge_lon"], ["lng", "lon"])
        if not (lat and lon):
            continue
        area = pick(df.columns, ["drain_sqkm", "area_sqkm", "drainage_area_sqkm", "area"], ["drain_sq", "area"])
        name = pick(df.columns, ["staname", "station_name", "gauge_name", "name"], ["staname", "name"])
        hrs = pick(df.columns, [], ["availab", "hours"])
        t = pd.DataFrame({"id": df[idc].map(norm_id),
                          "lat": pd.to_numeric(df[lat], errors="coerce"),
                          "lon": pd.to_numeric(df[lon], errors="coerce")})
        t["area_km2"] = pd.to_numeric(df[area], errors="coerce") if area else np.nan
        t["name"] = df[name] if name else ""
        t["hours"] = pd.to_numeric(df[hrs], errors="coerce") if hrs else np.nan
        t["file"] = os.path.basename(f)
        tabs.append(t)
    if not tabs:
        sys.exit(f"No CSV with gauge coordinates found under {raw}. Run download_camelsh.py --which attributes first, "
                 f"then `find {raw} -name '*.csv' | head` and tell me the file names.")
    g = pd.concat(tabs, ignore_index=True)
    g = g.dropna(subset=["lat", "lon"])
    # one row per gauge: keep the first non-null value of each field across files
    agg = g.groupby("id").agg({"lat": "first", "lon": "first", "area_km2": "first", "name": "first",
                               "hours": "max", "file": "first"}).reset_index()
    print(f"{len(agg)} gauges with coordinates (from {g.file.nunique()} table(s): {sorted(g.file.unique())[:6]})")
    # hours of observed flow may live in a different table (e.g. info.csv) that has no coordinates
    if agg.hours.isna().all():
        for f in files:
            try:
                df = pd.read_csv(f, dtype=str, low_memory=False)
            except Exception:  # noqa: BLE001
                continue
            hc = pick(df.columns, [], ["availab", "hours"])
            if not hc:
                continue
            idc = pick(df.columns, ["staid", "gauge_id", "site_no", "id", "station_id"]) or df.columns[0]
            h = pd.Series(pd.to_numeric(df[hc], errors="coerce").values, index=df[idc].map(norm_id).values)
            h = h[~h.index.duplicated()]
            agg["hours"] = agg["id"].map(h)
            if agg.hours.notna().any():
                print(f"flow-availability column '{hc}' taken from {os.path.basename(f)}")
                break
    if agg.area_km2.isna().all():
        print("NOTE: no drainage-area column found, area comparison unavailable")
    if agg.hours.isna().all():
        print("NOTE: no 'hours of data' column found in these tables; check flow availability in the flow zip instead")
    return agg


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="the --out directory of download_camelsh.py")
    ap.add_argument("--radius_km", type=float, default=15.0)
    ap.add_argument("--out", default="camelsh_candidates.csv")
    args = ap.parse_args()

    g = load_gauges(args.raw)
    rows = []
    pd.set_option("display.width", 200)
    for n, v in NODES.items():
        d = haversine_km(v["lat"], v["lon"], g.lat.values, g.lon.values)
        c = g.assign(node=n, dist_km=d, tva_area_km2=v["area_sqmi"] * SQMI_TO_SQKM)
        c["area_ratio"] = c.area_km2 / c.tva_area_km2
        c = c[c.dist_km <= args.radius_km].sort_values("dist_km")
        c["plausible"] = (c.dist_km <= 5) & c.area_ratio.between(0.7, 1.4)
        print(f"\n=== {n}: {v['name']} ({v['area_sqmi']} sq mi = {v['area_sqmi'] * SQMI_TO_SQKM:.0f} km2) ===")
        if c.empty:
            print(f"  no CAMELSH gauge within {args.radius_km} km")
        else:
            print(c[["id", "name", "dist_km", "area_km2", "area_ratio", "hours", "plausible"]].head(8).round(2).to_string(index=False))
        rows.append(c)
    pd.concat(rows).to_csv(args.out, index=False)
    print(f"\nAll candidates written to {args.out}")
    print("Pick the gauges that are the same reach (few km away, similar area), put their IDs in extra_ids.txt and run:\n"
          "  python download_camelsh.py --out", args.raw, "--basins_file extra_ids.txt --which flow")


if __name__ == "__main__":
    main()
