#!/usr/bin/env python3
"""
Download the parts of CAMELSH needed for the TVA headwater basins (run on Bridges-2,
which has internet access) and extract ONLY the requested basins.

Zenodo records used (see https://zenodo.org/records/<id>):
  15066778  attributes.7z, shapefiles.7z, timeseries.7z   (basins with observed flow when v2 was made)
  15070091  timeseries_nonobs.7z                          (forcing for the remaining basins)
  16729675  Hourly2.zip                                   (observed streamflow + water level, 5,767 gauges)

Steps (each can be run separately with --which):
  attributes  attributes.7z           (~19 MB)  -> <out>/attributes/
  obs         timeseries.7z           (~21 GB)  -> <out>/timeseries/<ID>.nc   (only requested IDs)
  nonobs      timeseries_nonobs.7z    (large)   -> <out>/timeseries/<ID>.nc   (only IDs still missing)
  flow        Hourly2.zip             (streamflow files; only requested IDs, via HTTP range requests
                                        if `remotezip` is installed, otherwise the whole zip is downloaded)

Needs the `7z` command for the .7z archives:   conda install -c conda-forge p7zip
Downloads resume automatically if interrupted (re-run the same command).

Example
  python download_camelsh.py --out /ocean/projects/<grant>/$USER/camelsh_raw \
      --basins_file ../data/headwater_basin_ids.txt --which attributes obs nonobs flow
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

REC_OBS, REC_NONOBS, REC_FLOW = 15066778, 15070091, 16729675


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def list_record_files(rec):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
        meta = json.load(r)
    files = meta.get("files", [])
    if isinstance(files, dict):  # some API versions
        files = files.get("entries", [])
    out = {}
    for f in files:
        key = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or f"https://zenodo.org/records/{rec}/files/{key}?download=1"
        out[key] = dict(url=url, size=f.get("size"), md5=(f.get("checksum") or "").replace("md5:", ""))
    return out


def find_file(files, pattern):
    hits = [k for k in files if pattern.lower() in k.lower()]
    if not hits:
        sys.exit(f"No file matching '{pattern}' in record. Available: {sorted(files)}")
    return hits[0]


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, size=None, md5=None, attempts=30):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        log(f"already downloaded: {dest}")
        return dest
    part = dest + ".part"
    for attempt in range(1, attempts + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if size and done == size:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (done and r.status == 206) else "wb"
                if mode == "wb":
                    done = 0
                with open(part, mode) as f:
                    last = time.time()
                    while True:
                        b = r.read(1 << 23)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if time.time() - last > 30:
                            log(f"  {os.path.basename(dest)}: {done / 1e9:.2f}" + (f" / {size / 1e9:.2f}" if size else "") + " GB")
                            last = time.time()
            if not size or os.path.getsize(part) == size:
                break
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range past end: already complete
                break
            log(f"  HTTP {e.code}, retry {attempt}/{attempts}")
            time.sleep(min(60, 5 * attempt))
        except Exception as e:  # noqa: BLE001
            log(f"  {type(e).__name__}: {e}; retry {attempt}/{attempts}")
            time.sleep(min(60, 5 * attempt))
    if size and os.path.getsize(part) != size:
        sys.exit(f"Download of {dest} incomplete ({os.path.getsize(part)} of {size} bytes). Re-run to resume.")
    os.replace(part, dest)
    if md5:
        log(f"  verifying md5 of {os.path.basename(dest)} ...")
        if md5sum(dest) != md5:
            sys.exit(f"md5 mismatch for {dest}; delete it and re-run.")
    return dest


def find_7z():
    for n in ("7z", "7zz", "7za", "7zr"):
        p = shutil.which(n)
        if p:
            return p
    sys.exit("No 7z executable found. Install with:  conda install -c conda-forge p7zip")


def extract_7z(archive, out_dir, ids=None):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [find_7z(), "x", archive, f"-o{out_dir}", "-y"]
    if ids:
        cmd += [f"-ir!*{i}.nc" for i in ids]
    log("extracting:", " ".join(cmd[:5]), f"... ({len(ids) if ids else 'all'} files)")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def id_in_name(name, sid):
    """True if the basin ID appears in the file's base name (not inside a longer digit string)."""
    return re.search(rf"(?<!\d){re.escape(sid)}(?!\d)", os.path.basename(name)) is not None


def collect_nc(root, ids):
    """Locate NetCDF files whose base name contains a requested ID; return {id: path}."""
    found = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".nc", ".nc4")):
                for sid in ids:
                    if sid not in found and id_in_name(fn, sid):
                        found[sid] = os.path.join(dp, fn)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for downloads and extracted files")
    ap.add_argument("--basins_file", required=True)
    ap.add_argument("--which", nargs="+", default=["attributes", "obs", "nonobs", "flow"],
                    choices=["attributes", "obs", "nonobs", "flow"])
    ap.add_argument("--delete_archives", action="store_true", help="remove big archives after extraction")
    ap.add_argument("--download_only", action="store_true",
                    help="only download the timeseries archive (network-bound; safe on a login node). "
                         "Run again without this flag, inside a job, to extract. nonobs is skipped in this mode.")
    args = ap.parse_args()

    ids = open(args.basins_file).read().split()
    out = os.path.abspath(args.out)
    dl = os.path.join(out, "downloads")
    os.makedirs(dl, exist_ok=True)
    ts_dir = os.path.join(out, "timeseries")
    flow_dir = os.path.join(out, "flow")
    log(f"{len(ids)} basins requested; output in {out}")

    if "attributes" in args.which:
        files = list_record_files(REC_OBS)
        k = find_file(files, "attributes")
        p = download(files[k]["url"], os.path.join(dl, k), files[k]["size"], files[k]["md5"])
        extract_7z(p, os.path.join(out, "attributes_raw"))
        log("attributes extracted to", os.path.join(out, "attributes_raw"))

    for step, rec, pattern in (("obs", REC_OBS, "timeseries.7z"), ("nonobs", REC_NONOBS, "nonobs")):
        if step not in args.which:
            continue
        have = collect_nc(ts_dir, set(ids)) if os.path.isdir(ts_dir) else {}
        missing = [i for i in ids if i not in have]
        if not missing:
            log(f"[{step}] all forcing files already present, skipping")
            continue
        if args.download_only and step == "nonobs":
            log("[nonobs] skipped in --download_only mode (only needed if basins are missing after extracting obs)")
            continue
        files = list_record_files(rec)
        k = find_file(files, pattern)
        p = download(files[k]["url"], os.path.join(dl, k), files[k]["size"], files[k]["md5"])
        if args.download_only:
            log(f"[{step}] downloaded {p}; extract later by re-running without --download_only")
            continue
        extract_7z(p, ts_dir, missing)
        have = collect_nc(ts_dir, set(ids))
        log(f"[{step}] forcing files found so far: {len(have)}/{len(ids)}; still missing: "
            f"{[i for i in ids if i not in have]}")
        if args.delete_archives:
            os.remove(p)

    if "flow" in args.which:
        have = collect_nc(flow_dir, set(ids)) if os.path.isdir(flow_dir) else {}
        missing = [i for i in ids if i not in have]
        if missing:
            files = list_record_files(REC_FLOW)
            k = find_file(files, "hourly2") if any("hourly2" in f.lower() for f in files) else find_file(files, ".zip")
            os.makedirs(flow_dir, exist_ok=True)
            def pick_members(names):
                sel, nomatch = {}, []
                for i in missing:
                    m = [n for n in names if id_in_name(n, i) and n.lower().endswith((".nc", ".nc4", ".csv"))]
                    m.sort(key=lambda n: (not n.lower().endswith((".nc", ".nc4")), len(n)))
                    if m:
                        sel[i] = m[0]
                    else:
                        nomatch.append(i)
                if nomatch:
                    exts = sorted({os.path.splitext(n)[1].lower() for n in names})
                    log(f"  [flow] no member matched {len(nomatch)} IDs (e.g. {nomatch[:3]}). "
                        f"Zip has {len(names)} members, extensions {exts}. First names:")
                    for n in names[:15]:
                        log("      ", n)
                    hits = [n for n in names if nomatch[0][-6:] in n][:5]
                    log(f"  [flow] members containing '{nomatch[0][-6:]}': {hits}")
                return sel

            try:
                from remotezip import RemoteZip
                with RemoteZip(files[k]["url"]) as z:
                    for i, n in pick_members(z.namelist()).items():
                        z.extract(n, flow_dir)
            except ImportError:
                log("`remotezip` not installed -> downloading the whole zip (pip install remotezip to avoid this)")
                p = download(files[k]["url"], os.path.join(dl, k), files[k]["size"], files[k]["md5"])
                import zipfile
                with zipfile.ZipFile(p) as z:
                    for i, n in pick_members(z.namelist()).items():
                        z.extract(n, flow_dir)
            have = collect_nc(flow_dir, set(ids))
        log(f"[flow] streamflow files: {len(have)}/{len(ids)}; missing: {[i for i in ids if i not in have]}")

    log("done. Next: python build_headwater_parquet.py --raw", out)


if __name__ == "__main__":
    main()