# TVA observed and forecast data for HUC8 06010105 (Upper French Broad)

This folder contains the original TVA NetCDF files for the TVA forecast nodes in
HUC8 06010105. The nodes are the ones linked to the 12 CAMELSH gauges selected in
[`Example/Correctmap.ipynb`](../../Correctmap.ipynb). The files are
**byte-for-byte copies** of the source files, in their original format; nothing
has been resampled or renamed. There are three kinds of data:

| Data | Collection(s) | Variable `long_name` | Units |
|---|---|---|---|
| Observed streamflow | `Observed_1H` | `River Flow` | CMS (m³ s⁻¹) |
| Observed stage | `Observed_1H` | `River Stage` | M |
| Observed precipitation (QPE → basin MAP) | `Observed_1H` | `Precipitation Areal Mean` | MM |
| Observed local inflow | `Observed_6H` | `Estimated Local Flow` | CMS |
| Forecast streamflow | `Forecast_6H`, `Forecast_1H` | `Adjusted River Flow`, `Simulated Total Flow…`, `Sum Routed Flow`, `Routed Flow` | CMS |
| **Forecast precipitation** | `Forecast_6H` | `Precipitation Areal Mean` (`MergeMAP_FrenchBroad_Forecast_MAP_[]_main`) | MM |

`variable_inventory.csv` lists every variable in every file and assigns it a
`role` (`observed_flow`, `observed_precipitation`, `observed_stage`,
`forecast_flow`, `forecast_precipitation`). Use it to select variables by role,
because the variable names change from node to node (for example
`ADJUSTQ_AVLN7_Forecast_QR_["Adj"]_main`).

## Layout

```text
TVA/
  Observed_1H/<node>.nc   9 files  hourly observations (flow, stage, precipitation)
  Observed_6H/<node>.nc   4 files  6-hourly estimated local flow (2018-02 → 2026-08)
  Forecast_6H/<node>.nc   9 files  forecast flow + forecast precipitation, 56 leads × 6 h (0–330 h)
  Forecast_1H/<node>.nc   1 file   NWPT1 only, routed flow, 336 leads × 1 h (0–335 h)
  gauges.csv              the 12 CAMELSH gauges → TVA node crosswalk
  tva_nodes.csv           the 9 in-basin TVA nodes, linked gauge, collection/role availability
  file_inventory.csv      per-file time dimension, record count, start/end, lead times
  variable_inventory.csv  per-file variables, long_name, units, role, dims
  file_manifest.csv       source path, destination, size, SHA-256, copy status
  manifest.json           run summary (source root, counts, total bytes)
  clone_tva_data.py       script that selects and copies the data
```

The total size is about 41 MB.

## Gauge ↔ TVA node crosswalk

Eight of the 12 gauges have TVA data. Each TVA NetCDF has a `usgs_id` global
attribute, which is used for the match when it is present. `IVYN7` and `FLCN7`
have no `usgs_id`, so they are matched to the gauge within 500 m.

| USGS gauge | Name | TVA node | Match | Obs Q | Obs P | Fcst Q | Fcst P |
|---|---|---|---|:-:|:-:|:-:|:-:|
| 03455000 | French Broad nr Newport, TN | NWPT1 | usgs_id | ✓ | ✓ | ✓ (6H + 1H) | ✓ |
| 03453500 | French Broad at Marshall | MARN7 | usgs_id | ✓ | ✓ | ✓ | ✓ |
| 03453000 | Ivy Creek nr Marshall | IVYN7 | co-located (0 m) | – | ✓ | ✓ | ✓ |
| 03451500 | French Broad at Asheville | AVLN7 | usgs_id | ✓ | ✓ | ✓ | ✓ |
| 03451000 | Swannanoa at Biltmore | BLTN7 | usgs_id | ✓ | ✓ | ✓ | ✓ |
| 03447687 | French Broad nr Fletcher | FLCN7 | co-located (444 m) | – | ✓ | ✓ | ✓ |
| 03443000 | French Broad at Blantyre | BLAN7 | usgs_id | ✓ | ✓ | ✓ | ✓ |
| 03439000 | French Broad at Rosman | ROSN7 | usgs_id | ✓ | ✓ | ✓ | ✓ |
| 0344894205 | N Fk Swannanoa nr Walkertown | — | none (nearest BLTN7, 23 km) | | | | |
| 03450000 | Beetree Creek nr Swannanoa | — | none (nearest BLTN7, 16 km) | | | | |
| 03446000 | Mills River nr Mills River | — | none (nearest FLCN7, 5 km) | | | | |
| 03441000 | Davidson River nr Brevard | — | none (nearest BLAN7, 8 km) | | | | |

`HSPN7` (French Broad at Hot Springs, USGS 03454500) is also in the basin, and
its files are included. It is not linked to any of the 12 gauges because
CAMELSH has no hourly file for 03454500.

## File structure

**Observed (`Observed_1H`, `Observed_6H`)**: one dimension, `time`, with a 1-D
series per variable. The `Observed_1H` time axis starts in 1869, but that early
part is mostly precipitation. River flow and stage start in 1985 (at AVLN7, for
example). The axis ends on 2026-08-15. Observations are not on a continuous
grid, so drop NaNs or reindex before resampling.

**Forecast (`Forecast_6H`, `Forecast_1H`)**: variables have dims
`(forecast_time, ensembleId, ensembleMemberId, lead_time)`. Both ensemble
dimensions have size 1, so these are deterministic forecasts.
`forecast_time` is the issue (reference) time, from 2016-10-05 to 2026-08-21.
Issues are not evenly spaced. `lead_time` is a timedelta. The valid time is
`forecast_time + lead_time`.

```python
import xarray as xr, pandas as pd

ds = xr.open_dataset("Forecast_6H/AVLN7.nc")
roles = pd.read_csv("variable_inventory.csv").query("node_id == 'AVLN7' and collection == 'Forecast_6H'")
p_name = roles.query("role == 'forecast_precipitation'")["variable"].item()
q_name = roles.query("long_name == 'Adjusted River Flow'")["variable"].item()

fcst_p = ds[p_name].squeeze(drop=True)   # (forecast_time, lead_time), mm per 6 h step
fcst_q = ds[q_name].squeeze(drop=True)   # (forecast_time, lead_time), m³/s
valid_time = ds["forecast_time"] + ds["lead_time"]

obs = xr.open_dataset("Observed_1H/AVLN7.nc")
obs_q = obs[[v for v in obs.data_vars if obs[v].attrs.get("long_name") == "River Flow"][0]]
```

Global attributes of every file carry `station_id`, `station_name`,
`usgs_id` (not zero-padded, may be absent), station lat/lon,
`time_coverage_start/end` and `time_frequency`.

## Notes and caveats

- The forecast precipitation variable has the same name in every file
  (`MergeMAP_FrenchBroad_…`), but its values differ by node. For example, the
  mean is 0.23 mm per step at AVLN7 and 0.36 at ROSN7, and correlations between
  nodes range from 0.75 to 0.97. It is the mean areal precipitation for each
  node's local area.
- Observed_6H `Estimated Local Flow` is the local (incremental) inflow for the
  node, not the total river flow.
- `IVYN7` and `FLCN7` have no observed river flow. For those gauges, use the
  CAMELSH series in `Example/TVA/Streamflow_hourly/`.
- The time zone is not stated in the files. Before comparing with CAMELSH
  (UTC) at the hourly scale, check it against the paired gauges.

## Reproduce / refresh

```bash
cd Example/huc8_06010105/TVA
python clone_tva_data.py --dry-run      # show crosswalk and copy plan
python clone_tva_data.py                # copy + write inventories (idempotent)
python clone_tva_data.py --overwrite    # replace files if the TVA source changed
```

By default, the source is
`/Users/xnt/Documents/MODEL/TVA_InflowForecast/Data/TVA_Data_new`. Override it
with `--tva-root` or the `HYDROORBIT_TVA_ROOT` environment variable. The script
selects nodes with the same rule as the notebook: TVA nodes from the screening
availability table that lie inside `data2/real/huc8.geojson` and have data. The
12 gauge IDs are fixed in `SELECTED_GAUGES`, because the notebook picks them
with a live USGS query that can change over time.
