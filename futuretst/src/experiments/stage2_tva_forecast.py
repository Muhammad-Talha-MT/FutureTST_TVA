"""
Stage 2: use TVA's FORECAST rain at TVA's actual issue times.

  --mode eval      score the stage-1 checkpoint (no fine-tuning) on the test period
  --mode finetune  fine-tune the stage-1 checkpoint on the TVA forecast archive
                   (issues Oct 2016 - 2018), select by validation (2019), then score on the test period

Time alignment (checked against the data): an issue at time T has history steps up to the window ENDING at T.
Forecast lead k (valid time T + 6k h) is the 6 h window ending at T + 6k h, i.e. forecast step j = k - 1
(0-based), and the TVA flow forecast at lead k is compared with the observed flow of that same step.
Future rain: TVA forecast precipitation for the first --rain_lead steps (where finite), 0 mm + "unavailable" after.

Scores are computed on the SAME windows for every method: the observed flow must exist, the TVA forecast flow
must exist, and the last observed flow (for the persistence baseline) must exist.

Columns of the report:  NSE_stage1 / NSE_finetuned : our model driven by TVA forecast rain
                        NSE_stage1_obsrain         : stage 1 driven by OBSERVED rain (upper bound, what stage 1 reported)
                        NSE_TVA                    : TVA's own forecast flow (river_flow)
                        NSE_persistence            : last observed flow held constant

Run from the futuretst/ directory:
  PYTHONPATH=./ python3 src/experiments/stage2_tva_forecast.py --mode eval \
      --npz ../data_processing/data/dataset_6h.npz --tva_dir /path/TVA_headwater --stage1_dir results/pretrain_6h_full
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.models.FutureTST import FutureTST
from src.experiments.pretrain_6h import load_data, make_batch

STEPS = [1, 2, 4, 8, 12, 20, 28, 40]


def nse(o, p):
    return 1 - np.sum((p - o) ** 2) / max(np.sum((o - o.mean()) ** 2), 1e-12)


def build_model(cfg, c_in, device):
    return FutureTST(context_window_size=cfg["window"], patch_size=cfg["patch_size"], stride_len=cfg["patch_stride"],
                     d_model=cfg["d_model"], num_transformer_layers=cfg["num_layers"], mlp_size=cfg["mlp_size"],
                     num_heads=cfg["num_heads"], mlp_dropout=cfg["mlp_dropout"], pred_size=cfg["pred_len"],
                     embedding_dropout=cfg["embedding_dropout"], input_channels=c_in).to(device)


def load_forecast_table(tva_dir, data, P, R):
    """One row per (node, TVA issue time) that lies on the 6-hourly axis."""
    names, times = data["basin_names"], data["times"]
    rm, rs = float(data["rain_mean"]), float(data["rain_std"])
    out = {k: [] for k in ("b", "t", "Rz", "Av", "Qf", "Rmm")}
    for b, n in enumerate(names):
        with xr.open_dataset(os.path.join(tva_dir, "forecast", f"{n}.nc")) as ds:
            ft = pd.DatetimeIndex(ds.forecast_time.values)
            lv = ds.lead_time.values   # stored as integer hours (units: hours); also accept timedelta64
            lead_h = (np.rint(lv / np.timedelta64(1, "h")) if np.issubdtype(lv.dtype, np.timedelta64) else lv).astype(int)
            pr = np.asarray(ds.precipitation.values, float)
            qf = np.asarray(ds.river_flow.values, float) if "river_flow" in ds else np.full(pr.shape, np.nan)
        assert pr.ndim == 2, f"{n}: unexpected precipitation dims {pr.shape}"
        col = {h: i for i, h in enumerate(lead_h)}
        cols = [col[6 * (j + 1)] for j in range(P)]
        t = np.searchsorted(times, ft.values.astype("datetime64[ns]"))
        ok = (t < len(times)) & (times[np.minimum(t, len(times) - 1)] == ft.values.astype("datetime64[ns]")) & ~ft.duplicated()
        pr, qf, t = pr[ok][:, cols], qf[ok][:, cols], t[ok]
        av = np.isfinite(pr) & (np.arange(P)[None, :] < R)
        rmm = np.where(av, pr, 0.0)
        out["b"].append(np.full(len(t), b, np.int64)); out["t"].append(t.astype(np.int64))
        out["Rz"].append(((rmm - rm) / rs).astype(np.float32)); out["Av"].append(av.astype(np.float32))
        out["Qf"].append(qf.astype(np.float32)); out["Rmm"].append(rmm.astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def select(tab, data, lo, hi, W, P, min_hist, min_tgt):
    times = data["times"]; n = len(times)
    lo_i = 0 if lo is None else int(np.searchsorted(times, np.datetime64(pd.Timestamp(lo))))
    hi_i = n - 1 if hi is None else int(np.searchsorted(times, np.datetime64(pd.Timestamp(hi)), side="right")) - 1
    b, t = tab["b"], tab["t"]
    ok = (t >= max(lo_i, W)) & (t + P - 1 <= hi_i)
    tt = np.where(ok, t, W)
    B = data["y_obs_mask"].shape[0]
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(data["y_obs_mask"], axis=1)], axis=1)
    cr = np.concatenate([np.zeros((B, 1)), np.cumsum(1 - data["rain_ok"], axis=1)], axis=1)
    ok &= ((cs[b, tt] - cs[b, tt - W]) >= min_hist * W) & ((cs[b, tt + P] - cs[b, tt]) >= min_tgt * P) \
          & ((cr[b, tt] - cr[b, tt - W]) == 0)
    return {k: v[ok] for k, v in tab.items()}


def batch_fc(data, S, ix, W, P, R, device, oracle=False):
    x, tgt, m = make_batch(data, S["b"][ix], S["t"][ix], W, P, R, device)
    if not oracle:
        x[:, W:, 0] = torch.from_numpy(S["Rz"][ix]).to(device)
        x[:, W:, 1] = torch.from_numpy(S["Av"][ix]).to(device)
    return x, tgt, m


@torch.no_grad()
def predict(model, data, S, cfg, device, bs, oracle=False):
    model.eval(); out = []
    for i in range(0, len(S["b"]), bs):
        ix = np.arange(i, min(i + bs, len(S["b"])))
        x, _, _ = batch_fc(data, S, ix, cfg["window"], cfg["pred_len"], cfg["rain_lead"], device, oracle)
        out.append(model(x).float().squeeze(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def val_loss(model, data, S, cfg, device, bs):
    model.eval(); tot = cnt = 0.0
    for i in range(0, len(S["b"]), bs):
        ix = np.arange(i, min(i + bs, len(S["b"])))
        x, tgt, m = batch_fc(data, S, ix, cfg["window"], cfg["pred_len"], cfg["rain_lead"], device)
        p = model(x).float().squeeze(-1)
        tot += float((((p - tgt) ** 2) * m).sum()); cnt += float(m.sum())
    return tot / max(cnt, 1.0)


def score(data, S, preds, P, min_n=30, require_tva=True):
    names = data["basin_names"]; b, t = S["b"], S["t"]
    mu, sd = data["y_mean"][b], data["y_std"][b]
    tc = t[:, None] + np.arange(P)[None, :]
    obs = data["y"][b[:, None], tc] * sd[:, None] + mu[:, None]
    msk = data["y_obs_mask"][b[:, None], tc].astype(bool)
    last = data["y"][b, t - 1] * sd + mu; lok = data["y_obs_mask"][b, t - 1].astype(bool)
    series = {k: v * sd[:, None] + mu[:, None] for k, v in preds.items()}
    if require_tva:
        series["TVA"] = S["Qf"].astype(float)
    rows = []
    for bb in np.unique(b):
        s = b == bb
        for k in STEPS:
            if k > P:
                continue
            ok = msk[s, k - 1] & lok[s]
            if require_tva:
                ok = ok & np.isfinite(S["Qf"][s, k - 1])
            if ok.sum() < min_n:
                continue
            o = obs[s, k - 1][ok]
            r = dict(basin=names[bb], lead_h=6 * k, n=int(ok.sum()), NSE_persistence=nse(o, last[s][ok]))
            for name, arr in series.items():
                p = arr[s, k - 1][ok]
                r["NSE_" + name] = nse(o, p); r["RMSE_" + name] = float(np.sqrt(np.mean((p - o) ** 2)))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "finetune"], default="eval")
    ap.add_argument("--npz", default="../data_processing/data/dataset_6h.npz")
    ap.add_argument("--tva_dir", required=True)
    ap.add_argument("--stage1_dir", default="results/pretrain_6h_full")
    ap.add_argument("--out_dir", default="results/stage2")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ft_start", default="2016-10-01 00:00:00"); ap.add_argument("--ft_end", default="2018-12-31 18:00:00")
    ap.add_argument("--val_start", default="2019-01-01 00:00:00"); ap.add_argument("--val_end", default="2019-12-31 18:00:00")
    ap.add_argument("--test_start", default="2020-01-01 00:00:00"); ap.add_argument("--test_end", default="2025-12-31 18:00:00")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--rain_lead", type=int, default=None,
                    help="override the number of future steps with usable rain (default: the value stage 1 was trained with)")
    ap.add_argument("--extra_ckpt", nargs="*", default=[],
                    help="score additional saved checkpoints without retraining, as label=path (e.g. ft_val=results/stage2_ft/best_model.pt)")
    ap.add_argument("--save_preds", default=None,
                    help="write per-window predictions (all leads, m3/s) for the plotting notebook, e.g. results/stage2_eval/export.npz")
    ap.add_argument("--fixed_epochs", type=int, default=0,
                    help=">0: train exactly this many epochs with no validation (use with --ft_end 2019-12-31 18:00:00)")
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr_min", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=5e-4); ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--max_windows", type=int, default=0, help="debug: subsample windows in every split")
    a = ap.parse_args()

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True)
        open(os.path.join(a.out_dir, "output.log"), "a").write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")

    cfg = json.load(open(os.path.join(a.stage1_dir, "args.json")))
    if a.rain_lead is not None:
        cfg["rain_lead"] = a.rain_lead
    W, P, R = cfg["window"], cfg["pred_len"], cfg["rain_lead"]
    print(f"rain reach: {R} steps ({6 * R} h)")
    data = load_data(a.npz)
    c_in = data["static"].shape[1] + 4
    tab = load_forecast_table(a.tva_dir, data, P, R)
    log(f"{len(data['basin_names'])} nodes, {len(tab['b'])} TVA issues on the 6-hourly axis, device={device}")

    sel = lambda lo, hi: select(tab, data, lo, hi, W, P, cfg["min_hist_frac"], cfg["min_tgt_frac"])
    def sub(S):
        if not a.max_windows or len(S["b"]) <= a.max_windows:
            return S
        ix = np.sort(rng.choice(len(S["b"]), a.max_windows, replace=False))   # one draw, applied to every array
        return {k: v[ix] for k, v in S.items()}
    te = sub(sel(a.test_start, a.test_end)); va = sub(sel(a.val_start, a.val_end)); ft = sub(sel(a.ft_start, a.ft_end))
    log(f"windows: fine-tune={len(ft['b'])} val={len(va['b'])} test={len(te['b'])}")

    # alignment diagnostic: forecast step 0 should look like the observed step t (rain) and flow at step t
    tt, bb = te["t"], te["b"]
    rz = data["rain"][bb, tt] * float(data["rain_std"]) + float(data["rain_mean"])
    q0 = data["y"][bb, tt] * data["y_std"][bb] + data["y_mean"][bb]; ok0 = data["y_obs_mask"][bb, tt].astype(bool) & np.isfinite(te["Qf"][:, 0])
    log(f"alignment check on test windows: corr(TVA forecast rain lead 6h, observed rain same step) = "
        f"{np.corrcoef(rz, te['Rmm'][:, 0])[0, 1]:.3f}; corr(TVA forecast flow lead 6h, observed flow same step) = "
        f"{np.corrcoef(q0[ok0], te['Qf'][ok0, 0])[0, 1]:.3f}   (expect about 0.8 and 0.95+)")

    model = build_model(cfg, c_in, device)
    model.load_state_dict(torch.load(os.path.join(a.stage1_dir, "best_model.pt"), map_location=device))
    preds = {"stage1": predict(model, data, te, cfg, device, a.eval_batch_size),
             "stage1_obsrain": predict(model, data, te, cfg, device, a.eval_batch_size, oracle=True)}

    for item in a.extra_ckpt:
        label, path = item.split("=", 1)
        m2 = build_model(cfg, c_in, device)
        m2.load_state_dict(torch.load(path, map_location=device))
        preds[label] = predict(m2, data, te, cfg, device, a.eval_batch_size)
        log(f"scored extra checkpoint '{label}' from {path}")
        del m2

    if a.mode == "finetune":
        fixed = a.fixed_epochs > 0      # fixed schedule: no validation data, train exactly N epochs, keep the last weights
        if len(ft["b"]) == 0 or (not fixed and len(va["b"]) == 0):
            raise SystemExit("no fine-tune / validation windows")
        n_ep = a.fixed_epochs if fixed else a.epochs
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=a.lr_min)
        best_path = os.path.join(a.out_dir, "best_model.pt"); best, bad = float("inf"), 0
        if fixed:
            log(f"FIXED schedule: {n_ep} epochs on issues {a.ft_start} .. {a.ft_end}, no validation, last weights are used")
        else:
            best = val_loss(model, data, va, cfg, device, a.eval_batch_size)
            log(f"validation loss before fine-tuning (stage 1, TVA forecast rain): {best:.5f}")
            torch.save(model.state_dict(), best_path)
        for ep in range(n_ep):
            if not fixed and bad >= a.patience:
                log(f"early stopping: no improvement for {a.patience} epochs"); break
            t0 = time.time(); model.train(); perm = rng.permutation(len(ft["b"])); ls = []
            for i in range(0, len(perm), a.batch_size):
                ix = perm[i:i + a.batch_size]
                x, tgt, m = batch_fc(data, ft, ix, W, P, R, device)
                opt.zero_grad(set_to_none=True)
                loss = (((model(x).float().squeeze(-1) - tgt) ** 2) * m).sum() / m.sum().clamp(min=1.0)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); ls.append(loss.item())
            sched.step()
            if fixed:
                log(f"epoch {ep + 1}/{n_ep} train={np.mean(ls):.5f} {time.time() - t0:.0f}s")
                continue
            v = val_loss(model, data, va, cfg, device, a.eval_batch_size)
            if v < best:
                best, bad = v, 0; torch.save(model.state_dict(), best_path); tag = "  *best*"
            else:
                bad += 1; tag = f"  (no improvement {bad}/{a.patience})"
            log(f"epoch {ep + 1}/{n_ep} train={np.mean(ls):.5f} val={v:.5f}{tag} {time.time() - t0:.0f}s")
        if fixed:
            torch.save(model.state_dict(), best_path)
        else:
            model.load_state_dict(torch.load(best_path, map_location=device)); log(f"best validation loss {best:.5f}")
        preds["finetuned"] = predict(model, data, te, cfg, device, a.eval_batch_size)

    if a.save_preds:
        b, t = te["b"], te["t"]; mu, sd = data["y_mean"][b], data["y_std"][b]; tc = t[:, None] + np.arange(P)[None, :]
        obs = np.where(data["y_obs_mask"][b[:, None], tc] == 1, data["y"][b[:, None], tc] * sd[:, None] + mu[:, None], np.nan)
        last = np.where(data["y_obs_mask"][b, t - 1] == 1, data["y"][b, t - 1] * sd + mu, np.nan)
        full = np.where(data["y_obs_mask"] == 1, data["y"] * data["y_std"][:, None] + data["y_mean"][:, None], np.nan)
        out = dict(node_idx=b, node_names=np.array(data["basin_names"]), issue_time=data["times"][t], step_hours=6,
                   obs=obs.astype(np.float32), last_obs=last.astype(np.float32), tva=te["Qf"].astype(np.float32),
                   axis_times=data["times"], obs_full=full.astype(np.float32))
        for k, v in preds.items():
            out["pred_" + k] = (v * sd[:, None] + mu[:, None]).astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(a.save_preds)), exist_ok=True)
        np.savez_compressed(a.save_preds, **out)
        log(f"saved per-window predictions for the notebook: {a.save_preds} ({len(b)} windows, models: {list(preds)})")

    tab_out = score(data, te, preds, P)
    tab_out.to_csv(os.path.join(a.out_dir, "metrics_test_by_node.csv"), index=False)
    skip = {"NSE_TVA", "NSE_persistence"}
    mcols_all = [c for c in tab_out.columns if c.startswith("NSE_") and c not in skip]
    order = ["NSE_finetuned"] + [c for c in mcols_all if c not in ("NSE_finetuned", "NSE_stage1", "NSE_stage1_obsrain")] + ["NSE_stage1", "NSE_stage1_obsrain"]
    cols = [c for c in order if c in tab_out.columns] + ["NSE_TVA", "NSE_persistence"]
    summ = tab_out.groupby("lead_h")[cols].median().round(3)
    log("test (issues 2020-2025 at TVA issue times), median NSE over nodes, identical windows for every column:\n" + summ.to_string())
    for c in [c for c in mcols_all if c != "NSE_stage1_obsrain"]:
        wins = tab_out.assign(win=tab_out[c] > tab_out["NSE_TVA"]).groupby("lead_h").win.mean().round(2)
        log(f"share of nodes where {c} beats TVA, by lead hour:\n" + wins.to_string())
    # model-only scoring: no need for TVA's forecast flow, so every node with forecast rain and observed flow is scored
    tab_m = score(data, te, preds, P, require_tva=False)
    tab_m.to_csv(os.path.join(a.out_dir, "metrics_test_by_node_model_only.csv"), index=False)
    mcols = [c for c in tab_m.columns if c.startswith("NSE_")]
    log(f"MODEL-ONLY scoring (TVA flow not required): {tab_m.basin.nunique()} nodes, median NSE over nodes:\n"
        + tab_m.groupby("lead_h")[mcols].median().round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()