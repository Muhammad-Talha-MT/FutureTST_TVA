"""
Stage 1: pretrain FutureTST at 6-hourly resolution on the TVA headwater nodes.

  * history  : --window steps (default 120 = 30 days) of TVA observed rain + observed flow (with an observed-mask)
  * horizon  : --pred_len steps (default 40 = 10 days, lead 6 h ... 240 h)
  * future rain: observed rain for the first --rain_lead steps (default 20 = 120 h), then 0 mm with the
                 "rain available" channel set to 0. This mimics TVA's forecast, whose rain stops after ~5 days.
  * statics  : StreamCat attributes + log area (built by data_processing/build_6h_dataset.py)
  * loss     : MSE on observed target steps only; checkpoint / early stopping on VALIDATION loss
  * windows  : indexed by forecast start step t; history may reach back across split boundaries, targets never do
  * reports  : NSE / KGE / RMSE by lead time in m3/s for validation and test, next to a persistence baseline

Run from the futuretst/ directory:
    PYTHONPATH=./ python3 src/experiments/pretrain_6h.py --npz ../data_processing/data/dataset_6h.npz
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.models.FutureTST import FutureTST


def load_data(path):
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["basin_names"] = [str(b) for b in data["basin_names"]]
    return data


def tidx(times, ts, side="left"):
    return int(np.searchsorted(times, np.datetime64(pd.Timestamp(ts)), side=side))


def build_index(data, lo, hi, W, P, R, stride, min_hist, min_tgt):
    """Valid (basin, t) pairs. t = first forecast step; history [t-W, t); targets [t, t+P) inside [lo, hi]."""
    times = data["times"]
    n = len(times)
    lo_i = 0 if lo is None else tidx(times, lo)
    hi_i = n - 1 if hi is None else tidx(times, hi, "right") - 1
    t_first, t_last = max(lo_i, W), hi_i - P + 1
    if t_last < t_first:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    span_h = stride * 6
    if 24 % span_h == 0:  # align issue times to the clock: stride 4 -> 00 UTC, stride 2 -> 00/12 UTC, stride 1 -> 00/06/12/18
        hrs = pd.DatetimeIndex(times[t_first:t_first + 24]).hour.values
        t_first += int(np.flatnonzero(hrs % span_h == 0)[0])
    ts = np.arange(t_first, t_last + 1, stride)
    B = data["y_obs_mask"].shape[0]
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(data["y_obs_mask"], axis=1)], axis=1)
    cr = np.concatenate([np.zeros((B, 1)), np.cumsum(1 - data["rain_ok"], axis=1)], axis=1)   # missing-rain counter
    bs, tt = [], []
    for b in range(B):
        ok = ((cs[b, ts] - cs[b, ts - W]) >= min_hist * W) & ((cs[b, ts + P] - cs[b, ts]) >= min_tgt * P) \
             & ((cr[b, ts + R] - cr[b, ts - W]) == 0)
        bs.append(np.full(ok.sum(), b, np.int64)); tt.append(ts[ok].astype(np.int64))
    return np.concatenate(bs), np.concatenate(tt)


def make_batch(data, bidx, tix, W, P, R, device):
    """x channels: [rain, rain_available, statics(S), hist_flow_observed_mask, Y];  T = W + P steps."""
    B, T = len(bidx), W + P
    cols = (tix - W)[:, None] + np.arange(T)[None, :]
    b2 = bidx[:, None]
    rain = data["rain"][b2, cols].copy()
    avail = np.ones((B, T), np.float32)
    avail[:, W + R:] = 0.0
    rain[:, W + R:] = data["rain_zero_std"]          # beyond the forecast reach: 0 mm, flagged unavailable
    mask_h = np.zeros((B, T), np.float32); y_h = np.zeros((B, T), np.float32)
    mask_h[:, :W] = data["y_obs_mask"][b2, cols[:, :W]]
    y_h[:, :W] = data["y"][b2, cols[:, :W]]
    tc = tix[:, None] + np.arange(P)[None, :]
    tgt = data["y"][b2, tc]; tm = data["y_obs_mask"][b2, tc].astype(np.float32)
    st = torch.from_numpy(data["static"][bidx]).to(device)
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    x = torch.cat([f(rain).unsqueeze(-1), f(avail).unsqueeze(-1), st.unsqueeze(1).expand(-1, T, -1),
                   f(mask_h).unsqueeze(-1), f(y_h).unsqueeze(-1)], dim=-1)
    return x, f(tgt), f(tm)


@torch.no_grad()
def run_eval(model, data, b, t, a, device, keep=False):
    model.eval(); tot = cnt = 0.0; out = []
    for i in range(0, len(b), a.eval_batch_size):
        x, tgt, m = make_batch(data, b[i:i + a.eval_batch_size], t[i:i + a.eval_batch_size], a.window, a.pred_len, a.rain_lead, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
            p = model(x).float().squeeze(-1)
        tot += float((((p - tgt) ** 2) * m).sum()); cnt += float(m.sum())
        if keep:
            out.append(p.cpu().numpy())
    loss = tot / max(cnt, 1.0)
    return (loss, np.concatenate(out)) if keep else loss


def skill_table(data, b, t, pred, W, P, steps):
    """Per-basin NSE/KGE/RMSE (m3/s) at selected lead steps, plus persistence (last observed flow held constant)."""
    rows = []
    tc = t[:, None] + np.arange(P)[None, :]
    obs = data["y"][b[:, None], tc]; msk = data["y_obs_mask"][b[:, None], tc].astype(bool)
    last_ok = data["y_obs_mask"][b, t - 1].astype(bool)
    last = data["y"][b, t - 1]
    for bb in np.unique(b):
        s = b == bb; mu, sd = data["y_mean"][bb], data["y_std"][bb]
        for k in steps:
            ok = msk[s, k - 1]
            if ok.sum() < 30:
                continue
            o = obs[s, k - 1][ok] * sd + mu; p = pred[s, k - 1][ok] * sd + mu
            nse = 1 - ((p - o) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-12)
            r = np.corrcoef(o, p)[0, 1] if o.std() > 0 and p.std() > 0 else np.nan
            kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
            ok2 = ok & last_ok[s]
            if ok2.sum() >= 30:
                o2 = obs[s, k - 1][ok2] * sd + mu; q2 = last[s][ok2] * sd + mu
                pers = 1 - ((q2 - o2) ** 2).sum() / max(((o2 - o2.mean()) ** 2).sum(), 1e-12)
            else:
                pers = np.nan
            rows.append(dict(basin=data["basin_names"][bb], lead_h=6 * k, n=int(ok.sum()), NSE=nse, KGE=kge,
                             RMSE=float(np.sqrt(((p - o) ** 2).mean())), NSE_persistence=pers))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../data_processing/data/dataset_6h.npz")
    ap.add_argument("--out_dir", default="results/pretrain_6h")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=int, default=120); ap.add_argument("--pred_len", type=int, default=40)
    ap.add_argument("--rain_lead", type=int, default=20, help="steps of future rain that are real (20 = 120 h)")
    ap.add_argument("--stride_train", type=int, default=4, help="steps between training windows (4 = daily)")
    ap.add_argument("--stride_eval", type=int, default=2, help="steps between eval windows (2 = 00 and 12 UTC)")
    ap.add_argument("--min_hist_frac", type=float, default=0.5); ap.add_argument("--min_tgt_frac", type=float, default=0.5)
    ap.add_argument("--train_start", default=None); ap.add_argument("--train_end", default="2018-12-31 18:00:00")
    ap.add_argument("--val_start", default="2019-01-01 00:00:00"); ap.add_argument("--val_end", default="2019-12-31 18:00:00")
    ap.add_argument("--test_start", default="2020-01-01 00:00:00"); ap.add_argument("--test_end", default="2025-12-31 18:00:00")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr_min", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=5e-4); ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_train_windows", type=int, default=0)
    ap.add_argument("--patch_size", type=int, default=16); ap.add_argument("--patch_stride", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2); ap.add_argument("--mlp_size", type=int, default=128)
    ap.add_argument("--mlp_dropout", type=float, default=0.2); ap.add_argument("--embedding_dropout", type=float, default=0.1)
    a = ap.parse_args()

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True)
        open(os.path.join(a.out_dir, "output.log"), "a").write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")

    json.dump(vars(a), open(os.path.join(a.out_dir, "args.json"), "w"), indent=2)
    data = load_data(a.npz)
    nb = len(data["basin_names"])
    log(f"{nb} nodes, {len(data['times'])} six-hourly steps, {data['static'].shape[1]} statics, device={device}")
    if str(data["scale_train_end"]) != a.train_end:
        log(f"WARNING: scaling used train_end={data['scale_train_end']} but --train_end={a.train_end}")

    kw = dict(W=a.window, P=a.pred_len, R=a.rain_lead, min_hist=a.min_hist_frac, min_tgt=a.min_tgt_frac)
    tr_b, tr_t = build_index(data, a.train_start, a.train_end, stride=a.stride_train, **kw)
    va_b, va_t = build_index(data, a.val_start, a.val_end, stride=a.stride_eval, **kw)
    te_b, te_t = build_index(data, a.test_start, a.test_end, stride=a.stride_eval, **kw)
    log(f"windows: train={len(tr_b)} val={len(va_b)} test={len(te_b)}")
    if len(tr_b) == 0 or len(va_b) == 0:
        raise SystemExit("no training/validation windows - check dates and data")
    log("train windows per node: " + ", ".join(f"{n}:{c}" for n, c in zip(data["basin_names"], np.bincount(tr_b, minlength=nb))))
    if a.max_train_windows:
        sel = rng.choice(len(tr_b), min(a.max_train_windows, len(tr_b)), replace=False); tr_b, tr_t = tr_b[sel], tr_t[sel]
        log(f"DEBUG: {len(tr_b)} training windows")

    C = data["static"].shape[1] + 4
    model = FutureTST(context_window_size=a.window, patch_size=a.patch_size, stride_len=a.patch_stride, d_model=a.d_model,
                      num_transformer_layers=a.num_layers, mlp_size=a.mlp_size, num_heads=a.num_heads, mlp_dropout=a.mlp_dropout,
                      pred_size=a.pred_len, embedding_dropout=a.embedding_dropout, input_channels=C).to(device)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()):,} (channels C={C})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr_min)
    best_path, last_path = os.path.join(a.out_dir, "best_model.pt"), os.path.join(a.out_dir, "last.pt")
    ep0, best, bad = 0, float("inf"), 0
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best, bad = ck["epoch"], ck["best_val"], ck["bad"]; log(f"resumed at epoch {ep0}, best_val={best:.5f}")

    for ep in range(ep0, a.epochs):
        if bad >= a.patience:
            log(f"early stopping: no validation improvement for {a.patience} epochs"); break
        t0 = time.time(); model.train(); perm = rng.permutation(len(tr_b)); ls = []
        for i in range(0, len(perm), a.batch_size):
            ix = perm[i:i + a.batch_size]
            x, tgt, m = make_batch(data, tr_b[ix], tr_t[ix], a.window, a.pred_len, a.rain_lead, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
                p = model(x)
            loss = (((p.float().squeeze(-1) - tgt) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); ls.append(loss.item())
        sched.step()
        val = run_eval(model, data, va_b, va_t, a, device)
        if val < best:
            best, bad = val, 0; torch.save(model.state_dict(), best_path); tag = "  *best*"
        else:
            bad += 1; tag = f"  (no improvement {bad}/{a.patience})"
        log(f"epoch {ep + 1}/{a.epochs} train={np.mean(ls):.5f} val={val:.5f}{tag} lr={sched.get_last_lr()[0]:.1e} {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "best_val": best, "bad": bad}, last_path)

    model.load_state_dict(torch.load(best_path, map_location=device)); log(f"best validation loss {best:.5f}")
    steps = [k for k in (1, 2, 4, 8, 12, 20, 28, 40) if k <= a.pred_len]
    for name, (b, t) in {"val": (va_b, va_t), "test": (te_b, te_t)}.items():
        if len(b) == 0:
            log(f"{name}: no windows"); continue
        loss, pred = run_eval(model, data, b, t, a, device, keep=True)
        tab = skill_table(data, b, t, pred, a.window, a.pred_len, steps); tab.to_csv(os.path.join(a.out_dir, f"metrics_{name}.csv"), index=False)
        np.savez(os.path.join(a.out_dir, f"predictions_{name}.npz"), preds=pred.astype(np.float32), basin_idx=b, t_idx=t,
                 issue_time=data["times"][t], basin_names=np.array(data["basin_names"]), y_mean=data["y_mean"], y_std=data["y_std"])
        log(f"{name}: masked MSE (z-units) {loss:.5f}, windows {len(b)}")
        if len(tab):
            log(f"{name}: median over nodes by lead\n" + tab.groupby("lead_h")[["NSE", "NSE_persistence", "KGE", "RMSE"]].median().round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()
