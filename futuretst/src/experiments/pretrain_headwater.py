"""
FutureTST pretraining on the TVA headwater basins (CAMELSH observed data).

Setup
  * hourly data, history window (default 720 h) + forecast horizon (default 240 h)
  * exogenous channels: observed Rainf (history AND "future"), 24 static
    attributes, flow-observed mask (history only)
  * endogenous channel: observed streamflow history (future part zero-padded)
  * loss: MSE over observed target hours only (missing targets are ignored)
  * checkpoint / early stopping on VALIDATION loss (not training loss)
  * windows are indexed by forecast start time t; history may reach back
    across split boundaries (targets never do)

Run from the futuretst/ directory:
    PYTHONPATH=./ python3 src/experiments/pretrain_headwater.py \
        --npz ../data_processing/data/pretrain_headwater.npz
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.models.FutureTST import FutureTST


# ----------------------------------------------------------------- data
def load_data(path):
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["basin_names"] = [str(b) for b in data["basin_names"]]
    return data


def time_index(times, ts, side="left"):
    return int(np.searchsorted(times, np.datetime64(pd.Timestamp(ts)), side=side))


def build_index(data, lo, hi, window, pred_len, stride, min_hist_frac, min_tgt_frac):
    """Return (basin_idx, t_idx) arrays of valid forecast windows.

    t is the first forecast hour. The history is [t-window, t), the target is
    [t, t+pred_len). Targets must lie inside [lo, hi]; history may start earlier.
    """
    times = data["times"]
    n = len(times)
    lo_i = 0 if lo is None else time_index(times, lo)
    hi_i = n - 1 if hi is None else time_index(times, hi, side="right") - 1
    t_first = max(lo_i, window)
    t_last = hi_i - pred_len + 1
    if t_last < t_first:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # align issue times to the clock (e.g. stride 24 -> 00 UTC, stride 6 -> 00/06/12/18)
    if stride <= 24 and 24 % stride == 0:
        hours = pd.DatetimeIndex(times[t_first:t_first + 24]).hour.values
        shift = int(np.flatnonzero(hours % stride == 0)[0])
        t_first += shift
    ts = np.arange(t_first, t_last + 1, stride)

    cs = np.concatenate([np.zeros((data["y_obs_mask"].shape[0], 1)),
                         np.cumsum(data["y_obs_mask"], axis=1)], axis=1)
    bs, tt = [], []
    for b in range(cs.shape[0]):
        hist = cs[b, ts] - cs[b, ts - window]
        tgt = cs[b, ts + pred_len] - cs[b, ts]
        ok = (hist >= min_hist_frac * window) & (tgt >= min_tgt_frac * pred_len)
        bs.append(np.full(ok.sum(), b, dtype=np.int64))
        tt.append(ts[ok].astype(np.int64))
    return np.concatenate(bs), np.concatenate(tt)


def make_batch(data, bidx, tidx, window, pred_len, device):
    """Assemble one batch on the CPU with fancy indexing, then move to device.

    Channels of x: [Rainf, statics(S), hist_flow_observed_mask, Y]  -> C = S + 3
    x has T = window + pred_len time steps; Y and the mask are zero in the future.
    """
    B = len(bidx)
    T = window + pred_len
    cols = (tidx - window)[:, None] + np.arange(T)[None, :]            # (B, T)
    b2 = bidx[:, None]
    rain = data["rain"][b2, cols]                                       # (B, T)
    mask_h = np.zeros((B, T), dtype=np.float32)
    y_h = np.zeros((B, T), dtype=np.float32)
    mask_h[:, :window] = data["y_obs_mask"][b2, cols[:, :window]]
    y_h[:, :window] = data["y"][b2, cols[:, :window]]
    tcols = tidx[:, None] + np.arange(pred_len)[None, :]
    tgt = data["y"][b2, tcols]
    tmask = data["y_obs_mask"][b2, tcols].astype(np.float32)
    static = data["static"][bidx]

    rain = torch.from_numpy(rain).to(device)
    mask_h = torch.from_numpy(mask_h).to(device)
    y_h = torch.from_numpy(y_h).to(device)
    static = torch.from_numpy(static).to(device)
    x = torch.cat([rain.unsqueeze(-1),
                   static.unsqueeze(1).expand(-1, T, -1),
                   mask_h.unsqueeze(-1),
                   y_h.unsqueeze(-1)], dim=-1)
    return x, torch.from_numpy(tgt).to(device), torch.from_numpy(tmask).to(device)


def masked_mse(pred, tgt, m):
    return (((pred.squeeze(-1) - tgt) ** 2) * m).sum() / m.sum().clamp(min=1.0)


# ----------------------------------------------------------- evaluation
@torch.no_grad()
def run_eval(model, data, bidx, tidx, args, device, keep_preds=False):
    model.eval()
    tot, cnt = 0.0, 0.0
    preds = []
    for i in range(0, len(bidx), args.eval_batch_size):
        x, tgt, m = make_batch(data, bidx[i:i + args.eval_batch_size],
                               tidx[i:i + args.eval_batch_size],
                               args.window, args.pred_len, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            p = model(x).float()
        tot += float((((p.squeeze(-1) - tgt) ** 2) * m).sum())
        cnt += float(m.sum())
        if keep_preds:
            preds.append(p.squeeze(-1).cpu().numpy())
    loss = tot / max(cnt, 1.0)
    return (loss, np.concatenate(preds, axis=0)) if keep_preds else loss


def skill_table(data, bidx, tidx, preds, pred_len, leads):
    """Per-basin NSE / KGE / RMSE at selected lead hours, in original flow units."""
    rows = []
    y_mean, y_std = data["y_mean"], data["y_std"]
    tcols = tidx[:, None] + np.arange(pred_len)[None, :]
    obs_n = data["y"][bidx[:, None], tcols]
    msk = data["y_obs_mask"][bidx[:, None], tcols].astype(bool)
    for b in np.unique(bidx):
        sel = bidx == b
        mu, sd = y_mean[b], y_std[b]
        for L in leads:
            k = L - 1
            ok = msk[sel, k]
            if ok.sum() < 30:
                continue
            o = obs_n[sel, k][ok] * sd + mu
            p = preds[sel, k][ok] * sd + mu
            nse = 1 - ((p - o) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-12)
            r = np.corrcoef(o, p)[0, 1] if o.std() > 0 and p.std() > 0 else np.nan
            kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
            rows.append(dict(basin=data["basin_names"][b], lead_h=L, n=int(ok.sum()),
                             NSE=nse, KGE=kge, RMSE=float(np.sqrt(((p - o) ** 2).mean()))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../data_processing/data/pretrain_headwater.npz")
    ap.add_argument("--out_dir", default="results/pretrain_headwater")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    # windows
    ap.add_argument("--window", type=int, default=720)
    ap.add_argument("--pred_len", type=int, default=240)
    ap.add_argument("--stride_train", type=int, default=24)
    ap.add_argument("--stride_eval", type=int, default=24)
    ap.add_argument("--min_hist_frac", type=float, default=0.5,
                    help="min fraction of observed flow hours in the history window")
    ap.add_argument("--min_tgt_frac", type=float, default=0.5,
                    help="min fraction of observed flow hours in the target window")
    # periods (targets must lie inside)
    ap.add_argument("--train_start", default=None)
    ap.add_argument("--train_end", default="2018-12-31 23:00:00")
    ap.add_argument("--val_start", default="2019-01-01 00:00:00")
    ap.add_argument("--val_end", default="2019-12-31 23:00:00")
    ap.add_argument("--test_start", default="2020-01-01 00:00:00")
    ap.add_argument("--test_end", default=None)
    # optimisation
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_train_windows", type=int, default=0, help="debug: subsample training windows")
    # model
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--patch_stride", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--mlp_size", type=int, default=128)
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--embedding_dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if str(device) != args.device:
        print(f"WARNING: {args.device} not available, using {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "output.log")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")

    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    data = load_data(args.npz)
    n_basins = len(data["basin_names"])
    log(f"data: {n_basins} basins, {len(data['times'])} hourly steps, "
        f"statics={data['static'].shape[1]}, scaling train_end={data['scale_train_end']}")
    if str(data["scale_train_end"]) != args.train_end:
        log(f"WARNING: scaling used train_end={data['scale_train_end']} but --train_end={args.train_end}")

    kw = dict(window=args.window, pred_len=args.pred_len,
              min_hist_frac=args.min_hist_frac, min_tgt_frac=args.min_tgt_frac)
    tr_b, tr_t = build_index(data, args.train_start, args.train_end, stride=args.stride_train, **kw)
    va_b, va_t = build_index(data, args.val_start, args.val_end, stride=args.stride_eval, **kw)
    te_b, te_t = build_index(data, args.test_start, args.test_end, stride=args.stride_eval, **kw)
    log(f"windows: train={len(tr_b)}  val={len(va_b)}  test={len(te_b)}")
    if len(tr_b) == 0 or len(va_b) == 0:
        raise SystemExit("No training or validation windows - check dates / data coverage.")
    per_b = np.bincount(tr_b, minlength=n_basins)
    log("train windows per basin: " + ", ".join(f"{n}:{c}" for n, c in zip(data["basin_names"], per_b)))
    if args.max_train_windows:
        sel = rng.choice(len(tr_b), min(args.max_train_windows, len(tr_b)), replace=False)
        tr_b, tr_t = tr_b[sel], tr_t[sel]
        log(f"DEBUG: using {len(tr_b)} training windows")

    C = data["static"].shape[1] + 3
    model = FutureTST(
        context_window_size=args.window, patch_size=args.patch_size, stride_len=args.patch_stride,
        d_model=args.d_model, num_transformer_layers=args.num_layers, mlp_size=args.mlp_size,
        num_heads=args.num_heads, mlp_dropout=args.mlp_dropout, pred_size=args.pred_len,
        embedding_dropout=args.embedding_dropout, input_channels=C).to(device)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()):,}  (input channels C={C})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)

    best_path = os.path.join(args.out_dir, "best_model.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    start_epoch, best_val, bad = 0, float("inf"), 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_val, bad = ck["epoch"], ck["best_val"], ck["bad"]
        log(f"resumed from epoch {start_epoch}, best_val={best_val:.5f}")

    for epoch in range(start_epoch, args.epochs):
        if bad >= args.patience:
            log(f"early stopping: no validation improvement for {args.patience} epochs")
            break
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(tr_b))
        losses = []
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, tgt, m = make_batch(data, tr_b[idx], tr_t[idx], args.window, args.pred_len, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                pred = model(x)
            loss = masked_mse(pred.float(), tgt, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = run_eval(model, data, va_b, va_t, args, device)
        improved = val < best_val
        if improved:
            best_val, bad = val, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
        log(f"epoch {epoch + 1}/{args.epochs}  train_loss={np.mean(losses):.5f}  val_loss={val:.5f}"
            f"{'  *best*' if improved else f'  (no improvement {bad}/{args.patience})'}"
            f"  lr={sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best_val": best_val, "bad": bad}, last_path)

    # ------------------------------------------------------------ test
    if not os.path.exists(best_path):
        raise SystemExit("No best model saved.")
    model.load_state_dict(torch.load(best_path, map_location=device))
    log(f"best validation loss: {best_val:.5f}")
    leads = [L for L in (1, 6, 12, 24, 48, 72, 120, 168, 240) if L <= args.pred_len]
    for name, (bb, tt) in {"val": (va_b, va_t), "test": (te_b, te_t)}.items():
        if len(bb) == 0:
            log(f"{name}: no windows (period outside data range)")
            continue
        loss, preds = run_eval(model, data, bb, tt, args, device, keep_preds=True)
        tab = skill_table(data, bb, tt, preds, args.pred_len, leads)
        tab.to_csv(os.path.join(args.out_dir, f"metrics_{name}.csv"), index=False)
        np.savez(os.path.join(args.out_dir, f"predictions_{name}.npz"),
                 preds=preds.astype(np.float32), basin_idx=bb, t_idx=tt,
                 issue_time=data["times"][tt], basin_names=np.array(data["basin_names"]),
                 y_mean=data["y_mean"], y_std=data["y_std"])
        log(f"{name}: masked MSE (normalized) = {loss:.5f}, windows={len(bb)}")
        if len(tab):
            summ = tab.groupby("lead_h")[["NSE", "KGE", "RMSE"]].median().round(3)
            log(f"{name}: median across basins by lead hour\n{summ.to_string()}")
    log("done")


if __name__ == "__main__":
    main()
