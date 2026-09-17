"""
Ablation solver for the Dual Physics Constraint (0 m prediction).

Differences from deploy_pisr.py
  * masked data losses  : L_low / L_high are evaluated only on pixels flagged as observed
  * explicit weights    : --w_low / --w_high (set 0 to drop a constraint)
  * high-constraint mode: none | derived (FFT upward of the low grid) | file (independent grid)
  * block hold-out      : a fraction of observed low-altitude blocks is removed from the loss
                          AND replaced in the input prior, then used as an independent check
  * independent check   : U_high(g0) is always compared with --indep_high on its valid pixels
  * reproducibility     : --seed fixes torch / numpy RNG

Typical configurations (see run_ablation.py):
  A single_low   : --high_mode none
  B dual_derived : --high_mode derived          (current paper setting)
  C dual_indep   : --high_mode file --file_high <filled 5156 LSC> --mask_high <unfilled 5156 LSC>
  D single_high  : as C with --w_low 0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deploy_pisr import (SimpleUNet, UpwardContinuationLayer, EarlyStopper,
                         device, extract_coastline_gmt, plot_coastline)

NA_VALUES = ['NaN', 'nan', 'NAN']


# ------------------------------------------------------------------
# Grid I/O
# ------------------------------------------------------------------
def detect_encoding(path):
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(2000)
            return enc
        except Exception:
            continue
    return 'utf-8'


def load_grid(path):
    """Load lon lat val -> (lons, lats, 2D array with NaN preserved)."""
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#', engine='python', na_values=NA_VALUES,
                     encoding=detect_encoding(path))
    df = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce')
    df.columns = ['lon', 'lat', 'val']
    df['lon'] = df['lon'].round(6)
    df['lat'] = df['lat'].round(6)
    lons = np.sort(df['lon'].unique())
    lats = np.sort(df['lat'].unique())
    piv = df.pivot_table(index='lat', columns='lon', values='val', aggfunc='mean', dropna=False)
    piv = piv.reindex(index=lats, columns=lons)
    return lons, lats, piv.values.astype(float)


def fill_nearest(grid):
    """Replace NaN by nearest valid value (used only to avoid NaN leakage in interpolation)."""
    nan = np.isnan(grid)
    if not nan.any():
        return grid.copy()
    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def resample(lons, lats, grid, size):
    """Resample to size x size on linspace(min,max) axes (same convention as deploy_pisr)."""
    if grid.shape == (size, size):
        return grid.copy()
    tl = np.linspace(lons.min(), lons.max(), size)
    ta = np.linspace(lats.min(), lats.max(), size)
    LO, LA = np.meshgrid(tl, ta)
    pts = np.stack([LA.ravel(), LO.ravel()], axis=-1)
    vals = RegularGridInterpolator((lats, lons), fill_nearest(grid), method='linear')(pts).reshape(size, size)
    valid = RegularGridInterpolator((lats, lons), (~np.isnan(grid)).astype(float), method='nearest')(pts).reshape(size, size)
    vals[valid < 0.5] = np.nan
    return vals


def save_xyz(path, lons, lats, grid, header):
    XX, YY = np.meshgrid(lons, lats)
    out = np.column_stack((XX.ravel(), YY.ravel(), grid.ravel()))
    np.savetxt(path, out, fmt='%.6f', header=header)


def make_holdout(shape, block_px, frac, seed):
    """Random block mask (True = held out)."""
    if frac <= 0 or block_px <= 0:
        return np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(seed)
    ny, nx = shape
    nby, nbx = int(np.ceil(ny / block_px)), int(np.ceil(nx / block_px))
    blocks = rng.random((nby, nbx)) < frac
    return np.kron(blocks, np.ones((block_px, block_px), dtype=bool))[:ny, :nx]


def rmse(a, b, m):
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float('nan'), 0
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d ** 2))), int(m.sum())


def grad_rms(g, m):
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.sqrt(np.mean(mag[m] ** 2))) if m.sum() else float('nan')


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Dual-constraint ablation solver (0 m prediction)")
    p.add_argument("--size", type=int, default=301)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lag", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", type=str, default="ablation/abl_")
    p.add_argument("--margin", type=int, default=20, help="ROI margin (px) excluded from statistics")
    # low altitude (input prior + shallow constraint)
    p.add_argument("--file_low", type=str, default="merged_gravity_1620m_filled.xyz", help="gap-free low grid (input prior)")
    p.add_argument("--mask_low", type=str, default="1500(10s)_new.xyg_pred_grid_0.1.xyz", help="grid whose NaNs define un-observed low pixels ('' = all observed)")
    p.add_argument("--fill_low", type=str, default="TW_EGM_grid_g_res_h1620.xyz", help="grid used (bias-corrected) to replace held-out pixels in the prior ('' = nearest inpaint)")
    p.add_argument("--obs_low", type=float, default=1620.0)
    p.add_argument("--w_low", type=float, default=1.0)
    # high altitude constraint
    p.add_argument("--high_mode", choices=["none", "derived", "file"], default="derived")
    p.add_argument("--file_high", type=str, default="5000(10s)_new.xyg_pred_grid_0.05_filled.xyz")
    p.add_argument("--mask_high", type=str, default="5000(10s)_new.xyg_pred_grid_0.05.xyz", help="grid whose NaNs define un-observed high pixels ('' = all)")
    p.add_argument("--obs_high", type=float, default=5156.0)
    p.add_argument("--w_high", type=float, default=1.0)
    # independent validation at high altitude (always evaluated, never trained on unless high_mode=file)
    p.add_argument("--indep_high", type=str, default="5000(10s)_new.xyg_pred_grid_0.05.xyz")
    # regularisation
    p.add_argument("--lam_tv", type=float, default=0.05)
    p.add_argument("--w_bg", type=float, default=0.0,
                   help="weak background anchor: w_bg * MSE(U_low(g0), prior) on pixels NOT in the low training mask "
                        "(held-out, 5156-only and EGM-filled regions). 0 = off. Prevents free drift where no data term acts.")
    # hold-out
    p.add_argument("--holdout_block", type=int, default=0, help="block size in px (0 = no hold-out)")
    p.add_argument("--holdout_frac", type=float, default=0.0)
    p.add_argument("--holdout_seed", type=int, default=12345, help="fixed so every config sees the same blocks")
    p.add_argument("--holdout_file", type=str, default="", help="load hold-out mask (lon lat 0/1) instead of generating")
    p.add_argument("--tag", type=str, default="", help="config label stored in metadata")
    # learning-curve diagnostics
    p.add_argument("--eval_every", type=int, default=0, help="every N epochs evaluate fit / hold-out / indep RMSE (0 = off)")
    p.add_argument("--no_stop", action="store_true", help="run all epochs; only record when early stopping WOULD have fired")
    args = p.parse_args()

    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    S = args.size
    # ---------------- load grids ----------------
    lons0, lats0, low_raw = load_grid(args.file_low)
    low = resample(lons0, lats0, low_raw, S)
    low = fill_nearest(low)  # gap-free by construction; safety
    lons = np.linspace(lons0.min(), lons0.max(), S)
    lats = np.linspace(lats0.min(), lats0.max(), S)
    dx_km = (lons[1] - lons[0]) * 102.0
    dy_km = (lats[1] - lats[0]) * 111.0
    print(f"Grid {S}x{S}, dx={dx_km:.3f} km, dy={dy_km:.3f} km")

    if args.mask_low:
        l0, a0, g = load_grid(args.mask_low)
        obs_low = ~np.isnan(resample(l0, a0, g, S))
    else:
        obs_low = np.ones((S, S), dtype=bool)
    print(f"Observed low pixels: {obs_low.sum()} / {S*S} ({100*obs_low.mean():.1f}%)")

    indep_high = None
    obs_indep = np.zeros((S, S), dtype=bool)
    if args.indep_high and os.path.exists(args.indep_high):
        l0, a0, g = load_grid(args.indep_high)
        indep_high = resample(l0, a0, g, S)
        obs_indep = ~np.isnan(indep_high)
        print(f"Independent high pixels: {obs_indep.sum()} ({100*obs_indep.mean():.1f}%)")

    # ---------------- hold-out ----------------
    if args.holdout_file:
        l0, a0, g = load_grid(args.holdout_file)
        holdout = resample(l0, a0, g, S) > 0.5
    else:
        holdout = make_holdout((S, S), args.holdout_block, args.holdout_frac, args.holdout_seed)
    holdout &= obs_low                       # only observed pixels can be held out
    train_low = obs_low & ~holdout
    print(f"Hold-out pixels: {holdout.sum()} ({100*holdout.sum()/max(obs_low.sum(),1):.1f}% of observed low)")

    low_obs_values = low.copy()              # keep observed values for validation
    if holdout.any():
        if args.fill_low and os.path.exists(args.fill_low):
            l0, a0, g = load_grid(args.fill_low)
            fill = fill_nearest(resample(l0, a0, g, S))
            bias = float(np.mean(low[train_low] - fill[train_low]))
            low[holdout] = fill[holdout] + bias
            print(f"Held-out prior pixels replaced by {args.fill_low} + bias {bias:.3f} mGal")
        else:
            tmp = low.copy()
            tmp[holdout] = np.nan
            low = fill_nearest(tmp)
            print("Held-out prior pixels replaced by nearest-neighbour inpainting")

    # ---------------- high target ----------------
    up_low = UpwardContinuationLayer(dx_km, dy_km, height_diff_km=args.obs_low / 1000.0).to(device)
    up_high = UpwardContinuationLayer(dx_km, dy_km, height_diff_km=args.obs_high / 1000.0).to(device)

    if args.high_mode == "derived":
        up_l2h = UpwardContinuationLayer(dx_km, dy_km, height_diff_km=(args.obs_high - args.obs_low) / 1000.0).to(device)
        with torch.no_grad():
            t = torch.tensor(low, dtype=torch.float32)[None, None].to(device)
            high = up_l2h(t).squeeze().cpu().numpy()
        obs_high = np.ones((S, S), dtype=bool)
        w_high = args.w_high
    elif args.high_mode == "file":
        l0, a0, g = load_grid(args.file_high)
        high = fill_nearest(resample(l0, a0, g, S))
        if args.mask_high:
            l0, a0, g = load_grid(args.mask_high)
            obs_high = ~np.isnan(resample(l0, a0, g, S))
        else:
            obs_high = np.ones((S, S), dtype=bool)
        w_high = args.w_high
    else:
        high = np.zeros((S, S))
        obs_high = np.zeros((S, S), dtype=bool)
        w_high = 0.0
    print(f"High mode: {args.high_mode}, observed high pixels: {obs_high.sum()}, w_low={args.w_low}, w_high={w_high}")

    # ---------------- tensors ----------------
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None, None].to(device)
    x_in = T(low)
    tgt_low = T(low)
    tgt_high = T(high)
    m_low = T(train_low.astype(np.float32))
    m_bg = T((~train_low).astype(np.float32))
    m_high = T(obs_high.astype(np.float32))

    model = SimpleUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.lag, min_delta=1e-5)

    roi = np.zeros((S, S), dtype=bool)
    mg = args.margin
    roi[mg:-mg, mg:-mg] = True

    def evaluate(pred_t):
        """eval-mode RMSEs: fit / hold-out at low, independent at high (ROI)."""
        cl = up_low(pred_t).squeeze().cpu().numpy()
        ch = up_high(pred_t).squeeze().cpu().numpy()
        r_fit, _ = rmse(cl, low_obs_values, train_low & roi)
        r_hold, _ = rmse(cl, low_obs_values, holdout & roi)
        r_ind = rmse(ch, indep_high, obs_indep & roi)[0] if indep_high is not None else float('nan')
        return r_fit, r_hold, r_ind

    hist, curve = [], []
    would_stop = None
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad()
        pred = model(x_in)
        chk_l = up_low(pred) if (args.w_low > 0 or args.w_bg > 0) else None
        l_low = masked_mse(chk_l, tgt_low, m_low) if args.w_low > 0 else torch.zeros((), device=device)
        l_bg = masked_mse(chk_l, tgt_low, m_bg) if args.w_bg > 0 else torch.zeros((), device=device)
        l_high = masked_mse(up_high(pred), tgt_high, m_high) if w_high > 0 else torch.zeros((), device=device)
        tv = torch.mean(torch.abs(pred[..., :, 1:] - pred[..., :, :-1])) + torch.mean(torch.abs(pred[..., 1:, :] - pred[..., :-1, :]))
        loss = args.w_low * l_low + w_high * l_high + args.w_bg * l_bg + args.lam_tv * tv
        loss.backward()
        opt.step()

        stab = 0.0
        if ep >= args.lag and hist[ep - args.lag][1] and loss.item() != 0:
            stab = 1.0 - abs(loss.item() - hist[ep - args.lag][1]) / loss.item()
        hist.append([ep, loss.item(), l_low.item(), l_high.item(), l_bg.item(), tv.item(), stab])
        if ep % 500 == 0:
            print(f"Epoch {ep}: total={loss.item():.5f} L_low={l_low.item():.5f} L_high={l_high.item():.5f} L_bg={l_bg.item():.5f} TV={tv.item():.4f} S={stab:.4f}")
        if args.eval_every and (ep % args.eval_every == 0 or ep == args.epochs - 1):
            model.eval()
            with torch.no_grad():
                r_fit, r_hold, r_ind = evaluate(model(x_in))
            model.train()
            curve.append([ep, loss.item(), r_fit, r_hold, r_ind])
        stopper(loss.item())
        if stopper.early_stop:
            if would_stop is None:
                would_stop = ep
                print(f"Early-stopping criterion met at epoch {ep}" + (" (continuing, --no_stop)" if args.no_stop else ""))
            if not args.no_stop:
                break
    elapsed = time.time() - t0

    # ---------------- evaluation ----------------
    model.eval()
    with torch.no_grad():
        pred = model(x_in)
        g0 = pred.squeeze().cpu().numpy()
        chk_low = up_low(pred).squeeze().cpu().numpy()
        chk_high = up_high(pred).squeeze().cpu().numpy()

    metrics = {
        "tag": args.tag, "seed": args.seed, "size": S, "epochs_run": int(len(hist)), "time_sec": float(elapsed),
        "would_stop_epoch": would_stop,
        "high_mode": args.high_mode, "w_low": args.w_low, "w_high": float(w_high), "w_bg": args.w_bg, "lam_tv": args.lam_tv,
        "holdout_block": args.holdout_block, "holdout_frac": args.holdout_frac,
        "n_obs_low": int(obs_low.sum()), "n_holdout": int(holdout.sum()), "n_obs_high": int(obs_high.sum()),
        "final_total_loss": float(hist[-1][1]),
    }
    # fit residuals
    metrics["rmse_low_fit"], metrics["n_low_fit"] = rmse(chk_low, low_obs_values, train_low & roi)
    metrics["rmse_low_holdout"], metrics["n_low_holdout"] = rmse(chk_low, low_obs_values, holdout & roi)
    metrics["rmse_high_fit"], metrics["n_high_fit"] = rmse(chk_high, high, obs_high & roi) if args.high_mode != "none" else (float('nan'), 0)
    # independent 5156 m check (validation for A/B, fit for C/D)
    if indep_high is not None:
        metrics["rmse_high_indep"], metrics["n_high_indep"] = rmse(chk_high, indep_high, obs_indep & roi)
        metrics["bias_high_indep"] = float(np.mean((chk_high - indep_high)[obs_indep & roi]))
    # region statistics of the 0 m field
    reg = {
        "R1_low_covered": obs_low & roi,
        "R2_high_only": (~obs_low) & obs_indep & roi,
        "R3_egm_only": (~obs_low) & (~obs_indep) & roi,
    }
    for k, m in reg.items():
        metrics[f"{k}_npix"] = int(m.sum())
        metrics[f"{k}_std_0m"] = float(np.std(g0[m])) if m.sum() else float('nan')
        metrics[f"{k}_gradrms_0m"] = grad_rms(g0, m)
    metrics["std_0m_roi"] = float(np.std(g0[roi]))
    metrics["range_0m_roi"] = [float(g0[roi].min()), float(g0[roi].max())]
    if curve:
        cv = np.array(curve, dtype=float)
        ok = np.isfinite(cv[:, 3])
        if ok.any():
            i = int(np.nanargmin(cv[:, 3]))
            metrics["best_holdout_epoch"] = int(cv[i, 0])
            metrics["best_holdout_rmse"] = float(cv[i, 3])
            metrics["fit_rmse_at_best_holdout"] = float(cv[i, 2])
        pd.DataFrame(cv, columns=["Epoch", "Total", "rmse_fit", "rmse_holdout", "rmse_high_indep"]).to_csv(f"{args.prefix}_curve.csv", index=False)

    print("\n--- Metrics (ROI) ---")
    for k in ["rmse_low_fit", "rmse_low_holdout", "rmse_high_fit", "rmse_high_indep", "bias_high_indep", "std_0m_roi"]:
        if k in metrics:
            print(f"{k:18s}: {metrics[k]}")

    # ---------------- save ----------------
    pf = args.prefix
    with open(f"{pf}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(hist, columns=["Epoch", "Total", "L_low", "L_high", "L_bg", "TV", "Stability"]).to_csv(f"{pf}_loss.csv", index=False)
    np.savez_compressed(f"{pf}_fields.npz", lons=lons, lats=lats, g0=g0, chk_low=chk_low, chk_high=chk_high,
                        low_obs=low_obs_values, high=high, indep_high=indep_high if indep_high is not None else np.full((S, S), np.nan),
                        obs_low=obs_low, holdout=holdout, obs_high=obs_high, obs_indep=obs_indep, roi=roi)
    save_xyz(f"{pf}_0m.xyz", lons, lats, g0, "lon lat gravity_0m")
    if holdout.any() and not args.holdout_file:
        save_xyz(f"{pf}_holdout.xyz", lons, lats, holdout.astype(float), "lon lat holdout(1=held out)")
    torch.save(model.state_dict(), f"{pf}_model.pth")

    # ---------------- figure ----------------
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    has_coast = extract_coastline_gmt(extent, "coastline.txt")
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))

    def show(a, arr, title, lim=None, cmap='RdYlBu_r'):
        if lim is None:
            lim = np.nanmax(np.abs(arr[roi])) if np.isfinite(arr[roi]).any() else 1
        im = a.imshow(arr, origin='lower', extent=extent, cmap=cmap, vmin=-lim, vmax=lim)
        a.set_title(title)
        if has_coast:
            plot_coastline(a, "coastline.txt")
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    show(ax[0, 0], g0, f"Pred 0 m [{args.tag}] seed {args.seed}\nstd(ROI)={metrics['std_0m_roi']:.2f}")
    d_low = np.where(obs_low, low_obs_values - chk_low, np.nan)
    show(ax[0, 1], d_low, f"Obs-Up at {int(args.obs_low)} m\nfit RMSE {metrics['rmse_low_fit']:.2f} | hold-out {metrics['rmse_low_holdout']:.2f}", cmap='RdBu_r')
    if holdout.any():
        ax[0, 1].contour(lons, lats, holdout.astype(float), levels=[0.5], colors='k', linewidths=0.6)
    if indep_high is not None:
        d_hi = np.where(obs_indep, indep_high - chk_high, np.nan)
        show(ax[0, 2], d_hi, f"Indep LSC - Up at {int(args.obs_high)} m\nRMSE {metrics.get('rmse_high_indep', float('nan')):.2f}", cmap='RdBu_r')
    else:
        ax[0, 2].axis('off')
    show(ax[1, 0], low_obs_values, f"Input prior {int(args.obs_low)} m")
    show(ax[1, 1], chk_low, f"Up(Pred) -> {int(args.obs_low)} m")
    show(ax[1, 2], chk_high, f"Up(Pred) -> {int(args.obs_high)} m")
    for a in ax.ravel():
        a.set_xlabel('Longitude'); a.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f"{pf}_result.png", dpi=130)
    plt.close()
    print(f"Saved {pf}_metrics.json / _fields.npz / _0m.xyz / _result.png  ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
