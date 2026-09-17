"""
blind_pisr_v2.py  --  revised single-constraint PI-ZSSR solver for the synthetic (Chapter 3) experiments.

Changes w.r.t. blind_restoration_pisr.py
  * de-normalisation uses ONLY the input (high-altitude) statistics -> amplification is learned, not injected
  * Gaussian stabilisation layer defined in km (scale-invariant), --gauss_km 0 disables it
  * noise: white or along-track correlated (--noise_type corr --corr_len_km L)
  * spatial block hold-out on the input grid; held-out pixels are removed from the physics loss and
    linearly in-painted in the network input
  * learning curves every --eval_every epochs: fit RMSE, hold-out RMSE, truth RMSE
  * three stopping rules recorded side by side (none of them stops the run when --no_stop):
      old   : loss-plateau EarlyStopper (paper, patience=--lag)
      disc  : discrepancy principle, first epoch with fit RMSE <= sigma_noise (effective, after prefilter)
      hold  : hold-out minimum (with --patience_holdout evaluations without improvement)
    prediction snapshots at each rule's epoch are kept and scored against truth
  * optional Tikhonov/Wiener FFT baseline with alpha chosen by discrepancy principle and by oracle (truth)
  * --seed, gradient clipping

Example (E1):
  python blind_pisr_v2.py --noise 1.0 --holdout_block 15 --holdout_frac 0.2 --eval_every 25 --no_stop --seed 0 --prefix e1/e1_s0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blind_restoration_pisr import SimpleUNet, UpwardContinuationLayer, GaussianSmoothing, EarlyStopper, device
from deploy_pisr_ablation import load_grid, resample, fill_nearest, make_holdout, rmse, masked_mse, save_xyz


# ------------------------------------------------------------------
def make_noise(shape, sigma, kind, corr_len_km, dy_km, rng):
    n = rng.standard_normal(shape)
    if kind == "corr" and corr_len_km > 0:
        n = gaussian_filter1d(n, sigma=corr_len_km / dy_km, axis=0, mode="reflect")  # along N-S flight lines
        n = n / n.std()
    return n * sigma


def inpaint_linear(grid, mask):
    """Replace grid[mask] by linear interpolation from the un-masked pixels."""
    if not mask.any():
        return grid.copy()
    ny, nx = grid.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    out = grid.copy()
    vals = griddata((yy[~mask], xx[~mask]), grid[~mask], (yy[mask], xx[mask]), method="linear")
    out[mask] = vals
    if np.isnan(out).any():
        out = fill_nearest(out)
    return out


def tikhonov_fft(g_high, dx_km, dy_km, h_km, alpha):
    """Regularised downward continuation: minimise ||U g - d||^2 + alpha ||g||^2 (Wiener-type filter)."""
    ny, nx = g_high.shape
    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dy_km)
    KX, KY = np.meshgrid(kx, ky)
    k = 2 * np.pi * np.sqrt(KX ** 2 + KY ** 2)
    U = np.exp(-k * h_km)
    H = U / (U ** 2 + alpha)
    return np.real(np.fft.ifft2(np.fft.fft2(g_high) * H))


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PI-ZSSR v2 synthetic solver (single constraint)")
    p.add_argument("--file_high", default="TW_EGM_grid_g_res_h5000.xyz")
    p.add_argument("--file_low", default="TW_EGM_grid_g_res_h1500.xyz", help="hidden truth ('' = none)")
    p.add_argument("--obs_mask", default="",
                   help="XYZ mask of LSC coverage Ω (finite = observation; empty = full grid)")
    p.add_argument("--obs_h", type=float, default=5000.0)
    p.add_argument("--target_h", type=float, default=1500.0)
    p.add_argument("--size", type=int, default=301)
    p.add_argument("--margin", type=int, default=20)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="e1/run")
    # noise
    p.add_argument("--noise", type=float, default=0.0, help="noise sigma (mGal) added to the high-altitude input")
    p.add_argument("--noise_type", choices=["white", "corr"], default="white")
    p.add_argument("--corr_len_km", type=float, default=20.0)
    p.add_argument("--noise_seed", type=int, default=-1, help="-1 = same as --seed")
    p.add_argument("--prefilter_km", type=float, default=0.0, help="Gaussian pre-filter of the noisy input (km, 0 = off)")
    # regularisation
    p.add_argument("--gauss_km", type=float, default=1.8, help="stabilisation Gaussian sigma in km (0 = off)")
    p.add_argument("--lam_tik", type=float, default=0.1)
    p.add_argument("--lam_stat", type=float, default=0.0, help="paper's unit-variance term (default off in v2)")
    # hold-out
    p.add_argument("--holdout_block", type=int, default=15)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--holdout_seed", type=int, default=12345)
    # stopping
    p.add_argument("--lag", type=int, default=300, help="patience of the paper's loss-plateau rule")
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--patience_holdout", type=int, default=20, help="evaluations without hold-out improvement before stopping")
    p.add_argument("--stop_rule", choices=["none", "old", "disc", "hold"], default="none",
                   help="rule that actually stops the run ('none' = run all epochs, only record)")
    p.add_argument("--no_stop", action="store_true", help="alias for --stop_rule none")
    p.add_argument("--sigma_disc", type=float, default=-1, help="sigma for discrepancy rule (-1 = effective noise sigma)")
    p.add_argument("--floor_pat", type=int, default=80,
                   help="epochs without fit-RMSE improvement before treating residual floor as discrepancy")
    # baseline
    p.add_argument("--baseline", action="store_true", help="also run Tikhonov/Wiener FFT baseline")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.no_stop:
        args.stop_rule = "none"

    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(args.seed if args.noise_seed < 0 else args.noise_seed)

    # ---------------- data ----------------
    S = args.size
    lons0, lats0, hi_raw = load_grid(args.file_high)
    clean_high = fill_nearest(resample(lons0, lats0, hi_raw, S))
    lons = np.linspace(lons0.min(), lons0.max(), S); lats = np.linspace(lats0.min(), lats0.max(), S)
    dx_km = (lons[1] - lons[0]) * 102.0; dy_km = (lats[1] - lats[0]) * 111.0
    omega = np.ones((S, S), dtype=bool)
    if args.obs_mask:
        ml, ma, mraw = load_grid(args.obs_mask)
        omega = np.isfinite(resample(ml, ma, mraw, S))
        print(f"Ω mask: {omega.sum()} / {omega.size} pixels are LSC observations")
    truth = None
    if args.file_low:
        l0, a0, lo_raw = load_grid(args.file_low)
        truth = resample(l0, a0, lo_raw, S)   # keep NaNs; never a training target
    roi = np.zeros((S, S), dtype=bool); roi[args.margin:-args.margin, args.margin:-args.margin] = True
    h_km = (args.obs_h - args.target_h) / 1000.0
    print(f"Grid {S}x{S} dx={dx_km:.3f} dy={dy_km:.3f} km, h={h_km:.3f} km, truth={'yes' if truth is not None else 'no'}")

    # noise + prefilter
    noise = make_noise((S, S), args.noise, args.noise_type, args.corr_len_km, dy_km, rng) if args.noise > 0 else np.zeros((S, S))
    obs_high = clean_high + noise
    if args.prefilter_km > 0:
        spx = args.prefilter_km / (0.5 * (dx_km + dy_km))
        obs_high = gaussian_filter(obs_high, sigma=spx, mode="reflect")
        noise_eff = gaussian_filter(noise, sigma=spx, mode="reflect")
    else:
        noise_eff = noise
    sigma_eff = float(np.std(noise_eff[roi])) if args.noise > 0 else 0.0
    sigma_disc = args.sigma_disc if args.sigma_disc >= 0 else sigma_eff
    print(f"noise: {args.noise_type} sigma={args.noise} -> effective sigma (after prefilter) = {sigma_eff:.3f} mGal")

    # hold-out on the observed grid; L_z only on Ω ∩ (~holdout) ∩ ROI
    holdout = make_holdout((S, S), args.holdout_block, args.holdout_frac, args.holdout_seed)
    train_mask = omega & (~holdout) & roi
    net_input = inpaint_linear(obs_high, holdout)          # network never sees held-out values
    print(f"hold-out: {holdout.sum()} px ({100*holdout.mean():.1f}%)  "
          f"L_z pixels: {train_mask.sum()} (Ω & ROI, not held out)")

    # leak-free normalisation: statistics on Ω only (far-field fill is not an observation)
    stat_m = omega & np.isfinite(net_input)
    mean_h = float(net_input[stat_m].mean())
    std_h = float(net_input[stat_m].std())
    print(f"input stats on Ω: mean={mean_h:.3f}  std={std_h:.3f} mGal")
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None, None].to(device)
    x_in = T((net_input - mean_h) / std_h)
    tgt = T(obs_high)
    m_train = T(train_mask.astype(np.float32))

    # layers
    up = UpwardContinuationLayer(dx_km, dy_km, h_km).to(device)
    if args.gauss_km > 0:
        spx = args.gauss_km / (0.5 * (dx_km + dy_km))
        ks = int(2 * np.ceil(2 * spx) + 1); ks = max(ks, 3)
        smooth = GaussianSmoothing(1, ks, spx).to(device)
        print(f"Gaussian layer: sigma={args.gauss_km} km = {spx:.2f} px, kernel {ks}x{ks}")
    else:
        smooth = torch.nn.Identity()
    model = SimpleUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    old_rule = EarlyStopper(patience=args.lag, min_delta=1e-5)

    def predict_real(train_mode=False):
        if not train_mode:
            model.eval()
        with torch.no_grad():
            pr = smooth(model(x_in)) * std_h + mean_h
        if not train_mode:
            model.train()
        return pr

    def score(pr_t):
        pr = pr_t.squeeze().cpu().numpy()
        upn = up(pr_t).squeeze().cpu().numpy()
        r_fit, _ = rmse(upn, obs_high, train_mask)
        r_hold, _ = rmse(upn, obs_high, holdout & omega & roi)
        eval_m = (omega & roi) if args.obs_mask else roi
        r_truth = rmse(pr, truth, eval_m)[0] if truth is not None else float("nan")
        return pr, r_fit, r_hold, r_truth

    # ---------------- training ----------------
    curve, snaps, rule_epoch = [], {}, {"old": None, "disc": None, "hold": None}
    best_hold, since_best = np.inf, 0
    best_fit, ep_best_fit = np.inf, 0
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad()
        pred_n = smooth(model(x_in))
        pred_r = pred_n * std_h + mean_h
        l_phy = masked_mse(up(pred_r), tgt, m_train)
        dh = pred_n[..., :, 1:] - pred_n[..., :, :-1]; dv = pred_n[..., 1:, :] - pred_n[..., :-1, :]
        l_tik = torch.mean(dh ** 2) + torch.mean(dv ** 2)
        l_stat = (torch.std(pred_n) - 1.0) ** 2 if args.lam_stat > 0 else torch.zeros((), device=device)
        loss = l_phy + args.lam_tik * l_tik + args.lam_stat * l_stat
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        stop_now = False
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            pr, r_fit, r_hold, r_truth = score(predict_real())
            curve.append([ep, loss.item(), l_phy.item(), r_fit, r_hold, r_truth])
            if r_hold < best_hold - 1e-4:
                best_hold, since_best = r_hold, 0
                snaps["hold"] = pr.copy(); rule_epoch["hold"] = ep
            else:
                since_best += 1
                if since_best >= args.patience_holdout and args.stop_rule == "hold":
                    stop_now = True
            if r_fit < best_fit - 1e-3:
                best_fit, ep_best_fit = r_fit, ep
            floor = (sigma_disc > 0) and (best_fit > sigma_disc) and (ep - ep_best_fit >= args.floor_pat)
            if rule_epoch["disc"] is None and sigma_disc > 0 and (r_fit <= sigma_disc or floor):
                rule_epoch["disc"] = ep; snaps["disc"] = pr.copy()
                why = f"fit RMSE {r_fit:.3f} <= sigma {sigma_disc:.3f}" if r_fit <= sigma_disc else \
                      f"residual floor {r_fit:.3f} > sigma {sigma_disc:.3f} (no improve for {args.floor_pat} ep)"
                print(f"[disc] {why} at epoch {ep}")
                if args.stop_rule == "disc":
                    stop_now = True
            if ep % 500 == 0:
                print(f"ep {ep}: loss={loss.item():.4f} fit={r_fit:.3f} hold={r_hold:.3f} truth={r_truth:.3f}")
        old_rule(loss.item())
        if old_rule.early_stop and rule_epoch["old"] is None:
            rule_epoch["old"] = ep; snaps["old"] = score(predict_real())[0]
            print(f"[old] loss-plateau rule met at epoch {ep}")
            if args.stop_rule == "old":
                stop_now = True
        if stop_now:
            print(f"stopped by rule '{args.stop_rule}' at epoch {ep}")
            break
    elapsed = time.time() - t0
    final_pr, f_fit, f_hold, f_truth = score(predict_real())
    snaps["final"] = final_pr; rule_epoch["final"] = ep

    # ---------------- metrics ----------------
    cv = np.array(curve, dtype=float)
    met = {"tag": args.tag, "seed": args.seed, "size": S, "epochs_run": int(ep + 1), "time_sec": float(elapsed),
           "noise": args.noise, "noise_type": args.noise_type, "corr_len_km": args.corr_len_km,
           "sigma_eff": sigma_eff, "sigma_disc": sigma_disc, "prefilter_km": args.prefilter_km,
           "gauss_km": args.gauss_km, "lam_tik": args.lam_tik, "lam_stat": args.lam_stat,
           "holdout_block": args.holdout_block, "holdout_frac": args.holdout_frac, "stop_rule": args.stop_rule,
           "n_omega": int(omega.sum()), "n_Lz": int(train_mask.sum()),
           "std_input_clean": float(np.std(clean_high[stat_m & roi])),
           "truth_amp_ratio": (float(np.nanstd(truth[roi]) / np.std(clean_high[stat_m & roi]))
                               if truth is not None else None)}
    for k, pr in snaps.items():
        e = rule_epoch[k]
        met[f"{k}_epoch"] = int(e) if e is not None else None
        met[f"{k}_rmse_truth"] = rmse(pr, truth, (omega & roi) if args.obs_mask else roi)[0] if truth is not None else None
        met[f"{k}_amp_ratio"] = float(np.std(pr[roi]) / np.std(clean_high[stat_m & roi]))
        upn = up(T(pr)).squeeze().cpu().numpy()
        met[f"{k}_rmse_fit"] = rmse(upn, obs_high, train_mask)[0]
        met[f"{k}_rmse_holdout"] = rmse(upn, obs_high, holdout & omega & roi)[0]
        if truth is not None and args.noise > 0:
            met[f"{k}_noise_amp_factor"] = met[f"{k}_rmse_truth"] / args.noise
    if truth is not None and len(cv):
        i = int(np.nanargmin(cv[:, 5]))
        met["oracle_epoch"] = int(cv[i, 0]); met["oracle_rmse_truth"] = float(cv[i, 5])

    # ---------------- baseline ----------------
    if args.baseline and truth is not None:
        alphas = np.logspace(-4, 1, 41)
        res = []
        for a in alphas:
            g = tikhonov_fft(obs_high, dx_km, dy_km, h_km, a)
            upn = up(T(g)).squeeze().cpu().numpy()
            res.append([a, rmse(upn, obs_high, roi)[0], rmse(g, truth, roi)[0], float(np.std(g[roi]) / np.std(clean_high[roi]))])
        res = np.array(res)
        j_or = int(np.argmin(res[:, 2]))
        cand = np.where(res[:, 1] <= max(sigma_disc, 1e-6))[0]
        j_disc = int(cand[-1]) if len(cand) else int(np.argmin(np.abs(res[:, 1] - sigma_disc)))  # largest alpha satisfying fit<=sigma
        met["baseline"] = {"alpha_oracle": float(res[j_or, 0]), "rmse_truth_oracle": float(res[j_or, 2]), "amp_oracle": float(res[j_or, 3]),
                           "alpha_disc": float(res[j_disc, 0]), "rmse_truth_disc": float(res[j_disc, 2]), "amp_disc": float(res[j_disc, 3]),
                           "rmse_fit_disc": float(res[j_disc, 1])}
        pd.DataFrame(res, columns=["alpha", "rmse_fit", "rmse_truth", "amp_ratio"]).to_csv(f"{args.prefix}_baseline.csv", index=False)
        snaps["tikhonov_disc"] = tikhonov_fft(obs_high, dx_km, dy_km, h_km, res[j_disc, 0])
        print(f"baseline Tikhonov: oracle alpha={res[j_or,0]:.2e} truthRMSE={res[j_or,2]:.3f} | disc alpha={res[j_disc,0]:.2e} truthRMSE={res[j_disc,2]:.3f}")

    # ---------------- save ----------------
    with open(f"{args.prefix}_metrics.json", "w") as f:
        json.dump(met, f, indent=2)
    pd.DataFrame(cv, columns=["Epoch", "Total", "L_phy", "rmse_fit", "rmse_holdout", "rmse_truth"]).to_csv(f"{args.prefix}_curve.csv", index=False)
    np.savez_compressed(f"{args.prefix}_fields.npz", lons=lons, lats=lats, clean_high=clean_high, obs_high=obs_high, noise=noise,
                        truth=truth if truth is not None else np.full((S, S), np.nan), holdout=holdout, roi=roi, omega=omega,
                        **{f"pred_{k}": v for k, v in snaps.items()})
    snap_key = args.stop_rule if args.stop_rule in snaps else ("disc" if "disc" in snaps else "final")
    save_xyz(f"{args.prefix}_{int(args.target_h)}m.xyz", lons, lats, snaps.get(snap_key, final_pr),
             f"lon lat gravity ({snap_key} snapshot)")

    print("\n--- summary (truth RMSE, ROI) ---")
    for k in ["old", "disc", "hold", "final"]:
        if met.get(f"{k}_epoch") is not None:
            print(f"{k:6s} epoch {met[f'{k}_epoch']:5d}  truth RMSE {met[f'{k}_rmse_truth']:.3f}  fit {met[f'{k}_rmse_fit']:.3f}  hold {met[f'{k}_rmse_holdout']:.3f}  amp {met[f'{k}_amp_ratio']:.3f}")
    if "oracle_epoch" in met:
        print(f"oracle epoch {met['oracle_epoch']:5d}  truth RMSE {met['oracle_rmse_truth']:.3f}   (truth amp ratio {met['truth_amp_ratio']:.3f})")

    # ---------------- figure ----------------
    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    lim = np.nanmax(np.abs(truth[roi])) if truth is not None else np.nanmax(np.abs(final_pr[roi]))
    def show(a, arr, title, lim=lim, cmap="RdYlBu_r"):
        im = a.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim); a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    show(ax[0, 0], obs_high, f"Input {int(args.obs_h)} m (noise {args.noise} {args.noise_type})", lim=np.nanmax(np.abs(obs_high[roi])))
    ax[0, 0].contour(lons, lats, omega.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    pred_show = snaps.get(snap_key, final_pr)
    if truth is not None:
        show(ax[0, 1], np.where(np.isfinite(truth), truth, np.nan), f"Truth {int(args.target_h)} m")
    show(ax[0, 2], pred_show, f"Pred ({snap_key} rule, ep {rule_epoch.get(snap_key)})\ntruth RMSE {met.get(f'{snap_key}_rmse_truth', float('nan')):.2f}")
    if truth is not None:
        d = truth - pred_show; dl = np.nanmax(np.abs(d[roi]))
        show(ax[1, 0], d, f"Truth - Pred({snap_key} rule)", lim=dl, cmap="RdBu_r")
        d2 = truth - final_pr
        show(ax[1, 1], d2, f"Truth - Pred(final, ep {ep})\ntruth RMSE {f_truth:.2f}", lim=dl, cmap="RdBu_r")
    a = ax[1, 2]
    a.plot(cv[:, 0], cv[:, 3], label="fit RMSE (observed)"); a.plot(cv[:, 0], cv[:, 4], label="hold-out RMSE")
    if truth is not None:
        a.plot(cv[:, 0], cv[:, 5], label="truth RMSE")
    for k, c in zip(["old", "disc", "hold"], ["k", "m", "g"]):
        if rule_epoch[k] is not None:
            a.axvline(rule_epoch[k], color=c, ls="--", lw=0.9, label=f"{k} rule (ep {rule_epoch[k]})")
    if sigma_disc > 0:
        a.axhline(sigma_disc, color="grey", ls=":", lw=0.8)
    a.set_xlabel("epoch"); a.set_ylabel("mGal"); a.set_yscale("log"); a.grid(alpha=0.3); a.legend(fontsize=7)
    a.set_title("learning curves")
    plt.tight_layout(); plt.savefig(f"{args.prefix}_result.png", dpi=130); plt.close()
    print(f"saved {args.prefix}_result.png  ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
