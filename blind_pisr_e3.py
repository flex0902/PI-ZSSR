"""
E3 dual-constraint solver: predict 0 m from masked 1500 m / 5000 m EGM residuals.

Coverage comes from the real LSC NaN patterns; values are synthetic (leak-free gap fill).

  python blind_pisr_e3.py --config A --seed 0 --prefix e3/A_s0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blind_restoration_pisr import SimpleUNet, UpwardContinuationLayer, GaussianSmoothing, EarlyStopper, device
from deploy_pisr_ablation import load_grid, resample, fill_nearest, rmse, masked_mse, save_xyz
from blind_pisr_v2 import make_noise, inpaint_linear, tikhonov_fft


def load_field(path, size, lons_ref=None, lats_ref=None):
    lons, lats, g = load_grid(path)
    g = resample(lons, lats, g, size)
    if lons_ref is None:
        lons_ref = np.linspace(lons.min(), lons.max(), size)
        lats_ref = np.linspace(lats.min(), lats.max(), size)
    return lons_ref, lats_ref, g


def region_rmse(pred, truth, mask):
    return rmse(pred, truth, mask)[0]


def tikhonov_disc(obs, mask, dx, dy, h_km, sigma, truth, roi, up_layer, T):
    """Largest alpha whose fit RMSE on `mask` is <= sigma; also truth-optimal alpha."""
    alphas = np.logspace(-4, 1, 41)
    rows = []
    for a in alphas:
        g = tikhonov_fft(obs, dx, dy, h_km, a)
        upn = up_layer(T(g)).squeeze().cpu().numpy()
        rows.append([a, rmse(upn, obs, mask & roi)[0], rmse(g, truth, roi)[0]])
    res = np.array(rows)
    cand = np.where(res[:, 1] <= max(sigma, 1e-6))[0]
    j_disc = int(cand[-1]) if len(cand) else int(np.argmin(np.abs(res[:, 1] - sigma)))
    j_or = int(np.argmin(res[:, 2]))
    g_disc = tikhonov_fft(obs, dx, dy, h_km, res[j_disc, 0])
    g_or = tikhonov_fft(obs, dx, dy, h_km, res[j_or, 0])
    return {
        "alpha_disc": float(res[j_disc, 0]), "rmse_fit_disc": float(res[j_disc, 1]),
        "g_disc": g_disc, "g_oracle": g_or,
        "alpha_oracle": float(res[j_or, 0]),
    }


def main():
    p = argparse.ArgumentParser(description="E3 dual-constraint 0 m solver")
    p.add_argument("--config", choices=["A", "C", "D"], default="C")
    p.add_argument("--file_0", default="TW_EGM_grid_g_res_h0.xyz")
    p.add_argument("--file_1500", default="TW_EGM_grid_g_res_h1500.xyz")
    p.add_argument("--file_5000", default="TW_EGM_grid_g_res_h5000.xyz")
    p.add_argument("--mask_low", default="1500(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--mask_high", default="5000(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--z1", type=float, default=1500.0)
    p.add_argument("--z2", type=float, default=5000.0)
    p.add_argument("--size", type=int, default=301)
    p.add_argument("--margin", type=int, default=20)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="e3/run")
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--noise_type", choices=["white", "corr"], default="white")
    p.add_argument("--corr_len_km", type=float, default=20.0)
    p.add_argument("--gauss_km", type=float, default=1.8)
    p.add_argument("--lam_tik", type=float, default=0.1)
    p.add_argument("--w_bg", type=float, default=0.1)
    p.add_argument("--lag", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--stop_rule", choices=["disc", "none"], default="disc")
    args = p.parse_args()

    w1, w2 = {"A": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}[args.config]
    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(args.seed)

    S = args.size
    lons, lats, truth0 = load_field(args.file_0, S)
    _, _, t1500 = load_field(args.file_1500, S, lons, lats)
    _, _, t5000 = load_field(args.file_5000, S, lons, lats)
    t1500 = fill_nearest(t1500)
    t5000 = fill_nearest(t5000)
    truth0 = fill_nearest(truth0)

    _, _, mlow_g = load_field(args.mask_low, S, lons, lats)
    _, _, mhi_g = load_field(args.mask_high, S, lons, lats)
    obs_low = np.isfinite(mlow_g)
    obs_high = np.isfinite(mhi_g)

    dx_km = (lons[1] - lons[0]) * 102.0
    dy_km = (lats[1] - lats[0]) * 111.0
    roi = np.zeros((S, S), dtype=bool)
    roi[args.margin:-args.margin, args.margin:-args.margin] = True
    R1 = obs_low & roi
    R2 = (~obs_low) & obs_high & roi
    R3 = (~obs_low) & (~obs_high) & roi
    print(f"Grid {S}x{S} dx={dx_km:.3f} km  R1={R1.sum()} R2={R2.sum()} R3={R3.sum()}  cfg={args.config}")

    n_low = make_noise((S, S), args.noise, args.noise_type, args.corr_len_km, dy_km, rng) if args.noise > 0 else np.zeros((S, S))
    n_high = make_noise((S, S), args.noise, args.noise_type, args.corr_len_km, dy_km, rng) if args.noise > 0 else np.zeros((S, S))
    noisy_low = t1500.copy()
    noisy_low[obs_low] = t1500[obs_low] + n_low[obs_low]
    filled_low = inpaint_linear(noisy_low, ~obs_low)
    noisy_high = t5000.copy()
    noisy_high[obs_high] = t5000[obs_high] + n_high[obs_high]
    filled_high = inpaint_linear(noisy_high, ~obs_high)

    sigma_low = float(np.std(n_low[obs_low & roi])) if args.noise > 0 else 0.0
    sigma_high = float(np.std(n_high[obs_high & roi])) if args.noise > 0 else 0.0
    print(f"sigma_low={sigma_low:.3f}  sigma_high={sigma_high:.3f}  w1={w1} w2={w2} w_bg={args.w_bg}")

    mean_h, std_h = float(filled_low.mean()), float(filled_low.std())
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None, None].to(device)
    x_in = T((filled_low - mean_h) / std_h)
    tgt_low, tgt_high = T(noisy_low), T(noisy_high)
    tgt_bg = T(filled_low)
    m_low = T(obs_low.astype(np.float32))
    m_high = T(obs_high.astype(np.float32))
    m_bg = T((~obs_low).astype(np.float32))

    up1 = UpwardContinuationLayer(dx_km, dy_km, args.z1 / 1000.0).to(device)
    up2 = UpwardContinuationLayer(dx_km, dy_km, args.z2 / 1000.0).to(device)
    if args.gauss_km > 0:
        spx = args.gauss_km / (0.5 * (dx_km + dy_km))
        ks = max(int(2 * np.ceil(2 * spx) + 1), 3)
        smooth = GaussianSmoothing(1, ks, spx).to(device)
        print(f"Gaussian {args.gauss_km} km = {spx:.2f} px, kernel {ks}")
    else:
        smooth = torch.nn.Identity()
    model = SimpleUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    plateau = EarlyStopper(patience=args.lag, min_delta=1e-5)

    def predict_real():
        model.eval()
        with torch.no_grad():
            pr = smooth(model(x_in)) * std_h + mean_h
        model.train()
        return pr

    def score(pr_t):
        pr = pr_t.squeeze().cpu().numpy()
        u1 = up1(pr_t).squeeze().cpu().numpy()
        u2 = up2(pr_t).squeeze().cpu().numpy()
        r1 = rmse(u1, noisy_low, obs_low & roi)[0]
        r2 = rmse(u2, noisy_high, obs_high & roi)[0]
        rt = {
            "ROI": region_rmse(pr, truth0, roi),
            "R1": region_rmse(pr, truth0, R1),
            "R2": region_rmse(pr, truth0, R2),
            "R3": region_rmse(pr, truth0, R3),
        }
        return pr, r1, r2, rt

    def disc_ok(r1, r2, floor1=False, floor2=False):
        ok1 = (w1 == 0) or (sigma_low <= 0) or (r1 <= sigma_low) or floor1
        ok2 = (w2 == 0) or (sigma_high <= 0) or (r2 <= sigma_high) or floor2
        return ok1 and ok2

    curve, snaps, rule_epoch = [], {}, {"disc": None, "old": None, "oracle": None}
    best_truth = np.inf
    best_r1, best_r2, ep_best_r1, ep_best_r2 = np.inf, np.inf, 0, 0
    floor_pat = 80
    t0 = time.time()
    last_ep = -1
    for ep in range(args.epochs):
        last_ep = ep
        opt.zero_grad()
        pred_n = smooth(model(x_in))
        pred_r = pred_n * std_h + mean_h
        l1 = masked_mse(up1(pred_r), tgt_low, m_low) if w1 > 0 else torch.zeros((), device=device)
        l2 = masked_mse(up2(pred_r), tgt_high, m_high) if w2 > 0 else torch.zeros((), device=device)
        lbg = masked_mse(up1(pred_r), tgt_bg, m_bg) if args.w_bg > 0 else torch.zeros((), device=device)
        dh = pred_n[..., :, 1:] - pred_n[..., :, :-1]
        dv = pred_n[..., 1:, :] - pred_n[..., :-1, :]
        l_tik = torch.mean(dh ** 2) + torch.mean(dv ** 2)
        loss = w1 * l1 + w2 * l2 + args.w_bg * lbg + args.lam_tik * l_tik
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        stop_now = False
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            pr, r1, r2, rt = score(predict_real())
            curve.append([ep, loss.item(), l1.item(), l2.item(), lbg.item(), r1, r2, rt["ROI"], rt["R1"], rt["R2"], rt["R3"]])
            if rt["ROI"] < best_truth:
                best_truth = rt["ROI"]
                snaps["oracle"] = pr.copy()
                rule_epoch["oracle"] = ep
            # residual-floor detection: upward continuation cannot reproduce white noise at large h
            if r1 < best_r1 - 1e-3:
                best_r1, ep_best_r1 = r1, ep
            if r2 < best_r2 - 1e-3:
                best_r2, ep_best_r2 = r2, ep
            floor1 = (w1 > 0) and (sigma_low > 0) and (best_r1 > sigma_low) and (ep - ep_best_r1 >= floor_pat)
            floor2 = (w2 > 0) and (sigma_high > 0) and (best_r2 > sigma_high) and (ep - ep_best_r2 >= floor_pat)
            if rule_epoch["disc"] is None and disc_ok(r1, r2, floor1, floor2):
                rule_epoch["disc"] = ep
                snaps["disc"] = pr.copy()
                why = []
                if w1 > 0:
                    why.append(f"r1500={r1:.3f}{' (floor)' if floor1 else ''}")
                if w2 > 0:
                    why.append(f"r5000={r2:.3f}{' (floor)' if floor2 else ''}")
                print(f"[disc] ep {ep}  {'  '.join(why)}  truth ROI={rt['ROI']:.3f}")
                if args.stop_rule == "disc":
                    stop_now = True
            if ep % 200 == 0:
                print(f"ep {ep}: loss={loss.item():.4f} r1={r1:.3f} r2={r2:.3f} truth={rt['ROI']:.3f}")
        plateau(loss.item())
        if plateau.early_stop and rule_epoch["old"] is None:
            rule_epoch["old"] = ep
            snaps["old"] = score(predict_real())[0]
            print(f"[old] loss plateau at epoch {ep}")
            if args.stop_rule == "disc" and rule_epoch["disc"] is None:
                stop_now = True
        if stop_now:
            break
    elapsed = time.time() - t0
    final_pr, f1, f2, ft = score(predict_real())
    snaps["final"] = final_pr
    rule_epoch["final"] = last_ep
    if "disc" not in snaps:
        snaps["disc"] = final_pr
        print("WARNING: discrepancy did not fire; using final snapshot")

    def pack(pr):
        u1 = up1(T(pr)).squeeze().cpu().numpy()
        u2 = up2(T(pr)).squeeze().cpu().numpy()
        out = {
            "rmse_fit_1500": rmse(u1, noisy_low, obs_low & roi)[0],
            "rmse_fit_5000": rmse(u2, noisy_high, obs_high & roi)[0],
            "rmse_ROI": region_rmse(pr, truth0, roi),
            "rmse_R1": region_rmse(pr, truth0, R1),
            "rmse_R2": region_rmse(pr, truth0, R2),
            "rmse_R3": region_rmse(pr, truth0, R3),
            "std_0": float(np.std(pr[roi])),
        }
        return out

    met = {
        "tag": args.config, "seed": args.seed, "size": S, "epochs_run": int(last_ep + 1),
        "time_sec": float(elapsed), "noise": args.noise, "noise_type": args.noise_type,
        "sigma_low": sigma_low, "sigma_high": sigma_high, "w1": w1, "w2": w2, "w_bg": args.w_bg,
        "gauss_km": args.gauss_km, "lam_tik": args.lam_tik, "stop_rule": args.stop_rule,
        "n_R1": int(R1.sum()), "n_R2": int(R2.sum()), "n_R3": int(R3.sum()),
        "std_truth0": float(np.std(truth0[roi])),
    }
    for k, pr in snaps.items():
        rec = pack(pr)
        met[f"{k}_epoch"] = int(rule_epoch[k]) if rule_epoch.get(k) is not None else None
        for kk, vv in rec.items():
            met[f"{k}_{kk}"] = vv

    tik1 = tikhonov_disc(filled_low, obs_low, dx_km, dy_km, args.z1 / 1000.0, sigma_low, truth0, roi, up1, T)
    tik2 = tikhonov_disc(filled_high, obs_high, dx_km, dy_km, args.z2 / 1000.0, sigma_high, truth0, roi, up2, T)
    for name, tk in (("tik_1500", tik1), ("tik_5000", tik2)):
        g = tk["g_disc"]
        met[f"{name}_alpha_disc"] = tk["alpha_disc"]
        met[f"{name}_rmse_ROI"] = region_rmse(g, truth0, roi)
        met[f"{name}_rmse_R1"] = region_rmse(g, truth0, R1)
        met[f"{name}_rmse_R2"] = region_rmse(g, truth0, R2)
        met[f"{name}_rmse_R3"] = region_rmse(g, truth0, R3)
        snaps[name] = g

    cv = np.array(curve, dtype=float)
    with open(f"{args.prefix}_metrics.json", "w") as f:
        json.dump(met, f, indent=2)
    pd.DataFrame(cv, columns=["Epoch", "Total", "L1", "L2", "Lbg", "r1500", "r5000",
                              "rmse_ROI", "rmse_R1", "rmse_R2", "rmse_R3"]).to_csv(
        f"{args.prefix}_curve.csv", index=False)
    np.savez_compressed(
        f"{args.prefix}_fields.npz", lons=lons, lats=lats, truth0=truth0,
        filled_low=filled_low, noisy_low=noisy_low, noisy_high=noisy_high,
        obs_low=obs_low, obs_high=obs_high, roi=roi, R1=R1, R2=R2, R3=R3,
        **{f"pred_{k}": v for k, v in snaps.items()},
    )
    save_xyz(f"{args.prefix}_0m.xyz", lons, lats, snaps["disc"], "lon lat gravity_0m")

    print("\n--- E3 summary (0 m RMSE) ---")
    for k in ("disc", "oracle", "old", "final"):
        if f"{k}_rmse_ROI" in met:
            print(f"{k:6s} ep {str(met.get(f'{k}_epoch')):>5s}  ROI {met[f'{k}_rmse_ROI']:.3f}  "
                  f"R1 {met[f'{k}_rmse_R1']:.3f}  R2 {met[f'{k}_rmse_R2']:.3f}  R3 {met[f'{k}_rmse_R3']:.3f}")
    print(f"tik1500 R2 {met['tik_1500_rmse_R2']:.3f} | tik5000 R2 {met['tik_5000_rmse_R2']:.3f}  ({elapsed:.0f} s)")

    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    lim = np.nanmax(np.abs(truth0[roi]))

    def show(a, arr, title, lim=lim, cmap="RdYlBu_r"):
        im = a.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
        a.set_title(title, fontsize=10)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    cov = np.zeros((S, S))
    cov[R1] = 1
    cov[R2] = 2
    cov[R3] = 3
    ax[0, 0].imshow(np.where(roi, cov, np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=3)
    ax[0, 0].set_title("coverage  R1=1 R2=2 R3=3")
    show(ax[0, 1], truth0, "truth 0 m")
    show(ax[0, 2], snaps["disc"], f"{args.config} disc ep {rule_epoch['disc']}\nROI {met['disc_rmse_ROI']:.2f}")
    d = truth0 - snaps["disc"]
    show(ax[1, 0], d, f"truth - {args.config} (disc)\nR2 {met['disc_rmse_R2']:.2f}",
         lim=np.nanmax(np.abs(d[roi])), cmap="RdBu_r")
    show(ax[1, 1], filled_low, "input: inpainted 1500 m", lim=np.nanmax(np.abs(filled_low[roi])))
    a = ax[1, 2]
    if len(cv):
        a.plot(cv[:, 0], cv[:, 5], label="r 1500")
        a.plot(cv[:, 0], cv[:, 6], label="r 5000")
        a.plot(cv[:, 0], cv[:, 7], label="truth ROI")
        a.plot(cv[:, 0], cv[:, 9], label="truth R2")
        if sigma_low > 0:
            a.axhline(sigma_low, color="C0", ls=":", lw=0.8)
        if sigma_high > 0:
            a.axhline(sigma_high, color="C1", ls=":", lw=0.8)
        if rule_epoch["disc"] is not None:
            a.axvline(rule_epoch["disc"], color="m", ls="--", lw=0.9)
        if rule_epoch["oracle"] is not None:
            a.axvline(rule_epoch["oracle"], color="r", ls="--", lw=0.9)
    a.set_yscale("log")
    a.legend(fontsize=7)
    a.set_title("learning curves")
    a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.prefix}_result.png", dpi=130)
    plt.close()
    print("saved", args.prefix + "_result.png")


if __name__ == "__main__":
    main()
