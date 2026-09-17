"""
Real-data dual-constraint 0 m solver (Taiwan airborne).

Fork of blind_pisr_e3.py: no hidden 0 m truth, no injected noise, independent
LSC examiners at 1620 m and 5156 m. EGM far-field fill is input only; L_z is
restricted to each altitude's LSC coverage Ω.

  python blind_pisr_real0.py --config C --prefix ch4/real0_C
  python blind_pisr_real0.py --config A --prefix ch4/real0_A
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


def load_field(path, size, lons_ref=None, lats_ref=None):
    lons, lats, g = load_grid(path)
    g = resample(lons, lats, g, size)
    if lons_ref is None:
        lons_ref = np.linspace(lons.min(), lons.max(), size)
        lats_ref = np.linspace(lats.min(), lats.max(), size)
    return lons_ref, lats_ref, g


def main():
    p = argparse.ArgumentParser(description="Real-data dual-constraint 0 m PI-ZSSR")
    p.add_argument("--config", choices=["A", "C", "D"], default="C")
    p.add_argument("--file_low", default="ch4/1500_pred_grid_0.1_filled_feather20.xyz",
                   help="complete z1 prior (three-layer filled 1620 m LSC)")
    p.add_argument("--mask_low", default="1500(10s)_new.xyg_pred_grid_0.1.xyz",
                   help="unfilled 1620 m LSC (finite = Ω_z1)")
    p.add_argument("--file_high", default="5000(10s)_new.xyg_pred_grid_0.1.xyz",
                   help="independent 5156 m LSC values")
    p.add_argument("--mask_high", default="5000(10s)_new.xyg_pred_grid_0.1.xyz",
                   help="unfilled 5156 m LSC (finite = Ω_z2)")
    p.add_argument("--z1", type=float, default=1620.0)
    p.add_argument("--z2", type=float, default=5156.0)
    p.add_argument("--size", type=int, default=301)
    p.add_argument("--margin", type=int, default=20)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="ch4/real0")
    p.add_argument("--gauss_km", type=float, default=1.8)
    p.add_argument("--lam_tik", type=float, default=0.1)
    p.add_argument("--w_bg", type=float, default=0.1)
    p.add_argument("--lag", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--floor_pat", type=int, default=80)
    p.add_argument("--stop_rule", choices=["disc", "none"], default="disc")
    p.add_argument("--sigma_low", type=float, default=0.9,
                   help="discrepancy sigma at z1 (1620 m LSC pred_std median)")
    p.add_argument("--sigma_high", type=float, default=0.7,
                   help="discrepancy sigma at z2 (5156 m LSC pred_std median)")
    args = p.parse_args()

    w1, w2 = {"A": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}[args.config]
    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    S = args.size
    lons, lats, filled_low = load_field(args.file_low, S)
    filled_low = fill_nearest(filled_low)
    _, _, g_low = load_field(args.mask_low, S, lons, lats)
    _, _, g_high = load_field(args.file_high, S, lons, lats)
    if args.mask_high and args.mask_high != args.file_high:
        _, _, mhi_g = load_field(args.mask_high, S, lons, lats)
        obs_high = np.isfinite(mhi_g)
    else:
        obs_high = np.isfinite(g_high)
    obs_low = np.isfinite(g_low)
    val_low = np.where(obs_low, g_low, 0.0)
    val_high = np.where(obs_high, g_high, 0.0)

    dx_km = (lons[1] - lons[0]) * 102.0
    dy_km = (lats[1] - lats[0]) * 111.0
    roi = np.zeros((S, S), dtype=bool)
    roi[args.margin:-args.margin, args.margin:-args.margin] = True
    train_low = obs_low & roi
    train_high = obs_high & roi
    R1 = obs_low & roi
    R2 = (~obs_low) & obs_high & roi
    R3 = (~obs_low) & (~obs_high) & roi
    print(f"Grid {S}x{S} dx={dx_km:.3f} dy={dy_km:.3f} km  "
          f"z1={args.z1:.0f} z2={args.z2:.0f}  cfg={args.config}")
    print(f"Ω_z1={obs_low.sum()}  Ω_z2={obs_high.sum()}  "
          f"L_z1={train_low.sum()}  L_z2={train_high.sum()}  "
          f"R1={R1.sum()} R2={R2.sum()} R3={R3.sum()}")

    sigma_low = float(args.sigma_low) if w1 > 0 else 0.0
    sigma_high = float(args.sigma_high) if w2 > 0 else 0.0
    print(f"sigma_low={sigma_low:.3f}  sigma_high={sigma_high:.3f}  "
          f"w1={w1} w2={w2} w_bg={args.w_bg}")

    stat_m = obs_low & np.isfinite(filled_low)
    mean_h = float(filled_low[stat_m].mean())
    std_h = float(filled_low[stat_m].std())
    print(f"input stats on Ω_z1: mean={mean_h:.3f}  std={std_h:.3f} mGal")

    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))[None, None].to(device)
    x_in = T((filled_low - mean_h) / std_h)
    tgt_low, tgt_high = T(val_low), T(val_high)
    tgt_bg = T(filled_low)
    m_low = T(train_low.astype(np.float32))
    m_high = T(train_high.astype(np.float32))
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
        r1 = rmse(u1, val_low, train_low)[0]
        r2 = rmse(u2, val_high, train_high)[0]
        return pr, r1, r2, u1, u2

    def disc_ok(r1, r2, floor1=False, floor2=False):
        ok1 = (w1 == 0) or (sigma_low <= 0) or (r1 <= sigma_low) or floor1
        ok2 = (w2 == 0) or (sigma_high <= 0) or (r2 <= sigma_high) or floor2
        return ok1 and ok2

    curve, snaps, rule_epoch = [], {}, {"disc": None, "old": None}
    best_r1, best_r2, ep_best_r1, ep_best_r2 = np.inf, np.inf, 0, 0
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
            pr, r1, r2, _, _ = score(predict_real())
            curve.append([ep, loss.item(), l1.item(), l2.item(), lbg.item(), r1, r2,
                          float(np.std(pr[roi]))])
            if r1 < best_r1 - 1e-3:
                best_r1, ep_best_r1 = r1, ep
            if r2 < best_r2 - 1e-3:
                best_r2, ep_best_r2 = r2, ep
            floor1 = (w1 > 0) and (sigma_low > 0) and (best_r1 > sigma_low) and (ep - ep_best_r1 >= args.floor_pat)
            floor2 = (w2 > 0) and (sigma_high > 0) and (best_r2 > sigma_high) and (ep - ep_best_r2 >= args.floor_pat)
            if rule_epoch["disc"] is None and disc_ok(r1, r2, floor1, floor2):
                rule_epoch["disc"] = ep
                snaps["disc"] = pr.copy()
                why = []
                if w1 > 0:
                    why.append(f"r{int(args.z1)}={r1:.3f}{' (floor)' if floor1 else ''}")
                if w2 > 0:
                    why.append(f"r{int(args.z2)}={r2:.3f}{' (floor)' if floor2 else ''}")
                print(f"[disc] ep {ep}  {'  '.join(why)}")
                if args.stop_rule == "disc":
                    stop_now = True
            if ep % 200 == 0:
                print(f"ep {ep}: loss={loss.item():.4f} r1={r1:.3f} r2={r2:.3f} std0={np.std(pr[roi]):.3f}")
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
    final_pr, f1, f2, _, _ = score(predict_real())
    snaps["final"] = final_pr
    rule_epoch["final"] = last_ep
    if "disc" not in snaps:
        snaps["disc"] = final_pr
        print("WARNING: discrepancy did not fire; using final snapshot")

    def pack(pr):
        u1 = up1(T(pr)).squeeze().cpu().numpy()
        u2 = up2(T(pr)).squeeze().cpu().numpy()
        return {
            "rmse_fit_z1": rmse(u1, val_low, train_low)[0],
            "rmse_fit_z2": rmse(u2, val_high, train_high)[0],
            "std_0": float(np.std(pr[roi])),
            "std_0_R1": float(np.std(pr[R1])) if R1.any() else None,
            "std_0_R2": float(np.std(pr[R2])) if R2.any() else None,
            "std_0_R3": float(np.std(pr[R3])) if R3.any() else None,
        }

    met = {
        "tag": args.config, "seed": args.seed, "size": S, "epochs_run": int(last_ep + 1),
        "time_sec": float(elapsed),
        "sigma_low": sigma_low, "sigma_high": sigma_high, "w1": w1, "w2": w2, "w_bg": args.w_bg,
        "gauss_km": args.gauss_km, "lam_tik": args.lam_tik, "stop_rule": args.stop_rule,
        "z1": args.z1, "z2": args.z2,
        "n_omega_z1": int(obs_low.sum()), "n_omega_z2": int(obs_high.sum()),
        "n_R1": int(R1.sum()), "n_R2": int(R2.sum()), "n_R3": int(R3.sum()),
        "mean_input_omega": mean_h, "std_input_omega": std_h,
        "file_low": args.file_low, "file_high": args.file_high,
    }
    for k, pr in snaps.items():
        rec = pack(pr)
        met[f"{k}_epoch"] = int(rule_epoch[k]) if rule_epoch.get(k) is not None else None
        for kk, vv in rec.items():
            met[f"{k}_{kk}"] = vv

    cv = np.array(curve, dtype=float)
    with open(f"{args.prefix}_metrics.json", "w") as f:
        json.dump(met, f, indent=2)
    pd.DataFrame(cv, columns=["Epoch", "Total", "L1", "L2", "Lbg", "r_z1", "r_z2", "std_0"]).to_csv(
        f"{args.prefix}_curve.csv", index=False)
    np.savez_compressed(
        f"{args.prefix}_fields.npz", lons=lons, lats=lats, filled_low=filled_low,
        val_low=val_low, val_high=val_high,
        obs_low=obs_low, obs_high=obs_high, roi=roi, R1=R1, R2=R2, R3=R3,
        **{f"pred_{k}": v for k, v in snaps.items()},
    )
    save_xyz(f"{args.prefix}_0m.xyz", lons, lats, snaps["disc"], "lon lat residual_gravity_0m")

    print("\n--- real 0 m summary (fit RMSE on Ω, not 0 m truth) ---")
    for k in ("disc", "old", "final"):
        if f"{k}_rmse_fit_z1" in met:
            print(f"{k:6s} ep {str(met.get(f'{k}_epoch')):>5s}  "
                  f"fit z1 {met[f'{k}_rmse_fit_z1']:.3f}  fit z2 {met[f'{k}_rmse_fit_z2']:.3f}  "
                  f"std0 {met[f'{k}_std_0']:.3f}  ({elapsed:.0f} s)")

    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    pr = snaps["disc"]
    u1 = up1(T(pr)).squeeze().cpu().numpy()
    u2 = up2(T(pr)).squeeze().cpu().numpy()
    lim0 = np.nanmax(np.abs(pr[roi]))

    def show(a, arr, title, lim=lim0, cmap="RdYlBu_r"):
        im = a.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
        a.set_title(title, fontsize=10)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    cov = np.full((S, S), np.nan)
    cov[R1] = 1
    cov[R2] = 2
    cov[R3] = 3
    ax[0, 0].imshow(cov, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=3)
    ax[0, 0].set_title("coverage  R1=1620  R2=5156 only  R3=none")
    show(ax[0, 1], filled_low, "input: filled 1620 m", lim=np.nanmax(np.abs(filled_low[roi])))
    show(ax[0, 2], pr, f"{args.config} ĝ_0 disc ep {rule_epoch['disc']}\nstd {met['disc_std_0']:.2f} mGal")
    d1 = np.where(train_low, val_low - u1, np.nan)
    d2 = np.where(train_high, val_high - u2, np.nan)
    show(ax[1, 0], d1, f"1620 LSC − U(ĝ_0)  RMSE {met['disc_rmse_fit_z1']:.2f}",
         lim=np.nanmax(np.abs(d1[train_low])) if train_low.any() else 1, cmap="RdBu_r")
    show(ax[1, 1], d2, f"5156 LSC − U(ĝ_0)  RMSE {met['disc_rmse_fit_z2']:.2f}",
         lim=np.nanmax(np.abs(d2[train_high])) if train_high.any() else 1, cmap="RdBu_r")
    a = ax[1, 2]
    if len(cv):
        a.plot(cv[:, 0], cv[:, 5], label=f"fit {int(args.z1)} m")
        a.plot(cv[:, 0], cv[:, 6], label=f"fit {int(args.z2)} m")
        if sigma_low > 0:
            a.axhline(sigma_low, color="C0", ls=":", lw=0.8)
        if sigma_high > 0:
            a.axhline(sigma_high, color="C1", ls=":", lw=0.8)
        if rule_epoch["disc"] is not None:
            a.axvline(rule_epoch["disc"], color="m", ls="--", lw=0.9, label="disc")
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
