"""
Aggregate E4 (grid-size consistency): e4/g<size>_n<noise>_s<seed>_metrics.json

Three series per grid size S (spacing dx_S):
  * noise-free                       (noise = 0)
  * fixed per-pixel noise            (sigma = 1 mGal on every grid)     -> noise PSD  ~ dx_S^2, decreases with refinement
  * PSD-matched noise                (sigma = 1 mGal * dx_301 / dx_S)   -> same in-band noise power on every grid
Outputs e4/E4_table.md/.csv, e4/E4_runs.csv, e4/E4_vs_grid.png

  python summarize_e4.py --dir e4
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REF = 301


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e4")
    args = ap.parse_args()
    rows = []
    for fn in sorted(glob.glob(os.path.join(args.dir, "g*_metrics.json"))):
        m = json.load(open(fn)); b = m["baseline"]
        S = m["size"]
        rows.append({"grid_n": S, "noise": m["noise"], "seed": m["seed"], "time_sec": m["time_sec"], "epochs": m["epochs_run"],
                     "sigma_eff": m["sigma_eff"],
                     "disc_epoch": m.get("disc_epoch"), "disc_rmse": m.get("disc_rmse_truth"), "disc_amp": m.get("disc_amp_ratio"),
                     "disc_fit": m.get("disc_rmse_fit"),
                     "oracle_epoch": m.get("oracle_epoch"), "oracle_rmse": m.get("oracle_rmse_truth"),
                     "final_rmse": m.get("final_rmse_truth"), "final_amp": m.get("final_amp_ratio"), "final_fit": m.get("final_rmse_fit"),
                     "tik_disc_rmse": b["rmse_truth_disc"], "tik_disc_amp": b["amp_disc"], "tik_or_rmse": b["rmse_truth_oracle"],
                     "truth_amp": m["truth_amp_ratio"]})
    df = pd.DataFrame(rows).sort_values(["grid_n", "noise", "seed"])
    df["psd_ratio"] = df.noise * (REF - 1) / (df.grid_n - 1)          # = 1 for PSD-matched series
    df["is_none"] = df.noise == 0; df["is_fixed"] = np.isclose(df.noise, 1.0); df["is_psd"] = np.isclose(df.psd_ratio, 1.0, atol=0.05)
    df.to_csv(os.path.join(args.dir, "E4_runs.csv"), index=False, float_format="%.4f")

    sizes = sorted(df.grid_n.unique())
    z = np.load(glob.glob(os.path.join(args.dir, f"g{sizes[0]}_n0_s0_fields.npz"))[0])
    ext_lon = z["lons"].max() - z["lons"].min(); ext_lat = z["lats"].max() - z["lats"].min()
    dx = {S: 0.5 * (ext_lon * 102.0 + ext_lat * 111.0) / (S - 1) for S in sizes}

    def ms(x, f="{:.2f}"):
        x = pd.Series(x).dropna()
        if x.empty:
            return "–"
        return (f + " ± " + f).format(x.mean(), x.std()) if len(x) > 1 else f.format(x.mean())

    tab = []
    for S in sizes:
        d = df[df.grid_n == S]; d0 = d[d.is_none]; d1 = d[d.is_fixed]; d2 = d[d.is_psd]
        r = {"grid": f"{S}x{S}", "dx (km)": f"{dx[S]:.2f}", "pixels": f"{S*S:,}",
             "s / epoch": f"{d.time_sec.sum() / d.epochs.sum():.2f}",
             "noise-free: PI-ZSSR": ms(d0.final_rmse, "{:.3f}"), "noise-free: Tikhonov": ms(d0.tik_disc_rmse, "{:.3f}"),
             "noise-free: amp": ms(d0.final_amp),
             "σ=1: PI-ZSSR disc": ms(d1.disc_rmse), "σ=1: epoch": ms(d1.disc_epoch, "{:.0f}"), "σ=1: amp": ms(d1.disc_amp),
             "σ=1: oracle": ms(d1.oracle_rmse), "σ=1: Tikhonov disc": ms(d1.tik_disc_rmse), "σ=1: Tikhonov best": ms(d1.tik_or_rmse),
             "PSD-matched σ": f"{d2.noise.mean():g}" if len(d2) else "–",
             "PSD: PI-ZSSR disc": ms(d2.disc_rmse), "PSD: epoch": ms(d2.disc_epoch, "{:.0f}"), "PSD: amp": ms(d2.disc_amp),
             "PSD: oracle": ms(d2.oracle_rmse), "PSD: Tikhonov disc": ms(d2.tik_disc_rmse), "PSD: Tikhonov amp": ms(d2.tik_disc_amp),
             "PSD: Tikhonov best": ms(d2.tik_or_rmse), "truth amp": ms(d.truth_amp)}
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(args.dir, "E4_table.csv"), index=False)
    try:
        md = tab.to_markdown(index=False)
    except Exception:
        md = tab.to_string(index=False)
    open(os.path.join(args.dir, "E4_table.md"), "w", encoding="utf-8").write(md)
    print(md)

    # figure: RMSE vs spacing for the three series
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    xs = [dx[S] for S in sizes]
    def line(a, ser, key, lab, col, ls, mk):
        g = df[df["is_" + ser]].groupby("grid_n")[key]
        m_, s_ = g.mean().reindex(sizes), g.std().reindex(sizes).fillna(0)
        ok = m_.notna().values
        a.errorbar(np.array(xs)[ok], m_.values[ok], yerr=s_.values[ok], color=col, ls=ls, marker=mk, ms=5, capsize=3, label=lab)
    line(ax[0], "none", "final_rmse", "PI-ZSSR (1500 epochs)", "C3", "-", "o")
    line(ax[0], "none", "tik_disc_rmse", "Tikhonov-FFT", "C0", "-", "o")
    ax[0].set_title("noise-free")
    for a, ser, ttl in ((ax[1], "fixed", "fixed per-pixel noise, sigma = 1 mGal on every grid"),
                        (ax[2], "psd", "PSD-matched noise, sigma = 1 mGal x dx_301/dx")):
        line(a, ser, "disc_rmse", "PI-ZSSR (discrepancy)", "C3", "-", "o")
        line(a, ser, "oracle_rmse", "PI-ZSSR (oracle)", "C3", ":", "^")
        line(a, ser, "tik_disc_rmse", "Tikhonov-FFT (alpha by discrepancy)", "C0", "-", "o")
        line(a, ser, "tik_or_rmse", "Tikhonov-FFT (best alpha)", "C0", ":", "^")
        a.set_title(ttl)
    for a in ax:
        a.invert_xaxis(); a.grid(alpha=0.3); a.legend(fontsize=8); a.set_ylim(bottom=0)
        a.set_xlabel("grid spacing (km)"); a.set_xticks(xs); a.set_xticklabels([f"{d:.2f}\n({S}²)" for d, S in zip(xs, sizes)])
    ax[0].set_ylabel("RMSE vs hidden truth (mGal)")
    plt.tight_layout(); out = os.path.join(args.dir, "E4_vs_grid.png"); plt.savefig(out, dpi=140); plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()

