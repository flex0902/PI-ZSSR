"""
Aggregate E3: regional RMSE table and C-A difference maps.

  python summarize_e3.py --dir e3
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


KEYS = ["disc_rmse_ROI", "disc_rmse_R1", "disc_rmse_R2", "disc_rmse_R3",
        "oracle_rmse_ROI", "oracle_rmse_R2", "disc_epoch",
        "tik_1500_rmse_R1", "tik_1500_rmse_R2", "tik_5000_rmse_R1", "tik_5000_rmse_R2"]


def fmt(m, s):
    if not np.isfinite(m):
        return "–"
    if s < 5e-3:
        return f"{m:.2f}"
    return f"{m:.2f} ± {s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e3")
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*_metrics.json"))):
        m = json.load(open(p, encoding="utf-8"))
        rows.append(m)
    if not rows:
        print("no metrics"); return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.dir, "E3_runs.csv"), index=False)

    lines = ["# E3 dual-constraint regional RMSE (0 m truth, mGal)\n"]
    g = df.groupby("tag")
    tab = pd.DataFrame(index=sorted(g.groups))
    for k in KEYS:
        tab[k] = [fmt(g[k].mean().loc[c], g[k].std().fillna(0).loc[c]) for c in tab.index]
        tab[k + "_n"] = [int(g[k].count().loc[c]) for c in tab.index]
    md = tab[[c for c in tab.columns if not c.endswith("_n")]].to_string()
    lines.append(md)
    open(os.path.join(args.dir, "E3_table.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

    # maps from seed 0 if present
    fa = os.path.join(args.dir, "A_s0_fields.npz")
    fc = os.path.join(args.dir, "C_s0_fields.npz")
    fd = os.path.join(args.dir, "D_s0_fields.npz")
    if not (os.path.exists(fa) and os.path.exists(fc)):
        return
    A = np.load(fa, allow_pickle=True)
    C = np.load(fc, allow_pickle=True)
    lons = np.asarray(A["lons"])
    lats = np.asarray(A["lats"])
    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    roi = A["roi"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    lim = np.nanmax(np.abs(A["truth0"][roi]))

    def show(a, arr, title, lim=lim, cmap="RdYlBu_r"):
        im = a.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
        a.set_title(title, fontsize=10)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    LON, LAT = np.meshgrid(lons, lats)
    # every 2nd pixel: circles stay visible instead of filling a solid block
    step = 2
    colors = {"R1": "#3b528b", "R2": "#5ec962", "R3": "#fde725"}
    a0 = ax[0, 0]
    for name, mask in (("R3", A["R3"]), ("R1", A["R1"]), ("R2", A["R2"])):
        m = np.asarray(mask)[::step, ::step]
        a0.scatter(LON[::step, ::step][m], LAT[::step, ::step][m],
                   s=3.5, c=colors[name], marker="o", linewidths=0, rasterized=True, zorder=2)
    a0.set_xlim(ext[0], ext[1])
    a0.set_ylim(ext[2], ext[3])
    a0.set_title("coverage")
    a0.set_facecolor("#f4f4f4")

    def label_at(mask, lon0, lat0, text, color, z=4):
        m = np.asarray(mask)
        d = (LON - lon0) ** 2 + (LAT - lat0) ** 2
        d = np.where(m, d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a0.text(LON[i, j], LAT[i, j], text, color=color, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=z,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    label_at(A["R1"], 119.85, 23.55, "R1", colors["R1"])   # land / offshore, west of R1–R2 boundary
    label_at(A["R2"], 121.05, 23.70, "R2", "#1b7a3d")      # mountains
    label_at(A["R3"], 123.25, 23.40, "R3", "#8a6d00")      # yellow, no survey
    show(ax[0, 1], A["truth0"], "truth 0 m")
    show(ax[0, 2], C["pred_disc"], "C (discrepancy)")
    show(ax[1, 0], A["pred_disc"], "A (discrepancy)")
    d = C["pred_disc"] - A["pred_disc"]
    show(ax[1, 1], d, "C − A", lim=np.nanmax(np.abs(d[roi])), cmap="RdBu_r")
    err = A["truth0"] - C["pred_disc"]
    show(ax[1, 2], err, "truth − C", lim=np.nanmax(np.abs(err[roi])), cmap="RdBu_r")
    plt.tight_layout()
    plt.savefig(os.path.join(args.dir, "E3_maps_s0.png"), dpi=140)
    plt.close()
    print("saved", os.path.join(args.dir, "E3_maps_s0.png"))
    if os.path.exists(fd):
        print("D_s0 present")


if __name__ == "__main__":
    main()
