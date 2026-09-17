"""Paper figures for Section 4.4."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_res_g_compare import extract_coastline_gmt, plot_coastline

NA = ["nan", "NaN", "NAN"]


def load_xyz(path):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=["lon", "lat", "v"], na_values=NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())
    g = df.pivot_table(index="lat", columns="lon", values="v", aggfunc="mean")
    g = g.reindex(index=lats, columns=lons).values.astype(float)
    return lons, lats, g


def show(ax, arr, ext, title, lim, cmap="RdYlBu_r", contour_int=None, lons=None, lats=None):
    im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim, aspect="equal")
    if contour_int and lons is not None:
        pos = np.arange(contour_int, lim + contour_int, contour_int)
        neg = np.arange(-lim, 0, contour_int)
        if len(pos):
            ax.contour(lons, lats, arr, levels=pos, colors="k", linewidths=0.45)
        if len(neg):
            ax.contour(lons, lats, arr, levels=neg, colors="k", linewidths=0.45, linestyles="--")
        ax.contour(lons, lats, arr, levels=[0], colors="k", linewidths=0.6)
    plot_coastline(ax)
    ax.set_xlim(119, 124)
    ax.set_ylim(21, 26)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mGal")


def main():
    A = np.load("ch4/real0_A_fields.npz")
    C = np.load("ch4/real0_C_fields.npz")
    lons, lats = C["lons"], C["lats"]
    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    extract_coastline_gmt(list(ext))
    gA, gC = A["pred_disc"], C["pred_disc"]
    d = gC - gA
    roi, R2 = C["roi"], C["R2"]
    lim = np.nanmax(np.abs(gC[roi]))

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.7), sharex=True, sharey=True)
    show(ax[0], gA, ext, r"A: $\hat g_0$ (1620 m examiner only)", lim)
    show(ax[1], gC, ext, r"C: $\hat g_0$ (1620 m + 5156 m)", lim)
    show(ax[2], d, ext, r"C $-$ A", np.nanmax(np.abs(d[roi])), cmap="RdBu_r")
    ax[2].contour(lons, lats, R2.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    fig.tight_layout()
    fig.savefig("ch4/fig18_g0_AC.png", dpi=180)
    plt.close()
    print("saved ch4/fig18_g0_AC.png")

    paths = [
        ("ch4/real0_C_full_g_h0.xyz", "0 m (PI-ZSSR C)"),
        ("ch4/real0_lsc_full_g_h1620.xyz", "1620 m LSC"),
        ("ch4/real0_lsc_full_g_h5156.xyz", "5156 m LSC"),
    ]
    grids = []
    for p, t in paths:
        lo, la, g = load_xyz(p)
        grids.append((lo, la, g, t))
    limf = max(np.nanmax(np.abs(g[np.isfinite(g)])) for _, _, g, _ in grids)
    limf = min(limf, 350.0)
    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.7), sharex=True, sharey=True)
    for a, (lo, la, g, t) in zip(ax, grids):
        e = (lo.min(), lo.max(), la.min(), la.max())
        show(a, g, e, t, limf, contour_int=50, lons=lo, lats=la)
    fig.tight_layout()
    fig.savefig("ch4/fig19_full_g.png", dpi=180)
    plt.close()
    print("saved ch4/fig19_full_g.png  lim", limf)


if __name__ == "__main__":
    main()
