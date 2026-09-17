"""Paper figures for Section 4.3: common-point difference maps, difference PSDs, field PSDs."""
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


def radial_psd(grid, dx_km, dy_km):
    ny, nx = grid.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    g = (grid - np.nanmean(grid)) * win
    g = np.nan_to_num(g, nan=0.0)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)
    df = 0.5 * (1.0 / (nx * dx_km) + 1.0 / (ny * dy_km))
    nb = int(np.ceil(kr.max() / df))
    bins = np.linspace(0.0, nb * df, nb + 1)
    idx = np.digitize(kr.ravel(), bins) - 1
    p = np.bincount(idx, spec.ravel(), minlength=nb) / np.maximum(np.bincount(idx, minlength=nb), 1)
    f = 0.5 * (bins[:-1] + bins[1:])
    return f, p


def band_power(f, p, lam_hi, lam_lo):
    m = (f >= 1.0 / lam_hi) & (f < 1.0 / lam_lo) & (f > 0)
    if not np.any(m):
        return np.nan
    return float(np.trapezoid(p[m], f[m]))


def main():
    lons, lats, g1620 = load_xyz("1500(10s)_new.xyg_pred_grid_0.1.xyz")
    _, _, g5156 = load_xyz("5000(10s)_new.xyg_pred_grid_0.1.xyz")
    _, _, gpiz = load_xyz("ch4/pizssr_5156to1620_1620m.xyz")
    _, _, glsc = load_xyz("ch4/lsc_dc_5156to1620.xyz")
    common = np.isfinite(g1620) & np.isfinite(g5156)
    print(f"common pixels: {common.sum()}")

    dx_km = (lons[1] - lons[0]) * 111.0 * np.cos(np.radians(lats.mean()))
    dy_km = (lats[1] - lats[0]) * 111.0
    ext = (lons.min(), lons.max(), lats.min(), lats.max())
    extract_coastline_gmt(list(ext))

    d_piz = np.where(common, g1620 - gpiz, np.nan)
    d_lsc = np.where(common, g1620 - glsc, np.nan)
    lim = np.nanmax(np.abs(np.r_[d_piz[common], d_lsc[common]]))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    titles = [
        f"1620 m LSC − PI-ZSSR\nRMSE {np.sqrt(np.mean(d_piz[common]**2)):.2f} mGal",
        f"1620 m LSC − LSC-only DC\nRMSE {np.sqrt(np.mean(d_lsc[common]**2)):.2f} mGal",
    ]
    for a, d, t in zip(ax, [d_piz, d_lsc], titles):
        im = a.imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        plot_coastline(a)
        a.set_xlim(119, 124); a.set_ylim(21, 26)
        a.set_xlabel("Longitude"); a.set_title(t)
        a.set_ylabel("Latitude")
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="mGal")
    fig.tight_layout()
    fig.savefig("ch4/fig15_diff_maps.png", dpi=180)
    plt.close()
    print("saved ch4/fig15_diff_maps.png")

    fields = {
        "1620 m LSC": np.where(common, g1620, np.nan),
        "PI-ZSSR": np.where(common, gpiz, np.nan),
        "LSC-only DC": np.where(common, glsc, np.nan),
        "5156 m LSC": np.where(common, g5156, np.nan),
    }
    diffs = {"PI-ZSSR": d_piz, "LSC-only DC": d_lsc}
    psd_f = {}
    for name, g in {**fields, **{f"diff {k}": v for k, v in diffs.items()}}.items():
        psd_f[name] = radial_psd(g, dx_km, dy_km)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, col in [("diff PI-ZSSR", "C0"), ("diff LSC-only DC", "C3")]:
        f, p = psd_f[name]
        ax.loglog(f[1:], p[1:], color=col, lw=1.6, label=name.replace("diff ", ""))
    ax.set_xlabel("Frequency (cycles / km)")
    ax.set_ylabel("Power")
    ax.set_title("PSD of differences vs 1620 m LSC (common pixels)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    ymin, ymax = ax.get_ylim()
    for lam in [50, 20, 10, 5, 2]:
        ax.axvline(1.0 / lam, color="0.5", ls="--", lw=0.7)
        ax.text(1.0 / lam, ymax, f" {lam} km", rotation=90, va="top", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig("ch4/fig16_diff_psd.png", dpi=180)
    plt.close()
    print("saved ch4/fig16_diff_psd.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = [("1620 m LSC", "k", 2.0), ("PI-ZSSR", "C0", 1.6), ("LSC-only DC", "C3", 1.6), ("5156 m LSC", "0.55", 1.2)]
    for name, col, lw in styles:
        f, p = psd_f[name]
        ax.loglog(f[1:], p[1:], color=col, lw=lw, label=name)
    ax.set_xlabel("Frequency (cycles / km)")
    ax.set_ylabel("Power")
    ax.set_title("Residual-field PSD on common pixels")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    ymin, ymax = ax.get_ylim()
    for lam in [50, 20, 10, 5, 2]:
        ax.axvline(1.0 / lam, color="0.6", ls="--", lw=0.7)
        ax.text(1.0 / lam, ymax, f" {lam} km", rotation=90, va="top", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig("ch4/fig17_field_psd.png", dpi=180)
    plt.close()
    print("saved ch4/fig17_field_psd.png")

    f_ref, p_ref = psd_f["1620 m LSC"]
    print("\nband-power ratios to 1620 m LSC")
    for name in ["PI-ZSSR", "LSC-only DC", "5156 m LSC"]:
        f, p = psd_f[name]
        row = [name]
        for hi, lo in [(50, 20), (20, 10), (10, 5)]:
            r = band_power(f, p, hi, lo) / band_power(f_ref, p_ref, hi, lo)
            row.append(f"{hi}-{lo} km: {r:.2f}")
        print("  ", "  ".join(row))
    print("\ndifference band power (PI-ZSSR / LSC-only)")
    fp, pp = psd_f["diff PI-ZSSR"]
    fl, pl = psd_f["diff LSC-only DC"]
    for hi, lo in [(100, 50), (50, 20), (20, 10), (10, 5)]:
        r = band_power(fp, pp, hi, lo) / band_power(fl, pl, hi, lo)
        print(f"  {hi}-{lo} km: {r:.2f}")


if __name__ == "__main__":
    main()
