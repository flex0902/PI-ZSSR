"""
Section 4.5: PSD and coherence of the 0 m PI-ZSSR field vs XGM2019e / EGM2008.

Compares on the prediction's native 1-arcmin grid (models are interpolated
onto those nodes; the prediction is never upsampled to 30"). Statistics use
the 20-pixel FFT margin. Residual mode subtracts EGM2008 n<=2000 from every
field so the comparison is not dominated by the RCR restore.

The first degree where coherence drops below 0.5 is reported as a crossing,
not as an "effective resolution".
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata

NA = ["nan", "NaN", "NAN", "None"]
EARTH_KM = 40030.0
EGM_NMAX = 2160


def load_xyz(path):
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, names=["lon", "lat", "v"], na_values=NA
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["lon", "lat"])
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())
    g = df.pivot_table(index="lat", columns="lon", values="v", aggfunc="mean")
    g = g.reindex(index=lats, columns=lons).values.astype(float)
    return lons, lats, g


def sample_on(lons_src, lats_src, z_src, lons_dst, lats_dst):
    """Interpolate a source grid onto the destination lon/lat nodes."""
    if (
        lons_src.shape == lons_dst.shape
        and lats_src.shape == lats_dst.shape
        and np.allclose(lons_src, lons_dst, atol=1e-6)
        and np.allclose(lats_src, lats_dst, atol=1e-6)
        and z_src.shape == (len(lats_dst), len(lons_dst))
    ):
        return z_src.copy()

    finite = np.isfinite(z_src)
    if finite.all() and lats_src.size > 1 and lons_src.size > 1:
        interp = RegularGridInterpolator(
            (lats_src, lons_src), z_src, bounds_error=False, fill_value=np.nan
        )
        LAT, LON = np.meshgrid(lats_dst, lons_dst, indexing="ij")
        out = interp(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    else:
        LAT_s, LON_s = np.meshgrid(lats_src, lons_src, indexing="ij")
        LAT, LON = np.meshgrid(lats_dst, lons_dst, indexing="ij")
        pts = np.column_stack([LON_s[finite], LAT_s[finite]])
        out = griddata(pts, z_src[finite], (LON, LAT), method="linear")
    if np.isnan(out).any():
        LAT_s, LON_s = np.meshgrid(lats_src, lons_src, indexing="ij")
        LAT, LON = np.meshgrid(lats_dst, lons_dst, indexing="ij")
        finite = np.isfinite(z_src)
        near = griddata(
            np.column_stack([LON_s[finite], LAT_s[finite]]),
            z_src[finite],
            (LON, LAT),
            method="nearest",
        )
        out = np.where(np.isfinite(out), out, near)
    return out


def finite_mask_on(path, lons, lats):
    lo, la, g = load_xyz(path)
    g = sample_on(lo, la, g, lons, lats)
    return np.isfinite(g)


def roi_slice(ny, nx, margin):
    return slice(margin, ny - margin), slice(margin, nx - margin)


def grid_spacing_km(lons, lats):
    lat0 = 0.5 * (lats[0] + lats[-1])
    dy = (lats[1] - lats[0]) * 111.32
    dx = (lons[1] - lons[0]) * 111.32 * np.cos(np.radians(lat0))
    return float(dx), float(dy)


def radial_psd_coh(z1, z2, dx_km, dy_km):
    """Hanning-windowed 2-D PSD and magnitude-squared coherence, radially averaged."""
    a = np.asarray(z1, float)
    b = np.asarray(z2, float)
    good = np.isfinite(a) & np.isfinite(b)
    a = np.where(good, a, np.nanmean(a[good]))
    b = np.where(good, b, np.nanmean(b[good]))
    a = a - a.mean()
    b = b - b.mean()
    ny, nx = a.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    a = a * win
    b = b * win
    F1 = np.fft.fftshift(np.fft.fft2(a))
    F2 = np.fft.fftshift(np.fft.fft2(b))
    S11 = np.abs(F1) ** 2
    S22 = np.abs(F2) ** 2
    S12 = F1 * np.conj(F2)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)
    df = 0.5 * (1.0 / (nx * dx_km) + 1.0 / (ny * dy_km))
    nb = max(int(np.ceil(kr.max() / df)), 1)
    bins = np.linspace(0.0, nb * df, nb + 1)
    idx = np.digitize(kr.ravel(), bins) - 1
    w = np.ones_like(kr.ravel(), dtype=float)
    n = np.bincount(idx, w, minlength=nb)
    n = np.maximum(n, 1.0)
    P11 = np.bincount(idx, S11.ravel(), minlength=nb) / n
    P22 = np.bincount(idx, S22.ravel(), minlength=nb) / n
    C12r = np.bincount(idx, np.real(S12).ravel(), minlength=nb) / n
    C12i = np.bincount(idx, np.imag(S12).ravel(), minlength=nb) / n
    coh = (C12r ** 2 + C12i ** 2) / (P11 * P22 + 1e-30)
    freqs = 0.5 * (bins[:-1] + bins[1:])
    return freqs, P11, P22, np.clip(coh, 0.0, 1.0)


def to_degree(freqs):
    f = np.asarray(freqs)
    lam = np.where(f > 0, 1.0 / f, np.inf)
    deg = np.where(np.isfinite(lam), EARTH_KM / lam, 0.0)
    return deg, lam


def smooth(y, w=5):
    if y.size < w:
        return y.copy()
    box = np.ones(w) / w
    return np.convolve(y, box, mode="same")


def first_half_crossing(deg, coh, deg_max, deg_min=50):
    m = (deg > deg_min) & (deg < deg_max) & np.isfinite(coh)
    if not np.any(m):
        return None
    d, c = deg[m], coh[m]
    below = np.where(c < 0.5)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    return {"degree": float(d[i]), "wavelength_km": float(EARTH_KM / d[i]), "coherence": float(c[i])}


def band_mean(deg, y, lo, hi):
    m = (deg >= lo) & (deg < hi) & np.isfinite(y)
    return float(np.mean(y[m])) if np.any(m) else None


def region_stats(pred, model, mask):
    d = pred - model
    m = mask & np.isfinite(d)
    if not np.any(m):
        return {"n": 0, "mean": None, "rmse": None, "std": None}
    r = d[m]
    return {
        "n": int(m.sum()),
        "mean": float(r.mean()),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "std": float(r.std()),
    }


def wavelength_axis(ax, max_deg):
    def fwd(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.asarray(x) == 0, np.inf, EARTH_KM / np.asarray(x))

    sec = ax.secondary_xaxis("top", functions=(fwd, fwd))
    sec.set_xlabel("Wavelength (km)")
    ticks = [2000, 1000, 500, 200, 100, 50, 30, 20, 15, 10, 8]
    ticks = [L for L in ticks if EARTH_KM / L <= max_deg]
    sec.set_xticks(ticks)
    sec.set_xticklabels([str(L) for L in ticks])
    return sec


def plot_psd_coh(deg, curves, title, out_png, max_deg, annotate=None, ylim=None):
    """
    curves: list of dicts with keys kind in {psd, coh}, y, label, color, ls, nmax
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    trim = 2
    d = deg[trim:-trim] if deg.size > 2 * trim else deg

    for c in curves:
        y = c["y"]
        y = smooth(y, 5)
        y = y[trim:-trim] if y.size > 2 * trim else y
        dd = d
        if c.get("nmax") is not None:
            keep = dd <= c["nmax"]
            dd, y = dd[keep], y[keep]
        if c["kind"] == "psd":
            ax1.semilogy(dd, y, color=c["color"], ls=c.get("ls", "-"), lw=1.8, label=c["label"])
        else:
            ax2.plot(dd, y, color=c["color"], ls=c.get("ls", "-"), lw=1.8, label=c["label"])

    ax1.set_xlabel("Spherical harmonic degree (d/o)")
    ax1.set_ylabel("Power spectral density")
    ax1.set_title("Radially averaged PSD")
    ax1.grid(True, which="both", ls="--", alpha=0.45)
    ax1.legend(frameon=False)
    ax1.set_xlim(max(np.min(d), 1), max_deg)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.axvline(2000, color="0.5", ls=":", lw=0.8)
    ax1.axvline(EGM_NMAX, color="0.5", ls=":", lw=0.8)
    wavelength_axis(ax1, max_deg)

    ax2.axhline(0.5, color="k", ls="--", lw=0.9, label="coherence = 0.5")
    ax2.set_xlabel("Spherical harmonic degree (d/o)")
    ax2.set_ylabel("Magnitude-squared coherence")
    ax2.set_title("Spatial coherence")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, ls="--", alpha=0.45)
    ax2.legend(frameon=False)
    ax2.set_xlim(max(np.min(d), 1), max_deg)
    ax2.axvline(2000, color="0.5", ls=":", lw=0.8)
    ax2.axvline(EGM_NMAX, color="0.5", ls=":", lw=0.8)
    wavelength_axis(ax2, max_deg)

    if annotate:
        ax2.plot(annotate["degree"], 0.5, "ko", ms=6)
        ax2.annotate(
            f"first <0.5 vs XGM\nd/o {annotate['degree']:.0f} (~{annotate['wavelength_km']:.1f} km)",
            xy=(annotate["degree"], 0.5),
            xytext=(min(annotate["degree"] + 250, max_deg * 0.55), 0.62),
            arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
            fontsize=8,
        )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out_png}")


def pack_spectrum(name, deg, P_pred, P_ref, coh, max_deg, deg_min=50):
    bands = [(2, 2000), (2000, 2160), (2160, 3000), (3000, 4000)]
    out = {
        "first_half_crossing": first_half_crossing(deg, coh, max_deg, deg_min=deg_min),
        "coherence_bands": {f"{a}-{b}": band_mean(deg, coh, a, b) for a, b in bands},
        "psd_ratio_bands": {
            f"{a}-{b}": (
                band_mean(deg, P_pred, a, b) / band_mean(deg, P_ref, a, b)
                if band_mean(deg, P_ref, a, b) not in (None, 0)
                else None
            )
            for a, b in bands
        },
    }
    return out


def crop_roi(z, slat, slon):
    return z[slat, slon]


def main():
    p = argparse.ArgumentParser(description="§4.5 PSD/coherence on the native 1' prediction grid")
    p.add_argument("--pred_file", default="ch4/real0_C_full_g_h0.xyz", help="configuration C, full g at 0 m")
    p.add_argument("--pred_b", default="", help="optional second prediction (A, full g) for C vs A")
    p.add_argument("--xgm_file", default="TW_XGM_grid_g_d5540_h0.xyz")
    p.add_argument("--egm_file", default="TW_EGM_grid_g_d2160_h0.xyz")
    p.add_argument("--ref_n2000", default="TW_EGM_grid_g_d2000_h0.xyz", help="EGM2008 n<=2000 at 0 m")
    p.add_argument("--mask_z1", default="1500(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--mask_z2", default="5000(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--margin", type=int, default=20)
    p.add_argument("--max_deg", type=int, default=5500)
    p.add_argument("--output_prefix", default="ch4/fig20")
    p.add_argument(
        "--mode",
        choices=["full", "residual", "both"],
        default="both",
        help="full g, residual after removing n<=2000, or both",
    )
    args = p.parse_args()

    for path in [args.pred_file, args.xgm_file, args.egm_file, args.ref_n2000]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    lons, lats, zC = load_xyz(args.pred_file)
    print(f"master grid from {args.pred_file}: {len(lats)} x {len(lons)}  dlon={lons[1]-lons[0]:.6f} deg")

    zX = sample_on(*load_xyz(args.xgm_file), lons, lats)
    zE = sample_on(*load_xyz(args.egm_file), lons, lats)
    z0 = sample_on(*load_xyz(args.ref_n2000), lons, lats)

    zA = None
    if args.pred_b:
        if not os.path.exists(args.pred_b):
            raise FileNotFoundError(args.pred_b)
        zA = sample_on(*load_xyz(args.pred_b), lons, lats)

    ny, nx = zC.shape
    slat, slon = roi_slice(ny, nx, args.margin)
    roi = np.zeros((ny, nx), dtype=bool)
    roi[slat, slon] = True

    om1 = finite_mask_on(args.mask_z1, lons, lats)
    om2 = finite_mask_on(args.mask_z2, lons, lats)
    R1 = om1
    R2 = om2 & ~om1
    R3 = ~om1 & ~om2
    print(f"pixels  R1={R1.sum()}  R2={R2.sum()}  R3={R3.sum()}  ROI={roi.sum()}")

    dx, dy = grid_spacing_km(lons, lats)
    print(f"spacing  dx={dx:.3f} km  dy={dy:.3f} km")

    fields = {
        "full": {"C": zC, "XGM": zX, "EGM": zE, "A": zA},
        "residual": {"C": zC - z0, "XGM": zX - z0, "EGM": zE - z0, "A": None if zA is None else zA - z0},
    }

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stats = {
        "pred_file": args.pred_file,
        "pred_b": args.pred_b or None,
        "grid": {"nlat": int(len(lats)), "nlon": int(len(lons)), "dx_km": dx, "dy_km": dy, "margin": args.margin},
        "counts": {"R1": int(R1.sum()), "R2": int(R2.sum()), "R3": int(R3.sum()), "ROI": int(roi.sum())},
        "note": "first_half_crossing is the first d/o>50 where coherence vs XGM falls below 0.5; not a resolution claim.",
    }

    modes = ["full", "residual"] if args.mode == "both" else [args.mode]
    for mode in modes:
        F = fields[mode]
        C_roi = crop_roi(F["C"], slat, slon)
        X_roi = crop_roi(F["XGM"], slat, slon)
        E_roi = crop_roi(F["EGM"], slat, slon)

        freqs, Pc, Px, cohX = radial_psd_coh(C_roi, X_roi, dx, dy)
        _, _, Pe, cohE = radial_psd_coh(C_roi, E_roi, dx, dy)
        deg, _ = to_degree(freqs)
        valid = freqs > 0
        deg, Pc, Px, Pe, cohX, cohE = deg[valid], Pc[valid], Px[valid], Pe[valid], cohX[valid], cohE[valid]

        deg_min = 2000 if mode == "residual" else 50
        crossing = first_half_crossing(deg, cohX, args.max_deg, deg_min=deg_min)
        stats[mode] = {
            "C_vs_XGM": pack_spectrum("C-XGM", deg, Pc, Px, cohX, args.max_deg, deg_min=deg_min),
            "C_vs_EGM": pack_spectrum("C-EGM", deg, Pc, Pe, cohE, min(args.max_deg, EGM_NMAX), deg_min=deg_min),
            "spatial_vs_XGM": {
                k: region_stats(F["C"], F["XGM"], m)
                for k, m in [("ROI", roi), ("R1", R1 & roi), ("R2", R2 & roi), ("R3", R3 & roi)]
            },
            "spatial_vs_EGM": {
                k: region_stats(F["C"], F["EGM"], m)
                for k, m in [("ROI", roi), ("R1", R1 & roi), ("R2", R2 & roi), ("R3", R3 & roi)]
            },
        }

        title = (
            "Full gravity at 0 m (ROI, native 1')"
            if mode == "full"
            else "Residual gravity at 0 m after removing EGM2008 n≤2000 (ROI, native 1')"
        )
        plot_psd_coh(
            deg,
            [
                {"kind": "psd", "y": Pc, "label": "PI-ZSSR C (0 m)", "color": "C2", "ls": "-"},
                {"kind": "psd", "y": Px, "label": "XGM2019e (d/o 5540)", "color": "C0", "ls": "--"},
                {"kind": "psd", "y": Pe, "label": "EGM2008 (d/o 2160)", "color": "C3", "ls": "--", "nmax": EGM_NMAX},
                {"kind": "coh", "y": cohX, "label": "C vs XGM2019e", "color": "C0", "ls": "-"},
                {"kind": "coh", "y": cohE, "label": "C vs EGM2008", "color": "C3", "ls": "--", "nmax": EGM_NMAX},
            ],
            title,
            f"{args.output_prefix}_{mode}_psd_coh.png",
            args.max_deg,
            annotate=crossing if mode == "full" else crossing,
        )

        if F["A"] is not None:
            A_roi = crop_roi(F["A"], slat, slon)
            _, Pa, _, cohAX = radial_psd_coh(A_roi, X_roi, dx, dy)
            _, _, _, cohAE = radial_psd_coh(A_roi, E_roi, dx, dy)
            Pa, cohAX, cohAE = Pa[valid], cohAX[valid], cohAE[valid]
            stats[mode]["A_vs_XGM"] = pack_spectrum("A-XGM", deg, Pa, Px, cohAX, args.max_deg, deg_min=deg_min)
            stats[mode]["spatial_A_vs_XGM"] = {
                k: region_stats(F["A"], F["XGM"], m)
                for k, m in [("ROI", roi), ("R1", R1 & roi), ("R2", R2 & roi), ("R3", R3 & roi)]
            }
            plot_psd_coh(
                deg,
                [
                    {"kind": "psd", "y": Pc, "label": "PI-ZSSR C", "color": "C2", "ls": "-"},
                    {"kind": "psd", "y": Pa, "label": "PI-ZSSR A", "color": "C1", "ls": "-"},
                    {"kind": "psd", "y": Px, "label": "XGM2019e (d/o 5540)", "color": "C0", "ls": "--"},
                    {"kind": "psd", "y": Pe, "label": "EGM2008 (d/o 2160)", "color": "C3", "ls": "--", "nmax": EGM_NMAX},
                    {"kind": "coh", "y": cohX, "label": "C vs XGM", "color": "C2", "ls": "-"},
                    {"kind": "coh", "y": cohAX, "label": "A vs XGM", "color": "C1", "ls": "-"},
                    {"kind": "coh", "y": cohE, "label": "C vs EGM", "color": "C3", "ls": "--", "nmax": EGM_NMAX},
                ],
                ("Full gravity at 0 m, C vs A (ROI, native 1')" if mode == "full"
             else "Residual gravity at 0 m after removing EGM2008 n≤2000, C vs A (ROI, native 1')"),
                f"{args.output_prefix}_{mode}_CA_psd_coh.png",
                args.max_deg,
            )

        print(f"\n=== {mode} ===")
        print(f"  first coherence<0.5 vs XGM: {crossing}")
        for pair, key in [("C vs XGM", "spatial_vs_XGM"), ("C vs EGM", "spatial_vs_EGM")]:
            print(f"  {pair} RMSE (mGal):")
            for reg, rec in stats[mode][key].items():
                print(f"    {reg:4s}  n={rec['n']:6d}  mean={rec['mean']:+7.3f}  rmse={rec['rmse']:.3f}")
        print("  coherence bands vs XGM:", stats[mode]["C_vs_XGM"]["coherence_bands"])

    stats_path = f"{args.output_prefix}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nsaved {stats_path}")


if __name__ == "__main__":
    main()
