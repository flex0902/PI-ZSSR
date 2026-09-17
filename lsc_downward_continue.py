"""
LSC downward continuation of a regular residual gravity grid.

Fits a Tscherning-Rapp covariance at the observation height (from flight-line
residuals), then applies a local LSC kernel on the filled high-altitude grid
to predict the same lon/lat at a lower height.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform

from airborne_data_process import GravityGridDataset, load_observation_data
from lsc_covariance_fit import (
    R_EARTH,
    calculate_empirical_covariance,
    compute_covariance_tscherning_rapp,
    fit_tscherning_rapp,
    load_tr_parameters,
    save_tr_parameters,
)
from lsc_covariance_fit_B import fit_with_fixed_B, precompute_legendre


def load_regular_xyz(path):
    print(f"Loading grid: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", names=["lon", "lat", "dg"])
    df = df.apply(pd.to_numeric, errors="coerce")
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())
    grid = (
        df.pivot(index="lat", columns="lon", values="dg")
        .reindex(index=lats, columns=lons)
        .to_numpy(dtype=float)
    )
    print(f"  Grid {grid.shape[0]}x{grid.shape[1]}, "
          f"lon[{lons.min():.4f}, {lons.max():.4f}], "
          f"lat[{lats.min():.4f}, {lats.max():.4f}], "
          f"NaNs={np.isnan(grid).sum()}")
    return lons, lats, grid


def save_regular_xyz(path, lons, lats, grid):
    LON, LAT = np.meshgrid(lons, lats)
    data = np.column_stack([LON.ravel(), LAT.ravel(), grid.ravel()])
    np.savetxt(path, data, fmt="%.6f %.6f %.3f", header="lon lat value", comments="# ")
    print(f"Saved {path}")


def spherical_xyz(lat_deg, lon_deg):
    lat = np.radians(np.asarray(lat_deg))
    lon = np.radians(np.asarray(lon_deg))
    return np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ))


def chord_to_psi_deg(chord):
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def find_best_B_at_height(cov_df, flight_height, n_min, n_max, b_min=4, b_max=10000):
    """Integer-B search that actually uses the observation height."""
    print(f"--- Optimizing B at H={flight_height:.1f} m (range {b_min}-{b_max}) ---")
    data = cov_df.dropna()
    psi_vals = np.radians(data["psi_deg"].values)
    emp_cov = data["cov"].values
    P_matrix = precompute_legendre(psi_vals, n_max)

    best_B = None
    best_params = None
    lowest_ssr = np.inf
    prev_ssr = None

    for b_val in range(b_min, b_max + 1):
        A, depth, curve, ssr = fit_with_fixed_B(
            psi_vals, emp_cov, P_matrix, b_val, n_min, n_max, flight_height=flight_height
        )
        if A is None:
            continue
        if ssr < lowest_ssr:
            lowest_ssr = ssr
            best_B = b_val
            best_params = (A, depth, curve)
        if prev_ssr is not None and ssr < prev_ssr and (prev_ssr - ssr) / prev_ssr < 1e-4:
            print(f"   Stopping B search at B={b_val}: SSR flattened.")
            break
        prev_ssr = ssr
        if b_val % 200 == 0:
            print(f"   B={b_val}: SSR={ssr:.4f} (best B={best_B}, SSR={lowest_ssr:.4f})")

    print(f"   Best B={best_B}, SSR={lowest_ssr:.4f}")
    return best_B, best_params


def fit_covariance_from_tracks(obs_file, egm_file, obs_h, n_min, n_max, param_out,
                               b_min=4, b_max=10000, emp_cov_file="empirical_covariance_data_5156m.csv"):
    if emp_cov_file and os.path.exists(emp_cov_file):
        print(f"Reusing empirical covariance: {emp_cov_file}")
        cov_df = pd.read_csv(emp_cov_file)
    else:
        df_obs = load_observation_data(obs_file)
        egm = GravityGridDataset(egm_file, size=(301, 301))
        df_resid = pd.DataFrame({
            "lon": df_obs["lon"].values,
            "lat": df_obs["lat"].values,
            "dg": df_obs["dg"].values - egm.get_values_at(df_obs["lon"], df_obs["lat"]),
        })
        print(f"Residual stats: mean={df_resid['dg'].mean():.3f}, std={df_resid['dg'].std():.3f} mGal")
        cov_df = calculate_empirical_covariance(df_resid, max_dist_deg=0.2, bin_size_sec=40)
        cov_df.to_csv(emp_cov_file, index=False)

    print("Fitting T-R with fixed B=24 ...")
    fit_tscherning_rapp(cov_df, n_min=n_min, n_max=n_max, fixed_B=24, flight_height=obs_h)

    best_B, params = find_best_B_at_height(
        cov_df, obs_h, n_min, n_max, b_min=b_min, b_max=b_max
    )
    if best_B is None:
        raise RuntimeError("T-R B optimization failed.")
    A, depth, _ = params
    R_B = R_EARTH - depth
    save_tr_parameters(A, R_B, best_B, obs_h, filename=param_out)
    print(f"Fitted A={A:.5f} mGal^2, B={best_B}, R_B={R_B:.2f} m, depth={depth/1000:.4f} km")
    return A, R_B, best_B


def load_or_fit_params(args):
    if args.params and os.path.exists(args.params):
        p = load_tr_parameters(args.params)
        print(f"Loaded T-R parameters from {args.params}")
        print(f"  A={p['A']:.5f}, B={p['B']}, R_B={p['R_B']:.2f}, H={p.get('flight_height', 'NA')}")
        if abs(p.get("flight_height", args.obs_h) - args.obs_h) > 1.0:
            print(f"  Warning: parameter file height {p.get('flight_height')} differs from --obs_h {args.obs_h}")
        return p["A"], p["R_B"], int(p["B"])
    if not args.obs or not args.egm:
        raise ValueError("Need --params, or both --obs and --egm to fit a 5156 m covariance.")
    return fit_covariance_from_tracks(
        args.obs, args.egm, args.obs_h, args.n_min, args.n_max, args.param_out,
        b_min=args.b_min, b_max=args.b_max
    )


def stencil_offsets(lat0, dlon, dlat, max_dist_deg):
    nlat = int(np.ceil(max_dist_deg / dlat)) + 1
    lon_scale = max(np.cos(np.radians(lat0)), 0.2)
    nlon = int(np.ceil(max_dist_deg / (dlon * lon_scale))) + 2
    di, dj = np.meshgrid(np.arange(-nlat, nlat + 1), np.arange(-nlon, nlon + 1), indexing="ij")
    lat = lat0 + di * dlat
    lon = 0.0 + dj * dlon
    xyz_c = spherical_xyz([lat0], [0.0])[0]
    xyz = spherical_xyz(lat.ravel(), lon.ravel())
    psi = chord_to_psi_deg(np.linalg.norm(xyz - xyz_c, axis=1)).reshape(lat.shape)
    mask = psi <= max_dist_deg + 1e-12
    return di[mask], dj[mask]


def build_lsc_kernel(lat0, dlon, dlat, max_dist_deg, A, R_B, B, n_max, obs_h, pred_h, noise_std):
    di, dj = stencil_offsets(lat0, dlon, dlat, max_dist_deg)
    lat_s = lat0 + di * dlat
    lon_s = dj * dlon
    xyz = spherical_xyz(lat_s, lon_s)

    chord = pdist(xyz, metric="euclidean")
    psi_off = chord_to_psi_deg(chord)
    cov_off = compute_covariance_tscherning_rapp(
        psi_off, A, R_B, B, n_max=n_max, cov_type="gg", h1=obs_h, h2=obs_h
    )
    C = squareform(cov_off)
    var_obs = compute_covariance_tscherning_rapp(
        0.0, A, R_B, B, n_max=n_max, cov_type="gg", h1=obs_h, h2=obs_h
    )
    np.fill_diagonal(C, var_obs)
    C = C + np.eye(len(di)) * (noise_std ** 2)

    xyz_p = spherical_xyz([lat0], [0.0])[0]
    psi_cross = chord_to_psi_deg(np.linalg.norm(xyz - xyz_p, axis=1))
    c_cross = compute_covariance_tscherning_rapp(
        psi_cross, A, R_B, B, n_max=n_max, cov_type="gg", h1=obs_h, h2=pred_h
    )
    var_pred = compute_covariance_tscherning_rapp(
        0.0, A, R_B, B, n_max=n_max, cov_type="gg", h1=pred_h, h2=pred_h
    )

    try:
        weights = solve(C, c_cross, assume_a="pos")
    except np.linalg.LinAlgError:
        C = C + np.eye(len(di)) * 1e-6
        weights = solve(C, c_cross, assume_a="sym")

    pred_var = float(var_pred - np.dot(c_cross, weights))
    pred_std = float(np.sqrt(max(pred_var, 0.0)))
    return di.astype(int), dj.astype(int), weights, pred_std


def continue_grid(lons, lats, grid, A, R_B, B, n_max, obs_h, pred_h, max_dist_deg, noise_std):
    dlon = float(np.median(np.diff(lons)))
    dlat = float(np.median(np.diff(lats)))
    nlat, nlon = grid.shape
    pred = np.full_like(grid, np.nan, dtype=float)
    pred_std = np.full_like(grid, np.nan, dtype=float)

    print(f"Building per-latitude LSC kernels (dlat={dlat:.5f} deg, dlon={dlon:.5f} deg, "
          f"radius={max_dist_deg} deg) ...")

    kernel_cache = {}
    for i, lat0 in enumerate(lats):
        key = round(float(lat0), 8)
        if key not in kernel_cache:
            kernel_cache[key] = build_lsc_kernel(
                float(lat0), dlon, dlat, max_dist_deg, A, R_B, B, n_max, obs_h, pred_h, noise_std
            )
        di, dj, w, pstd = kernel_cache[key]
        ii = i + di
        for j in range(nlon):
            jj = j + dj
            if ii.min() < 0 or ii.max() >= nlat or jj.min() < 0 or jj.max() >= nlon:
                continue
            vals = grid[ii, jj]
            if np.isnan(vals).any():
                continue
            pred[i, j] = np.dot(w, vals)
            pred_std[i, j] = pstd
        if i % 50 == 0 or i == nlat - 1:
            print(f"  Row {i+1}/{nlat} (lat={lat0:.4f}), stencil={len(w)}, pred_std={pstd:.3f} mGal")

    n_valid = np.count_nonzero(~np.isnan(pred))
    print(f"Continuation done: {n_valid}/{pred.size} valid points")
    return pred, pred_std


def main():
    parser = argparse.ArgumentParser(description="LSC downward continuation of a residual gravity grid.")
    parser.add_argument("--input", type=str, required=True, help="High-altitude filled residual grid (XYZ)")
    parser.add_argument("--output", type=str, default="lsc_dc_h1620.xyz", help="Output continued residual grid")
    parser.add_argument("--obs", type=str, default=None, help="Flight-line XYZ/XYG used to fit covariance")
    parser.add_argument("--egm", type=str, default=None, help="EGM grid at observation height (for residuals)")
    parser.add_argument("--params", type=str, default=None, help="Existing T-R JSON; skip fitting if present")
    parser.add_argument("--param_out", type=str, default="tr_parameters_5156m_optimized.json")
    parser.add_argument("--obs_h", type=float, default=5156.0, help="Observation height (m)")
    parser.add_argument("--pred_h", type=float, default=1620.0, help="Prediction height (m)")
    parser.add_argument("--max_dist_deg", type=float, default=0.1, help="LSC search radius (degrees)")
    parser.add_argument("--noise_std", type=float, default=3.0,
                        help="Observation noise sigma in mGal (default 3.0, near airborne crossover error)")
    parser.add_argument("--b_min", type=int, default=4, help="Minimum integer B for T-R search")
    parser.add_argument("--b_max", type=int, default=10000, help="Maximum integer B for T-R search")
    parser.add_argument("--n_min", type=int, default=3)
    parser.add_argument("--n_max", type=int, default=3000)
    args = parser.parse_args()

    A, R_B, B = load_or_fit_params(args)
    lons, lats, grid = load_regular_xyz(args.input)
    pred, pred_std = continue_grid(
        lons, lats, grid, A, R_B, B, args.n_max,
        args.obs_h, args.pred_h, args.max_dist_deg, args.noise_std
    )
    save_regular_xyz(args.output, lons, lats, pred)
    std_path = os.path.splitext(args.output)[0] + "_std.xyz"
    save_regular_xyz(std_path, lons, lats, pred_std)

    meta = {
        "input": args.input,
        "output": args.output,
        "A": float(A),
        "R_B": float(R_B),
        "B": int(B),
        "obs_h": float(args.obs_h),
        "pred_h": float(args.pred_h),
        "max_dist_deg": float(args.max_dist_deg),
        "noise_std": float(args.noise_std),
        "b_min": int(args.b_min),
        "b_max": int(args.b_max),
        "n_max": int(args.n_max),
        "valid_points": int(np.count_nonzero(~np.isnan(pred))),
        "mean_pred_std": float(np.nanmean(pred_std)),
    }
    meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
