
import numpy as np
import pandas as pd
import argparse
import lsc_covariance_plotting as lscp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from blind_restoration_pisr import GaussianSmoothing

# Additional imports for Coastline
import subprocess
import os

def extract_coastline_gmt(extent, output_file="coastline.txt"):
    """
    Uses GMT to extract coastline for the given extent.
    extent: [lon_min, lon_max, lat_min, lat_max]
    """
    # Check if file exists and has content (User provided?)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
        print(f"Using existing coastline file: {output_file}")
        return True

    region = f"{extent[0]}/{extent[1]}/{extent[2]}/{extent[3]}"
    cmd = f"gmt pscoast -R{region} -JM10c -W1p,black -M -Df > {output_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Coastline extracted to {output_file}")
        return True
    except subprocess.CalledProcessError:
        print("Warning: GMT pscoast failed.")
        return False
    except FileNotFoundError:
        print("Warning: GMT not found.")
        return False

def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1):
    if not os.path.exists(coast_file):
        return

    segment_x = []
    segment_y = []
    seg_count = 0
    
    with open(coast_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if segment_x:
                    ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
                    seg_count += 1
                    segment_x, segment_y = [], []
            elif line.startswith('#'):
                continue
            else:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        val_x = float(parts[0])
                        val_y = float(parts[1])
                        # Simple bounds check debug (disabled)
                        segment_x.append(val_x)
                        segment_y.append(val_y)
                    except ValueError:
                        pass
        if segment_x:
            ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
            seg_count += 1
    
    # Debug info
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # print(f"Plot Coastline: Drawn {seg_count} segments. Ax limits: X={xlim}, Y={ylim}")

def load_xyz(file_path):
    """
    Load .xyz file. 
    Expected format: lon lat value
    Supports automatic delimiter detection/handling if Pandas engine='python'
    """
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path, sep=r'\s+', names=['lon', 'lat', 'value'], engine='python',
                         comment='#', na_values=['nan', 'NaN', 'NAN'])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    return df

def main():
    parser = argparse.ArgumentParser(description='Fill gaps in LSC grid using EGM residuals with bias correction.')
    parser.add_argument('--lsc_grid', type=str, required=True, help='Path to LSC prediction grid (Airborne residuals)')
    parser.add_argument('--egm_grid', type=str, required=True, help='Path to EGM residual grid (EGM_d2160 - EGM_d2000)')
    parser.add_argument('--output', type=str, default='merged_grid.xyz', help='Output merged grid filename')
    parser.add_argument('--smooth', type=float, default=0.0, help='Gaussian smoothing radius in km (0.0 = no smoothing)')
    parser.add_argument('--feather_px', type=int, default=20,
                        help='Cosine blend width in pixels from LSC coverage (0 = hard EGM patch)')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    
    args = parser.parse_args()

    # 1. Load Data
    df_lsc = load_xyz(args.lsc_grid)
    df_egm = load_xyz(args.egm_grid)
    
    if df_lsc is None or df_egm is None:
        print("Failed to load input files.")
        return

    print(f"LSC Grid: {len(df_lsc)} points, Range: {df_lsc['value'].min():.2f} to {df_lsc['value'].max():.2f}")
    print(f"EGM Grid: {len(df_egm)} points, Range: {df_egm['value'].min():.2f} to {df_egm['value'].max():.2f}")

    # 2. Alignment / Pivot to Grid
    # We assume both are effectively grids, but might be in list form.
    # To perform point-wise operations, checking if coordinates match is crucial.
    # We'll pivot them to 2D arrays for easier manipulation, or merge on lon/lat.
    
    print("Merging datasets on (lon, lat)...")
    # Round coordinates to avoid floating point mismatch issues (e.g. 120.0000001 vs 120.0)
    df_lsc['lon_r'] = df_lsc['lon'].round(6)
    df_lsc['lat_r'] = df_lsc['lat'].round(6)
    df_egm['lon_r'] = df_egm['lon'].round(6)
    df_egm['lat_r'] = df_egm['lat'].round(6)
    
    # Merge: Keep all EGM points (source of filling) and LSC points
    # Outer join to ensure we have the union of both grids
    df_merged = pd.merge(df_lsc, df_egm, on=['lon_r', 'lat_r'], how='outer', suffixes=('_lsc', '_egm'))
    
    # Check coverage
    n_common = df_merged.dropna(subset=['value_lsc', 'value_egm']).shape[0]
    n_lsc_only = df_merged[~df_merged['value_lsc'].isna()].shape[0]
    n_egm_only = df_merged[~df_merged['value_egm'].isna()].shape[0]
    
    print(f"Common points (Overlap): {n_common}")
    print(f"Total LSC points: {n_lsc_only}")
    
    if n_common == 0:
        print("Error: No overlapping points found between LSC and EGM grids. Check coordinate systems or rounding.")
        # Attempt fallback: Interpolate EGM to LSC coordinates
        print("Attempting to interpolate EGM to LSC coordinates...")
        # (This block would require more complex interpolation logic if grid systems are totally different)
        return

    # 3. Bias Estimation
    # Calculate difference in overlapping region
    # Mask: Valid LSC and Valid EGM
    mask_common = ~df_merged['value_lsc'].isna() & ~df_merged['value_egm'].isna()
    diffs = df_merged.loc[mask_common, 'value_lsc'] - df_merged.loc[mask_common, 'value_egm']
    
    bias_mean = diffs.mean()
    bias_std = diffs.std()
    
    print("\nBias Estimation (LSC - EGM):")
    print(f"  Mean Bias: {bias_mean:.4f} mGal")
    print(f"  Std Dev  : {bias_std:.4f} mGal")
    print("  (A low Std Dev implies EGM captures the trend well, just shifted)")

    # 4. Gap filling: LSC on Ω; bias-corrected EGM in the far field;
    #    cosine blend in a feather_px buffer. EGM is never an observation.
    from scipy.ndimage import distance_transform_edt

    df_merged['final_lon'] = df_merged['lon_lsc'].combine_first(df_merged['lon_egm'])
    df_merged['final_lat'] = df_merged['lat_lsc'].combine_first(df_merged['lat_egm'])
    df_merged['egm_bc'] = df_merged['value_egm'] + bias_mean
    omega = ~df_merged['value_lsc'].isna()

    lons_g = np.sort(df_merged['final_lon'].unique())
    lats_g = np.sort(df_merged['final_lat'].unique())
    piv_lsc = df_merged.pivot_table(index='final_lat', columns='final_lon', values='value_lsc', aggfunc='mean')
    piv_egm = df_merged.pivot_table(index='final_lat', columns='final_lon', values='egm_bc', aggfunc='mean')
    piv_lsc = piv_lsc.reindex(index=lats_g, columns=lons_g)
    piv_egm = piv_egm.reindex(index=lats_g, columns=lons_g)
    g_lsc = piv_lsc.values.astype(float)
    g_egm = piv_egm.values.astype(float)
    om = np.isfinite(g_lsc)
    dist = distance_transform_edt(~om)

    w = np.ones_like(dist, dtype=float)
    if args.feather_px > 0:
        band = (dist > 0) & (dist <= args.feather_px)
        far = dist > args.feather_px
        w[band] = 0.5 * (1.0 + np.cos(np.pi * dist[band] / args.feather_px))
        w[far] = 0.0
        print(f"\nFill: LSC on Ω ({om.sum()} px); cosine feather {args.feather_px} px "
              f"({band.sum()} px); EGM far field ({far.sum()} px)")
    else:
        w[~om] = 0.0
        print(f"\nFill: hard patch, {(~om).sum()} gap pixels <- bias-corrected EGM")

    # Feather band has no LSC: blend nearest-LSC with bias-corrected EGM.
    g_lsc_nn = g_lsc.copy()
    if (~om).any():
        idx = distance_transform_edt(~om, return_distances=False, return_indices=True)
        g_lsc_nn = g_lsc[tuple(idx)]
    merged = np.where(om, g_lsc, w * g_lsc_nn + (1.0 - w) * g_egm)
    still_nan = np.isnan(merged)
    if still_nan.any():
        print(f"  {still_nan.sum()} pixels still NaN (no EGM); leaving as NaN")

    # write merged_value / omega back to the dataframe
    LON, LAT = np.meshgrid(lons_g, lats_g)
    df_fill = pd.DataFrame({
        'final_lon': LON.ravel(),
        'final_lat': LAT.ravel(),
        'merged_value': merged.ravel(),
        'omega': np.where(om.ravel(), 1.0, np.nan),
        'blend_w': w.ravel(),
    })
    df_merged = df_merged.drop(columns=['merged_value'], errors='ignore')
    df_merged = pd.merge(df_merged, df_fill, on=['final_lon', 'final_lat'], how='left')

    omega_path = args.output.replace('.xyz', '_omega.xyz')
    np.savetxt(omega_path, df_fill[['final_lon', 'final_lat', 'omega']].values,
               fmt='%.6f %.6f %.3f', delimiter=' ', header='lon lat omega')
    print(f"  Ω mask saved to {omega_path}")
    
    # 4.5 Smoothing (Optional)
    if args.smooth > 0:
        print(f"\nApplying Gaussian Smoothing (sigma={args.smooth} km)...")
        import torch

        # Ensure coordinates are ready
        lons = np.sort(df_merged['final_lon'].unique())
        lats = np.sort(df_merged['final_lat'].unique())
        
        if len(lons) < 2 or len(lats) < 2:
            print("  Warning: Not enough points for smoothing. Skipping.")
        else:
            # Calculate grid step (deg)
            dx = np.median(np.diff(lons))
            dy = np.median(np.diff(lats))
            
            # Convert to km (Approx at mean lat)
            mean_lat = np.mean(lats)
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))
            
            dx_km = dx * km_per_deg_lon
            dy_km = dy * km_per_deg_lat
            
            print(f"  Grid Step: {dx:.4f} x {dy:.4f} deg  (~{dx_km:.2f} x {dy_km:.2f} km)")
            
            # Sigma in pixels
            sigma_px_x = args.smooth / dx_km
            sigma_px_y = args.smooth / dy_km
            sigma_px = (sigma_px_x + sigma_px_y) / 2.0
            
            print(f"  Sigma in pixels: {sigma_px:.2f}")
            
            if sigma_px < 0.1:
                print("  Sigma too small (< 0.1 px). Skipping.")
            else:
                # Pivot
                # Handle duplicates if any (mean)
                grid_pivot = df_merged.pivot_table(index='final_lat', columns='final_lon', values='merged_value', aggfunc='mean')
                
                # Re-index to ensure full regular grid if some points missing
                grid_pivot = grid_pivot.reindex(index=lats, columns=lons)

                # Fill NaNs if any (simple inpainting or mean)
                if grid_pivot.isnull().values.any():
                    print(f"  Warning: Filling {grid_pivot.isnull().sum().sum()} NaNs in grid before smoothing.")
                    grid_pivot = grid_pivot.fillna(method='bfill').fillna(method='ffill').fillna(0)

                grid_np = grid_pivot.values
                h, w = grid_np.shape
                
                # Prepare Tensor
                inp_tensor = torch.from_numpy(grid_np).float().unsqueeze(0).unsqueeze(0)
                
                # Create Smoother
                k_size = int(4 * sigma_px + 1)
                if k_size % 2 == 0: k_size += 1
                if k_size < 3: k_size = 3
                
                print(f"  Kernel Size: {k_size}x{k_size}")
                smoother = GaussianSmoothing(channels=1, kernel_size=k_size, sigma=sigma_px)
                
                with torch.no_grad():
                    out_tensor = smoother(inp_tensor)
                
                out_np = out_tensor.squeeze().numpy()
                
                # Map back
                # Create a temporary DF with the smoothed values
                grid_pivot[:] = out_np
                
                # Unstack to get a long format: indices will be final_lat, final_lon
                smoothed_series = grid_pivot.stack()
                smoothed_series.name = 'smoothed_value'
                
                df_smoothed = smoothed_series.reset_index()
                
                # Update main dataframe
                # merge on final_lat, final_lon
                df_merged = pd.merge(df_merged, df_smoothed, on=['final_lat', 'final_lon'], how='left')
                
                # Use smoothed value where available
                df_merged['merged_value'] = df_merged['smoothed_value'].fillna(df_merged['merged_value'])
                print("  Smoothing applied.")
    
    output_df = df_merged[['final_lon', 'final_lat', 'merged_value']].dropna()
    output_df.columns = ['lon', 'lat', 'value']
    
    # Sort for tidiness
    output_df = output_df.sort_values(by=['lat', 'lon'])
    
    print(f"\nSaving merged grid to {args.output}...")
    # Use standard format
    np.savetxt(args.output, output_df.values, fmt='%.6f %.6f %.3f', delimiter=' ', header='lon lat value')
    print("Done.")

    # 6. Plotting (Optional)
    if args.plot:
        plot_results(df_merged, mask_common, bias_mean, args.output)

def plot_results(df, common_mask, bias, output_base):
    print("\nGenerating plots...")
    
    # 1. Scatter Plot of Overlap
    plt.figure(figsize=(8, 6))
    lsc_vals = df.loc[common_mask, 'value_lsc']
    egm_vals = df.loc[common_mask, 'value_egm']
    
    plt.scatter(egm_vals, lsc_vals, alpha=0.5, s=5)
    
    # Reference line y=x
    lims = [min(lsc_vals.min(), egm_vals.min()), max(lsc_vals.max(), egm_vals.max())]
    plt.plot(lims, lims, 'k--', label='1:1 Line')
    
    # Bias Corrected Fit
    plt.plot(lims, [l + bias for l in lims], 'r-', label=f'Bias Corrected (+{bias:.2f})')
    
    plt.xlabel('EGM Residuals')
    plt.ylabel('LSC Residuals (Airborne)')
    plt.title('Correlation Check: EGM vs Airborne Residuals')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_base.replace('.xyz', '_correlation.png'))
    plt.close()
    
    # 2. Map View
    plt.figure(figsize=(15, 5))
    
    # Determine bounds
    lon_min, lon_max = 119.0, 124.0 #df['final_lon'].min(), df['final_lon'].max()
    lat_min, lat_max = 21.0, 26.0 #df['final_lat'].min(), df['final_lat'].max()

    # --- EXTRACT COASTLINE ---
    coast_file = "coastline.txt"
    extent = [lon_min, lon_max, lat_min, lat_max]
    has_coast = extract_coastline_gmt(extent, coast_file)

    # Use pivot table for grid plotting (easier to visualize)
    # Assuming somewhat regular grid
    
    # Plot 1: LSC (Original)
    max_lsc = np.nanmax(abs(df['value_lsc']))
    plt.subplot(1, 3, 1)
    plt.scatter(df['final_lon'], df['final_lat'], c=df['value_lsc'], cmap='RdYlBu_r', marker='s', s=0.6, vmin=-max_lsc, vmax=max_lsc)
    plt.title('LSC Predictions')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if has_coast:
        plot_coastline(plt.gca(), coast_file, color='black', linewidth=1)
    plt.colorbar(label='mGal', fraction=0.046, pad=0.04)
    
    # Plot 2: EGM (Original)
    max_egm = np.nanmax(abs(df['value_egm']))
    plt.subplot(1, 3, 2)
    plt.scatter(df['final_lon'], df['final_lat'], c=df['value_egm'], cmap='RdYlBu_r', marker='s', s=0.6, vmin=-max_egm, vmax=max_egm)
    plt.title('EGM Residuals')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if has_coast:
        plot_coastline(plt.gca(), coast_file, color='black', linewidth=1)
    plt.colorbar(label='mGal', fraction=0.046, pad=0.04)
    
    # Plot 3: Merged
    max_merged = np.nanmax(abs(df['merged_value']))
    plt.subplot(1, 3, 3)
    plt.scatter(df['final_lon'], df['final_lat'], c=df['merged_value'], cmap='RdYlBu_r', marker='s', s=0.6, vmin=-max_merged, vmax=max_merged)
    plt.title('Merged & Filled')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if has_coast:
        plot_coastline(plt.gca(), coast_file, color='black', linewidth=1)
    plt.colorbar(label='mGal', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_base.replace('.xyz', '_map.png'))
    plt.close()
    print("Plots saved.")

if __name__ == "__main__":
    main()
