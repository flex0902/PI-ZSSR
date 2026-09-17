import pandas as pd
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import os

# Import plotting utils if available in the same directory
try:
    from plot_res_g_compare import extract_coastline_gmt, plot_coastline
except ImportError:
    # Fallback if not importable
    def extract_coastline_gmt(extent, output_file="coastline.txt"): return False
    def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1): pass

def clean_data(df, col_name):
    """Ensure data is numeric and handle potential issues"""
    # Coerce to numeric
    val = pd.to_numeric(df[col_name], errors='coerce')
    # Filter out extreme outliers if any (optional, but good for plotting)
    # val[np.abs(val) > 10000] = np.nan
    return val

def calculate_gravity_difference(truth_file, pred_file, output_prefix="diff_g", common_file=None):
    print(f"Calculating Gravity Difference (Truth - Pred)...")
    print(f"  Truth File: {truth_file}")
    print(f"  Pred File:  {pred_file}")
    
    # 1. Read Truth File
    # Try reading with header inference or manual
    # We assume standard 3 column format: lon lat val
    try:
        # Check first line
        with open(truth_file, 'r') as f: line = f.readline()
        if line.startswith('#') or "lon" in line.lower():
            # Header exists
            df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#')
            if len(df_truth.columns) >= 3:
                cols = df_truth.columns.tolist()
                df_truth.rename(columns={cols[0]: 'lon', cols[1]: 'lat', cols[2]: 'truth_g'}, inplace=True)
            else:
                 # Fallback
                 df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#', header=None, names=['lon', 'lat', 'truth_g'])
        else:
            # No header
            df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#', header=None, names=['lon', 'lat', 'truth_g'])
            
        df_truth['truth_g'] = clean_data(df_truth, 'truth_g')
        
    except Exception as e:
        print(f"Error reading truth file: {e}")
        return

    # 2. Read Pred File
    try:
        # Check first line
        with open(pred_file, 'r') as f: line = f.readline()
        if line.startswith('#') or "lon" in line.lower():
            df_pred = pd.read_csv(pred_file, sep=r'\s+', comment='#')
            if len(df_pred.columns) >= 3:
                cols = df_pred.columns.tolist()
                df_pred.rename(columns={cols[0]: 'lon', cols[1]: 'lat', cols[2]: 'pred_g'}, inplace=True)
            else:
                 df_pred = pd.read_csv(pred_file, sep=r'\s+', comment='#', header=None, names=['lon', 'lat', 'pred_g'])
        else:
            df_pred = pd.read_csv(pred_file, sep=r'\s+', comment='#', header=None, names=['lon', 'lat', 'pred_g'])

        df_pred['pred_g'] = clean_data(df_pred, 'pred_g')

    except Exception as e:
        print(f"Error reading pred file: {e}")
        return
        
    print(f"  Truth Points: {len(df_truth)}")
    print(f"  Pred Points:  {len(df_pred)}")
    
    # 3. Merge
    # Round coordinates to match grids
    df_truth['lon_r'] = df_truth['lon'].round(5)
    df_truth['lat_r'] = df_truth['lat'].round(5)
    
    df_pred['lon_r'] = df_pred['lon'].round(5)
    df_pred['lat_r'] = df_pred['lat'].round(5)
    
    # Inner merge to compare overlapping points only
    df_merged = pd.merge(df_truth, df_pred[['lon_r', 'lat_r', 'pred_g']], on=['lon_r', 'lat_r'], how='inner')
    
    print(f"  Matched Points: {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("Error: No overlapping points found.")
        return

    # 4. Calculate Difference
    # Diff = Truth - Pred
    df_merged['diff'] = df_merged['truth_g'] - df_merged['pred_g']
    
    # Stats
    valid_diff = df_merged['diff'].dropna()
    rms = np.sqrt(np.mean(valid_diff**2))
    mean_diff = np.mean(valid_diff)
    std_diff = np.std(valid_diff)
    min_diff = np.min(valid_diff)
    max_diff = np.max(valid_diff)
    
    print(f"  RMS Error:  {rms:.4f} mGal")
    print(f"  Mean Error: {mean_diff:.4f} mGal")
    print(f"  Std Dev:    {std_diff:.4f} mGal")
    print(f"  Min Error:  {min_diff:.4f} mGal")
    print(f"  Max Error:  {max_diff:.4f} mGal")
    
    # 5. Plot Combined Figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # --- Left: Difference Map ---
    ax_map = axes[0]
    
    lon = df_merged['lon'].values
    lat = df_merged['lat'].values
    val = df_merged['diff'].values
    
    grid_size = int(np.sqrt(len(df_merged))) # Approx
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Grid Interpolation
    print("  Interpolating for Map...")
    # Try to infer grid shape
    unique_lon = np.unique(lon)
    unique_lat = np.unique(lat)
    if len(unique_lon) * len(unique_lat) == len(df_merged):
         # It matches perfectly, reshape
         # Sort by lat then lon
         df_sorted = df_merged.sort_values(by=['lat', 'lon'])
         ZI = df_sorted['diff'].values.reshape(len(unique_lat), len(unique_lon))
    else:
        # Irregular or missing points, interpolate
        xi = np.linspace(lon_min, lon_max, 500)
        yi = np.linspace(lat_min, lat_max, 500)
        XI, YI = np.meshgrid(xi, yi)
        interp = LinearNDInterpolator(list(zip(lon, lat)), val)
        ZI = interp(XI, YI)

    # Plot Map
    extract_coastline_gmt(extent)
    
    v_max_abs = np.nanmax(np.abs(ZI))
    im = ax_map.imshow(ZI, origin='lower', extent=extent, cmap='RdBu_r', vmin=-v_max_abs, vmax=v_max_abs)
    plot_coastline(ax_map)
    
    # Colorbar for map
    plt.colorbar(im, ax=ax_map, label='Difference (mGal)', fraction=0.046, pad=0.04)
    ax_map.set_title(f"Gravity Difference (Truth - Pred)\nRMS: {rms:.2f} mGal | Mean: {mean_diff:.2f} | Std: {std_diff:.2f}")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    
    # --- Right: Histogram ---
    ax_hist = axes[1]
    ax_hist.hist(valid_diff, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    # Plot normal distribution curve
    xmin, xmax = ax_hist.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1/(std_diff * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_diff) / std_diff)**2)
    ax_hist.plot(x, p, 'k', linewidth=2, label=f'Normal Dist\n($\mu$={mean_diff:.2f}, $\sigma$={std_diff:.2f})')
    
    ax_hist.set_title(f"Distribution of Gravity Differences")
    ax_hist.set_xlabel("Difference (mGal)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Save
    output_combined = f"{output_prefix}_analysis.png"
    plt.tight_layout()
    plt.savefig(output_combined, dpi=300)
    print(f"  Saved Combined Analysis to {output_combined}")
    plt.close()

    #6. statistics of the differences at common points
    if common_file and os.path.exists(common_file):
        print(f"\nCalculating Statistics at Common Points from {common_file}...")
        try:
            # Read common points (assuming first 2 columns are lon, lat)
            df_common = pd.read_csv(common_file, sep=r'\s+', header=None, usecols=[0, 1], names=['lon', 'lat'])
            df_common['lon_r'] = df_common['lon'].round(5)
            df_common['lat_r'] = df_common['lat'].round(5)
            
            # Merge with calculated differences
            # Inner join to keep only points present in both
            df_common_diff = pd.merge(df_merged, df_common[['lon_r', 'lat_r']], on=['lon_r', 'lat_r'], how='inner')
            
            if len(df_common_diff) > 0:
                valid_diff_common = df_common_diff['diff'].dropna()
                
                rms_c = np.sqrt(np.mean(valid_diff_common**2))
                mean_c = np.mean(valid_diff_common)
                std_c = np.std(valid_diff_common)
                min_c = np.min(valid_diff_common)
                max_c = np.max(valid_diff_common)
                
                print(f"  Shape of common points: {len(df_common)}")
                print(f"  Matched common points:  {len(df_common_diff)}")
                print(f"  --- Common Points Statistics ---")
                print(f"  RMS Error:  {rms_c:.4f} mGal")
                print(f"  Mean Error: {mean_c:.4f} mGal")
                print(f"  Std Dev:    {std_c:.4f} mGal")
                print(f"  Min Error:  {min_c:.4f} mGal")
                print(f"  Max Error:  {max_c:.4f} mGal")

                # Plot the common points difference and histogram side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                extract_coastline_gmt(extent)
                
                # --- Left: Common Points Map ---
                ax = axes[0]
                # Use common points data
                lon_c = df_common_diff['lon'].values
                lat_c = df_common_diff['lat'].values
                val_c = df_common_diff['diff'].values
                
                # Plot common points as squares (marker='s')
                # Use scatter instead of interpolation
                v_max_abs_c = np.nanmax(np.abs(val_c))
                sc = ax.scatter(lon_c, lat_c, c=val_c, cmap='RdBu_r', marker='s', s=2, vmin=-v_max_abs_c, vmax=v_max_abs_c)
                plot_coastline(ax)
                
                plt.colorbar(sc, ax=ax, label='Difference (mGal)', fraction=0.046, pad=0.04)
                ax.set_title(f"Common Points Gravity Diff\n Range: {min_c:.2f} to {max_c:.2f} | RMS: {rms_c:.2f} mGal")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                # --- Right: Histogram ---
                ax_hist = axes[1]
                ax_hist.hist(valid_diff_common, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
                
                # Plot normal distribution curve
                xmin, xmax = ax_hist.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1/(std_c * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_c) / std_c)**2)
                ax_hist.plot(x, p, 'k', linewidth=2, label=f'Normal Dist\n($\mu$={mean_c:.2f}, $\sigma$={std_c:.2f})')
                
                ax_hist.set_title(f"Distribution of Gravity Differences")
                ax_hist.set_xlabel("Difference (mGal)")
                ax_hist.set_ylabel("Density")
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                
                # Save
                output_map_common = f"{output_prefix}_common_map.png"
                plt.tight_layout()
                plt.savefig(output_map_common, dpi=300)
                print(f"  Saved Common Points Difference Map (with Hist) to {output_map_common}")
                plt.close()

                # --- 6.1 PSD Analysis for Common Points ---
                print("  Computing PSD for Common Points Differences...")
                try:
                    # Interpolate to grid for FFT
                    # Use extent from earlier
                    # Determine grid spacing (approx) to get km
                    lon_range = extent[1] - extent[0]
                    lat_range = extent[3] - extent[2]
                    
                    # Assume roughly square pixels or use mean
                    # 500x500 grid as used in main block
                    grid_n = 500
                    xi_c = np.linspace(extent[0], extent[1], grid_n)
                    yi_c = np.linspace(extent[2], extent[3], grid_n)
                    XI_c, YI_c = np.meshgrid(xi_c, yi_c)
                    
                    # Interpolate
                    interp_c = LinearNDInterpolator(list(zip(lon_c, lat_c)), val_c)
                    ZI_c = interp_c(XI_c, YI_c)
                    
                    # Fill NaNs with 0 (or mean) for FFT
                    # Better: fill with mean of valid data to minimize edge effects
                    ZI_c_filled = np.nan_to_num(ZI_c, nan=np.nanmean(ZI_c))
                    
                    # Calculate dx in km (center lat)
                    center_lat = (extent[2] + extent[3]) / 2.0
                    km_per_deg_lat = 111.0
                    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
                    
                    dx_deg = lon_range / grid_n
                    dx_km = dx_deg * km_per_deg_lon
                    
                    # 2D FFT
                    F = np.fft.fft2(ZI_c_filled)
                    Fshift = np.fft.fftshift(F)
                    magnitude = np.abs(Fshift)
                    power = magnitude**2
                    
                    # Radial Averaging
                    npix_y, npix_x = ZI_c_filled.shape
                    y_idx, x_idx = np.indices((npix_y, npix_x))
                    center_y, center_x = (npix_y-1)/2, (npix_x-1)/2
                    
                    r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
                    r_int = r.astype(int)
                    
                    # Binning
                    tbin = np.bincount(r_int.ravel(), power.ravel())
                    nr = np.bincount(r_int.ravel())
                    radial_profile = tbin / nr
                    
                    # Frequency Axis
                    # Max freq is Nyquist = 1 / (2 * dx)
                    # This corresponds to radius = npix / 2 (approx)
                    # So freq = r_int * (1 / (npix * dx)) roughly
                    # More precisely: freq step df = 1 / (N * dx)
                    # radial index i corresponds to frequency f = i * df
                    
                    df = 1.0 / (grid_n * dx_km)
                    freqs = np.arange(len(radial_profile)) * df
                    
                    # Plot PSD
                    output_psd = f"{output_prefix}_common_psd.png"
                    plt.figure(figsize=(10, 6))
                    plt.semilogy(freqs[1:], radial_profile[1:], 'b-', linewidth=1) # Skip DC
                    plt.title(f"Radially Averaged PSD of Differences (Common Points)\nMin Freq: {freqs[1]:.4f} c/km, Max: {freqs[-1]:.4f} c/km")
                    plt.xlabel("Frequency (cycles/km)")
                    plt.ylabel("Power Spectral Density")
                    plt.grid(True, which="both", ls="-", alpha=0.5)
                    
                    # Annotate specific wavelengths if needed
                    # e.g., 100km, 50km, 20km
                    # f = 1/lambda
                    for lam in [100, 50, 20, 10, 5, 2]:
                        f_mark = 1.0/lam
                        if freqs[0] < f_mark < freqs[-1]:
                           plt.axvline(x=f_mark, color='r', linestyle='--', alpha=0.5)
                           plt.text(f_mark, plt.ylim()[0], f"{lam}km", rotation=90, verticalalignment='bottom', color='r')

                    plt.savefig(output_psd, dpi=300)
                    plt.close()
                    print(f"  Saved PSD Analysis to {output_psd}")
                    
                except Exception as e:
                    print(f"  Error computing PSD: {e}")

            else:
                print("  No matching common points found.")
        except Exception as e:
            print(f"  Error processing common points file: {e}")
    else:
        print(f"\nCommon points file {common_file} not found. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot gravity difference.")
    parser.add_argument("--truth", type=str, required=True, help="Truth gravity grid file (XYZ)")
    parser.add_argument("--pred", type=str, required=True, help="Predicted gravity grid file (XYZ)")
    parser.add_argument("--output_prefix", type=str, default="gravity_diff", help="Prefix for output plots")
    parser.add_argument("--common", type=str, default="check_grid_overlap.xyz", help="Common points file")
    args = parser.parse_args()
    
    calculate_gravity_difference(args.truth, args.pred, args.output_prefix, args.common)
