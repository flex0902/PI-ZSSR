import pandas as pd
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, griddata

# Import plotting utils if available in the same directory
try:
    from plot_res_g_compare import extract_coastline_gmt, plot_coastline
except ImportError:
    # Fallback if not importable
    def extract_coastline_gmt(extent, output_file="coastline.txt"): return False
    def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1): pass

def plot_full_gravity(df, output_png, draw_contour=False):
    """
    Plots the Full Gravity Grid with optional contour lines
    """
    print(f"Plotting Full Gravity to {output_png}...")
    
    lon = df['lon'].values
    lat = df['lat'].values
    val = df['full_g'].values
    
    # Grid parameters
    # Assume regular grid, or interpolate
    grid_size_lon = len(np.unique(lon))
    grid_size_lat = len(np.unique(lat))
    
    # If not perfectly regular, we use grid_size ~ sqrt(N)
    if grid_size_lon * grid_size_lat != len(df):
        grid_size = int(np.sqrt(len(df)))
        grid_size_lon = grid_size
        grid_size_lat = grid_size
    
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    
    xi = np.linspace(lon_min, lon_max, grid_size_lon)
    yi = np.linspace(lat_min, lat_max, grid_size_lat)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate
    print("  Interpolating...")
    points = list(zip(lon, lat))
    interp = LinearNDInterpolator(points, val)
    ZI = interp(XI, YI)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Extract Coastline
    extract_coastline_gmt(extent)
    
    v_min = np.nanmin(ZI)
    v_max = np.nanmax(ZI)
    range_val = np.maximum(np.abs(v_min), np.abs(v_max))
    im = ax.imshow(ZI, origin='lower', extent=extent, cmap='RdYlBu_r', vmin=-range_val, vmax=range_val)
    
    if draw_contour is not None and draw_contour > 0:
        print(f"  Drawing Contour Lines (Interval: {draw_contour} mGal)...")
        # Define levels based on specified interval starting from 0 extending outwards
        # Generate levels spanning the full data range based on the interval
        min_level = np.floor(v_min / draw_contour) * draw_contour
        max_level = np.ceil(v_max / draw_contour) * draw_contour
        levels = np.arange(min_level, max_level + draw_contour, draw_contour)
        
        cs = ax.contour(XI, YI, ZI, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
        
    plot_coastline(ax)
    
    plt.colorbar(im, ax=ax, label='mGal')
    ax.set_title(f"Gravity Anomaly\nRange: {v_min:.2f} ~ {v_max:.2f} mGal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"  Saved plot to {output_png}")


def restore_egm(resg_file, egm_file, output_file, do_plot=False):
    print(f"Restoring Full Gravity...")
    print(f"  Residual File: {resg_file}")
    print(f"  EGM File:      {egm_file}")
    
    # 1. Read Residual Gravity (Has Header starting with #)
    try:
        # Use header=None because comment='#' skips the first line (which is the header)
        # If we let pandas infer header, it will take the first data line as header, losing 1 point.
        df_res = pd.read_csv(resg_file, sep='\s+', comment='#', header=None)
        
        # Manually assign columns
        # We expect at least 3 columns: lon, lat, res_g
        if len(df_res.columns) >= 3:
            df_res.rename(columns={0: 'lon', 1: 'lat', 2: 'res_g'}, inplace=True)
            # Ensure res_g is numeric, coercing errors (like 'None') to NaN
            df_res['res_g'] = pd.to_numeric(df_res['res_g'], errors='coerce')
        else:
            print("Error: Residual file has fewer than 3 columns.")
            return
            
    except Exception as e:
        print(f"Error reading residual file: {e}")
        return

    # 2. Read EGM Gravity (No Header)
    try:
        df_egm = pd.read_csv(egm_file, sep='\s+', header=None, names=['lon', 'lat', 'egm_g'])
    except Exception as e:
        print(f"Error reading EGM file: {e}")
        return
        
    print(f"  Res Points: {len(df_res)}")
    print(f"  EGM Points: {len(df_egm)}")
    
    # 3. Interpolate EGM onto Residual Grid coordinates
    print("  Interpolating EGM data onto Residual Grid coordinates...")
    try:
        egm_pts = (df_egm['lon'], df_egm['lat'])
        egm_vals = df_egm['egm_g']
        res_pts = (df_res['lon'], df_res['lat'])
        
        interp_egm = griddata(egm_pts, egm_vals, res_pts, method='linear')
        df_res['egm_g'] = interp_egm
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return
        
    valid_count = df_res['egm_g'].notna().sum()
    print(f"  Interpolated valid EGM points: {valid_count}")
    
    if valid_count == 0:
        print("Error: No overlapping points found after interpolation. Check coordinates.")
        return

    # 4. Calculate Full Gravity
    # If res_g or egm_g is NaN, full_g will be NaN
    df_res['full_g'] = df_res['res_g'] + df_res['egm_g']
    
    # 5. Save
    output_df = df_res[['lon', 'lat', 'full_g']]
    output_df.to_csv(output_file, sep=' ', index=False, header=False, float_format='%.6f', na_rep='NaN')
    print(f"Saved Full Gravity to: {output_file}")
    
    # 6. Plot
    if do_plot:
        plot_file = output_file.replace('.xyz', '.png')
        if plot_file == output_file: plot_file += ".png"
        plot_full_gravity(output_df, plot_file, draw_contour=args.contour)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Residual Gravity to EGM Gravity to restore Full Gravity.")
    parser.add_argument("--resg", type=str, required=True, help="Residual gravity grid file (XYZ format)")
    parser.add_argument("--egm", type=str, required=True, help="EGM gravity grid file (XYZ format)")
    parser.add_argument("--output", type=str, required=True, help="Output Full Gravity file (XYZ format)")
    parser.add_argument("--plot", action='store_true', help="Plot the result")
    parser.add_argument("--contour", type=float, default=None, help="Draw contour lines with the specified interval (e.g., --contour 50 for 50 mGal spacing)")
    
    args = parser.parse_args()
    
    restore_egm(args.resg, args.egm, args.output, args.plot)
