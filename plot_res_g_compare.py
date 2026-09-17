
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import os
import subprocess
from plot_utils import plot_cross_sections

def detect_file_encoding(file_path):
    encodings = ['utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: f.read(1000)
            return enc
        except: continue
    return 'utf-8'

def extract_coastline_gmt(extent, output_file="coastline.txt"):
    if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
        return True
    region = f"{extent[0]}/{extent[1]}/{extent[2]}/{extent[3]}"
    cmd = f"gmt pscoast -R{region} -JM10c -W1p,black -M -Df > {output_file}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except:
        return False

def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1):
    if not os.path.exists(coast_file): return
    segment_x, segment_y = [], []
    with open(coast_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if segment_x:
                    ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
                    segment_x, segment_y = [], []
            elif not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        segment_x.append(float(parts[0]))
                        segment_y.append(float(parts[1]))
                    except: pass
        if segment_x: ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)

def plot_gravity_map(file_1 = "5000(10s).xyg_pred_grid_0.1.xyz", file_2 = "1500(10s)_new.xyg_pred_grid_0.1.xyz",
                     output_png = "res_g_grid_5156m_1620m.png", title_1 = "Residual Gravity Anomaly (5156m)", 
                     title_2 = "Residual Gravity Anomaly (1620m)", cross_section_path = None,
                     target_lat = None, target_lon = None):

    print(f"Loading {file_1}...")
    df_1 = pd.read_csv(file_1, sep='\s+', header=None, names=['lon', 'lat', 'dg'], encoding=detect_file_encoding(file_1))
    print(f"Loading {file_2}...")
    df_2 = pd.read_csv(file_2, sep='\s+', header=None, names=['lon', 'lat', 'dg'], encoding=detect_file_encoding(file_2))

    # Grid parameters (same as standard 301x301)
    grid_size = 301
    lon_min, lon_max = df_1['lon'].min(), df_1['lon'].max()
    lat_min, lat_max = df_1['lat'].min(), df_1['lat'].max()
    
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    LO, LA = np.meshgrid(lon_grid, lat_grid)
    extent = [lon_min, lon_max, lat_min, lat_max]

    print("Gridding data...")
    interp_1 = LinearNDInterpolator(list(zip(df_1['lon'], df_1['lat'])), df_1['dg'])
    img_1 = interp_1(LO, LA)
    img_1 = np.nan_to_num(img_1, nan=0.0)

    interp_2 = LinearNDInterpolator(list(zip(df_2['lon'], df_2['lat'])), df_2['dg'])
    img_2 = interp_2(LO, LA)
    img_2 = np.nan_to_num(img_2, nan=0.0)

    # Extract Coastline
    extract_coastline_gmt(extent)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 5000m Plot
    v_max_5k = np.abs(img_1).max()
    im1 = axes[0].imshow(img_1, origin='lower', cmap='RdBu_r', vmin= -v_max_5k, vmax= v_max_5k, extent=extent)
    # 2000m Plot
    v_max_2k = np.abs(img_2).max()
    im2 = axes[1].imshow(img_2, origin='lower', cmap='RdBu_r', vmin= -v_max_2k, vmax= v_max_2k, extent=extent)    
    
    axes[0].set_title(f"{title_1}\nRange: {np.nanmin(img_1):.2f} ~ {np.nanmax(img_1):.2f} mGal")
    plot_coastline(axes[0])
    plt.colorbar(im2, ax=axes[0], label='mGal') # use the same colorbar for both plots
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    axes[1].set_title(f"{title_2}\nRange: {np.nanmin(img_2):.2f} ~ {np.nanmax(img_2):.2f} mGal")
    plot_coastline(axes[1])
    plt.colorbar(im2, ax=axes[1], label='mGal')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Figure saved to {output_png}")
    if cross_section_path:
        plot_cross_sections([img_1, img_2], 
                            LO, LA, save_path=cross_section_path,
                            labels=[f'{title_1}', f'{title_2}'],
                            target_lat=target_lat, target_lon=target_lon)

def plot_gravity_grid_comparison(grid_1, grid_2, extent,
                                 output_png="comparison.png",
                                 title_1="Grid 1", title_2="Grid 2",
                                 cross_section_path=None,
                                 target_lat=None, target_lon=None,
                                 lon_grid=None, lat_grid=None):
    """
    Plots two gravity grids side-by-side. 
    grid_5, grid_2: 2D numpy arrays
    extent: [lon_min, lon_max, lat_min, lat_max]
    lon_grid, lat_grid: 2D meshgrids (optional, for cross_section)
    """

    # Extract Coastline
    extract_coastline_gmt(extent)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Grid 1 Plot
    v_max_1 = np.nanmax(np.abs(grid_1))
    im1 = axes[0].imshow(grid_1, origin='lower', cmap='RdYlBu_r', vmin= -v_max_1, vmax= v_max_1, extent=extent)
    
    # Grid 2 Plot
    v_max_2 = np.nanmax(np.abs(grid_2))
    im2 = axes[1].imshow(grid_2, origin='lower', cmap='RdYlBu_r', vmin= -v_max_2, vmax= v_max_2, extent=extent)    
    
    axes[0].set_title(f"{title_1}\nRange: {np.nanmin(grid_1):.2f} ~ {np.nanmax(grid_1):.2f} mGal")
    plot_coastline(axes[0])
    plt.colorbar(im1, ax=axes[0], label='mGal') 
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    axes[1].set_title(f"{title_2}\nRange: {np.nanmin(grid_2):.2f} ~ {np.nanmax(grid_2):.2f} mGal")
    plot_coastline(axes[1])
    plt.colorbar(im2, ax=axes[1], label='mGal')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Comparison Grid Figure saved to {output_png}")

    if cross_section_path and lon_grid is not None and lat_grid is not None:
         # Check if coords are provided
        if target_lat is None: target_lat = (extent[2] + extent[3])/2
        if target_lon is None: target_lon = (extent[0] + extent[1])/2

        plot_cross_sections([grid_1, grid_2], 
                            lon_grid, lat_grid, save_path=cross_section_path,
                            labels=[title_1, title_2],
                            target_lat=target_lat, target_lon=target_lon)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_1", type=str, default="5000(10s).xyg_pred_grid_0.1.xyz")
    parser.add_argument("--file_2", type=str, default="1500(10s)_new.xyg_pred_grid_0.1.xyz")
    parser.add_argument("--output_png", type=str, default="grid_g_res_5156m_1620m.png")
    parser.add_argument("--title_1", type=str, default="Residual Gravity Anomaly (5156m)")
    parser.add_argument("--title_2", type=str, default="Residual Gravity Anomaly (1620m)")
    parser.add_argument("--cross_section", action="store_true", default=False)
    parser.add_argument("--cross_section_path", type=str, default="cross_section_g_res_5156m_1620m.png") # if None, no cross section will be saved
    parser.add_argument("--target_lat", type=float, default=23.0)
    parser.add_argument("--target_lon", type=float, default=122.0)
    args = parser.parse_args()

    if args.cross_section:
        plot_gravity_map(args.file_1, args.file_2, args.output_png, args.title_1, args.title_2, 
                        args.cross_section_path, args.target_lat, args.target_lon)
    else:
        plot_gravity_map(args.file_1, args.file_2, args.output_png, args.title_1, args.title_2, 
                        None, args.target_lat, args.target_lon)