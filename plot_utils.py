
import matplotlib.pyplot as plt
import numpy as np

def plot_cross_sections(grids, lon_grid, lat_grid, save_path="cross_section.png", labels=None, target_lat=23.5, target_lon=121.0):
    """
    繪製沿特定經緯度的剖面圖。
    grids: list of 2D arrays (must have same shape as lon_grid/lat_grid)
    labels: list of labels corresponding to grids
    """
    if not isinstance(grids, list):
        grids = [grids]
    
    if labels is None:
        labels = [f'Grid {i+1}' for i in range(len(grids))]

    # Find indices
    # Assuming lon_grid varies along columns, lat_grid along rows
    # Check if 1D or meshgrid
    if lon_grid.ndim == 2:
        lons = lon_grid[0, :]
        lats = lat_grid[:, 0]
    else:
         # Fallback if 1D passed
        lons = lon_grid
        lats = lat_grid
        
    dist_lon = np.abs(lons - target_lon)
    dist_lat = np.abs(lats - target_lat)
    
    idx_lon = np.argmin(dist_lon)
    idx_lat = np.argmin(dist_lat)
    
    actual_lon = lons[idx_lon]
    actual_lat = lats[idx_lat]
    
    # Setup Figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Colors/Styles for arbitrary number of lines
    styles = ['b--', 'y-', 'r--', 'g-', 'c-', 'm:']
    if len(grids) > len(styles):
        styles = styles * (len(grids) // len(styles) + 1)
        
    # ==========================
    # 1. LATITUDE PROFILE (East-West) at fixed Latitude
    # ==========================
    x_axis_lon = lons
    ax = axes[0]
    
    for i, grid in enumerate(grids):
        prof = grid[idx_lat, :]
        label = labels[i] if i < len(labels) else f'Grid {i+1}'
        style = styles[i]
        scale_linewidth = 2 if '--' in style else 1.5
        ax.plot(x_axis_lon, prof, style, alpha=0.8, linewidth=scale_linewidth, label=label)
        
    ax.set_title(f'Latitude Profile at {actual_lat:.1f}N')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Gravity (mGal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==========================
    # 2. LONGITUDE PROFILE (North-South) at fixed Longitude
    # ==========================
    x_axis_lat = lats
    ax = axes[1]
    
    for i, grid in enumerate(grids):
        prof_lon = grid[:, idx_lon]
        label = labels[i] if i < len(labels) else f'Grid {i+1}'
        style = styles[i]
        scale_linewidth = 2 if '--' in style else 1.5
        ax.plot(x_axis_lat, prof_lon, style, alpha=0.8, linewidth=scale_linewidth, label=label)
        
    ax.set_title(f'Longitude Profile at {actual_lon:.1f}E')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Gravity (mGal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    

