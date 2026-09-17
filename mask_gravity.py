import pandas as pd
import argparse
import numpy as np
import os
from restore_egm import plot_full_gravity

def mask_full_gravity(grid_file, full_file, output_file, do_plot=False):
    print(f"Masking Full Gravity...")
    print(f"  Grid File (Source of NaNs): {grid_file}")
    print(f"  Full File (Target):         {full_file}")
    
    # 1. Read Grid File (Source of NaNs)
    # Use usecols=[0,1,2] to handle potential extra columns
    try:
        # Use header=None because comment='#' skips any existing header lines starting with #
        # We manually assign names to handle both cases (with/without header)
        df_grid = pd.read_csv(grid_file, sep='\s+', comment='#', header=None, usecols=[0,1,2])
        if len(df_grid.columns) >= 3:
            df_grid.rename(columns={0: 'lon', 1: 'lat', 2: 'val'}, inplace=True)
            # Ensure val is numeric, coerce errors (like partial headers not starting with #) to NaN
            df_grid['val'] = pd.to_numeric(df_grid['val'], errors='coerce')
        else:
            print("Error: Grid file has fewer than 3 columns.")
            return
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return

    # 2. Read Full Gravity File
    try:
        df_full = pd.read_csv(full_file, sep='\s+', header=None, names=['lon', 'lat', 'full_g'], comment='#')
        # Also ensure full_g is numeric
        df_full['full_g'] = pd.to_numeric(df_full['full_g'], errors='coerce')
    except Exception as e:
        print(f"Error reading full file: {e}")
        return
        
    print(f"  Grid Points: {len(df_grid)}")
    print(f"  Full Points: {len(df_full)}")
    
    # 3. Merge
    # Round coordinates to match grids
    df_grid['lon_r'] = df_grid['lon'].round(5)
    df_grid['lat_r'] = df_grid['lat'].round(5)
    
    df_full['lon_r'] = df_full['lon'].round(5)
    df_full['lat_r'] = df_full['lat'].round(5)
    
    # Left merge on full to keep all full points, but bring in 'val' from grid
    # If a point in full doesn't exist in grid, val acts as NaN? No, merge behavior depends on join type.
    # We assume grids are identical in coverage. But if grid has gaps (missing rows? or NaN rows?), it matters.
    # The file "1500(10s).xyg_pred_grid_0.1.xyz" likely has ALL points, but some values are NaN.
    # If rows are missing, then left join results in NaN for 'val'.
    # If rows exist but value is NaN, then 'val' is NaN.
    # So checking 'val'.isna() covers both missing rows (gaps) and explicit NaNs.
    df_merged = pd.merge(df_full, df_grid[['lon_r', 'lat_r', 'val']], on=['lon_r', 'lat_r'], how='left')
    
    print(f"  Merged Points: {len(df_merged)}")
    
    # 4. Mask
    # Where 'val' is NaN, set 'full_g' to NaN
    mask = df_merged['val'].isna()
    count_masked = mask.sum()
    print(f"  Masking {count_masked} points where grid value is NaN...")
    
    df_merged.loc[mask, 'full_g'] = np.nan
    
    # 5. Save
    output_df = df_merged[['lon', 'lat', 'full_g']]
    output_df.to_csv(output_file, sep=' ', index=False, header=False, float_format='%.6f', na_rep='NaN')
    print(f"Saved Corrected Full Gravity to: {output_file}")

    # 6. Plot
    if do_plot:
        plot_file = output_file.replace('.xyz', '.png') 
        if plot_file == output_file: plot_file += "_masked.png"
        plot_full_gravity(output_df, plot_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask full gravity grid using NaNs from another grid.")
    parser.add_argument("--grid", type=str, required=True, help="Grid file with NaNs (XYZ format)")
    parser.add_argument("--full", type=str, required=True, help="Full gravity file to be corrected (XYZ format)")
    parser.add_argument("--output", type=str, required=True, help="Output corrected file (XYZ format)")
    parser.add_argument("--plot", action='store_true', help="Plot the result")
    args = parser.parse_args()
    
    mask_full_gravity(args.grid, args.full, args.output, args.plot)
