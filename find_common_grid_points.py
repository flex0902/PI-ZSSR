import argparse
import pandas as pd
import numpy as np

def load_grid(filepath):
    """Loads a 3-column space-separated content file (lon, lat, val)."""
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['lon', 'lat', 'val'],
                         comment='#', na_values=['NaN', 'nan', 'NAN'])
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['val'] = pd.to_numeric(df['val'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    p = argparse.ArgumentParser(description="Find grid points valid in both XYZ files.")
    p.add_argument("--file1", default="1500(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--file2", default="5000(10s)_new.xyg_pred_grid_0.1.xyz")
    p.add_argument("--output", default="check_grid_overlap.xyz")
    args = p.parse_args()
    file1, file2, output_file = args.file1, args.file2, args.output

    print(f"Loading {file1}...")
    df1 = load_grid(file1)
    if df1 is None: return

    print(f"Loading {file2}...")
    df2 = load_grid(file2)
    if df2 is None: return

    # Check dtypes and ensure consistency
    # Assuming float64 for all
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")

    # Merge on lon, lat
    # We round coordinates to avoid floating point mismatch if generated differently
    # But usually if from same grid spec they should be identical.
    # To be safe, round to 6 decimal places.
    df1['lon_r'] = df1['lon'].round(6)
    df1['lat_r'] = df1['lat'].round(6)
    df2['lon_r'] = df2['lon'].round(6)
    df2['lat_r'] = df2['lat'].round(6)

    print("Merging grids on coordinates...")
    merged = pd.merge(df1, df2, on=['lon_r', 'lat_r'], how='inner', suffixes=('_1', '_2'))
    
    print(f"Merged shape (common grid points): {merged.shape}")
    
    # Filter for valid values in BOTH
    valid_mask = (~merged['val_1'].isna()) & (~merged['val_2'].isna())
    common_valid = merged[valid_mask].copy()
    
    count = len(common_valid)
    print(f"Number of grid points valid in BOTH files: {count}")

    if count > 0:
        # Restore original lon/lat from file 1 (or 2)
        # We can drop the rounded cols
        common_valid = common_valid[['lon_1', 'lat_1', 'val_1', 'val_2']]
        #common_valid.rename(columns={'lon_1': 'lon', 'lat_1': 'lat', 'val_1': 'res_1500', 'val_2': 'res_5000'}, inplace=True)
        
        print(f"Saving common valid points to {output_file}...")
        common_valid.to_csv(output_file, sep=' ', index=False, header=False)
        
        # Calculate stats
        diff = common_valid['val_1'] - common_valid['val_2']
        print("\nStatistics of Difference (File1 - File2):")
        print(diff.describe())
        
        # Also print bounds
        print("\nCommon Area Bounds:")
        print(f"Lon: {common_valid['lon_1'].min():.6f} to {common_valid['lon_1'].max():.6f}")
        print(f"Lat: {common_valid['lat_1'].min():.6f} to {common_valid['lat_1'].max():.6f}")

    else:
        print("No common valid points found!")

if __name__ == "__main__":
    main()
