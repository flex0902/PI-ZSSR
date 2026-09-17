
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import argparse
import os

def log(msg):
    with open("debug_log.txt", "a") as f:
        f.write(str(msg) + "\n")

def detect_file_encoding(file_path):
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'gbk', 'big5']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(1000)
            return enc
        except:
            continue
    return 'utf-8'

def load_obs(file_path):
    log(f"Loading Observations: {file_path}")
    enc = detect_file_encoding(file_path)
    # Based on user description and previous file knowledge:
    # lon, lat, dg, sigma, orth_height, ell_height, line_number, line_direction
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, comment='#', encoding=enc)
        if df.shape[1] >= 7:
            df.columns = ['lon', 'lat', 'dg', 'sigma', 'orth_h', 'ell_h', 'line_number', 'line_dir'] + [f'col_{i}' for i in range(8, df.shape[1])]
            # specific columns of interest
            # specific columns of interest
            # df = df[['lon', 'lat', 'dg', 'line_number']] # Keep all columns for correction output
        else:
            log(f"Warning: Unexpected column count {df.shape[1]}. Trying first 3 as lon, lat, dg, and 4th as line?")
            df = df.iloc[:, :4]
            df.columns = ['lon', 'lat', 'dg', 'line_number']
            
        return df
    except Exception as e:
        log(f"Error loading obs: {e}")
        return None

def load_grid(file_path):
    log(f"Loading EGM Grid: {file_path}")
    enc = detect_file_encoding(file_path)
    try:
        # Assuming XYZ format
        df = pd.read_csv(file_path, sep='\s+', header=None, comment='#', names=['lon', 'lat', 'val'], encoding=enc)
        return df
    except Exception as e:
        log(f"Error loading grid: {e}")
        return None

def main():
    # Clear log
    with open("debug_log.txt", "w") as f: f.write("Start\n")
    
    parser = argparse.ArgumentParser(description="Analyze 1500m data vs EGM model by Line Number.")
    parser.add_argument('--obs', default='1500(10s).xyg', help='Observation file path')
    parser.add_argument('--egm', default='TW_EGM_grid_g_d2160_h1620.xyz', help='EGM grid file path')
    parser.add_argument('--output', default=None, help='Output corrected file') #'1500(10s)_new.xyg'
    
    args = parser.parse_args()

    # 1. Load Data
    df_obs = load_obs(args.obs)
    df_egm = load_grid(args.egm)
    
    if df_obs is None or df_egm is None:
        return

    log(f"Obs count: {len(df_obs)}")
    log(f"EGM grid count: {len(df_egm)}")

    # 2. Interpolate EGM to Obs
    log("Building Interpolator...")
    # Basic LinearNDInterpolator
    interp = LinearNDInterpolator(list(zip(df_egm['lon'], df_egm['lat'])), df_egm['val'])
    
    log("Interpolating...")
    df_obs['egm_val'] = interp(df_obs['lon'], df_obs['lat'])
    
    # Check for NaNs (points outside grid hull)
    n_nans = df_obs['egm_val'].isna().sum()
    if n_nans > 0:
        log(f"Warning: {n_nans} points are outside the EGM grid coverage (set to NaN).")
    
    # 3. Calculate Difference
    df_obs['diff'] = df_obs['dg'] - df_obs['egm_val']
    
    # 4. Group by Line Number and Calculate Stats
    log("Calculating Statistics by Line Number...")
    # Convert 'line_number' to string just in case
    df_obs['line_number'] = df_obs['line_number'].astype(str)
    
    stats = df_obs.groupby('line_number')['diff'].agg(
        Count='count',
        Min='min',
        Max='max',
        Mean='mean',
        Std='std'
    ).reset_index()
    
    # Add definition of Bias = Mean
    stats['Bias'] = stats['Mean']

    # 5. Output
    log(stats)
    output_line_stats = args.obs + "_line_statistics.csv"
    stats.to_csv(output_line_stats, index=False)
    log(f"\nStatistics saved to {output_line_stats}")

    # 6. Correct Data
    log("Applying Bias Correction...")
    log(f"Columns before merge: {df_obs.columns}")
    # Merge Bias
    df_obs = df_obs.merge(stats[['line_number', 'Bias']], on='line_number', how='left')
    log(f"Columns after merge: {df_obs.columns}")
    # Correct dg
    df_obs['dg_corrected'] = df_obs['dg'] - df_obs['Bias']
    
    # Save Corrected
    output_corrected = args.obs + "_bias_corrected.csv"
    df_obs.to_csv(output_corrected, index=False)
    log(f"Bias corrected data saved to {output_corrected}")

    # Save the diffs with line numbers for inspection
    #df_obs.to_csv('obs_with_diffs.csv', index=False)
    #log("Detailed diffs saved to obs_with_diffs.csv")
    
    if args.output is not None:
        # Save corrected dg, keep all columns, ASCII format
        # Format: 'lon', 'lat', 'dg_corrected', 'sigma', 'orth_h', 'ell_h', 'line_number', 'line_dir'
        output_cols = ['lon', 'lat', 'dg_corrected', 'sigma', 'orth_h', 'ell_h', 'line_number', 'line_dir']
 
        # Format specific columns as requested
        # lon, lat: 9 decimal places
        # dg_corrected: 2 decimal places
        # We apply formatting by converting to string with specific precision
        # This ensures to_csv writes exactly what we want without global float_format affecting others unpredictably
        df_obs['lon'] = df_obs['lon'].apply(lambda x: f"{x:.9f}")
        df_obs['lat'] = df_obs['lat'].apply(lambda x: f"{x:.9f}")
        df_obs['dg_corrected'] = df_obs['dg_corrected'].apply(lambda x: f"{x:.2f}")
        df_obs['sigma'] = df_obs['sigma'].apply(lambda x: f"{x:.3f}")
        df_obs['orth_h'] = df_obs['orth_h'].apply(lambda x: f"{x:.3f}")
        df_obs['ell_h'] = df_obs['ell_h'].apply(lambda x: f"{x:.3f}")
        #df_obs['line_number'] = df_obs['line_number'].apply(lambda x: f"{x:4.0f}")
        #df_obs['line_dir'] = df_obs['line_dir'].apply(lambda x: f"{x:2.0f}")

        # Check if all columns exist
        missing_cols = [c for c in output_cols if c not in df_obs.columns]
        if missing_cols:
            log(f"Warning: Missing columns for output: {missing_cols}")
            log(f"Available columns: {df_obs.columns}")
            # Fallback: Just save what we have from the requested list
        output_cols = [c for c in output_cols if c in df_obs.columns]
    
    log(f"Saving corrected file to: {args.output}")
    df_obs[output_cols].to_csv(args.output, index=False, sep=' ', header=False)
    log("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
