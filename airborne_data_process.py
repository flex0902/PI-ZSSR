import subprocess
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import LinearNDInterpolator
import lsc_covariance_plotting as lscp

# ==========================================
# 1. Config & Tools
# ==========================================
def detect_file_encoding(file_path):
    encodings = ['utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: f.read(1000)
            return enc
        except: continue
    return 'utf-8'

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

# ==========================================
# 2. Load Gravity Grid
# ==========================================
class GravityGridDataset(Dataset):
    def __init__(self, grid_xyz, size=(301, 301)):
        self.file_grid = grid_xyz
        self.size = size
        
        # 1. Load Data (Shared Attribute)
        self.df_grid = self._load_raw_data()
        
        # 2. Create Interpolator (Shared Attribute)
        print("Building Interpolator...")
        self.interpolator = LinearNDInterpolator(
            list(zip(self.df_grid['lon'], self.df_grid['lat'])), 
            self.df_grid['dg']
        )
        
        # 3. Create Image Grid (Optional, lazy load or initialized here if needed)
        # For compatibility with legacy code, we initialize it here
        self.img_grid, self.extent = self._create_image_grid()
        
        self.dx = (self.extent[1] - self.extent[0]) / size[1]
        self.dy = (self.extent[3] - self.extent[2]) / size[0]
        self.dx_km = self.dx * 102.0
        self.dy_km = self.dy * 111.0
        print(f"Grid Params: {size}, dx={self.dx_km:.3f} km, dy={self.dy_km:.3f} km")

    def _load_raw_data(self):
        print(f"Loading raw grid data from {self.file_grid}...")
        enc_h = detect_file_encoding(self.file_grid)
        df = pd.read_csv(self.file_grid, sep='\s+', header=None, names=['lon', 'lat', 'dg'], encoding=enc_h)
        return df

    def _create_image_grid(self):
        # Define uniform grid based on loaded data
        lon_min, lon_max = self.df_grid['lon'].min(), self.df_grid['lon'].max()
        lat_min, lat_max = self.df_grid['lat'].min(), self.df_grid['lat'].max()
        
        lon_grid = np.linspace(lon_min, lon_max, self.size[1])
        lat_grid = np.linspace(lat_min, lat_max, self.size[0])
        self.LO, self.LA = np.meshgrid(lon_grid, lat_grid)
        
        print("Interpolating to Image Grid (Regular Mesh)...")
        # Use the pre-built interpolator
        img_grid = self.interpolator(self.LO, self.LA)
        img_grid = np.nan_to_num(img_grid, nan=0.0)
        
        return img_grid, (lon_min, lon_max, lat_min, lat_max)

    def get_values_at(self, query_lon, query_lat):
        """
        Query gravity values at specific (lon, lat) points.
        Returns numpy array of values.
        """
        # Ensure input are arrays
        q_lon = np.array(query_lon)
        q_lat = np.array(query_lat)
        
        # Prepare points (N, 2)
        points = np.column_stack((q_lon, q_lat))
        
        vals = self.interpolator(points)
        return np.nan_to_num(vals, nan=0.0)

# ==========================================
# 3. Load Gravity Observation (Modified)
# ==========================================
def load_observation_data(data_file):
    """
    載入觀測點數據（從 .xyg 檔案）
    
    Parameters:
    -----------
    data_file : str
        數據檔案路徑（格式：lon lat dg sigma orthorth_height ellipsoidal_height line_number line_directin，空格分隔）
    Returns:
    --------
    df : DataFrame
        包含 'lon', 'lat', 'dg' 欄位的 DataFrame
    """
    print(f"\n載入觀測點數據: {data_file}")
    # Try multiple encodings to handle different file formats
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'gbk', 'big5']
    df = None
    for encoding in encodings:
        try:
            # Added comment='#' to skip header lines
            df = pd.read_csv(data_file, sep='\s+', header=None, encoding=encoding, comment='#')
            print(f"成功使用編碼: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # If it's not an encoding error, re-raise it
            raise
    
    if df is None:
        raise ValueError(f"無法使用任何編碼讀取檔案: {data_file}")
    
    # Handle different formats based on column count
    if df.shape[1] == 3:
        df.columns = ['lon', 'lat', 'dg']
    elif df.shape[1] == 8:
        # Format: lon lat dg sigma orth_height ellipsoidal_height line_number line_directin
        df.columns = ['lon', 'lat', 'dg', 'sigma', 'orth_height', 'ellipsoidal_height', 'line_number', 'line_directin']
        # Keep only required columns
        df = df[['lon', 'lat', 'dg']]
    elif df.shape[1] >= 3:
        print(f"警告: 檔案包含 {df.shape[1]} 欄位，將取前三欄作為 lon, lat, dg")
        df = df.iloc[:, :3]
        df.columns = ['lon', 'lat', 'dg']
    else:
        raise ValueError(f"檔案格式錯誤: 預期至少 3 個欄位，實際有 {df.shape[1]} 個")

    df['lon'] = df['lon'].astype(float)
    df['lat'] = df['lat'].astype(float)
    df['dg'] = df['dg'].astype(float)
    
    print(f"  載入 {len(df)} 個觀測點")
    print(f"  經度範圍: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
    print(f"  緯度範圍: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data file (lon, lat, dg)')
    parser.add_argument('--size', type=int, default=301, help="Grid size (e.g. 301, 601)")
    parser.add_argument('--grid', type=str, default=None, help="Path to grid file (e.g. grid.xyz)")
    parser.add_argument('--n_max', type=int, default=3000, help='Max degree')
    parser.add_argument('--n_min', type=int, default=3, help='Min degree')
    parser.add_argument('--max_dist_deg', type=float, default=0.20, help='Max search radius distance (degrees)')
    parser.add_argument('--pred_type', type=str, default="random", help="pred_type = random, all, grid")
    parser.add_argument('--flight_height', type=float, default=5000.0, help='Flight height in meters')
    parser.add_argument('--fit_only', action='store_true',
                        help='Stop after T-R covariance fitting; do not run LSC prediction')
    parser.add_argument('--params', type=str, default=None,
                        help='Load T-R JSON and skip covariance fitting')
    parser.add_argument('--noise-std', dest='noise_std', type=float, default=1.0,
                        help='LSC observation noise standard deviation (mGal)')
    args = parser.parse_args()
    
    n_max = args.n_max
    n_min = args.n_min
    max_dist_deg = args.max_dist_deg
    pred_type = args.pred_type
    grid_size = args.size
    flight_height = args.flight_height
    noise_std = args.noise_std

    # Load data
    df_obs = load_observation_data(args.data)
    # print(df_obs.head())
    if args.grid is None:
        df_obs = df_obs.dropna(subset=['dg']).copy()
        df_obs['diff'] = df_obs['dg']
        extent = [119.0, 124.0, 21.0, 26.0]
    else:
        # Initialize Grid Dataset (Loads data & builds interpolator)
        # Note: size is used for image generation, but interpolator uses raw data
        grid_dataset = GravityGridDataset(args.grid, size=(args.size, args.size))
        print("Interpolating grid data to observation points...")
        # Use the class method to get values
        grid_values = grid_dataset.get_values_at(df_obs['lon'], df_obs['lat'])
        df_obs['grid_value'] = grid_values
        # Calculate difference if needed (assuming 'dg' in obs is the same quantity as grid)
        df_obs['diff'] = df_obs['dg'] - df_obs['grid_value']
        print("Interpolation done. Result sample:")
        # Physical Extent
        extent = grid_dataset.extent
    print(df_obs.head())
    # Save result if needed
    #output_file = args.data.rsplit('.', 1)[0] + '_with_grid.csv'
    #df_obs.to_csv(output_file, index=False)
    #print(f"Saved result to {output_file}")

    # --- EXTRACT COASTLINE ---
    coast_file = "coastline.txt"
    has_coast = extract_coastline_gmt(extent, coast_file)

    # Plotting using external module
    lscp.plot_residual_difference(df_obs, coast_file=coast_file, output_file=f'{args.data}_result.png', title='Residuals (dg - grid_value)')

    # ==========================================
    # 4. LSC Covariance Fitting
    # ==========================================
    import lsc_covariance_fit as lsc
    import lsc_covariance_fit_B as lsc_B
    from scipy.spatial import cKDTree

    best_B = None
    params = None
    A_final = None
    RB_final = None
    fitted_curve = None

    if args.params:
        p = lsc.load_tr_parameters(args.params)
        best_B = int(p["B"])
        params = (float(p["A"]), float(p.get("depth_km", 0.0)) * 1000.0)
        A_final = float(p["A"])
        RB_final = float(p["R_B"])
        print(f"\nLoaded T-R from {args.params}")
        print(f"  A={A_final:.5f}  B={best_B}  R_B={RB_final:.2f}  h={p.get('flight_height', flight_height)}")
        if args.fit_only:
            print("--params with --fit_only: nothing to do.")
            raise SystemExit(0)
    else:
        print("\n--- Starting LSC Covariance Fitting ---")
        df_resid = pd.DataFrame({
            'lon': df_obs['lon'],
            'lat': df_obs['lat'],
            'dg': df_obs['diff']
        })

        cov_df = lsc.calculate_empirical_covariance(df_resid, max_dist_deg=0.2, bin_size_sec=40)
        h_tag = int(round(flight_height))
        cov_df.to_csv("empirical_covariance_data.csv", index=False)
        cov_df.to_csv(f"empirical_covariance_data_{h_tag}m.csv", index=False)
        print("   Empirical covariance data saved.")

        print(f"   Fitting with Flight Height = {flight_height} m")
        A_final, RB_final, fitted_curve = lsc.fit_tscherning_rapp(
            cov_df,
            n_min=n_min,
            n_max=n_max,
            fixed_B=24,
            flight_height=flight_height
        )

        if A_final is not None:
            lsc.save_tr_parameters(A_final, RB_final, 24, flight_height, filename="tr_parameters.json")
            print("\n" + "="*30)
            print(" FINAL RESULTS")
            print("="*30)
            print(f" Flight Height: {flight_height:.2f} m")
            print(f" A Parameter  : {A_final:.5f} mGal^2")
            print(f" R_B Parameter: {RB_final:.5f} m")
            print(f" Depth (R-R_B): {(lsc.R_EARTH - RB_final)/1000:.5f} km")
            print("="*30)

        print("\n--- Finding Best Integer B ---")
        best_B, params, best_curve, history = lsc_B.find_best_integer_B(
            cov_df,
            b_min=4,
            b_max=10000,
            n_min=n_min,
            n_max=n_max,
            flight_height=flight_height
        )

        if best_B is not None:
            A_best, depth_best = params
            RB_best = lsc.R_EARTH - depth_best
            print("\n" + "="*30)
            print(" FINAL RESULTS (Optimized B)")
            print("="*30)
            print(f" Best B       : {best_B}")
            print(f" A Parameter  : {A_best:.5f} mGal^2")
            print(f" R_B Parameter: {RB_best:.2f} m")
            print(f" Depth (R-R_B): {depth_best/1000:.5f} km")
            print(f" Flight Height: {flight_height:.2f} m")
            print(f" Sum of Squared Residuals: {history[1][-1]:.5f}")
            print("="*30)
            lsc.save_tr_parameters(A_best, RB_best, best_B, flight_height, filename="tr_parameters_optimized.json")
            lsc.save_tr_parameters(A_best, RB_best, best_B, flight_height,
                                   filename=f"tr_parameters_optimized_h{h_tag}.json")
            lscp.plot_optimized_fit_and_search(
                cov_df, best_curve, history, best_B, A_best, depth_best, flight_height, fitted_curve,
                output_file=f'lsc_covariance_fit_B_optimized_{args.data}.png'
            )
            if args.fit_only:
                print(f"\n--fit_only: covariance fit complete.")
                raise SystemExit(0)
        elif args.fit_only:
            print("--fit_only: B search failed; stopping before LSC prediction.")
            raise SystemExit(1)
    # Test for stopping before LSC prediction
    #exit()
    # 4. perform LSC prediction using the optimized parameters at the random observation points
    import lsc_covariance_predict as lsc_pred
    print("\n--- Performing LSC Prediction (Validation) ---")

    if pred_type == 'random':
        n_pred_points = 20
        n_sample = min(n_pred_points, len(df_obs))
        pred_indices = np.random.choice(df_obs.index, size=n_sample, replace=False)
        df_pred_points = df_obs.loc[pred_indices].copy()
        print(f"Selected {n_pred_points} random observation points for validation:")
        print(df_pred_points[['lon', 'lat', 'dg', 'diff']])
    elif pred_type == 'all':
        df_pred_points = df_obs.copy()
        n_pred_points = len(df_pred_points)
        print(f"Selected all {n_pred_points} observation points for prediction")
    elif pred_type == 'grid':
        print(f"Generating prediction grid: Lon[119-124], Lat[21-26], Size={grid_size}x{grid_size}")
        lon_lines = np.linspace(119, 124, grid_size)
        lat_lines = np.linspace(21, 26, grid_size)
        LON, LAT = np.meshgrid(lon_lines, lat_lines)
        df_pred_points = pd.DataFrame({
            'lon': LON.ravel(),
            'lat': LAT.ravel(),
            'dg': np.nan,
            'diff': np.nan,
        })
        n_pred_points = len(df_pred_points)
        print(f"Generated grid with {n_pred_points} points for prediction:")
        print(f"range lon: {df_pred_points['lon'].min():.4f} to {df_pred_points['lon'].max():.4f}")
        print(f"range lat: {df_pred_points['lat'].min():.4f} to {df_pred_points['lat'].max():.4f}")
        print(f"grid resolution: {grid_size}x{grid_size}")
    else:
        print(f"Unknown pred_type '{pred_type}', defaulting to random 20")
        n_pred_points = 20
        pred_indices = np.random.choice(df_obs.index, size=n_pred_points, replace=False)
        df_pred_points = df_obs.loc[pred_indices].copy()

    final_B = best_B if best_B is not None else 24
    if best_B is not None:
        final_A, final_depth = params
        final_RB = lsc.R_EARTH - final_depth
    else:
        final_A = A_final
        final_RB = RB_final

    search_radius_deg = max_dist_deg * 1.0
    print(f"\nLSC Configuration:")
    print(f"  Param A: {final_A:.4f}")
    print(f"  Param B: {final_B}")
    print(f"  Param R_B: {final_RB:.1f}")
    print(f"  Search Radius: {search_radius_deg} deg")
    print(f"  Covariance Type: 'gg' (Gravity-Gravity)")
    print(f"  Flight Height: {flight_height} m")
    print(f"  Noise std: {noise_std} mGal")

    lsc_obs_df = pd.DataFrame({
        'lon': df_obs['lon'].values,
        'lat': df_obs['lat'].values,
        'dg': df_obs['diff'].values,
    })
    obs_index = df_obs.index.to_numpy()
    index_to_pos = {lab: i for i, lab in enumerate(obs_index)}
    lat_rad = np.radians(lsc_obs_df['lat'].values)
    lon_rad = np.radians(lsc_obs_df['lon'].values)
    obs_coords = np.vstack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ]).T
    obs_tree = cKDTree(obs_coords)

    predictions = []
    uncertainties = []
    n_done = 0
    t_pred0 = time.time()
    for idx, row in df_pred_points.iterrows():
        p_lon, p_lat = row['lon'], row['lat']
        if pred_type == 'grid':
            exclude_pos = None
            true_resid_str = "NaN"
        else:
            exclude_pos = index_to_pos.get(idx)
            true_resid_str = f"{row['diff']:.4f}" if pd.notna(row.get('diff', np.nan)) else "NaN"

        val_pred, val_std, nearby = lsc_pred.perform_lsc_prediction_for_point(
            lsc_obs_df,
            p_lat, p_lon,
            final_A, final_RB, final_B,
            max_degree=n_max,
            max_dist_deg=search_radius_deg,
            target='gravity',
            noise_std=noise_std,
            obs_height=flight_height,
            pred_height=flight_height,
            tree=obs_tree,
            obs_coords=obs_coords,
            exclude_pos=exclude_pos,
        )
        predictions.append(val_pred)
        uncertainties.append(val_std)
        n_done += 1
        if n_done <= 3 or n_done % 500 == 0 or n_done == n_pred_points:
            pred_s = f"{val_pred:.4f}" if np.isfinite(val_pred) else "NaN"
            std_s = f"{val_std:.4f}" if np.isfinite(val_std) else "NaN"
            elapsed = time.time() - t_pred0
            print(f"  {n_done}/{n_pred_points}  True={true_resid_str} Pred={pred_s} std={std_s} nearby={len(nearby)}  {elapsed:.0f}s")

    # 4.4 Results
    df_pred_points['pred_resid'] = predictions 
    df_pred_points['pred_std'] = uncertainties
    
    # Calculate errors only if ground truth exists
    if pred_type != 'grid':
        df_pred_points['lsc_error'] = df_pred_points['diff'] - df_pred_points['pred_resid']
        # Save validation results # args.data.rsplit('.', 1)[0]
        if pred_type == 'random': output_pred_file = f"{args.data}_pred_validation_{max_dist_deg}.csv"
        if pred_type == 'all': output_pred_file = f"{args.data}_pred_all_{max_dist_deg}.csv"
        df_pred_points.to_csv(output_pred_file, index=False)
        print(f"Saved result to {output_pred_file}")    
    else:
        df_pred_points['lsc_error'] = np.nan
        # restore EGM grid.
        # If LSC prediction is NaN (no data nearby), treat residual as 0.0, so the result is just the EGM trend.
        #df_pred_points['pred_resid'] = df_pred_points['pred_resid'].fillna(0.0) + grid_dataset.get_values_at(df_pred_points['lon'], df_pred_points['lat'])
        # Save grid results (lon, lat, pred_resid + EGM)        
        output_pred_file = f"{args.data}_pred_grid_{max_dist_deg}.xyz"
        # Pandas to_csv doesn't support mixed float_format like '%.6f %.6f %.3f'
        # Use numpy savetxt instead
        np.savetxt(output_pred_file, df_pred_points[['lon', 'lat', 'pred_resid']].values, fmt='%.6f %.6f %.3f', delimiter=' ')
        print(f"Saved grid result to {output_pred_file}")        
    
    print("\nPrediction Results Summary:")
    cols_to_show = ['lon', 'lat', 'pred_resid', 'pred_std']
    if 'diff' in df_pred_points.columns and not df_pred_points['diff'].isna().all():
        cols_to_show.extend(['diff', 'lsc_error'])
    print(df_pred_points[cols_to_show].head())
    
    # 4.5 Common Plots
    
    # Determine plot title and filename suffix based on mode
    if pred_type == 'grid':
        plot_title = 'LSC Prediction (Grid)'
        suffix = 'pred_grid'
    elif pred_type == 'random':
        plot_title = 'LSC Validation (Random)'
        suffix = 'pred_validation'
    else:
        plot_title = 'LSC Prediction (Observations)'
        suffix = 'pred_all'

    lscp.plot_residual_difference(df_pred_points, coast_file=coast_file, output_file=f'{args.data}_{suffix}_{max_dist_deg}.png',
        title=plot_title, value_column='pred_resid', marker='s', s=0.5)
    lscp.plot_residual_difference(df_pred_points, coast_file=coast_file, output_file=f'{args.data}_{suffix}_{max_dist_deg}_std.png', 
        title=f'{plot_title} Uncertainty', value_column='pred_std', marker='s', s=0.5)

    # RMSE and LSC Error Plot (Only if we have ground truth)
    if pred_type != 'grid':
        lscp.plot_residual_difference(df_pred_points, coast_file=coast_file, output_file=f'{args.data}_{suffix}_{max_dist_deg}_error.png',
            title=f'{plot_title} Errors', value_column='lsc_error', marker='s', s=0.5)
        
        rmse = np.sqrt((df_pred_points['lsc_error']**2).mean())
        print(f"\nLSC Validation RMSE: {rmse:.4f} mGal")
    else:
        print("\nGrid prediction mode: Skipping RMSE calculation (no ground truth).")

    
