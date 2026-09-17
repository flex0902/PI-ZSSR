import numpy as np
import pandas as pd
import lsc_covariance_plotting as lscp
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import json
import argparse

# ==========================================
# PART 1: EMPIRICAL COVARIANCE CALCULATION
# ==========================================

def calculate_empirical_covariance(df, max_dist_deg, bin_size_sec):
    print(f"--- Step 1: Processing {len(df)} observations ---")
    
    lat_rad = np.radians(df['lat'].values)
    lon_rad = np.radians(df['lon'].values)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.vstack([x, y, z]).T

    print("   Building spatial index...")
    tree = cKDTree(coords)
    
    max_dist_rad = np.radians(max_dist_deg)
    max_chord_dist = 2 * np.sin(max_dist_rad / 2)
    
    print(f"   Querying pairs within {max_dist_deg:.3f} degrees...")
    pairs = tree.query_pairs(max_chord_dist, output_type='ndarray')
    print(f"   Found {len(pairs)} pairs.")
    
    chord_dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    psi_rad = 2 * np.arcsin(chord_dists / 2)
    psi_deg = np.degrees(psi_rad)
    
    print("   Computing anomaly products...")
    dg = df['dg'].values
    dg_products = dg[pairs[:, 0]] * dg[pairs[:, 1]]

    print("   Binning data...")
    bin_size_deg = bin_size_sec / 3600.0
    bins = np.arange(0, max_dist_deg + bin_size_deg, bin_size_deg)
    bin_indices = np.digitize(psi_deg, bins)

    results = []
    # Bin 0 (Variance)
    variance = np.mean(dg**2)
    results.append({'psi_deg': 0, 'cov': variance, 'count': len(dg)})

    for k in range(1, len(bins)):
        mask = (bin_indices == k)
        if np.any(mask):
            mean_cov = np.mean(dg_products[mask])
            count = np.sum(mask)
            results.append({
                'psi_deg': bins[k-1] + (bin_size_deg/2),
                'cov': mean_cov,
                'count': count
            })
            
    return pd.DataFrame(results)

# ==========================================
# PART 2: TSCHERNING-RAPP MODEL FITTING
# ==========================================

R_EARTH = 6371000.0 

def precompute_legendre(psi_rad, n_max):
    x = np.cos(psi_rad)
    P = np.zeros((n_max + 1, len(x)))
    P[0, :] = 1.0
    P[1, :] = x
    for n in range(1, n_max):
        P[n+1, :] = ((2*n + 1) * x * P[n, :] - n * P[n-1, :]) / (n + 1)
    return P

def fit_tscherning_rapp(cov_df, n_min=3, n_max=3000, fixed_B=24, flight_height=0.0):
    """
    Fits parameters A and R_B considering observation height.
    flight_height: Average height of observations in meters.
    """
    print(f"--- Step 2: Fitting T-R Model Parameters (H={flight_height}m, B={fixed_B}) ---")
    
    data = cov_df.dropna()
    psi_vals = np.radians(data['psi_deg'].values)
    emp_cov = data['cov'].values
    
    print(f"   Pre-computing Legendre Polynomials (N={n_max})...")
    P_matrix = precompute_legendre(psi_vals, n_max)
    
    # Define observation radius (Earth Radius + Flight Height)
    r_obs = R_EARTH + flight_height

    def model_func(psi_dummy, A, depth):
        # R_B is derived from Depth (relative to surface R)
        R_B = R_EARTH - depth
        
        # [CRITICAL UPDATE] Attenuation factor includes flight height
        # s = (R_B / r_obs)^2
        s_ratio = (R_B / r_obs) ** 2
        
        ns = np.arange(n_min, n_max + 1)
        
        # T-R Model Formula
        sigma_sq = (A * (ns - 1)) / ((ns - 2) * (ns + fixed_B))
        s_terms = s_ratio ** (ns + 1)
        
        P_sub = P_matrix[ns, :] 
        terms = sigma_sq * s_terms
        return np.sum(terms[:, np.newaxis] * P_sub, axis=0)

    # Initial Guesses (Variance, 10km depth)
    p0 = [np.var(emp_cov), 10000.0] 
    bounds = ([0, 0], [np.inf, R_EARTH])
    
    try:
        popt, pcov = curve_fit(model_func, psi_vals, emp_cov, p0=p0, bounds=bounds, maxfev=5000)
        A_fit, depth_fit = popt
        R_B_fit = R_EARTH - depth_fit
        
        # Generate fitted curve
        fitted_cov = model_func(psi_vals, A_fit, depth_fit)
        
        return A_fit, R_B_fit, fitted_cov
        
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None, None

# ==========================================
# PART 3: LSC COVARIANCE FUNCTION
# ==========================================
GAMMA = 979800.0

def compute_covariance_tscherning_rapp(psi_deg, A, R_B, B=24, n_min=3, n_max=3000, 
                                     cov_type='gg', h1=0.0, h2=0.0):
    """
    Computes T-R covariance between two points at heights h1 and h2.
    
    Parameters:
    -----------
    h1, h2 : float
        Heights of the two points in meters (default 0.0).
    """
    psi_rad = np.atleast_1d(np.radians(psi_deg))
    P_matrix = precompute_legendre(psi_rad, n_max)
    
    # [CRITICAL CHANGE]
    # Calculate radii for the two points
    r1 = R_EARTH + h1
    r2 = R_EARTH + h2
    
    # Calculate the s-ratio for varying heights
    # s = (R_B^2) / (r1 * r2)
    s_ratio = (R_B**2) / (r1 * r2)
    
    ns = np.arange(n_min, n_max + 1)
    
    # Scaling factor for functional types
    # Note: Functional relationships often depend on r.
    # For Gravity Anomaly (g), the term includes (n-1)/r.
    # Ideally, we use r_mean = sqrt(r1*r2) or R_EARTH for the denominator scale factor 
    # as the variation is small compared to the s_ratio power law.
    # Using R_EARTH in the denominator coefficient is standard approximation.
    
    term_common = A / ((ns - 2) * (ns + B))
    
    if cov_type == 'gg':
        # Gravity-Gravity
        sigma_coeff = (ns - 1) * term_common
    elif cov_type == 'gn':
        # Gravity-Geoid (Geoid is N, Gravity is g)
        # N ~ T/gamma, g ~ -dT/dr - 2T/r
        sigma_coeff = (R_EARTH / GAMMA) * A / ((ns - 2) * (ns + B)) 
    elif cov_type == 'nn':
        # Geoid-Geoid
        sigma_coeff = (R_EARTH**2 / GAMMA**2) * A / ((ns - 1) * (ns - 2) * (ns + B))
    else:
        raise ValueError("Unknown cov_type")

    # The attenuation term s^(n+1) handles the height difference
    s_terms = s_ratio ** (ns + 1)
    
    P_sub = P_matrix[ns, :]
    terms = sigma_coeff * s_terms
    cov = np.sum(terms[:, np.newaxis] * P_sub, axis=0)
    
    if np.isscalar(psi_deg):
        return cov[0]
    return cov

def save_tr_parameters(A, R_B, B, height, filename="tr_parameters.json"):
    params = {
        'A': float(A),
        'R_B': float(R_B),
        'B': int(B),
        'flight_height': float(height),
        'R_EARTH': float(R_EARTH),
        'depth_km': float((R_EARTH - R_B) / 1000.0)
    }
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"   T-R parameters saved to {filename}")

def load_tr_parameters(filename="tr_parameters.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process LSC Covariance Fitting')
    parser.add_argument('--data', type=str, required=True, help='Path to data file (lon, lat, dg)')
    parser.add_argument('--max_dist_deg', type=float, default=0.30, help='Max distance (degrees)')
    parser.add_argument('--bin_size_sec', type=int, default=30, help='Bin size (seconds)')
    parser.add_argument('--n_max', type=int, default=3000, help='Max degree')
    parser.add_argument('--n_min', type=int, default=3, help='Min degree')   
    parser.add_argument('--B', type=int, default=24, help='B parameter')
    parser.add_argument('--height', type=float, default=0.0, help='Flight height in meters (e.g. 5000)')
    
    args = parser.parse_args()    
    
    # 1. LOAD DATA
    print(f"Loading data from {args.data}...")
    try:
        # Robust loading: skip comment lines, handle extra columns
        df = pd.read_csv(args.data, sep='\s+', header=None, comment='#')
        if df.shape[1] > 3:
             print(f"Warning: File has {df.shape[1]} columns. Using first 3 as lon, lat, dg.")
             df = df.iloc[:, :3]
        
        if df.shape[1] == 3:
            df.columns = ['lon', 'lat', 'dg']
        else:
             raise ValueError(f"Expected at least 3 columns (lon, lat, dg), found {df.shape[1]}")
             
        df = df.astype(float)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    # 2. RUN STEP 1: CALCULATE EMPIRICAL COVARIANCE
    # [MODIFIED] Use args instead of hardcoded numbers
    cov_df = calculate_empirical_covariance(df, max_dist_deg=args.max_dist_deg, bin_size_sec=args.bin_size_sec)
    cov_df.to_csv("empirical_covariance_data.csv", index=False)
    print("   Empirical covariance data saved.")

    # 3. RUN STEP 2: FIT THE MODEL
    # [MODIFIED] Pass args.height, args.n_max, args.n_min, args.B
    A_final, RB_final, fitted_curve = fit_tscherning_rapp(
        cov_df, 
        n_min=args.n_min, 
        n_max=args.n_max, 
        fixed_B=args.B,
        flight_height=args.height
    )

    if A_final is not None:
        # Save fitted curve
        data_for_fit = cov_df.dropna()
        fitted_df = pd.DataFrame({
            'psi_deg': data_for_fit['psi_deg'].values,
            'cov': fitted_curve
        })
        fitted_df.to_csv("fitted_curve.csv", index=False)
        
        # Save Parameters
        save_tr_parameters(A_final, RB_final, args.B, args.height, filename="tr_parameters.json")
        
        # Save Summary CSV
        #params_df = pd.DataFrame({
        #    'parameter': ['A', 'R_B', 'B', 'Height', 'depth_km'],
        #    'value': [A_final, RB_final, args.B, args.height, (R_EARTH - RB_final)/1000.0],
        #    'unit': ['mGal^2', 'm', '', 'm', 'km']
        #})
        #params_df.to_csv("tr_parameters.csv", index=False)

        # 4. REPORT AND PLOT
        print("\n" + "="*30)
        print(" FINAL RESULTS")
        print("="*30)
        print(f" A Parameter  : {A_final:.5f} mGal^2")
        print(f" B Parameter  : {args.B}")
        print(f" Flight Height: {args.height} m")
        print(f" R_B Parameter: {RB_final:.2f} m")
        print(f" Depth (R-R_B): {(R_EARTH - RB_final)/1000:.5f} km")
        print("="*30)

        lscp.plot_covariance_fit(
            cov_df, fitted_curve, args.height, A_final, args.B, 
            output_file='lsc_covariance_fit.png'
        )
        print("Plot saved to lsc_covariance_fit.png")