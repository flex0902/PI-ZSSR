import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import lsc_covariance_plotting as lscp

# Constants
R_EARTH = 6371000.0 

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def precompute_legendre(psi_rad, n_max):
    # (Same as before)
    x = np.cos(psi_rad)
    P = np.zeros((n_max + 1, len(x)))
    P[0, :] = 1.0
    P[1, :] = x
    for n in range(1, n_max):
        P[n+1, :] = ((2*n + 1) * x * P[n, :] - n * P[n-1, :]) / (n + 1)
    return P

def get_ssr(observed, predicted):
    """Calculate Sum of Squared Residuals"""
    return np.sum((observed - predicted)**2)

# ==========================================
# 2. CORE FITTING ENGINE (Single B)
# ==========================================
def fit_with_fixed_B(psi_vals, emp_cov, P_matrix, fixed_B, n_min, n_max, flight_height=0.0):
    """
    Fits A and Depth for a SPECIFIC integer B.
    Returns: (A, Depth, Fitted_Curve, SSR_Error)
    """

    # Define observation radius (Earth Radius + Flight Height)
    r_obs = R_EARTH + flight_height

    def model_func(psi_dummy, A, depth):
        R_B = R_EARTH - depth
        #s_ratio = (R_B / R_EARTH) ** 2
        s_ratio = (R_B / r_obs) ** 2

        # Vectorized Summation
        ns = np.arange(n_min, n_max + 1)
        
        # T-R Model Formula with the current FIXED B
        sigma_sq = (A * (ns - 1)) / ((ns - 2) * (ns + fixed_B))
        s_terms = s_ratio ** (ns + 1)
        
        P_sub = P_matrix[ns, :] 
        terms = sigma_sq * s_terms
        return np.sum(terms[:, np.newaxis] * P_sub, axis=0)

    # Initial Guesses: A=Variance, Depth=10km
    p0 = [np.var(emp_cov), 10000.0] 
    bounds = ([0, 0], [np.inf, R_EARTH])
    
    try:
        popt, pcov = curve_fit(model_func, psi_vals, emp_cov, p0=p0, bounds=bounds, maxfev=5000)
        A_fit, depth_fit = popt
        
        # Calculate the curve and the error
        fitted_curve = model_func(psi_vals, A_fit, depth_fit)
        ssr = get_ssr(emp_cov, fitted_curve)
        
        return A_fit, depth_fit, fitted_curve, ssr
        
    except RuntimeError:
        return None, None, None, np.inf

# ==========================================
# 3. OPTIMIZER LOOP (Find Best B)
# ==========================================
def find_best_integer_B(cov_df, b_min=4, b_max=60, n_min=3, n_max=3000, flight_height=0.0):
    print(f"--- Optimizing B (Range: {b_min} to {b_max}), flight height = {flight_height:.1f} m ---")
    
    data = cov_df.dropna()
    psi_vals = np.radians(data['psi_deg'].values)
    emp_cov = data['cov'].values
    
    # Pre-compute Legendre once (heavy lifting)
    print(f"   Pre-computing Legendre Polynomials (N={n_max})...")
    P_matrix = precompute_legendre(psi_vals, n_max)
    
    best_B = None
    best_params = None # (A, Depth)
    best_curve = None
    lowest_ssr = np.inf
    
    # Store history to plot "Error vs B"
    history_B = []
    history_SSR = []

    # --- GRID SEARCH LOOP ---
    for b_val in range(b_min, b_max + 1):
        A, depth, curve, ssr = fit_with_fixed_B(
            psi_vals, emp_cov, P_matrix, b_val, n_min, n_max, flight_height=flight_height
        )
        
        if A is not None:
            history_B.append(b_val)
            history_SSR.append(ssr)
            if b_val == b_min or b_val % 50 == 0:
                print(f"   B={b_val}: A={A:.3f}, depth={depth/1000:.2f} km, SSR={ssr:.2f}")
            
            # Check if this is the new best record
            if ssr < lowest_ssr:
                lowest_ssr = ssr
                best_B = b_val
                best_params = (A, depth)
                best_curve = curve
            
            # Check for flattening trend (relative change < 1e-4)
            # Only check if we have at least 2 points and SSR is decreasing
            if len(history_SSR) >= 2:
                prev_ssr = history_SSR[-2]
                if ssr < prev_ssr and (prev_ssr - ssr)/prev_ssr < 1e-4:
                    print(f"Stopping search: SSR improvement flattened (< 1e-4).")
                    break
                
    print(f"   Optimization Complete. Best B found: {best_B} at SSR={lowest_ssr:.2f}")
    return best_B, best_params, best_curve, (history_B, history_SSR)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data from CSV (經驗共變異函數資料)
    # CSV 格式要求：
    #   - 必須包含兩個欄位：'psi_deg' (球面距離，單位：度) 和 'cov' (共變異值，單位：mGal²)
    #   - 範例檔案：example_covariance_data.csv
    
    # 選項 A: 從 CSV 載入實際資料
    print("Loading covariance data from CSV...")
    cov_df = pd.read_csv("empirical_covariance_data.csv") #example_covariance_data.csv
    # 確保欄位名稱正確
    assert 'psi_deg' in cov_df.columns and 'cov' in cov_df.columns, \
        "CSV must contain columns 'psi_deg' and 'cov'"
    
    # 選項 B: 使用合成資料（測試用，可註解掉選項 A 後使用）
    # print("Generating synthetic data...")
    # psi_dummy = np.linspace(0, 1.0, 30)
    # cov_dummy = 800 * np.exp(-psi_dummy * 6) + np.random.normal(0, 10, 30)
    # cov_df = pd.DataFrame({'psi_deg': psi_dummy, 'cov': cov_dummy})

    # 2. Find the best B
    best_B, params, best_curve, history = find_best_integer_B(cov_df, b_min=4, b_max=500)

    # 3. Visualization using external module
    A_final, depth_final = params
    R_B_final = R_EARTH - depth_final
    
    lscp.plot_optimized_fit_and_search(
        cov_df, best_curve, history, best_B, A_final, depth_final, 0.0,
        output_file='lsc_covariance_fit_B.png'
    )
    print("Covariance fit plot saved to lsc_covariance_fit_B.png")
    
    print("\n--- FINAL ESTIMATED PARAMETERS ---")
    print(f" A   = {A_final:.4f}")
    print(f" B   = {best_B} (Integer)")
    print(f" R_B = {R_B_final:.2f}")