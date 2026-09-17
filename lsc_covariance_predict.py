"""
範例：如何在 LSC (最小二乘配置) 中使用 T-R 模型參數計算共變異值

這個檔案展示如何載入保存的參數並在 LSC 計算中使用共變異函數。
"""

import numpy as np
import pandas as pd
import lsc_covariance_plotting as lscp
from lsc_covariance_fit import (
    compute_covariance_tscherning_rapp,
    load_tr_parameters,
    R_EARTH
)
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.linalg import solve
import argparse

# C(ψ) lookup: T-R at fixed (A, R_B, B, h1, h2) depends only on spherical distance.
_COV_TABLES = {}


def eval_tr_cov(psi_deg, A, R_B, B=24, n_max=1800, cov_type="gg", h1=0.0, h2=0.0):
    psi = np.atleast_1d(np.asarray(psi_deg, dtype=float))
    key = (round(float(A), 10), round(float(R_B), 3), int(B), int(n_max),
           str(cov_type), round(float(h1), 3), round(float(h2), 3))
    if key not in _COV_TABLES:
        psi_grid = np.linspace(0.0, 0.5, 5001)
        c_grid = np.asarray(
            compute_covariance_tscherning_rapp(
                psi_grid, A, R_B, B, n_max=n_max, cov_type=cov_type, h1=h1, h2=h2
            ),
            dtype=np.float64,
        )
        _COV_TABLES[key] = (psi_grid, c_grid)
        print(f"  Cached T-R C(psi): B={B}, h1={h1:.1f}, h2={h2:.1f}, n_max={n_max}")
    psi_grid, c_grid = _COV_TABLES[key]
    out = np.interp(np.clip(psi, 0.0, psi_grid[-1]), psi_grid, c_grid)
    if np.isscalar(psi_deg) or np.ndim(psi_deg) == 0:
        return float(out[0])
    return out

# ==========================================
# 步驟 1: 載入參數
# ==========================================

def load_parameters(param_file="tr_parameters.json"):
    """
    載入 T-R 模型參數 (修正版：包含飛行高度)
    """
    params = load_tr_parameters(param_file)
    A = params['A']
    R_B = params['R_B']
    B = params['B']
    # [新增] 讀取飛行高度，如果參數檔中沒有紀錄，預設為 0
    flight_height = params.get('flight_height', 0.0)
    
    print(f"\n載入的參數:")
    print(f"  A = {A:.5f} mGal^2")
    print(f"  R_B = {R_B:.2f} m")
    print(f"  B = {B}")
    print(f"  訓練時的飛行高度 (obs_height) = {flight_height:.1f} m")
    
    return A, R_B, B, flight_height

# ==========================================
# 步驟 2: 載入觀測點數據
# ==========================================

def load_observation_data(data_file):
    """
    載入觀測點數據（從 .xyg 檔案）
    
    Parameters:
    -----------
    data_file : str
        數據檔案路徑（格式：lon lat dg，空格分隔）
        
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
            df = pd.read_csv(data_file, sep='\s+', header=None, encoding=encoding)
            print(f"成功使用編碼: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # If it's not an encoding error, re-raise it
            raise
    
    if df is None:
        raise ValueError(f"無法使用任何編碼讀取檔案: {data_file}")
    
    df.columns = ['lon', 'lat', 'dg']
    df['lon'] = df['lon'].astype(float)
    df['lat'] = df['lat'].astype(float)
    df['dg'] = df['dg'].astype(float)
    
    print(f"  載入 {len(df)} 個觀測點")
    print(f"  經度範圍: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
    print(f"  緯度範圍: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
    
    return df

# ==========================================
# 步驟 3: 載入預測點
# ==========================================

def load_prediction_points(data_file, n_points=3, random_seed=42):
    """
    從數據檔案中隨機選取預測點
    
    Parameters:
    -----------
    data_file : str
        數據檔案路徑
    n_points : int
        預測點數量
    random_seed : int
        隨機種子
        
    Returns:
    --------
    pred_df : DataFrame
        包含 'lon', 'lat' 欄位的 DataFrame（預測點座標）
    """
    print(f"\n載入預測點（從 {data_file} 選取 {n_points} 個點）")
    
    # 載入完整數據
    df = pd.read_csv(data_file, sep='\s+', header=None)
    df.columns = ['lon', 'lat', 'dg']
    df['lon'] = df['lon'].astype(float)
    df['lat'] = df['lat'].astype(float)
    
    # 隨機選取預測點
    np.random.seed(random_seed)
    indices = np.random.choice(len(df), size=n_points, replace=False)
    pred_df = df.iloc[indices][['lon', 'lat']].copy()
    pred_df.reset_index(drop=True, inplace=True)
    
    print(f"  選取的預測點:")
    for i, row in pred_df.iterrows():
        print(f"    點 {i+1}: ({row['lon']:.6f}, {row['lat']:.6f})")
    
    return pred_df

# ==========================================
# 步驟 4: 根據預測點位置搜尋鄰近觀測點
# ==========================================

def find_nearby_observations(obs_df, pred_lat, pred_lon, max_dist_deg, tree=None, obs_coords=None):
    """
    根據預測點位置搜尋在 max_dist_deg 範圍內的觀測點
    """
    if obs_coords is None:
        obs_lat_rad = np.radians(obs_df['lat'].values)
        obs_lon_rad = np.radians(obs_df['lon'].values)
        obs_x = np.cos(obs_lat_rad) * np.cos(obs_lon_rad)
        obs_y = np.cos(obs_lat_rad) * np.sin(obs_lon_rad)
        obs_z = np.sin(obs_lat_rad)
        obs_coords = np.vstack([obs_x, obs_y, obs_z]).T

    pred_lat_rad = np.radians(pred_lat)
    pred_lon_rad = np.radians(pred_lon)
    pred_coord = np.array([
        np.cos(pred_lat_rad) * np.cos(pred_lon_rad),
        np.cos(pred_lat_rad) * np.sin(pred_lon_rad),
        np.sin(pred_lat_rad),
    ])

    if tree is None:
        tree = cKDTree(obs_coords)

    max_dist_rad = np.radians(max_dist_deg)
    max_chord_dist = 2 * np.sin(max_dist_rad / 2)
    indices = tree.query_ball_point(pred_coord, max_chord_dist)

    if len(indices) == 0:
        return pd.DataFrame(columns=['lon', 'lat', 'dg']), np.array([])

    nearby_df = obs_df.iloc[indices].copy()
    nearby_df.reset_index(drop=True, inplace=True)
    return nearby_df, np.array(indices)

# ==========================================
# 方法 1: 從 JSON 檔案載入參數（保留向後兼容）
# ==========================================

def example_load_and_use():
    """載入參數並計算共變異值的範例"""
    
    # 1. 載入保存的參數
    params = load_tr_parameters("tr_parameters.json")
    
    A = params['A']
    R_B = params['R_B']
    B = params['B']
    
    print(f"\n載入的參數:")
    print(f"  A = {A:.5f} mGal^2")
    print(f"  R_B = {R_B:.2f} m")
    print(f"  B = {B}")
    
    # 2. 計算單一距離的共變異值
    psi_single = 0.15  # 度
    cov_single = compute_covariance_tscherning_rapp(psi_single, A, R_B, B)
    print(f"\n距離 {psi_single} 度的共變異值: {cov_single:.4f} mGal^2")
    
    # 3. 計算多個距離的共變異值（向量化）
    psi_array = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])  # 度
    cov_array = compute_covariance_tscherning_rapp(psi_array, A, R_B, B)
    print(f"\n多個距離的共變異值:")
    for psi, cov in zip(psi_array, cov_array):
        print(f"  {psi:.3f} 度 -> {cov:.4f} mGal^2")
    
    return A, R_B, B

# ==========================================
# 方法 2: 在 LSC 中建立共變異矩陣
# ==========================================
#def build_covariance_matrix_for_lsc(obs_lat, obs_lon, A, R_B, B=24, cov_type='gg'):
def build_covariance_matrix_for_lsc(obs_lat, obs_lon, A, R_B, B=24, max_degree=1800, cov_type='gg', obs_height=0.0):    
    """
    為 LSC 建立觀測點之間的共變異矩陣
    
    Parameters:
    -----------
    obs_lat : array-like
        觀測點緯度（度）
    obs_lon : array-like
        觀測點經度（度）
    A : float
        T-R 參數 A
    R_B : float
        T-R 參數 R_B
    B : int, default=24
        T-R 參數 B
    max_degree : int, default=1800
        最大階數
    cov_type : str, default='gg'
        共變異類型 ('gg', 'gn', 'nn')
    obs_height : float, default=0.0
        觀測點高度（米），默認為 0.0
        
    Returns:
    --------
    C_obs : ndarray
        觀測點之間的共變異矩陣 (N x N)
    """
    n_obs = len(obs_lat)
    
    # 轉換為 3D 卡式座標
    lat_rad = np.radians(obs_lat)
    lon_rad = np.radians(obs_lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.vstack([x, y, z]).T
    
    # 計算所有點對之間的距離
    chord_dists = pdist(coords, metric='euclidean')
    psi_rad = 2 * np.arcsin(chord_dists / 2)
    psi_deg = np.degrees(psi_rad)
    
    # [修正點 1] 計算共變異值時傳入 cov_type
    #cov_values = compute_covariance_tscherning_rapp(psi_deg, A, R_B, B, cov_type=cov_type)
    # Pass obs_height for BOTH h1 and h2
    cov_values = eval_tr_cov(
        psi_deg, A, R_B, B, n_max=max_degree, cov_type=cov_type,
        h1=obs_height, h2=obs_height
    )

    C_obs = np.zeros((n_obs, n_obs))
    triu_indices = np.triu_indices(n_obs, k=1)
    C_obs[triu_indices] = cov_values
    C_obs = C_obs + C_obs.T

    variance = eval_tr_cov(
        0.0, A, R_B, B, n_max=max_degree, cov_type=cov_type,
        h1=obs_height, h2=obs_height
    )

    np.fill_diagonal(C_obs, variance)
    
    return C_obs

#def build_cross_covariance_matrix(obs_lat, obs_lon, pred_lat, pred_lon, A, R_B, B=24, cov_type='gg'):
def build_cross_covariance_matrix(obs_lat, obs_lon, pred_lat, pred_lon, A, R_B, B=24, max_degree=1800, cov_type='gg', 
                                obs_height=0.0, pred_height=0.0):
    """
    建立觀測點與預測點之間的交叉共變異矩陣
    
    Parameters:
    -----------
    obs_lat, obs_lon : array-like
        觀測點座標（度）
    pred_lat, pred_lon : array-like
        預測點座標（度）
    A, R_B, B : T-R 模型參數
    max_degree : int, default=1800
        最大階數
    cov_type : str, default='gg'
        共變異類型 ('gg', 'gn', 'nn')
    obs_height : float, default=0.0
        觀測點高度（米），默認為 0.0
    pred_height : float, default=0.0
        預測點高度（米），默認為 0.0
        
    Returns:
    --------
    C_cross : ndarray
        交叉共變異矩陣 (N_obs x N_pred)
    """
    n_obs = len(obs_lat)
    n_pred = len(pred_lat)
    
    # 轉換觀測點座標
    obs_lat_rad = np.radians(obs_lat)
    obs_lon_rad = np.radians(obs_lon)
    obs_x = np.cos(obs_lat_rad) * np.cos(obs_lon_rad)
    obs_y = np.cos(obs_lat_rad) * np.sin(obs_lon_rad)
    obs_z = np.sin(obs_lat_rad)
    obs_coords = np.vstack([obs_x, obs_y, obs_z]).T
    
    # 轉換預測點座標
    pred_lat_rad = np.radians(pred_lat)
    pred_lon_rad = np.radians(pred_lon)
    pred_x = np.cos(pred_lat_rad) * np.cos(pred_lon_rad)
    pred_y = np.cos(pred_lat_rad) * np.sin(pred_lon_rad)
    pred_z = np.sin(pred_lat_rad)
    pred_coords = np.vstack([pred_x, pred_y, pred_z]).T
    
    chord = np.linalg.norm(obs_coords[:, None, :] - pred_coords[None, :, :], axis=2)
    psi_deg = np.degrees(2 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
    C_cross = eval_tr_cov(
        psi_deg.ravel(), A, R_B, B, n_max=max_degree, cov_type=cov_type,
        h1=obs_height, h2=pred_height
    ).reshape(n_obs, n_pred)
    return C_cross

def build_prediction_covariance_matrix(pred_lat, pred_lon, A, R_B, B=24, max_degree=1800, cov_type='gg', pred_height=0.0):
    """
    建立預測點之間的共變異矩陣 (修正版)
    
    Parameters:
    -----------
    pred_lat, pred_lon : array-like
        預測點座標（度）
    A, R_B, B : T-R 模型參數
    max_degree : int, default=1800
        最大階數
    cov_type : str
        共變異類型 ('gg', 'gn', 'nn')
    pred_height : float
        預測點高度（米），默認為 0.0
        
    Returns:
    --------
    C_pred : ndarray
        預測點之間的共變異矩陣 (N_pred x N_pred)
    """
    n_pred = len(pred_lat)
    
    # 轉換為 3D 卡式座標
    lat_rad = np.radians(pred_lat)
    lon_rad = np.radians(pred_lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.vstack([x, y, z]).T
    
    # 計算所有點對之間的距離
    chord_dists = pdist(coords, metric='euclidean')
    psi_rad = 2 * np.arcsin(chord_dists / 2)
    psi_deg = np.degrees(psi_rad)
    
    # 計算非對角線元素的共變異值
    # Pass pred_height for BOTH h1 and h2 (since it's pred-pred covariance)
    cov_values = eval_tr_cov(
        psi_deg, A, R_B, B, n_max=max_degree, cov_type=cov_type,
        h1=pred_height, h2=pred_height
    )

    # [修正] 先建立矩陣，再填值
    C_pred = np.zeros((n_pred, n_pred))
    
    # 填入非對角線元素
    triu_indices = np.triu_indices(n_pred, k=1)
    C_pred[triu_indices] = cov_values
    C_pred = C_pred + C_pred.T  # 對稱化
    
    # [修正] 最後計算並填入對角線元素 (變異數)
    variance = eval_tr_cov(
        0.0, A, R_B, B, n_max=max_degree, cov_type=cov_type,
        h1=pred_height, h2=pred_height
    )
    np.fill_diagonal(C_pred, variance)
    
    return C_pred

#def perform_lsc_prediction(obs_lat, obs_lon, obs_values, pred_lat, pred_lon, A, R_B, B=24, 
#                           target='gravity', noise_std=1.0, verbose=False):
def perform_lsc_prediction(obs_lat, obs_lon, obs_values, pred_lat, pred_lon, A, R_B, B=24, max_degree=1800,
                           target='gravity', noise_std=1.0, 
                           obs_height=0.0, pred_height=0.0, verbose=False):
    """
    執行 LSC (最小二乘配置) 預測- 優化版
    新增參數: noise_std (觀測噪聲標準差, mGal)

    Parameters:
    -----------
    obs_lat, obs_lon : array-like
        觀測點座標（度）
    obs_values : array-like
        觀測值（mGal）
    pred_lat, pred_lon : array-like
        預測點座標（度）
    A, R_B, B : T-R 模型參數   
    max_degree : int
        最大階數
    target : str
        'gravity' : 推估重力異常 (單位 mGal)
        'geoid'   : 推估大地起伏 (單位 m)
    noise_std : float
        觀測噪聲 (mGal)
    verbose : bool
        是否打印詳細信息

    Returns:
    --------
    predicted : ndarray
        預測值 (N_pred,)
    pred_error_cov : ndarray
        預測誤差共變異矩陣 (N_pred x N_pred)
    """
# === [關鍵邏輯切換] ===
    if target == 'gravity':
        cross_type = 'gg'  # 觀測(g) -> 預測(g)
        pred_type  = 'gg'  # 預測量的自協方差 (g-g)
        if verbose:
            print("--- 執行模式: 重力異常推估 ---")
        
    elif target == 'geoid':
        cross_type = 'gn'  # 觀測(g) -> 預測(N)
        pred_type  = 'nn'  # 預測量的自協方差 (N-N)
        if verbose:
            print("--- 執行模式: 大地起伏推估 ---")
        
    else:
        raise ValueError("Target must be 'gravity' or 'geoid'")

    # 1. C_obs 永遠是 gg (因為輸入數據永遠是重力異常)
    C_obs = build_covariance_matrix_for_lsc(
        obs_lat, obs_lon, A, R_B, B, max_degree, cov_type='gg',
        obs_height=obs_height
        )

    # 2. C_cross 和 C_pred 根據目標改變
    C_cross = build_cross_covariance_matrix(
        obs_lat, obs_lon, pred_lat, pred_lon, A, R_B, B, max_degree, cov_type=cross_type,
        obs_height=obs_height, pred_height=pred_height
        )
    C_pred = build_prediction_covariance_matrix(
        pred_lat, pred_lon, A, R_B, B, max_degree, cov_type=pred_type, 
        pred_height=pred_height
        )
    
    # 3. 加入噪聲變異數 (Regularization)
    # C_total = C_obs + D (D 是噪聲共變異矩陣，通常為對角矩陣)
    # 這一步至關重要，能防止矩陣奇異並過濾高頻噪聲
    n_obs = len(obs_values)
    noise_variance = noise_std ** 2
    C_total = C_obs + np.eye(n_obs) * noise_variance
    
    # 4. 解線性方程組 (取代 inv 求逆)
    # 公式: predicted = C_cross.T * (C_obs + D)^-1 * l
    # 改寫為: (C_obs + D) * X = l -> 解 X -> predicted = C_cross.T * X
    
    # 計算權重向量 X (assume_a='pos' 表示矩陣為正定，計算更快)
    weights = solve(C_total, obs_values, assume_a='pos')
    
    # 計算預測值
    predicted = C_cross.T @ weights
    
    # 5. 計算誤差估計
    # Error Cov = C_pred - C_cross.T * (C_obs + D)^-1 * C_cross
    # 同樣使用 solve 避免求逆
    term2 = solve(C_total, C_cross, assume_a='pos')
    pred_error_cov = C_pred - C_cross.T @ term2
    
    return predicted, pred_error_cov, C_obs, C_cross, C_pred

def generate_synthetic_observations(obs_lat, obs_lon, signal_type='smooth', noise_level=1.0):
    """
    生成模擬的觀測值
    
    Parameters:
    -----------
    obs_lat, obs_lon : array-like
        觀測點座標（度）
    signal_type : str
        信號類型：'smooth' (平滑信號) 或 'random' (隨機信號)
    noise_level : float
        噪聲水平（標準差，mGal）
        
    Returns:
    --------
    obs_values : ndarray
        觀測值（mGal）
    true_values : ndarray
        真實值（無噪聲，用於驗證）
    """
    if signal_type == 'smooth':
        # 使用平滑的空間相關信號
        true_values = 20 * np.sin(obs_lat * 10) * np.cos(obs_lon * 10)
    else:
        # 隨機信號
        np.random.seed(42)
        true_values = np.random.normal(0, 10, len(obs_lat))
    
    # 添加觀測噪聲
    np.random.seed(123)
    noise = np.random.normal(0, noise_level, len(obs_lat))
    obs_values = true_values + noise
    
    return obs_values, true_values

# ==========================================
# 步驟 5-8: 完整的 LSC 預測流程（優化版）
# ==========================================

def perform_lsc_prediction_for_point(obs_df, pred_lat, pred_lon, A, R_B, B,
                                     max_degree=1800, max_dist_deg=0.2, target='gravity', noise_std=1.0,
                                     obs_height=0.0, pred_height=0.0,
                                     tree=None, obs_coords=None, exclude_pos=None):
    """
    對單一預測點執行 LSC 預測（使用局部觀測點）- 支援高度
    
    Parameters:
    -----------
    obs_df : DataFrame
        完整觀測點數據
    pred_lat : float
        預測點緯度（度）
    pred_lon : float
        預測點經度（度）
    A, R_B, B : T-R 模型參數
    max_degree : int
        最大階數
    max_dist_deg : float
        最大搜索距離（度）
    target : str
        'gravity' 或 'geoid'
    noise_std : float
        觀測噪聲標準差（mGal）
        
    Returns:
    --------
    predicted : float
        預測值
    pred_std : float
        預測標準誤差
    nearby_df : DataFrame
        使用的鄰近觀測點
    """
    # 搜尋鄰近觀測點
    nearby_df, indices = find_nearby_observations(
        obs_df, pred_lat, pred_lon, max_dist_deg, tree=tree, obs_coords=obs_coords
    )
    if exclude_pos is not None and len(indices) > 0:
        keep = indices != exclude_pos
        indices = indices[keep]
        nearby_df = obs_df.iloc[indices].copy()
        nearby_df.reset_index(drop=True, inplace=True)
    
    if len(nearby_df) == 0:
        return np.nan, np.nan, nearby_df
    
    # 提取觀測點座標和觀測值
    obs_lat = nearby_df['lat'].values
    obs_lon = nearby_df['lon'].values
    obs_values = nearby_df['dg'].values
    
    # 執行 LSC 預測
    # [修正] 將 obs_height 和 pred_height 傳入核心函式
    predicted, pred_error_cov, _, _, _ = perform_lsc_prediction(
        obs_lat, obs_lon, obs_values, 
        np.array([pred_lat]), np.array([pred_lon]), 
        A, R_B, B, max_degree,
        target=target,
        noise_std=noise_std,
        obs_height=obs_height,   # [傳遞]
        pred_height=pred_height, # [傳遞]
        verbose=False
    )
    
    pred_std = np.sqrt(pred_error_cov[0, 0])
    
    return predicted[0], pred_std, nearby_df

# ==========================================
# 主程序：完整的 LSC 使用流程
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LSC prediction')
    parser.add_argument('--data', type=str, required=True, help='Path to data file (lon, lat, dg)')
    parser.add_argument('--max_dist_deg', type=float, default=0.20, help='Max search radius distance (degrees)')
    parser.add_argument('--pred_height', type=float, help='Predicted height in meters (e.g. 0 = 預測目標在地面/大地水準面)')
    parser.add_argument('--target', type=str, default='gravity', help='Target type: gravity or geoid')
    parser.add_argument('--n_max', type=int, default=1800, help='Max degree')
    args = parser.parse_args()


    print("="*70)
    print("LSC 共變異函數使用範例（優化版 - 避免記憶體問題）")
    print("="*70)
    
    # ==========================================
    # 步驟 1: 載入參數
    # ==========================================
    print("\n" + "="*70)
    print("步驟 1: 載入參數")
    print("="*70)
    A, R_B, B, flight_height = load_parameters("tr_parameters.json")
    max_degree = args.n_max

    # ==========================================
    # 步驟 2: 載入觀測點
    # ==========================================
    print("\n" + "="*70)
    print("步驟 2: 載入觀測點")
    print("="*70)
    obs_df = load_observation_data(args.data) # "Sulawesi_airg1-EGM_d1800.xyg"
    
    # ==========================================
    # 步驟 3: 載入預測點
    # ==========================================
    print("\n" + "="*70)
    print("步驟 3: 載入預測點")
    print("="*70)
    pred_df = load_prediction_points("Sulawesi_airg1-EGM_d1800.xyg", n_points=3, random_seed=42)
    
    # ==========================================
    # 步驟 4-6: 對每個預測點執行 LSC 預測
    # ==========================================
    print("\n" + "="*70)
    print("步驟 4-6: 搜尋鄰近觀測點並執行 LSC 預測")
    print("="*70)
    
    max_dist_deg = args.max_dist_deg #0.2  # 最大搜索距離（度）
    target = args.target # 'gravity' 或 'geoid'
    noise_std = 3.0  # 觀測噪聲標準差（mGal）
    
    obs_height = flight_height  # 觀測點高度 (預設為飛行高度)
    pred_height = flight_height # 預測點高度 (預設為飛行高度)
    if args.pred_height is not None: # 0 = 預測目標在地面/大地水準面
        pred_height = args.pred_height
  
    print(f"\nLSC 預測設定:")
    print(f"  目標類型: {target}")
    print(f"  最大搜索距離: {max_dist_deg} 度")
    print(f"  最大階數: {max_degree}")
    print(f"  觀測噪聲標準差: {noise_std} mGal")
    print(f"  觀測高度 (Input): {obs_height:.1f} m")  # [顯示]
    print(f"  預測高度 (Target): {pred_height:.1f} m") # [顯示]

    if target == 'gravity':
        print("  執行模式: 重力異常推估")
    elif target == 'geoid':
        print("  執行模式: 大地起伏推估")
    else:
        raise ValueError("目標類型必須是 'gravity' 或 'geoid'")
    
    predictions = []
    pred_stds = []
    all_nearby_dfs = []
    
    for idx, row in pred_df.iterrows():
        print(f"\n處理預測點 {idx+1}/{len(pred_df)}:")
        pred_lat = row['lat']
        pred_lon = row['lon']
        
        # 搜尋鄰近觀測點並執行預測
        predicted, pred_std, nearby_df = perform_lsc_prediction_for_point(
            obs_df, pred_lat, pred_lon, A, R_B, B, max_degree=max_degree,
            max_dist_deg=max_dist_deg,
            target=target,
            noise_std=noise_std,
            obs_height=obs_height,   # [傳入]
            pred_height=pred_height,  # [傳入]
        )
        
        predictions.append(predicted)
        pred_stds.append(pred_std)
        all_nearby_dfs.append(nearby_df)
    
    # ==========================================
    # 步驟 7: 預測結果
    # ==========================================
    print("\n" + "="*70)
    print("步驟 7: 預測結果")
    print("="*70)
    
    pred_df['predicted_value'] = predictions
    pred_df['prediction_std'] = pred_stds
    pred_df['prediction_error'] = pred_stds
    
    print("\n預測結果:")
    print(pred_df.to_string(index=False))
    
    # 統計資訊
    print(f"\n統計資訊:")
    print(f"  預測點數量: {len(pred_df)}")
    print(f"  平均預測值: {np.nanmean(predictions):.4f} mGal")
    print(f"  平均預測誤差: {np.nanmean(pred_stds):.4f} mGal")
    for idx, nearby_df in enumerate(all_nearby_dfs):
        print(f"  預測點 {idx+1} 使用的觀測點數量: {len(nearby_df)}")
    
    # ==========================================
    # 步驟 8: 視覺化結果
    # ==========================================
    print("\n" + "="*70)
    print("步驟 8: 視覺化結果")
    print("="*70)
    
    # ==========================================
    # 步驟 8: 視覺化結果 (Using external module)
    # ==========================================
    print("\n" + "="*70)
    print("步驟 8: 視覺化結果")
    print("="*70)
    
    lscp.plot_prediction_results(
        obs_df, pred_df, all_nearby_dfs, max_dist_deg, target=target, 
        output_file='lsc_prediction_results.png'
    )
    
    # ==========================================
    # 保存結果到 CSV
    # ==========================================
    print("\n" + "="*70)
    print("保存結果到 CSV 檔案...")
    print("="*70)
    
    # 保存預測點資料
    pred_df.to_csv('prediction_points.csv', index=False)
    print("預測點資料已保存為 'prediction_points.csv'")
    
    # 保存使用的觀測點資料（合併所有鄰近觀測點）
    all_obs_used = pd.concat(all_nearby_dfs, ignore_index=True).drop_duplicates()
    all_obs_used.to_csv('observation_points.csv', index=False)
    print(f"使用的觀測點資料已保存為 'observation_points.csv' ({len(all_obs_used)} 個點)")
    
    print("\n" + "="*70)
    print("LSC 預測完成！")
    print("="*70)

