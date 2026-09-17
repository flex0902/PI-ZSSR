import matplotlib.pyplot as plt
import numpy as np
import os
import platform

# ==========================================
# 1. Font Setup
# ==========================================
def setup_chinese_font():
    """Sets up Chinese font support for matplotlib."""
    system = platform.system()
    
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_candidates = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
    else:  # Linux
        font_candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    plt.rcParams['font.sans-serif'] = font_candidates
    plt.rcParams['axes.unicode_minus'] = False
    # print(f"Font setup: {font_candidates[0]}")

# Initialize font on module load
setup_chinese_font()

# ==========================================
# 2. Coastline Plotting
# ==========================================
def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1):
    """
    Plots the coastline from a GMT-extracted text file onto the given axes.
    """
    if not os.path.exists(coast_file):
        return

    segment_x = []
    segment_y = []
    
    # Check file size to ensure it's not empty
    if os.path.getsize(coast_file) < 10:
        return

    try:
        with open(coast_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if segment_x:
                        ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
                        segment_x, segment_y = [], []
                elif line.startswith('#'):
                    continue
                else:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            val_x = float(parts[0])
                            val_y = float(parts[1])
                            segment_x.append(val_x)
                            segment_y.append(val_y)
                        except ValueError:
                            pass
            if segment_x:
                ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
    except Exception as e:
        print(f"Warning: Failed to plot coastline: {e}")

# ==========================================
# 3. Residual Difference Plot
# ==========================================
def plot_residual_difference(df_obs, coast_file=None, output_file=None, title=None, value_column='diff', marker='o', s=1):
    """
    Plots the residual difference (dg - grid_value) scattered on a map.
    marker: 'o' for circle, 's' for square.
    s: size of the marker.
    """
    plt.figure(figsize=(10, 8))
    # Check if column exists
    if value_column not in df_obs.columns:
        print(f"Warning: Column '{value_column}' not found in DataFrame. Skipping plot.")
        return
    c_max = np.abs(df_obs[value_column]).max()
    im = plt.scatter(df_obs['lon'], df_obs['lat'], c=df_obs[value_column], cmap='RdYlBu_r', vmin= -c_max, vmax= c_max,
        s=s, marker=marker)
    
    if coast_file and os.path.exists(coast_file):
        plot_coastline(plt.gca(), coast_file)
        
    plt.colorbar(im, label='[mGal]')
    plt.title(f"{title}\nRange: {df_obs[value_column].min():.2f}~{df_obs[value_column].max():.2f} (mGal)")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Residual plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

# ==========================================
# 4. Covariance Fit Plot (Single B)
# ==========================================
def plot_covariance_fit(cov_df, fitted_curve, flight_height, A, B, output_file=None):
    """
    Plots the empirical covariance vs the fitted T-R model curve.
    """
    plt.figure(figsize=(10, 6))
    data_for_fit = cov_df.dropna()
    
    plt.plot(cov_df['psi_deg'], cov_df['cov'], 'ko', label='Empirical Data', alpha=0.6)
    if fitted_curve is not None:
        plt.plot(data_for_fit['psi_deg'], fitted_curve, 'r-', linewidth=2.5, label='T-R Model Fit')

    plt.xlabel('Spherical Distance (degrees)')
    plt.ylabel('Covariance ($mGal^2$)')
    plt.title(f'LSC Covariance Fit\nH={flight_height}m, A={A:.1f}, B={B}')
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Covariance fit plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

# ==========================================
# 5. Optimized B Fit Plot
# ==========================================
def plot_optimized_fit_and_search(cov_df, best_curve, history, best_B, A, depth, flight_height, fitted_curve=None, output_file=None):
    """
    Plots 2 subplots:
    1. Empirical Covariance vs Best Fit Curve
    2. Optimization Landscape (SSR vs B)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Plot: The Covariance Fit
    ax1.plot(cov_df['psi_deg'], cov_df['cov'], 'ko', label='Empirical Data', alpha=0.6)
    if best_curve is not None:
        # Usually fitted on dropna() data
        data_for_fit = cov_df.dropna()
        ax1.plot(data_for_fit['psi_deg'], best_curve, 'g-', linewidth=2.5, label=f'Best Fit (B={best_B})')
    if fitted_curve is not None:
        ax1.plot(data_for_fit['psi_deg'], fitted_curve, 'r-', linewidth=2.5, label='T-R Model Fit (B=24)')
    ax1.set_xlabel('Spherical Distance (degrees)')
    ax1.set_ylabel('Covariance ($mGal^2$)')
    ax1.set_title(f'Optimized LSC Covariance Fit\nH={flight_height}m, A={A:.1f}, B={best_B}\nDepth={depth/1000:.2f} km')
    ax1.legend()
    ax1.grid(True)
    
    # Right Plot: The Search History (Error vs B)
    if history:
        history_B, history_SSR = history
        ax2.plot(history_B, history_SSR, 'b.-')
        # Highlight the winner
        if best_B is not None:
            try:
                min_ssr = min(history_SSR)
                min_ssr_24 = history_SSR[history_B.index(24)]
                ax2.plot(best_B, min_ssr, 'ro', markersize=10, label=f'Best Fit (B={best_B})')
                ax2.plot(24, min_ssr_24, 'ko', markersize=10, label='T-R Model (B=24)')
            except ValueError:
                pass

    ax2.set_title('Optimization: Error vs Parameter B')
    ax2.set_xlabel('Parameter B (Integer)')
    ax2.set_ylabel('Sum of Squared Residuals (SSR)')
    #ax2.axvline(x=24, color='k', linestyle='--', label='B=24 (standard)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Optimized fit plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

# ==========================================
# 6. Prediction Results Plot
# ==========================================
def plot_prediction_results(obs_df, pred_df, all_nearby_dfs, max_dist_deg, target='gravity', output_file='lsc_prediction_results.png'):
    """
    Plots prediction results:
    Left: Map of observations and prediction points.
    Right: Prediction values with error bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left Plot: Map
    ax1 = axes[0]
    
    # Plot all observations (light, small)
    ax1.scatter(obs_df['lon'], obs_df['lat'], c='lightgray', s=5, 
                alpha=0.3, label='All Observations', zorder=1)
    
    # Plot for each prediction point
    colors = plt.cm.tab10(np.linspace(0, 1, len(pred_df)))
    for idx, (_, pred_row) in enumerate(pred_df.iterrows()):
        nearby_df = all_nearby_dfs[idx]
        if len(nearby_df) > 0:
            # Nearby obs
            ax1.scatter(nearby_df['lon'], nearby_df['lat'], 
                       c=[colors[idx]], s=50, alpha=0.6, 
                       edgecolors='black', linewidth=0.5,
                       label=f'Obs for P{idx+1}', zorder=1)
        
        # Prediction point
        ax1.scatter(pred_row['lon'], pred_row['lat'], 
                   c=[colors[idx]], s=50, marker='s', 
                   edgecolors='black', linewidth=2,
                   label=f'Pred P{idx+1}', zorder=2)
        
        # Annotation
        val = pred_row.get('predicted_value', np.nan)
        std = pred_row.get('prediction_std', np.nan)
        ax1.annotate(f'P{idx+1}\n{val:.1f}±{std:.1f}', 
                    (pred_row['lon'], pred_row['lat']), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Longitude (deg)', fontsize=12)
    ax1.set_ylabel('Latitude (deg)', fontsize=12)
    ax1.set_title(f'LSC Prediction: Obs & Pred Points\n(Search Radius: {max_dist_deg} deg)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    # ax1.set_aspect('equal', adjustable='box') # Can cause issues with tight_layout depending on data range
    
    # Right Plot: Values & Errors
    ax2 = axes[1]
    predictions = pred_df['predicted_value'].values
    pred_stds = pred_df['prediction_std'].values
    x_pos = np.arange(len(predictions))
    valid_mask = ~np.isnan(predictions)
    
    if np.any(valid_mask):
        ax2.errorbar(x_pos[valid_mask], predictions[valid_mask], 
                    yerr=pred_stds[valid_mask],
                    fmt='o', markersize=12, capsize=10, capthick=2,
                    elinewidth=2, color='steelblue', 
                    markerfacecolor='steelblue', markeredgecolor='black',
                    markeredgewidth=1.5, alpha=0.8, label='Pred ± 1σ')
        
        # Val labels
        for i, (val, err) in enumerate(zip(predictions, pred_stds)):
            if not np.isnan(val):
                ax2.text(i + 0.02, val, f'{val:.2f}', 
                        ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Prediction Point ID', fontsize=12)
    if target == 'gravity':
        ax2.set_ylabel('Predicted Value (mGal)', fontsize=12)
    elif target == 'geoid':
        ax2.set_ylabel('Predicted Value (m)', fontsize=12)

    ax2.set_title('LSC Predictions & Uncertainties', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'P{i+1}' for i in range(len(predictions))])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {output_file}")
    else:
        plt.show()
    plt.close()
