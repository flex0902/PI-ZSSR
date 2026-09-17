import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from torch.utils.data import Dataset, DataLoader
from plot_utils import plot_cross_sections
import time
import json

# ==========================================
# 1. Config & Tools
# ==========================================
class EarlyStopper:
    def __init__(self, patience=200, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def detect_file_encoding(file_path):
    encodings = ['utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: f.read(1000)
            return enc
        except: continue
    return 'utf-8'

# Additional imports for Coastline
import subprocess
import os

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

def plot_coastline(ax, coast_file="coastline.txt", color='black', linewidth=1):
    if not os.path.exists(coast_file):
        return

    segment_x = []
    segment_y = []
    seg_count = 0
    
    with open(coast_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if segment_x:
                    ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
                    seg_count += 1
                    segment_x, segment_y = [], []
            elif line.startswith('#'):
                continue
            else:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        val_x = float(parts[0])
                        val_y = float(parts[1])
                        # Simple bounds check debug (disabled)
                        segment_x.append(val_x)
                        segment_y.append(val_y)
                    except ValueError:
                        pass
        if segment_x:
            ax.plot(segment_x, segment_y, color=color, linewidth=linewidth)
            seg_count += 1
    
    # Debug info
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # print(f"Plot Coastline: Drawn {seg_count} segments. Ax limits: X={xlim}, Y={ylim}")

# ==========================================
# 2. Physics Layers
# ==========================================
class UpwardContinuationLayer(nn.Module):
    def __init__(self, dx, dy, height_diff_km):
        super(UpwardContinuationLayer, self).__init__()
        self.dx = dx
        self.dy = dy
        self.h = height_diff_km
        
    def forward(self, x):
        """ FFT-based Upward Continuation """
        B, C, H, W = x.shape
        fft_x = torch.fft.fft2(x)
        
        kx = torch.fft.fftfreq(W, d=self.dx).to(x.device)
        ky = torch.fft.fftfreq(H, d=self.dy).to(x.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        
        K_rad = 2 * np.pi * torch.sqrt(KX**2 + KY**2)
        continuation_filter = torch.exp(-self.h * K_rad).unsqueeze(0).unsqueeze(0)
        
        filtered_fft = fft_x * continuation_filter
        out = torch.fft.ifft2(filtered_fft)
        return out.real

# ==========================================
# 3. Architecture (Same U-Net)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        if x.shape != x2.shape: x = torch.nn.functional.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        if x.shape != x1.shape: x = torch.nn.functional.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        return self.outc(x)

# ==========================================
# 4. Data Loading
# ==========================================
class GravityGridDataset(Dataset):
    def __init__(self, file_5000, file_2000, file_0000=None, size=(301, 301)):
        self.file_5k = file_5000
        self.file_2k = file_2000
        self.file_0k = file_0000
        self.size = size
        self.has_target = (file_0000 is not None and str(file_0000).lower() != 'none')
        
        self.img_5k, self.img_2k, self.img_0k, self.extent = self.load_and_grid()
        
        self.dx = (self.extent[1] - self.extent[0]) / size[1]
        self.dy = (self.extent[3] - self.extent[2]) / size[0]
        self.dx_km = self.dx * 102.0
        self.dy_km = self.dy * 111.0
        print(f"Grid: {size}, dx={self.dx_km:.3f}km, dy={self.dy_km:.3f}km")

    def load_and_grid(self):
        print(f"Loading data: {self.file_5k}, {self.file_2k}")
        if self.has_target: print(f"Target: {self.file_0k}")
        else: print("Target: None (Blind Mode)")

        df_5k = pd.read_csv(self.file_5k, sep='\s+', header=None, names=['lon','lat','dg'], encoding=detect_file_encoding(self.file_5k), comment='#')
        df_2k = pd.read_csv(self.file_2k, sep='\s+', header=None, names=['lon','lat','dg'], encoding=detect_file_encoding(self.file_2k), comment='#')
        
        lon_grid = np.linspace(df_5k['lon'].min(), df_5k['lon'].max(), self.size[1])
        lat_grid = np.linspace(df_5k['lat'].min(), df_5k['lat'].max(), self.size[0])
        LO, LA = np.meshgrid(lon_grid, lat_grid)
        
        img_5k = LinearNDInterpolator(list(zip(df_5k['lon'], df_5k['lat'])), df_5k['dg'])(LO, LA)
        img_2k = LinearNDInterpolator(list(zip(df_2k['lon'], df_2k['lat'])), df_2k['dg'])(LO, LA)
        
        if self.has_target:
            df_0k = pd.read_csv(self.file_0k, sep='\s+', header=None, names=['lon','lat','dg'], encoding=detect_file_encoding(self.file_0k), comment='#')
            img_0k = LinearNDInterpolator(list(zip(df_0k['lon'], df_0k['lat'])), df_0k['dg'])(LO, LA)
            img_0k = np.nan_to_num(img_0k, nan=0.0)
        else:
            img_0k = None
        
        img_5k = np.nan_to_num(img_5k, nan=0.0)
        img_2k = np.nan_to_num(img_2k, nan=0.0)
        
        return img_5k, img_2k, img_0k, (lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max())

    def __len__(self): return 1
    def __getitem__(self, idx):
        # Return tensors
        t_5 = torch.tensor(self.img_5k, dtype=torch.float32).unsqueeze(0)
        t_2 = torch.tensor(self.img_2k, dtype=torch.float32).unsqueeze(0)
        
        if self.has_target:
            t_0 = torch.tensor(self.img_0k, dtype=torch.float32).unsqueeze(0)
        else:
            t_0 = torch.zeros_like(t_5)
            
        return t_5, t_2, t_0

# ==========================================
# 5. Main Solver
# ==========================================
def deploy_0m_solver(file_high="TW_EGM_grid_g_res_h5000.xyz", file_low="TW_EGM_grid_g_res_h1500.xyz",
                     file_0k=None, obs_high=5000, obs_low=1500, 
                     grid_size=301, epochs=5000, output_prefix="deploy_pisr", lag=300):
    # FILES
    # file_5k = "TW_EGM_grid_g_res_h5000.xyz" 
    # file_2k = "TW_EGM_grid_g_res_h1500.xyz"
    # For validation
    # file_0k = "TW_EGM_grid_g_res_h0.xyz"
    
    dataset = GravityGridDataset(file_high, file_low, file_0k, size=(grid_size, grid_size))
    
    # PHYSICS LAYERS
    # 0m -> 2000m (Diff = 2.0km)
    up_layer_2k = UpwardContinuationLayer(dataset.dx_km, dataset.dy_km, height_diff_km=obs_low/1000).to(device)
    # 0m -> 5000m (Diff = 5.0km)
    up_layer_5k = UpwardContinuationLayer(dataset.dx_km, dataset.dy_km, height_diff_km=obs_high/1000).to(device)
    
    # MODEL
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002) # Higher LR for optimization
    
    # INPUT (GUIDANCE)
    # We encourage the model to output something structurally similar to 2000m initially
    # but we DO NOT enforce pixel loss on it.
    input_guidance = torch.tensor(dataset.img_2k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # TARGETS
    target_5k = torch.tensor(dataset.img_5k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    target_2k = torch.tensor(dataset.img_2k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    print("\n--- Start 0m Deployment (Dual Constraint) ---")
    print("Optimization Target: Find G_0m such that:")
    print(f"1. Upward(G_0m, {obs_low}km) == Obs_{obs_low}")
    print(f"2. Upward(G_0m, {obs_high}km) == Obs_{obs_high}")
    
    bbox = dataset.extent # (lon_min, lon_max, lat_min, lat_max)
    
    # History for CSV
    loss_history = []
    
    # Early Stopper
    early_stopper = EarlyStopper(patience=lag, min_delta=1e-5)

    #epochs = 5000
    
    history = []

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Predict 0m
        pred_0m = model(input_guidance) # Input is just a seed/guidance
        
        # 2. Upward Checks
        check_2k = up_layer_2k(pred_0m)
        check_5k = up_layer_5k(pred_0m)
        
        # 3. Dual Physics Loss
        loss_2k = nn.MSELoss()(check_2k, target_2k)
        loss_5k = nn.MSELoss()(check_5k, target_5k)
        
        # 4. Regularization (Total Variation to prevent high-freq noise explosion)
        # 0m is very sensitive, we need slight smoothness constraint
        # spatial gradients
        diff_h = torch.abs(pred_0m[:,:,:,1:] - pred_0m[:,:,:,:-1])
        diff_v = torch.abs(pred_0m[:,:,1:,:] - pred_0m[:,:,:-1,:])
        loss_tv = (torch.mean(diff_h) + torch.mean(diff_v))
        
        # Total Loss
        total_loss = loss_2k + loss_5k + 0.05 * loss_tv
        
        total_loss.backward()
        optimizer.step()
        
        history.append(total_loss.item())

        # Calculate Stability Score (Lag 300)
        stability_score = 0.0
        #lag = 300
        if epoch >= lag:
            prev_loss = loss_history[epoch - lag][1] # Index 1 is Total_Loss
            current_loss_val = total_loss.item()
            if current_loss_val != 0:
                change_ratio = abs(current_loss_val - prev_loss) / current_loss_val
                stability_score = 1.0 - change_ratio
        
        # Log History
        loss_history.append([epoch, total_loss.item(), loss_2k.item(), loss_5k.item(), stability_score])
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={total_loss.item():.5f} | L_2k={loss_2k.item():.5f} | L_5k={loss_5k.item():.5f}, Stability={stability_score:.4f}")
        
        # Check Early Stopping
        early_stopper(total_loss.item())
        if early_stopper.early_stop:
            print(f"Early stop at epoch {epoch}")
            break
        
    # ==========================
    # Verification Plot
    # ==========================
    model.eval()
    with torch.no_grad():
        final_0m = model(input_guidance).squeeze().cpu().numpy()
        final_check_2k = up_layer_2k(model(input_guidance)).squeeze().cpu().numpy()
        final_check_5k = up_layer_5k(model(input_guidance)).squeeze().cpu().numpy()
        
        # Metrics (For Post-Hoc Analysis)
        # CROP MARGINS to avoid FFT edge effects
        margin = 20    
        
        # Sliced versions for statistics
        final_0m_center = final_0m[margin:-margin, margin:-margin]
        final_check_2k_center = final_check_2k[margin:-margin, margin:-margin]
        final_check_5k_center = final_check_5k[margin:-margin, margin:-margin]
        target_2k_center = dataset.img_2k[margin:-margin, margin:-margin]
        target_5k_center = dataset.img_5k[margin:-margin, margin:-margin]
        
        # Calculate RMS for the central region
        rms_2k = np.sqrt(np.mean((final_check_2k_center - target_2k_center)**2))
        rms_5k = np.sqrt(np.mean((final_check_5k_center - target_5k_center)**2))
        
        if dataset.has_target:
            dataset.img_0k_center = dataset.img_0k[margin:-margin, margin:-margin]
            rms_0m = np.sqrt(np.mean((final_0m_center - dataset.img_0k_center)**2))
        else:
            rms_0m = 0.0
        
        # Calculate Relative Max Error for the central region
        rel_error_2k = np.max(np.abs(final_check_2k_center - target_2k_center)) / np.max(np.abs(target_2k_center))
        rel_error_5k = np.max(np.abs(final_check_5k_center - target_5k_center)) / np.max(np.abs(target_5k_center))
        
        if dataset.has_target:
            rel_error_0m = np.max(np.abs(final_0m_center - dataset.img_0k_center)) / np.max(np.abs(dataset.img_0k_center))
        else:
            rel_error_0m = 0.0
        
        # Print metrics
        print(f"\n--- Metrics for Central Region ---")
        print(f"RMS (1500m): {rms_2k:.5f}")
        print(f"RMS (5000m): {rms_5k:.5f}")
        if dataset.has_target: print(f"RMS (0m): {rms_0m:.5f}")
        else: print(f"RMS (0m): N/A (Blind)")
        
        print(f"Relative Max Error (1500m): {rel_error_2k:.5f}")
        print(f"Relative Max Error (5000m): {rel_error_5k:.5f}")
        if dataset.has_target: print(f"Relative Max Error (0m): {rel_error_0m:.5f}")
        
        # Plotting Setup (3 rows, 3 columns)
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Physical Extent
        extent = dataset.extent

        # --- EXTRACT COASTLINE ---
        coast_file = "coastline.txt"
        has_coast = extract_coastline_gmt(extent, coast_file)

        # Calculate physical coordinates for the crop box
        rect_x = extent[0] + margin * dataset.dx
        rect_y = extent[2] + margin * dataset.dy
        rect_w = (dataset.size[1] - 2*margin) * dataset.dx
        rect_h = (dataset.size[0] - 2*margin) * dataset.dy        

        # --- Row 1: The Result (0m) ---
        # Fixed Scale 0k
        vmin_0k = final_0m.min()
        vmax_0k = final_0m.max()
        limit_0k = np.max([np.abs(vmin_0k), np.abs(vmax_0k)])
        # 8. Obs 0m
        if dataset.has_target:
            im7 = axes[0, 0].imshow(dataset.img_0k, origin='lower', cmap='RdYlBu_r', vmin=-limit_0k, vmax=limit_0k, extent=extent)
            axes[0, 0].set_title(f'Truth (0m)\nRange: {dataset.img_0k.min():.1f} ~ {dataset.img_0k.max():.1f}')
            plt.colorbar(im7, ax=axes[0, 0], fraction=0.046, pad=0.04)
            axes[0, 0].set_xlabel('Longitude')
            axes[0, 0].set_ylabel('Latitude')
            if has_coast: plot_coastline(axes[0, 0], coast_file)
        else:
            axes[0, 0].axis('off')
            
        # 1. Prediction 0m
        im0 = axes[0, 1].imshow(final_0m, origin='lower', cmap='RdYlBu_r', vmin=-limit_0k, vmax=limit_0k, extent=extent)
        axes[0, 1].set_title(f'Pred (0m)\nRange: {final_0m.min():.1f} ~ {final_0m.max():.1f}')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[0, 1], coast_file)
        plt.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 9. Residual 0m
        if dataset.has_target:
            diff_0m = dataset.img_0k - final_0m
            diff_0m_center = dataset.img_0k_center - final_0m_center
            limit_diff_0m = np.max(np.abs(diff_0m_center))
            im9 = axes[0, 2].imshow(diff_0m, origin='lower', cmap='RdBu_r', vmin=-limit_diff_0m, vmax=limit_diff_0m, extent=extent)
            axes[0, 2].set_title(f'Error 0m (Truth - Pred)\nRange: {diff_0m_center.min():.2f}~{diff_0m_center.max():.2f} | RMSE: {rms_0m:.2f}')
            plt.colorbar(im9, ax=axes[0, 2], fraction=0.046, pad=0.04)
            axes[0, 2].set_xlabel('Longitude')
            axes[0, 2].set_ylabel('Latitude')
            if has_coast: plot_coastline(axes[0, 2], coast_file)

            # Add visual box for statistic region
            rect1 = matplotlib.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                             linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
            axes[0, 2].add_patch(rect1)
        else:
            axes[0, 2].axis('off')
            
        # --- Row 2: 2000m Level ---
        # Fixed Scale 2k
        vmin_2k = min(dataset.img_2k.min(), final_check_2k.min())
        vmax_2k = max(dataset.img_2k.max(), final_check_2k.max())
        limit_2k = np.max([np.abs(vmin_2k), np.abs(vmax_2k)])
        # 3. Obs 2k
        im2 = axes[1, 0].imshow(dataset.img_2k, origin='lower', cmap='RdYlBu_r', vmin=-limit_2k, vmax=limit_2k, extent=extent)
        axes[1, 0].set_title(f'Input {obs_low}m\nRange: {dataset.img_2k.min():.1f} ~ {dataset.img_2k.max():.1f}')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[1, 0], coast_file)
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 2. Check 2k
        im1 = axes[1, 1].imshow(final_check_2k, origin='lower', cmap='RdYlBu_r', vmin=-limit_2k, vmax=limit_2k, extent=extent)
        axes[1, 1].set_title(f'Upward (Pred -> {obs_low}m)\nRange: {final_check_2k.min():.1f} ~ {final_check_2k.max():.1f}')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[1, 1], coast_file)
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 4. Residual 2k
        diff_2k = dataset.img_2k - final_check_2k
        diff_2k_center = target_2k_center - final_check_2k_center
        limit_diff_2k = np.max(np.abs(diff_2k_center))
        im3 = axes[1, 2].imshow(diff_2k, origin='lower', cmap='RdBu_r', vmin=-limit_diff_2k, vmax=limit_diff_2k, extent=extent)
        axes[1, 2].set_title(f'Upward Error {obs_low}m (Truth - Check)\nRange: {diff_2k_center.min():.2f}~{diff_2k_center.max():.2f} | RMSE: {rms_2k:.2f}')
        axes[1, 2].set_xlabel('Longitude')
        axes[1, 2].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[1, 2], coast_file)
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Add visual box
        rect2 = matplotlib.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                             linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        axes[1, 2].add_patch(rect2)

        # --- Row 3: 5000m Level ---
        # Fixed Scale 5k
        vmin_5k = min(dataset.img_5k.min(), final_check_5k.min())
        vmax_5k = max(dataset.img_5k.max(), final_check_5k.max())
        limit_5k = np.max([np.abs(vmin_5k), np.abs(vmax_5k)])
        # 6. Obs 5k
        im5 = axes[2, 0].imshow(dataset.img_5k, origin='lower', cmap='RdYlBu_r', vmin=-limit_5k, vmax=limit_5k, extent=extent)
        axes[2, 0].set_title(f'Input {obs_high}m\nRange: {dataset.img_5k.min():.1f} ~ {dataset.img_5k.max():.1f}')
        axes[2, 0].set_xlabel('Longitude')
        axes[2, 0].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[2, 0], coast_file)
        plt.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # 5. Check 5k
        im4 = axes[2, 1].imshow(final_check_5k, origin='lower', cmap='RdYlBu_r', vmin=-limit_5k, vmax=limit_5k, extent=extent)
        axes[2, 1].set_title(f'Upward (Pred -> {obs_high}m)\nRange: {final_check_5k.min():.1f} ~ {final_check_5k.max():.1f}')
        axes[2, 1].set_xlabel('Longitude')
        axes[2, 1].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[2, 1], coast_file)
        plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, pad=0.04)
                
        # 7. Residual 5k
        diff_5k = dataset.img_5k - final_check_5k
        diff_5k_center = target_5k_center - final_check_5k_center
        limit_diff_5k = np.max(np.abs(diff_5k_center))
        im6 = axes[2, 2].imshow(diff_5k, origin='lower', cmap='RdBu_r', vmin=-limit_diff_5k, vmax=limit_diff_5k, extent=extent)
        axes[2, 2].set_title(f'Upward Error {obs_high}m (Truth - Check)\nRange: {diff_5k_center.min():.2f}~{diff_5k_center.max():.2f} | RMSE: {rms_5k:.2f}')
        axes[2, 2].set_xlabel('Longitude')
        axes[2, 2].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[2, 2], coast_file)
        plt.colorbar(im6, ax=axes[2, 2], fraction=0.046, pad=0.04)

        # Add visual box
        rect3 = matplotlib.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                             linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        axes[2, 2].add_patch(rect3)

        #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_result.png')
        print(f"Done. Saved to {output_prefix}_result.png")
        plt.close()

        # Cross-sections
        print("Plotting Cross Sections...")
        # Reconstruct meshgrid for plotting
        rows, cols = dataset.size
        lon_grid = np.linspace(extent[0], extent[1], cols)
        lat_grid = np.linspace(extent[2], extent[3], rows)
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        plot_cross_sections([dataset.img_5k, final_check_5k, dataset.img_2k, final_check_2k, final_0m], 
                           lon_2d, lat_2d, save_path=f"{output_prefix}_section.png",
                           labels=[f'Obs {obs_high}m', f'Upward {obs_high}m', f'Obs {obs_low}m', f'Upward {obs_low}m', f'Pred 0m'])
        print(f"Done. Saved to {output_prefix}_section.png")

        # ==========================
        # Save Data Products
        # ==========================
        
        # 1. Save Model Weights
        torch.save(model.state_dict(), f'{output_prefix}_model.pth')
        print(f"Saved model weights to {output_prefix}_model.pth")
        
        # 2. Save Loss History
        df_loss = pd.DataFrame(loss_history, columns=['Epoch', 'Total_Loss', 'Loss_2k', 'Loss_5k', 'Stability_Score'])
        df_loss.to_csv(f'{output_prefix}_loss.csv', index=False)
        print(f"Saved loss history to {output_prefix}_loss.csv")
        
        # 3. Save XYZ Grid (Final Verification)
        # Reconstruct coordinates for export
        rows, cols = final_0m.shape
        lon_grid = np.linspace(bbox[0], bbox[1], cols)
        lat_grid = np.linspace(bbox[2], bbox[3], rows)
        XX, YY = np.meshgrid(lon_grid, lat_grid) # Note: Meshgrid order default is xy
        
        # Flatten and save
        out_data = np.column_stack((XX.flatten(), YY.flatten(), final_0m.flatten()))
        np.savetxt(f'{output_prefix}_0m.xyz', out_data, fmt='%.6f', delimiter=' ', header='lon lat gravity_0m')
        print(f"Saved predicted grid to {output_prefix}_0m.xyz")
        
        # 4. Save Metadata (JSON)
        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time} seconds")
        metadata = {
            "Description": "This model is trained on 5000m and 1500m gravity data and predicts 0m gravity data.",
            "Model_type": "PI-BSR",
            "Model_path": f'{output_prefix}_model.pth',
            "Loss_history_path": f'{output_prefix}_loss.csv',
            "XYZ_path": f'{output_prefix}_0m.xyz',
            "Metadata_path": f'{output_prefix}_metadata.json',
            "Training_epochs": int(epoch),
            "Execution_time_sec": float(elapsed_time),
            "Grid_size": int(grid_size),
            "Pred_0m_rms_mgal": float(rms_0m),
            "Upward_1500_rms_mgal": float(rms_2k),
            "Upward_5000_rms_mgal": float(rms_5k),
            "Relative_Max_Error_0m": float(rel_error_0m),
            "Relative_Max_Error_1500": float(rel_error_2k),
            "Relative_Max_Error_5000": float(rel_error_5k)
        }
        with open(f'{output_prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {output_prefix}_metadata.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=301, help="Grid size (e.g. 301, 601)")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lag", type=int, default=300, help="Lag value")
    parser.add_argument("--prefix", type=str, default="deploy_pisr_", help="Prefix for output files")
    parser.add_argument("--file_high", type=str, default="TW_EGM_grid_g_res_h5000.xyz", help="High-altitude data file")
    parser.add_argument("--file_low", type=str, default="TW_EGM_grid_g_res_h1500.xyz", help="Low-altitude data file")
    parser.add_argument("--file_0k", type=str, default=None, help="ground truth data file (Optional)")
    parser.add_argument("--obs_high", type=float, default=5000, help="Observation height of high-altitude data in m")
    parser.add_argument("--obs_low", type=float, default=1500, help="Observation height of low-altitude data in m")
    #parser.add_argument("--prefilter", action='store_true', help="Apply prefiltering")
    #parser.add_argument("--noise", type=float, default=0.0, help="Noise level in mGal (sigma)")
    args = parser.parse_args()
    prefix = args.prefix+str(args.size)
    #file_0k = "TW_EGM_grid_g_res_h0.xyz"

    try:
        deploy_0m_solver(file_high=args.file_high, file_low=args.file_low, file_0k=args.file_0k, obs_high=args.obs_high, obs_low=args.obs_low, grid_size=args.size, epochs=args.epochs, output_prefix=prefix, lag=args.lag)
    except Exception as e:
        import traceback
        traceback.print_exc()
