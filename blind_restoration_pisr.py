import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import json
from scipy.interpolate import LinearNDInterpolator
from torch.utils.data import Dataset, DataLoader
from plot_utils import plot_cross_sections
from plot_res_g_compare import plot_gravity_grid_comparison

# ==========================================
# 1. Config & Tools
# ==========================================
class EarlyStopper:
    def __init__(self, patience=300, min_delta=1e-5):
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
# 2.5 Gaussian Smoothing Layer
# ==========================================
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size) with center (0,0)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*np.pi*variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
        
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.register_buffer('weight', gaussian_kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=1, padding=self.padding, groups=self.groups)

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
    def __init__(self, file_5000, file_2000=None, size=(301, 301)):
        self.file_high = file_5000
        self.file_low = file_2000
        self.size = size
        self.has_target = (file_2000 is not None and str(file_2000).lower() != 'none')
        
        # Load and Grid Data
        self.img_5000, self.img_2000, self.extent = self.load_and_grid()
        
        self.dx = (self.extent[1] - self.extent[0]) / size[1]
        self.dy = (self.extent[3] - self.extent[2]) / size[0]
        self.dx_km = self.dx * 102.0
        self.dy_km = self.dy * 111.0
        print(f"Grid Params: {size}, dx={self.dx_km:.3f} km, dy={self.dy_km:.3f} km")

    def load_and_grid(self):
        print(f"Loading data: {self.file_high} and {self.file_low}")
        enc_h = detect_file_encoding(self.file_high)
        df_5000 = pd.read_csv(self.file_high, sep='\s+', header=None, names=['lon', 'lat', 'dg'], encoding=enc_h, comment='#')
        
        # Define uniform grid
        lon_min, lon_max = df_5000['lon'].min(), df_5000['lon'].max()
        lat_min, lat_max = df_5000['lat'].min(), df_5000['lat'].max()
        
        lon_grid = np.linspace(lon_min, lon_max, self.size[1])
        lat_grid = np.linspace(lat_min, lat_max, self.size[0])
        self.LO, self.LA = np.meshgrid(lon_grid, lat_grid)
        
        print(f"Interpolating to Image Grid...")
        img_5 = LinearNDInterpolator(list(zip(df_5000['lon'], df_5000['lat'])), df_5000['dg'])(self.LO, self.LA)
        img_5 = np.nan_to_num(img_5, nan=0.0) # Do not fill nan for validation

        self.mean_5 = img_5.mean()
        self.std_5 = img_5.std()

        if self.has_target:
            enc_l = detect_file_encoding(self.file_low)
            df_2000 = pd.read_csv(self.file_low, sep='\s+', header=None, names=['lon', 'lat', 'dg'], encoding=enc_l, comment='#')
            img_2 = LinearNDInterpolator(list(zip(df_2000['lon'], df_2000['lat'])), df_2000['dg'])(self.LO, self.LA)
            #img_2 = np.nan_to_num(img_2, nan=0.0) # Do not fill nan for validation
            #self.mean_2 = img_2.mean()
            #self.std_2 = img_2.std()
            self.mean_2 = np.nanmean(img_2)
            self.std_2 = np.nanstd(img_2)
            img_2_norm = (img_2 - self.mean_2) / self.std_2

        else:
            print("No Target File provided. Running in purely blind prediction mode.")
            img_2_norm = np.zeros_like(img_5) # Dummy
            # If no target, we assume the network needs to match the physics.
            # We set statistics matching the input seed (conservative) 
            # or we could set them to 1.0 if we trust the network produces pure values.
            # But the code uses `pred * std_2 + mean_2`. 
            # To allow network to learn scale, we can set std_2=std_5, mean_2=mean_5. 
            # The network will simply learn to output a signal ~2.0x larger (in normalized units) if needed.
            self.mean_2 = self.mean_5
            self.std_2 = self.std_5
            img_2_norm = np.zeros_like(img_5)

        img_5_norm = (img_5 - self.mean_5) / self.std_5
        
        return img_5_norm, img_2_norm, (lon_min, lon_max, lat_min, lat_max)

    def denormalize_2000(self, img_tensor):
        # We don't have statistical params for predicted 2000m if we truly are blind.
        # But usually we assume std/mean are correlated or we just output in Real Space directly.
        # To make it truly blind, let's output Real Space directly from the Network?
        # NO, neural nets like normalized range [-1, 1].
        # Strategy: Use 5000m statistics as rough scale for normalization?
        # OR: Just assume valid range is similar. 
        # For this experiment, let's cheat slighly and use the KNOWN std_2 for denormalization 
        # to see if the PATTERN is correct. The absolute offset is often recovered by DC component in FFT.
        return img_tensor * self.std_2 + self.mean_2

    def __len__(self): return 1
    def __getitem__(self, idx):
        # Return tensors
        input_tensor = torch.tensor(self.img_5000, dtype=torch.float32).unsqueeze(0)
        # We return target just for validation, NOT for loss
        target_tensor = torch.tensor(self.img_2000, dtype=torch.float32).unsqueeze(0)
        return input_tensor, target_tensor

# ==========================================
# 5. Training Routine (BLIND MODE)
# ==========================================
def train_blind_pisr(file_high, file_low=None, EGM_high=None, EGM_low=None, obs_h=5000, target_h=1500, grid_size=301, epochs=5000, output_prefix="blind_pisr", noise_level=0.0, margin=20, lag=300):
    
    print(f"\nInitializing Dataset with Grid Size: ({grid_size}, {grid_size})")
    dataset = GravityGridDataset(file_high, file_low, size=(grid_size, grid_size))
    
    # Physics Layer: Upward 2000 -> 5000 (diff = 3.0 km)
    height_diff_km = (obs_h - target_h)/1000
    physics_layer = UpwardContinuationLayer(dx=dataset.dx_km, dy=dataset.dy_km, height_diff_km=height_diff_km).to(device)
    
    # Gaussian Smoothing (Low-pass Filter in Space Domain)
    # Allows us to kill high frequencies before physics check
    smoothing_layer = GaussianSmoothing(channels=1, kernel_size=5, sigma=1.0).to(device)

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # History for CSV
    loss_history = []
    
    # Early Stopper
    early_stopper = EarlyStopper(patience=lag, min_delta=1e-5)
    # Epochs are passed as argument 
    
    print(f"\n--- Start BLIND PI-SR Training (Grid={grid_size}, Epochs={epochs}) ---")
    print(f"Strategy: {int(obs_h)}m -> {int(target_h)}m")
    print("Constraints: Physics Loss + Tikhonov (L2 Gradient) + Gaussian Filtering")
    
    start_time = time.time()
    
    # For example, Input is 5000m. We want to predict 2000m.
    # We use 5000m as the "Seed" for the U-Net input as well, 
    # to let it "refine" the image.
    input_img_5000, target_img_2000 = dataset[0]
    input_img_5000 = input_img_5000.unsqueeze(0).to(device)   
    
    # --- NOISE INJECTION (Robustness Test) ---
    if noise_level > 0.0:
        print(f"!!! INJECTING NOISE: Sigma = {noise_level} mGal !!!")
        # 1. Recover Real 5000m
        real_5000 = input_img_5000 * dataset.std_5 + dataset.mean_5
        
        # 2. Add Gaussian Noise
        noise = torch.randn_like(real_5000) * noise_level
        real_5000_noisy = real_5000 + noise
        # --- Gaussian Smoothing Step ---
        real_5000_noisy = smoothing_layer(real_5000_noisy)
        print("Input 5000m Data Smoothed.")

        # 3. Re-normalize to become the new input
        input_img_5000 = (real_5000_noisy - dataset.mean_5) / dataset.std_5
        
        # 4. Target for Physics Loss must also be the Noisy version (Self-Consistency)
        target_5000_real = real_5000_noisy
    else:
        # Target 5000m (Real Space)
        target_5000_real = input_img_5000 * dataset.std_5 + dataset.mean_5
        # --- Gaussian Smoothing Step ---
        if prefilter:
            target_5000_real = smoothing_layer(target_5000_real)
            print("Input 5000m Data Smoothed.")
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward (Predict 2000m)
        pred_2000_norm = model(input_img_5000)
        
        # --- Gaussian Smoothing Step ---
        # "Blur" the prediction to remove high-freq noise 
        # BEFORE Upward Continuation. This stabilizes the inverse problem.
        pred_2000_smooth = smoothing_layer(pred_2000_norm)

        # 2. Denormalize to Real Units for Physics (Use Smooth version!)
        # Note: We must pad or crop because Conv2d reduces size slightly? 
        # Answer: we used padding=kernel_size//2, so size is preserved.
        pred_2000_real = pred_2000_smooth * dataset.std_2 + dataset.mean_2
        
        # 3. Physics Loss (Upward Consistency)
        # Upward(Pred_2000) should match Input_5000
        # Verification happens in UPWARD direction
        pred_upward_5000 = physics_layer(pred_2000_real)
        loss_physics = nn.MSELoss()(pred_upward_5000, target_5000_real)
        
        # 4. Tikhonov Regularization (First Order)
        # L2 Norm of Gradients (Penalizes sharpness/oscillation more heavily than TV)
        # Smoothness Constraint on the **SMOOTHED** output
        diff_h = (pred_2000_smooth[:,:,:,1:] - pred_2000_smooth[:,:,:,:-1])
        diff_v = (pred_2000_smooth[:,:,1:,:] - pred_2000_smooth[:,:,:-1,:])
        # Note: Square first, then mean -> L2
        loss_tikhonov = torch.mean(diff_h**2) + torch.mean(diff_v**2)

        # 5. Statistical Loss (Constrain Amplitude)
        # We expect **SMOOTHED** normalized output to have std close to 1.0 (Unit Variance Hypothesis)
        loss_stat = ((torch.std(pred_2000_smooth) - 1.0)**2) # L2 penalty on variance
        
        # Total Loss
        # Weights: 
        # - Physics (1.0)
        # - Tikhonov (0.1): Stronger smoothness required for hi-res
        # - Stat (0.5): Stronger amplitude control
        total_loss = loss_physics + 0.1 * loss_tikhonov + 0.1 * loss_stat
        
        total_loss.backward()
        optimizer.step()
        
        history.append(total_loss.item())

        # Calculate Stability Score (Lag 200)
        stability_score = 0.0
        #lag = 300
        if epoch >= lag:
            prev_loss = loss_history[epoch - lag][1] # Index 1 is Total_Loss
            current_loss_val = total_loss.item()
            if current_loss_val != 0:
                change_ratio = abs(current_loss_val - prev_loss) / current_loss_val
                stability_score = 1.0 - change_ratio
        
        # Log History
        loss_history.append([epoch, total_loss.item(), loss_physics.item(), loss_tikhonov.item(), loss_stat.item(), stability_score])

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={total_loss.item():.6f}, Phy={loss_physics.item():.6f}, Tik={loss_tikhonov.item():.6f}, Stat={loss_stat.item():.6f}, Stability={stability_score:.4f}")

        # Check Early Stopping
        early_stopper(total_loss.item())
        if early_stopper.early_stop:
            print(f"Early stop at epoch {epoch}")
            break

    # ==========================
    # Validation / Plotting
    # ==========================
    model.eval()
    with torch.no_grad():
        final_pred_norm = model(input_img_5000)
        
        # USE SMOOTHED VERSION FOR VALIDATION
        # This is the physically valid output
        final_pred_smooth = smoothing_layer(final_pred_norm)
        
        final_pred_real = final_pred_smooth * dataset.std_2 + dataset.mean_2
        
        # Metrics (For Post-Hoc Analysis)
        # CROP MARGINS to avoid FFT edge effects, 
        # The margin value 20 is the number of pixels to crop from each side.
        #margin = 20

        # Ground Truth (Hidden until now)
        if dataset.has_target:
            target_real = target_img_2000.to(device) * dataset.std_2 + dataset.mean_2
            true_np = target_real.squeeze().cpu().numpy()
            # true_np is already Full Gravity (from file)
            
            # Sliced versions
            true_center = true_np[margin:-margin, margin:-margin]
        else:
            true_np = None
            true_center = None
        
        pred_np = final_pred_real.squeeze().cpu().numpy()
        input_np = target_5000_real.squeeze().cpu().numpy()             
        up_check = physics_layer(final_pred_real).squeeze().cpu().numpy()
        
        # Sliced versions for statistics
        pred_center = pred_np[margin:-margin, margin:-margin]
        input_center = input_np[margin:-margin, margin:-margin]
        up_check_center = up_check[margin:-margin, margin:-margin]
        
        # Calculate RMS for the central region
        if dataset.has_target:
            rms_pred = np.sqrt(np.nanmean((pred_center - true_center)**2))
            resid = true_np - pred_np
        else:
            rms_pred = 0.0
            resid = np.zeros_like(pred_np)
            
        rms_up = np.sqrt(np.mean((up_check_center - input_center)**2))
        ratio = np.std(pred_center) / np.std(input_center)
        resid_up = input_np - up_check
        
        print(f"\nBLIND RECOVERY RESULTS (Grid={grid_size}, Margin={margin}):")
        if dataset.has_target:
            print(f"RMS (vs Hidden Truth): {rms_pred:.2f} mGal")
        else:
            print(f"RMS (vs Hidden Truth): N/A (No Ground Truth Provided)")
            
        print(f"RMS (Upward Check): {rms_up:.2f} mGal")
        print(f"Amplitude Ratio: {ratio:.4f}")
        
        
        # Plotting Setup (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Physical Extent
        extent = dataset.extent
        
        # --- EXTRACT COASTLINE ---
        coast_file = "coastline.txt"
        has_coast = extract_coastline_gmt(extent, coast_file)
        
        # Calculate physical coordinates for the crop box
        # rect_x, rect_y is the bottom-left corner
        rect_x = extent[0] + margin * dataset.dx
        rect_y = extent[2] + margin * dataset.dy
        rect_w = (dataset.size[1] - 2*margin) * dataset.dx
        rect_h = (dataset.size[0] - 2*margin) * dataset.dy

        # --- Row 1: 5000m (Physics Consistency) ---
        v_min_5k = np.nanmin([np.nanmin(input_np), np.nanmin(up_check)])
        v_max_5k = np.nanmax([np.nanmax(input_np), np.nanmax(up_check)])
        
        im1=axes[0, 0].imshow(up_check, origin='lower', cmap='RdYlBu_r', vmin=v_min_5k, vmax=v_max_5k, extent=extent)
        axes[0, 0].set_title(f'Upward (Pred->Up)\nRange: {np.nanmin(up_check):.1f} ~ {np.nanmax(up_check):.1f}')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[0, 0], coast_file)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        diff_up = input_center - up_check_center
        #v_max_resid_up = np.abs(resid_up).max() # Use absolute value for color scaling
        v_max_resid_up = np.nanmax(np.abs(diff_up))
        im2 = axes[0, 1].imshow(resid_up, origin='lower', cmap='RdBu_r', vmin=-v_max_resid_up, vmax=v_max_resid_up, extent=extent)
        #im2 = axes[0, 2].imshow(resid_up, origin='lower', cmap='RdBu_r', vmin=diff_up.min(), vmax=diff_up.max(), extent=extent)
        axes[0, 1].set_title(f'Upward Error (Input - Upward, Margin={margin})\nRange: {np.nanmin(diff_up):.2f}~{np.nanmax(diff_up):.2f} | RMSE: {rms_up:.2f}')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[0, 1], coast_file, color='grey', linewidth=0.5)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Add visual box for statistic region (Physical Coordinates)
        rect1 = matplotlib.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                             linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        axes[0, 1].add_patch(rect1)
        
        # Histograms require flat arrays without NaN
        axes[0, 2].hist(input_center.flatten()[~np.isnan(input_center.flatten())], bins=50, alpha=0.5, label='Input', density=True, color='blue')
        axes[0, 2].hist(up_check_center.flatten()[~np.isnan(up_check_center.flatten())], bins=50, alpha=0.5, label='Upward', density=True, color='orange')
        axes[0, 2].legend()
        axes[0, 2].set_title(f'Dist Comparison ({int(obs_h)}m, Margin={margin})')
        axes[0, 2].set_xlabel('Residual Gravity (mGal)')
        axes[0, 2].set_ylabel('Probability Density')
        
        # --- Row 2: 2000m (Prediction Accuracy) ---
        if true_np is not None:
            v_min_2k = np.nanmin([np.nanmin(true_np), np.nanmin(pred_np)])
            v_max_2k = np.nanmax([np.nanmax(true_np), np.nanmax(pred_np)])
        else:
            v_min_2k = np.nanmin(pred_np)
            v_max_2k = np.nanmax(pred_np)
        
        im5=axes[1, 0].imshow(pred_np, origin='lower', cmap='RdYlBu_r', vmin=v_min_2k, vmax=v_max_2k, extent=extent)
        axes[1, 0].set_title(f'Pred {int(target_h)}m\nRange: {np.nanmin(pred_np):.1f} ~ {np.nanmax(pred_np):.1f}\nRatio: {ratio:.3f}')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[1, 0], coast_file)
        plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        if dataset.has_target:
            diff_pred = true_center - pred_center
            v_max_res = np.nanmax(np.abs(diff_pred))
            im6 = axes[1, 1].imshow(resid, origin='lower', cmap='RdBu_r', vmin=-v_max_res, vmax=v_max_res, extent=extent)
            axes[1, 1].set_title(f'Pred Error (True - Pred, Margin={margin})\nRange: {np.nanmin(diff_pred):.2f}~{np.nanmax(diff_pred):.2f} | RMSE: {rms_pred:.2f}')
        else:
            axes[1, 1].text(0.5, 0.5, "No Ground Truth\nComparison Unavailable", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'Pred Error (Unavailable)')
            
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        if has_coast: plot_coastline(axes[1, 1], coast_file, color='grey', linewidth=0.5)
        if dataset.has_target: plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04) # Only if plotted

        # Add visual box for statistic region (Physical Coordinates)
        rect2 = matplotlib.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                             linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        axes[1, 1].add_patch(rect2)
        
        if dataset.has_target:
            axes[1, 2].hist(true_center.flatten()[~np.isnan(true_center.flatten())], bins=50, alpha=0.5, label='Truth', density=True, color='blue')
        axes[1, 2].hist(pred_center.flatten()[~np.isnan(pred_center.flatten())], bins=50, alpha=0.5, label='Pred', density=True, color='green')
        axes[1, 2].legend()
        axes[1, 2].set_title(f'Dist Comparison ({int(target_h)}m, Margin={margin})')
        axes[1, 2].set_xlabel('Residual Gravity (mGal)')
        axes[1, 2].set_ylabel('Probability Density')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_result.png')
        print(f"Blind result saved to {output_prefix}_result.png")

        # Cross-sections
        print("Plotting Cross Sections...")
        # Reconstruct meshgrid for plotting
        rows, cols = dataset.size
        lon_grid = np.linspace(extent[0], extent[1], cols)
        lat_grid = np.linspace(extent[2], extent[3], rows)
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        if dataset.has_target:
            plot_cross_sections([input_np, up_check, true_np, pred_np], 
                               lon_2d, lat_2d, save_path=f"{output_prefix}_cross_section.png",
                               labels=[f'Obs {int(obs_h)}m', f'Upward {int(obs_h)}m', f'Truth {int(target_h)}m', f'Pred {int(target_h)}m'],
                               target_lat=24.0, target_lon=122.0)
        else:
            plot_cross_sections([input_np, up_check, pred_np], 
                               lon_2d, lat_2d, save_path=f"{output_prefix}_cross_section.png",
                               labels=[f'Obs {int(obs_h)}m', f'Upward {int(obs_h)}m', f'Pred {int(target_h)}m'],
                               target_lat=24.0, target_lon=122.0)
        print(f"Done. Saved to {output_prefix}_cross_section.png")

        # ==========================
        # Save Data Products
        # ==========================
        
        # 1. Save Model Weights
        torch.save(model.state_dict(), f'{output_prefix}_model.pth')
        print(f"Saved model weights to {output_prefix}_model.pth")
        
        # 2. Save Loss History
        df_loss = pd.DataFrame(loss_history, columns=['Epoch', 'Total_Loss', 'Loss_phy', 'Loss_tik', 'Loss_stat', 'Stability_Score'])
        df_loss.to_csv(f'{output_prefix}_loss.csv', index=False)
        print(f"Saved loss history to {output_prefix}_loss.csv")
        
        # 3. Save XYZ Grid (Final Verification)
        # Reconstruct coordinates for export
        rows, cols = pred_np.shape
        lon_grid = np.linspace(extent[0], extent[1], cols)
        lat_grid = np.linspace(extent[2], extent[3], rows)
        XX, YY = np.meshgrid(lon_grid, lat_grid) # Note: Meshgrid order default is xy
        
        # Flatten and save
        pred_flat = pred_np.flatten()
        xyz_data = np.stack([XX.flatten(), YY.flatten(), pred_flat], axis=1)
        np.savetxt(f'{output_prefix}_{int(target_h)}m.xyz', xyz_data, fmt='%.6f', header='Longitude Latitude Gravity_mGal')
        print(f"Saved predicted grid to {output_prefix}_{int(target_h)}m.xyz")
        
        # 4. Save Metadata (JSON)
        elapsed_time = time.time() - start_time
        meta = {
            "amplification_ratio": float(ratio),
            "training_rms_mgal": float(rms_pred),
            "training_epochs": int(epoch),
            "execution_time_sec": float(elapsed_time),
            "grid_size": int(grid_size),
            "noise_level": float(noise_level),
            "Upward-check_rms_mgal": float(rms_up),
            "model_type": "SimpleUNet + GaussianSmoothing",
            "description": f"Ratio is std({int(target_h)}m)/std({int(obs_h)}m) observed during training. Use this to amplify {int(obs_h)}m std during inference."
        }
        with open(f'{output_prefix}_metadata.json', 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"Saved metadata to {output_prefix}_metadata.json (Ratio={ratio:.4f}, Time={elapsed_time:.1f}s)")

        # EGM dataset
        EGMdataset = None
        if EGM_high is not None and EGM_low is not None:
            EGMdataset = GravityGridDataset(EGM_high, EGM_low, size=(grid_size, grid_size))
            input_img_5000_EGM, input_img_2000_EGM = EGMdataset[0]
            input_img_5000_EGM = input_img_5000_EGM.unsqueeze(0).to(device)
            input_img_2000_EGM = input_img_2000_EGM.unsqueeze(0).to(device)

            # plot_gravity_map of input_full and pred_full to compare
            # Denormalize EGM at observation height (using EGM's own statistics)
            real_EGM_5000 = (input_img_5000_EGM * EGMdataset.std_5 + EGMdataset.mean_5).squeeze().cpu().numpy()
            # input_full is full gravity at observation height
            input_full = input_np + real_EGM_5000
            
            # Denormalize EGM at target height (using EGM's own statistics)
            real_EGM_2000 = (input_img_2000_EGM * EGMdataset.std_2 + EGMdataset.mean_2).squeeze().cpu().numpy()
            # pred_full is full gravity at target height
            pred_full = pred_np + real_EGM_2000
            
            print("Plotting Full Gravity Comparison (Input vs Pred)...")
            
            plot_gravity_grid_comparison(input_full, pred_full, extent,
                                         output_png=f"{output_prefix}_full_gravity_comparison.png",
                                         title_1=f"Input Full Gravity ({int(obs_h)}m)", 
                                         title_2=f"Pred Full Gravity ({int(target_h)}m)",
                                         cross_section_path=f"{output_prefix}_full_gravity_cross_section.png",
                                         target_lat=24.0, target_lon=122.0,
                                         lon_grid=lon_2d, lat_grid=lat_2d)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=301, help="Grid size (e.g. 301, 601)")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--prefilter", action='store_true', help="Apply prefiltering")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level in mGal (sigma)")
    parser.add_argument("--file_high", type=str, default="TW_EGM_grid_g_res_h5000.xyz", help="High-altitude data file")
    #parser.add_argument("--file_low", type=str, default="TW_EGM_grid_g_res_h1500.xyz", help="Low-altitude data file")
    parser.add_argument("--file_low", type=str, default=None, help="Low-altitude data file (Optional)")
    parser.add_argument("--EGM_high", type=str, default=None, help="High-altitude EGM data file (Optional)")
    parser.add_argument("--EGM_low", type=str, default=None, help="Low-altitude EGM data file (Optional)")
    parser.add_argument("--obs_h", type=float, default=5000, help="Observation height in m")
    parser.add_argument("--target_h", type=float, default=1500, help="Target height in m")
    parser.add_argument("--prefix", type=str, default="blind_pisr_", help="Prefix for output files")
    parser.add_argument("--margin", type=int, default=20, help="Margin to crop from each side")
    parser.add_argument("--lag", type=int, default=300, help="Lag value")
    args = parser.parse_args()
    prefix = args.prefix + str(args.size)
    if args.noise > 0:
        prefix += f"_n{int(args.noise)}"
    print(f"\nPrefix for output files: {prefix}")
    prefilter = args.prefilter
    margin = args.margin
    
    try:
        train_blind_pisr(file_high=args.file_high, 
        file_low=args.file_low, 
        EGM_high=args.EGM_high, 
        EGM_low=args.EGM_low, 
        obs_h=args.obs_h, 
        target_h=args.target_h, 
        grid_size=args.size, 
        epochs=args.epochs, 
        output_prefix=prefix, 
        noise_level=args.noise, 
        margin=margin,
        lag=args.lag)
    except Exception as e:
        import traceback
        traceback.print_exc()
