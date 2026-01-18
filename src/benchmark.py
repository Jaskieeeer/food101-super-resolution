import os
# ---------------------------------------------------------
# CRITICAL: This must be set BEFORE importing torch
# ---------------------------------------------------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your modules
from dataset import FoodSRDataset
from models import SRCNN, ResNetSR, AttentionSR # <--- Import all models
from utils import calculate_metrics

# --- CONFIG ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Choose the model file you want to test
MODEL_PATH = "weights/ResNet_Baseline_epoch_5.pth" 
SAVE_DIR = "reports/benchmark/resnet_mae+paper_bs16_vs_bicubic"

def get_model_from_filename(filename):
    """
    Automatically picks the right class based on the file name.
    """
    if "SRCNN" in filename:
        return SRCNN(scale_factor=2).to(DEVICE), "SRCNN"
    elif "ResNet" in filename:
        return ResNetSR(scale_factor=2).to(DEVICE), "ResNetSR"
    elif "Attention" in filename:
        return AttentionSR(scale_factor=2).to(DEVICE), "AttentionSR"
    else:
        raise ValueError(f"Could not guess architecture from filename: {filename}")

def benchmark():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load Data (Test Split)
    print("Loading Test Data...")
    ds = FoodSRDataset(split='test', crop_size=200, scale_factor=2)
    # Let's test on just 100 images to be quick
    ds = torch.utils.data.Subset(ds, range(100)) 
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 2. Load Your Model (Auto-Detected)
    print(f"Loading Model from: {MODEL_PATH}")
    try:
        model, model_name = get_model_from_filename(MODEL_PATH)
        # Load weights (suppress security warning)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_PATH}. Run train.py first!")
        return
    model.eval()
    
    # 3. Initialize Score Keepers
    bicubic_psnr = []
    bicubic_ssim = []
    model_psnr = []
    model_ssim = []
    
    print(f"Starting Benchmark: OpenCV Bicubic vs. {model_name}...")
    
    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(tqdm(loader)):
            # --- A. Neural Network Prediction ---
            lr_dev = lr_tensor.to(DEVICE)
            sr_tensor = model(lr_dev)
            
            # Clamp output to 0-1 (Crucial for correct metrics)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
            
            # Calculate NN Metrics
            m_psnr, m_ssim = calculate_metrics(sr_tensor, hr_tensor)
            model_psnr.append(m_psnr)
            model_ssim.append(m_ssim)
            
            # --- B. OpenCV Bicubic Prediction ---
            # Convert tensors to Numpy images for OpenCV
            # Shape: [1, 3, H, W] -> [H, W, 3]
            lr_np = lr_tensor.squeeze().permute(1, 2, 0).numpy()
            hr_np = hr_tensor.squeeze().permute(1, 2, 0).numpy()
            
            # Upscale using standard Bicubic Interpolation
            target_h, target_w = hr_np.shape[:2]
            bicubic_img = cv2.resize(lr_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to Tensor for consistent metric calculation
            bicubic_tensor = torch.from_numpy(bicubic_img).permute(2, 0, 1).unsqueeze(0)
            
            # Calculate Bicubic Metrics
            b_psnr, b_ssim = calculate_metrics(bicubic_tensor, hr_tensor)
            bicubic_psnr.append(b_psnr)
            bicubic_ssim.append(b_ssim)
            
            # --- Save Visual Comparison (First 3 images) ---
            if i < 3:
                visualize_comparison(lr_np, bicubic_img, sr_tensor, hr_np, i, model_name)

    # 4. Final Results
    avg_b_psnr = sum(bicubic_psnr) / len(bicubic_psnr)
    avg_m_psnr = sum(model_psnr) / len(model_psnr)
    
    print("\n" + "="*40)
    print(f"📢 BENCHMARK RESULTS (On {len(loader)} images)")
    print("="*40)
    print(f"🔹 OpenCV Bicubic : PSNR = {avg_b_psnr:.2f} | SSIM = {sum(bicubic_ssim)/len(bicubic_ssim):.4f}")
    print(f"🔹 Your {model_name:<9} : PSNR = {avg_m_psnr:.2f} | SSIM = {sum(model_ssim)/len(model_ssim):.4f}")
    print("="*40)
    
    if avg_m_psnr > avg_b_psnr:
        print(f"✅ SUCCESS: Your {model_name} beat standard OpenCV upscaling!")
    else:
        print(f"⚠️ WARNING: Your {model_name} is worse than standard upscaling.")

def visualize_comparison(lr, bicubic, sr_tensor, hr, idx, model_name):
    """
    Saves a plot: Input | Bicubic | Neural Net | Ground Truth
    """
    # Convert SR tensor back to numpy
    sr_img = sr_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Clip values to 0-1 range
    sr_img = np.clip(sr_img, 0, 1)
    bicubic = np.clip(bicubic, 0, 1)
    
    plt.figure(figsize=(15, 5))
    
    titles = ['Low Res Input', 'OpenCV Bicubic', f'Your {model_name}', 'High Res Ground Truth']
    images = [lr, bicubic, sr_img, hr]
    
    for j, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, j+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/comparison_{idx}.png")
    plt.close()

if __name__ == "__main__":
    benchmark()