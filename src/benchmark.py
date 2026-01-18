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
from models import SRCNN, ResNetSR, AttentionSR 
# Added calculate_lpips to imports
from utils import calculate_metrics, calculate_lpips 

# --- CONFIG ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Choose the model file you want to test
# Try testing your GAN weights here once you have them!
MODEL_PATH = "weights/SRGAN_SRCNN_ep6.pth" 
SAVE_DIR = "reports/benchmark/SRGAN_SRCNN_ep6_lpips_comparison"

def get_model_from_filename(filename):
    if "SRCNN" in filename:
        return SRCNN(scale_factor=2).to(DEVICE), "SRCNN"
    elif "ResNet" in filename:
        return ResNetSR(scale_factor=2).to(DEVICE), "ResNetSR"
    elif "Attention" in filename:
        return AttentionSR(scale_factor=2).to(DEVICE), "AttentionSR"
    # Fallback/Guess if naming is custom
    elif "GAN" in filename:
         # GAN Generator is usually ResNet-based in your setup
        return ResNetSR(scale_factor=2).to(DEVICE), "SRGAN"
    else:
        raise ValueError(f"Could not guess architecture from filename: {filename}")

def benchmark():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Test Data...")
    ds = FoodSRDataset(split='test', crop_size=200, scale_factor=2)
    ds = torch.utils.data.Subset(ds, range(100)) 
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 2. Load Model
    print(f"Loading Model from: {MODEL_PATH}")
    try:
        model, model_name = get_model_from_filename(MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_PATH}.")
        return
    model.eval()
    
    # 3. Score Keepers
    # Bicubic
    b_psnr, b_ssim, b_lpips = [], [], []
    # Model
    m_psnr, m_ssim, m_lpips = [], [], []
    
    print(f"Starting Benchmark: OpenCV Bicubic vs. {model_name}...")
    
    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(tqdm(loader)):
            # --- A. Neural Network Prediction ---
            lr_dev = lr_tensor.to(DEVICE)
            hr_dev = hr_tensor.to(DEVICE)
            
            sr_tensor = model(lr_dev)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
            
            # NN Metrics
            p, s = calculate_metrics(sr_tensor, hr_dev)
            l = calculate_lpips(sr_tensor, hr_dev, DEVICE) # <--- NEW LPIPS
            
            m_psnr.append(p)
            m_ssim.append(s)
            m_lpips.append(l)
            
            # --- B. OpenCV Bicubic Prediction ---
            lr_np = lr_tensor.squeeze().permute(1, 2, 0).numpy()
            hr_np = hr_tensor.squeeze().permute(1, 2, 0).numpy()
            target_h, target_w = hr_np.shape[:2]
            
            bicubic_img = cv2.resize(lr_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            bicubic_tensor = torch.from_numpy(bicubic_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # Bicubic Metrics
            # Ensure bicubic tensor is on same device for LPIPS
            bp, bs = calculate_metrics(bicubic_tensor.cpu(), hr_tensor)
            bl = calculate_lpips(bicubic_tensor, hr_dev, DEVICE) # <--- NEW LPIPS
            
            b_psnr.append(bp)
            b_ssim.append(bs)
            b_lpips.append(bl)
            
            if i < 3:
                visualize_comparison(lr_np, bicubic_img, sr_tensor, hr_np, i, model_name)

    # 4. Final Results
    print("\n" + "="*60)
    print(f"📢 BENCHMARK RESULTS ({len(loader)} images)")
    print("="*60)
    print(f"{'Method':<15} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10} | {'LPIPS (↓)':<10}")
    print("-" * 55)
    
    # Averages
    avg_b_psnr = sum(b_psnr)/len(b_psnr)
    avg_b_ssim = sum(b_ssim)/len(b_ssim)
    avg_b_lpips = sum(b_lpips)/len(b_lpips)
    
    avg_m_psnr = sum(m_psnr)/len(m_psnr)
    avg_m_ssim = sum(m_ssim)/len(m_ssim)
    avg_m_lpips = sum(m_lpips)/len(m_lpips)
    
    print(f"{'Bicubic':<15} | {avg_b_psnr:<10.2f} | {avg_b_ssim:<10.4f} | {avg_b_lpips:<10.4f}")
    print(f"{model_name:<15} | {avg_m_psnr:<10.2f} | {avg_m_ssim:<10.4f} | {avg_m_lpips:<10.4f}")
    print("="*60)
    
    # Interpretation for the User
    if avg_m_lpips < avg_b_lpips:
        print("✅ LPIPS SUCCESS: Your model is perceptually better (looks more real)!")
    else:
        print("⚠️ LPIPS WARNING: Your model looks less natural than bicubic scaling.")

def visualize_comparison(lr, bicubic, sr_tensor, hr, idx, model_name):
    sr_img = sr_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    sr_img = np.clip(sr_img, 0, 1)
    bicubic = np.clip(bicubic, 0, 1)
    
    plt.figure(figsize=(15, 5))
    titles = ['Low Res', 'Bicubic', f'{model_name}', 'Ground Truth']
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