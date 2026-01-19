import sys
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.getcwd())
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import your modules
from src.dataset import FoodSRDataset
from src.models import SRCNN, ResNetSR, AttentionSR, RRDBNet
from src.utils import calculate_metrics, calculate_lpips

# --- CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "chosen_models"
CSV_PATH = "reports/results_summary.csv"
IMG_OUTPUT_ROOT = "reports/images"  # <--- FIXED PATH

# How many images to save for visual comparison per model?
VISUAL_LIMIT = 20 

# Set to None for FULL test set (25k images), or 1000 for a quick test
TEST_LIMIT = None 

def get_model_from_filename(filepath):
    """
    Detects architecture from filename and loads weights safely.
    """
    filename = os.path.basename(filepath)
    model_name_ext = os.path.splitext(filename)[0] 
    
    # 1. Architecture Detection
    if "SRCNN" in filename:
        model = SRCNN(scale_factor=2)
    elif "ResNet" in filename:
        model = ResNetSR(scale_factor=2)
    elif "Attention" in filename:
        model = AttentionSR(scale_factor=2)
    elif "RRDB" in filename or "ESRGAN" in filename:
        model = RRDBNet(in_channels=3, out_channels=3, nf=64, nb=3, scale_factor=2)
    else:
        model = ResNetSR(scale_factor=2)

    model = model.to(DEVICE)
    
    # 2. Load Weights
    # print(f"🔄 Loading: {filename}...")
    try:
        state_dict = torch.load(filepath, map_location=DEVICE, weights_only=True)
        
        # Remove 'module.' prefix if present
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"❌ Error loading weights for {filename}: {e}")
        return None, None

    return model, model_name_ext

def get_processed_models_from_folders(img_root, csv_path):
    """
    Scans OUTPUT FOLDERS. Only counts a model as 'done' if 
    its folder contains at least 20 images.
    """
    processed = set()
    
    # 1. Check Directories
    if os.path.exists(img_root):
        print(f"📂 Scanning output folder: {img_root}")
        for item in os.listdir(img_root):
            path = os.path.join(img_root, item)
            
            if os.path.isdir(path):
                # Count the PNG files inside
                files = [f for f in os.listdir(path) if f.endswith('.png')]
                count = len(files)
                
                # STRICT CHECK: Only skip if we have 20+ images
                if count >= 20: 
                    processed.add(item)
                    # print(f"   ✅ Found {item} ({count} images)")
                else:
                    print(f"   ⚠️ Found incomplete folder '{item}' ({count}/20 images). Will re-run.")
    
    # 2. Special Check for Bicubic
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "Baseline_Bicubic" in content:
                    processed.add("Baseline_Bicubic")
        except:
            pass
            
    return processed

def benchmark_all():
    # 1. Setup
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Error: Directory '{MODELS_DIR}' not found.")
        return

    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")])
    if not model_files:
        print(f"❌ No .pth files found in {MODELS_DIR}")
        return

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    
    # --- CHECK FOLDERS ---
    processed_models = get_processed_models_from_folders(IMG_OUTPUT_ROOT, CSV_PATH)
    print(f"📋 Found {len(processed_models)} completed models: {list(processed_models)}")

    # 2. Load Data
    print(f"⏳ Loading Test Set (Limit: {TEST_LIMIT if TEST_LIMIT else 'ALL'})...")
    full_ds = FoodSRDataset(split='test', crop_size=200, scale_factor=2)
    if TEST_LIMIT:
        ds = torch.utils.data.Subset(full_ds, range(TEST_LIMIT))
    else:
        ds = full_ds
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    # Prepare CSV Header
    fieldnames = ["Model", "PSNR", "SSIM", "LPIPS"]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    need_bicubic = "Baseline_Bicubic" not in processed_models
    
    # --- MODEL LOOP ---
    for f_name in model_files:
        path = os.path.join(MODELS_DIR, f_name)
        
        # We need the model ID (filename without extension)
        temp_id = os.path.splitext(f_name)[0]
        
        if temp_id in processed_models:
            print(f"⏭️  Skipping {temp_id} (Already Done)")
            continue

        # Load Model
        print(f"🔄 Loading: {f_name}...")
        model, model_id = get_model_from_filename(path)
        
        if model is None: continue
        
        model.eval()
        
        # Setup specific image folder
        save_img_dir = os.path.join(IMG_OUTPUT_ROOT, model_id)
        os.makedirs(save_img_dir, exist_ok=True)
        
        # Accumulators
        m_psnr, m_ssim, m_lpips = [], [], []
        b_psnr, b_ssim, b_lpips = [], [], [] 
        
        print(f"🚀 Benchmarking {model_id}...")
        
        # --- TRY-CATCH BLOCK FOR INFERENCE ---
        try:
            with torch.no_grad():
                for i, (lr_tensor, hr_tensor) in enumerate(tqdm(loader, leave=False)):
                    lr_dev = lr_tensor.to(DEVICE)
                    hr_dev = hr_tensor.to(DEVICE)
                    
                    # A. Model Prediction
                    sr_tensor = model(lr_dev).clamp(0, 1)
                    
                    mp, ms = calculate_metrics(sr_tensor, hr_dev)
                    ml = calculate_lpips(sr_tensor, hr_dev, DEVICE)
                    
                    m_psnr.append(mp)
                    m_ssim.append(ms)
                    m_lpips.append(ml)
                    
                    # B. Bicubic Prediction (Only if needed)
                    bicubic_tensor = None
                    if need_bicubic:
                        lr_cpu = lr_dev.cpu()
                        bic_cpu = torch.nn.functional.interpolate(
                            lr_cpu, size=hr_dev.shape[2:], mode='bicubic', align_corners=False
                        )
                        bicubic_tensor = bic_cpu.to(DEVICE).clamp(0, 1)
                        
                        bp, bs = calculate_metrics(bicubic_tensor, hr_dev)
                        bl = calculate_lpips(bicubic_tensor, hr_dev, DEVICE)
                        
                        b_psnr.append(bp)
                        b_ssim.append(bs)
                        b_lpips.append(bl)

                    # C. Save Visuals
                    if i < VISUAL_LIMIT:
                        if bicubic_tensor is None:
                             lr_cpu = lr_dev.cpu()
                             bic_cpu = torch.nn.functional.interpolate(
                                lr_cpu, size=hr_dev.shape[2:], mode='bicubic', align_corners=False
                            )
                             bicubic_tensor = bic_cpu.to(DEVICE).clamp(0, 1)

                        save_comparison(
                            lr_tensor, bicubic_tensor, sr_tensor, hr_tensor, 
                            f"{save_img_dir}/img_{i:03d}.png", model_id
                        )

            # --- SAVE RESULTS ---
            
            # 1. Save Bicubic
            if need_bicubic and len(b_psnr) > 0:
                avg_bp = sum(b_psnr) / len(b_psnr)
                avg_bs = sum(b_ssim) / len(b_ssim)
                avg_bl = sum(b_lpips) / len(b_lpips)
                
                with open(CSV_PATH, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        "Model": "Baseline_Bicubic",
                        "PSNR": f"{avg_bp:.2f}",
                        "SSIM": f"{avg_bs:.4f}",
                        "LPIPS": f"{avg_bl:.4f}"
                    })
                print("✅ Saved Baseline_Bicubic results.")
                processed_models.add("Baseline_Bicubic")
                need_bicubic = False 

            # 2. Save Model Results
            avg_mp = sum(m_psnr) / len(m_psnr)
            avg_ms = sum(m_ssim) / len(m_ssim)
            avg_ml = sum(m_lpips) / len(m_lpips)
            
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "Model": model_id,
                    "PSNR": f"{avg_mp:.2f}",
                    "SSIM": f"{avg_ms:.4f}",
                    "LPIPS": f"{avg_ml:.4f}"
                })
            print(f"✅ Saved {model_id} results (PSNR: {avg_mp:.2f})")
            processed_models.add(model_id)

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR while testing {model_id}:")
            print(f"   {e}")
            print("   ⏭️  Skipping this model and continuing...")
            
        finally:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    print("\n" + "="*60)
    print("🏁 ALL DONE.")
    print(f"See results in: {CSV_PATH}")

def save_comparison(lr, bic, sr, hr, path, title):
    def to_np(t): return t.squeeze().cpu().permute(1, 2, 0).numpy()
    
    try:
        fig, ax = plt.subplots(1, 4, figsize=(16, 5))
        images = [to_np(lr), to_np(bic), to_np(sr), to_np(hr)]
        titles = ["Low Res", "Bicubic", title, "Ground Truth"]
        
        for a, img, t in zip(ax, images, titles):
            a.imshow(np.clip(img, 0, 1))
            a.set_title(t)
            a.axis("off")
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to save image {path}: {e}")

if __name__ == "__main__":
    benchmark_all()