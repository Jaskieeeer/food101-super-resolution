import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import numpy as np

# 1. Import your model
from src.models import AttentionSR

# --- CONFIGURATION ---
WEIGHTS_PATH = "weights/AttentionSR_Phase1.pth" 
SCALE_FACTOR = 4
PATH_FILE = "test.txt" # <--- The file you uploaded

# Default images (if test.txt is empty or missing)
TEST_IMAGES = [
    "data/food-101/images/pizza/12345.jpg",
]
# ---------------------

def load_paths_from_file(filename):
    """Reads image paths from a text file, one per line."""
    paths = []
    if not os.path.exists(filename):
        print(f"⚠️ Warning: '{filename}' not found. Using default list.")
        return []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Clean up lines (remove newlines, extra spaces)
    for line in lines:
        clean_path = line.strip()
        if clean_path and not clean_path.startswith("#"): # Skip empty lines or comments
            paths.append(clean_path)
            
    print(f"📂 Loaded {len(paths)} extra paths from {filename}")
    return paths

def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Benchmark on {device}...")

    # Load Model
    model = AttentionSR(scale_factor=SCALE_FACTOR).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("✅ Weights loaded.")
    except FileNotFoundError:
        print("❌ Weights not found! Check path.")
        return
    model.eval()

    # --- MERGE PATHS ---
    # Combine hardcoded defaults with paths from test.txt
    file_paths = load_paths_from_file(PATH_FILE)
    all_images = TEST_IMAGES + file_paths
    
    # Remove duplicates just in case
    all_images = list(set(all_images))

    if not all_images:
        print("❌ No images found to test! Check test.txt or default paths.")
        return

    for img_path in all_images:
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing file: {img_path}")
            continue

        print(f"Processing: {os.path.basename(img_path)}...")
        
        # 1. Prepare Data
        try:
            hr_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error opening {img_path}: {e}")
            continue
            
        w, h = hr_img.size
        
        # Ensure divisible by scale factor
        w, h = (w // SCALE_FACTOR) * SCALE_FACTOR, (h // SCALE_FACTOR) * SCALE_FACTOR
        hr_img = hr_img.resize((w, h), Image.BICUBIC)

        # Create Low Res Input
        lr_dims = (w // SCALE_FACTOR, h // SCALE_FACTOR)
        lr_img = hr_img.resize(lr_dims, Image.BICUBIC)

        # 2. CLASSIC CV METHOD (Bicubic)
        classic_cv_img = lr_img.resize((w, h), Image.BICUBIC)
        
        # 3. AI METHOD (AttentionSR)
        img_tensor = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
        
        output = torch.clamp(output, 0.0, 1.0).squeeze(0).cpu()
        ai_img = transforms.ToPILImage()(output)

        # 4. Calculate Scores
        psnr_classic = calculate_psnr(classic_cv_img, hr_img)
        psnr_ai = calculate_psnr(ai_img, hr_img)

        # 5. Visualize
        crop = 100
        cx, cy = w // 2, h // 2
        box = (cx - crop, cy - crop, cx + crop, cy + crop)

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        # A: Input
        axes[0].imshow(lr_img.resize((w, h), Image.NEAREST).crop(box))
        axes[0].set_title("Input (Pixels)", fontsize=12)
        
        # B: Classic
        axes[1].imshow(classic_cv_img.crop(box))
        axes[1].set_title(f"Classic CV\nPSNR: {psnr_classic:.2f} dB", fontsize=12, color='red')

        # C: AI
        axes[2].imshow(ai_img.crop(box))
        color = 'green' if psnr_ai > psnr_classic else 'black'
        axes[2].set_title(f"AttentionSR\nPSNR: {psnr_ai:.2f} dB", fontsize=12, color=color, fontweight='bold')

        # D: Truth
        axes[3].imshow(hr_img.crop(box))
        axes[3].set_title("Ground Truth", fontsize=12)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"Benchmark: {os.path.basename(img_path)} (x{SCALE_FACTOR})", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_benchmark()