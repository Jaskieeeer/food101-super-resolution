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

# Add your own image paths here
TEST_IMAGES = [
    "data/food-101/images/pizza/12345.jpg",
    "data/food-101/images/hamburger/56789.jpg"
]
# ---------------------

def calculate_psnr(img1, img2):
    # Convert images to numpy and float32
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def run_benchmark():
    # Setup Device
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

    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            continue

        print(f"Processing: {os.path.basename(img_path)}...")
        
        # 1. Prepare Data
        hr_img = Image.open(img_path).convert('RGB')
        w, h = hr_img.size
        
        # Make sure dimensions are divisible by scale factor (for clean comparison)
        w, h = (w // SCALE_FACTOR) * SCALE_FACTOR, (h // SCALE_FACTOR) * SCALE_FACTOR
        hr_img = hr_img.resize((w, h), Image.BICUBIC)

        # Create Low Res Input
        lr_dims = (w // SCALE_FACTOR, h // SCALE_FACTOR)
        lr_img = hr_img.resize(lr_dims, Image.BICUBIC)

        # 2. CLASSIC CV METHOD (Bicubic Upscale)
        # This is what Photoshop does. It smoothes pixels but adds no detail.
        classic_cv_img = lr_img.resize((w, h), Image.BICUBIC)
        
        # 3. AI METHOD (Your Model)
        img_tensor = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
        
        output = torch.clamp(output, 0.0, 1.0).squeeze(0).cpu()
        ai_img = transforms.ToPILImage()(output)

        # 4. Calculate Scores
        psnr_classic = calculate_psnr(classic_cv_img, hr_img)
        psnr_ai = calculate_psnr(ai_img, hr_img)

        # 5. Visualize (Zoomed Crop)
        # We crop the center to see details better
        crop = 100
        cx, cy = w // 2, h // 2
        box = (cx - crop, cy - crop, cx + crop, cy + crop)

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        # A: Low Res (Pixelated view)
        axes[0].imshow(lr_img.resize((w, h), Image.NEAREST).crop(box))
        axes[0].set_title("Input (Pixels)", fontsize=12)
        axes[0].set_xlabel("What the model sees")
        
        # B: Classic CV (Bicubic)
        axes[1].imshow(classic_cv_img.crop(box))
        axes[1].set_title(f"Classic CV (Bicubic)\nPSNR: {psnr_classic:.2f} dB", fontsize=12, color='red')
        axes[1].set_xlabel("Smooth but Blurry")

        # C: Your AI Model
        axes[2].imshow(ai_img.crop(box))
        # Green title if you beat classic (you should!)
        color = 'green' if psnr_ai > psnr_classic else 'black'
        axes[2].set_title(f"AttentionSR (Ours)\nPSNR: {psnr_ai:.2f} dB", fontsize=12, color=color, fontweight='bold')
        axes[2].set_xlabel("Sharper Edges & Details")

        # D: Ground Truth
        axes[3].imshow(hr_img.crop(box))
        axes[3].set_title("Ground Truth", fontsize=12)
        axes[3].set_xlabel("Perfect Image")

        # Hide ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"Benchmark: {os.path.basename(img_path)} (x{SCALE_FACTOR})", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_benchmark()