import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# 1. Import your model
from src.models import ResNetSR

# --- CONFIGURATION ---
WEIGHTS_PATH = "weights/gan_phase2_v6.pth"
SCALE_FACTOR = 4
NUM_EXAMPLES = 3  # How many windows to pop up
# ---------------------

def calculate_psnr(img1, img2):
    """Calculates PSNR (Peak Signal-to-Noise Ratio) between two images."""
    # Convert images to numpy float32
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def run_final_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading Official Test Set on {device}...")

    # 1. Load the Official Test Split (Guaranteed Unseen Data)
    test_dataset = datasets.Food101(
        root='./data', 
        split='test', 
        download=True,
        transform=transforms.ToTensor()
    )
    
    # 2. Load Your Model
    model = ResNetSR(scale_factor=SCALE_FACTOR).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"✅ Loaded weights: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Weights file not found at {WEIGHTS_PATH}")
        return

    model.eval()

    # 3. Pick Random Indices
    total_test_images = len(test_dataset)
    indices = random.sample(range(total_test_images), NUM_EXAMPLES)

    for idx in indices:
        hr_tensor, _ = test_dataset[idx] 
        
        # 4. Prepare Dimensions (must be divisible by 4)
        c, h, w = hr_tensor.shape
        h, w = (h // SCALE_FACTOR) * SCALE_FACTOR, (w // SCALE_FACTOR) * SCALE_FACTOR
        hr_tensor = hr_tensor[:, :h, :w]
        
        # Convert HR Tensor to PIL for Bicubic operations
        hr_img = transforms.ToPILImage()(hr_tensor)

        # 5. Create Low Res Input
        lr_size = (w // SCALE_FACTOR, h // SCALE_FACTOR)
        lr_img = hr_img.resize(lr_size, resample=0) # Use 0 (Nearest) to simulate raw pixels? No, usually Bicubic downsample is standard.
        # Standard degradation model: Bicubic Downsample
        lr_img = hr_img.resize(lr_size, resample=3) # 3 = BICUBIC

        # 6. Generate BASELINE (Bicubic Upscale)
        bicubic_img = lr_img.resize((w, h), resample=3) # 3 = BICUBIC
        
        # 7. Generate OUR MODEL (AttentionSR)
        lr_tensor_input = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor_input)
        
        sr_img = transforms.ToPILImage()(torch.clamp(sr_tensor.squeeze(0).cpu(), 0, 1))

        # 8. Calculate PSNR Scores
        psnr_bicubic = calculate_psnr(bicubic_img, hr_img)
        psnr_ours = calculate_psnr(sr_img, hr_img)

        # 9. Visualize
        plot_benchmark(lr_img, bicubic_img, sr_img, hr_img, psnr_bicubic, psnr_ours, idx)

def plot_benchmark(lr, bicubic, ours, hr, psnr_b, psnr_o, img_id):
    # Crop Center for better detail visibility
    w, h = hr.size
    crop_size = 120
    cx, cy = w // 2, h // 2
    box = (cx - crop_size, cy - crop_size, cx + crop_size, cy + crop_size)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # 1. Input
    axes[0].imshow(lr.resize(hr.size, resample=0).crop(box))
    axes[0].set_title(f"Input (Simulated)", fontsize=12)
    axes[0].set_xlabel("Pixelated Source")

    # 2. Bicubic
    axes[1].imshow(bicubic.crop(box))
    axes[1].set_title(f"Classic Bicubic\nPSNR: {psnr_b:.2f} dB", fontsize=12, color='red')
    axes[1].set_xlabel("Blurry Baseline")

    # 3. Our Model
    color = 'green' if psnr_o > psnr_b else 'black'
    axes[2].imshow(ours.crop(box))
    axes[2].set_title(f"AttentionSR (Ours)\nPSNR: {psnr_o:.2f} dB", fontsize=12, color=color, fontweight='bold')
    axes[2].set_xlabel("AI Reconstruction")

    # 4. Ground Truth
    axes[3].imshow(hr.crop(box))
    axes[3].set_title("Ground Truth", fontsize=12)
    axes[3].set_xlabel("Target Reality")

    # Hide ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Test Image ID: {img_id} (Official Test Set)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_final_benchmark()