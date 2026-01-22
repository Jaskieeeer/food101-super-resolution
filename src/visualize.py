import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. Import your model
from src.models import AttentionSR

# --- CONFIGURATION ---
WEIGHTS_PATH = "weights/AttentionSR_Phase1.pth"
SCALE_FACTOR = 4
NUM_EXAMPLES = 3  # How many test images to show
# ---------------------

def run_clean_test():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading Official Test Set on {device}...")

    # 1. Load the Official Test Split (Guaranteed Unseen Data)
    # We don't resize here; we want the original big image to crop from
    test_dataset = datasets.Food101(
        root='./data', 
        split='test',  # <--- CRITICAL: This ensures 0% pollution
        download=True,
        transform=transforms.ToTensor()
    )
    
    # 2. Load Your Model
    model = AttentionSR(scale_factor=SCALE_FACTOR).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"✅ Loaded weights: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Weights file not found at {WEIGHTS_PATH}")
        return

    model.eval()

    # 3. Pick Random Indices from the Test Set
    total_test_images = len(test_dataset)
    print(f"📚 Total Test Images Available: {total_test_images}")
    
    indices = random.sample(range(total_test_images), NUM_EXAMPLES)

    for i, idx in enumerate(indices):
        hr_img, _ = test_dataset[idx] # Get High Res Tensor
        
        # 4. Prepare Input (Create Low Res on the fly)
        # We process dimensions to ensure they are clean multiples of 4
        c, h, w = hr_img.shape
        h, w = (h // SCALE_FACTOR) * SCALE_FACTOR, (w // SCALE_FACTOR) * SCALE_FACTOR
        hr_img = hr_img[:, :h, :w] # Crop slightly to fit scale
        
        # Downscale to create input (Bicubic simulation)
        resize_down = transforms.Resize((h // SCALE_FACTOR, w // SCALE_FACTOR), interpolation=transforms.InterpolationMode.BICUBIC)
        lr_img = resize_down(hr_img)
        
        # Send to Model
        lr_tensor = lr_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # 5. Visualize
        plot_comparison(lr_tensor, sr_tensor, hr_img, idx)

def plot_comparison(lr, sr, hr, img_id):
    # Convert tensors to displayable images
    lr_img = transforms.ToPILImage()(torch.clamp(lr.squeeze(0).cpu(), 0, 1))
    sr_img = transforms.ToPILImage()(torch.clamp(sr.squeeze(0).cpu(), 0, 1))
    hr_img = transforms.ToPILImage()(torch.clamp(hr.cpu(), 0, 1))
    
    # Create a nice crop to see details (Center 100x100)
    w, h = hr_img.size
    crop_size = 120
    cx, cy = w // 2, h // 2
    box = (cx - crop_size, cy - crop_size, cx + crop_size, cy + crop_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Left: Input (Upscaled purely for display)
    axes[0].imshow(lr_img.resize(hr_img.size, resample=0).crop(box))
    axes[0].set_title(f"Input (Simulated {hr_img.size[0]//4}px)")
    axes[0].axis('off')

    # Middle: Your Model
    axes[1].imshow(sr_img.crop(box))
    axes[1].set_title("AttentionSR Output", color='green', fontweight='bold')
    axes[1].axis('off')

    # Right: Ground Truth
    axes[2].imshow(hr_img.crop(box))
    axes[2].set_title("Ground Truth (Real)")
    axes[2].axis('off')

    plt.suptitle(f"Test Image ID: {img_id} (Unseen Data)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_clean_test()