import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import math
import os

from src.models import get_model

SCALE_FACTOR = 4
NUM_EXAMPLES = 17
OUTPUT_DIR = "report"

WEIGHTS = {
    "SRCNN":           "weights/srcnn_nlpd_best.pth",
    "RESNET":          "weights/resnet_run_best.pth",
    "AttentionSR":     "weights/attentionsr_run_best.pth",
    "AttentionSR_GAN": "weights/attentionsr_gan_best.pth" 
}

def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_prediction(model_name, weight_path, lr_tensor, device):
    arch_name = "AttentionSR" if "AttentionSR" in model_name else model_name
    
    model = get_model(arch_name, scale_factor=SCALE_FACTOR, device=device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except FileNotFoundError:
        print(f"Warning: Could not find weights for {model_name} at {weight_path}")
        return None
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

    model.eval()
    
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    sr_img = transforms.ToPILImage()(torch.clamp(sr_tensor.squeeze(0).cpu(), 0, 1))
    return sr_img

def run_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing images on {device}...")

    test_dataset = datasets.Food101(root='./data', split='test', download=True, transform=transforms.ToTensor())
    
    indices = random.sample(range(len(test_dataset)), NUM_EXAMPLES)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, idx in enumerate(indices):
        print(f"\n--- Processing Image {i+1}/{NUM_EXAMPLES} (Index: {idx}) ---")
        
        save_path = os.path.join(OUTPUT_DIR, f"image_{idx}")
        os.makedirs(save_path, exist_ok=True)
        
        hr_tensor, _ = test_dataset[idx]
        
        c, h, w = hr_tensor.shape
        h, w = (h // SCALE_FACTOR) * SCALE_FACTOR, (w // SCALE_FACTOR) * SCALE_FACTOR
        hr_tensor = hr_tensor[:, :h, :w]
        hr_img = transforms.ToPILImage()(hr_tensor)

        lr_size = (w // SCALE_FACTOR, h // SCALE_FACTOR)
        lr_img = hr_img.resize(lr_size, resample=3) 
        lr_tensor = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)

        hr_img.save(os.path.join(save_path, "ground_truth.png"))
        lr_img.resize(hr_img.size, resample=0).save(os.path.join(save_path, "input_lr_resized.png")) 
        
        bicubic_img = lr_img.resize((w, h), resample=3)
        bicubic_img.save(os.path.join(save_path, "bicubic.png"))
        
        psnr_bicubic = calculate_psnr(bicubic_img, hr_img)
        print(f"Saved Baseline | Bicubic PSNR: {psnr_bicubic:.2f} dB")

        for name, path in WEIGHTS.items():
            sr_img = get_prediction(name, path, lr_tensor, device)
            
            if sr_img:
                psnr = calculate_psnr(sr_img, hr_img)
                
                filename = f"{name.lower()}.png"
                sr_img.save(os.path.join(save_path, filename))
                print(f"Saved {name} | PSNR: {psnr:.2f} dB")
            else:
                print(f"Skipped {name} (Model failed to load)")

    print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    run_comparison()