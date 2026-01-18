import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from torchvision.utils import save_image
import lpips

# Global instance to avoid reloading weights every batch
_LPIPS_MODEL = None

def get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        print("🧠 Loading LPIPS Metric (AlexNet)...")
        # 'alex' is faster and standard for evaluation
        _LPIPS_MODEL = lpips.LPIPS(net='alex').to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL

def calculate_lpips(img1, img2, device):
    """
    Inputs: Tensors [N, 3, H, W] in range [0, 1]
    Output: Scalar (Lower is Better)
    """
    model = get_lpips_model(device)
    
    # LPIPS expects inputs in range [-1, 1]
    # We assume img1/img2 are [0, 1]
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1
    
    with torch.no_grad():
        dist = model(img1_norm, img2_norm)
    
    # Return average distance across batch
    return dist.mean().item()
def calculate_metrics(img1, img2):
    """
    Calculates PSNR and SSIM for a batch of images.
    Input: PyTorch tensors [Batch, 3, H, W] (normalized 0-1)
    """
    # Detach from GPU/Graph and convert to Numpy
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    
    psnr_val = 0
    ssim_val = 0
    batch_size = img1.shape[0]
    
    for i in range(batch_size):
        # Scikit-image expects [H, W, 3], PyTorch is [3, H, W]
        im1 = np.transpose(img1[i], (1, 2, 0))
        im2 = np.transpose(img2[i], (1, 2, 0))
        
        # Calculate metrics
        psnr_val += psnr(im1, im2, data_range=1.0)
        # channel_axis=2 tells it that the 3rd dimension is color channels
        ssim_val += ssim(im1, im2, data_range=1.0, channel_axis=2)
        
    return psnr_val / batch_size, ssim_val / batch_size

def save_comparison(lr, sr, hr, epoch):
    """
    Saves a visual comparison (Input vs Output vs Target)
    """
    os.makedirs("reports/images", exist_ok=True)
    
    # Resize Low-Res (LR) to match HR size just for the picture
    lr_resized = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='nearest')
    
    # Stack images side-by-side: LR | Output | Target
    comparison = torch.cat((lr_resized, sr, hr), dim=3) 
    save_image(comparison, f"reports/images/epoch_{epoch}_sample.png")