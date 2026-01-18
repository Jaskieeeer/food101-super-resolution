import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from torchvision.utils import save_image

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