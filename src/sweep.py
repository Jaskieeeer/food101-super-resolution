import os
# Fix for Mac M4 MPS Fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb

# Local imports
from models import SRCNN, ResNetSR, AttentionSR
from dataset import FoodSRDataset
from loss import NLPDLoss 
from utils import calculate_metrics  # <--- Importing your existing function

# CONFIG
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(config=None):
    try: # Safety net for OOM crashes
        with wandb.init(config=config):
            config = wandb.config
            
            # 1. Build Model
            if config.model_type == "SRCNN":
                model = SRCNN(scale_factor=2).to(DEVICE)
            elif config.model_type == "ResNet":
                model = ResNetSR(scale_factor=2).to(DEVICE)
            elif config.model_type == "AttentionSR":
                model = AttentionSR(scale_factor=2).to(DEVICE)
                
            # 2. Select Loss
            if config.loss_type == "MSE":
                criterion = nn.MSELoss()
            elif config.loss_type == "Paper_NLPD":
                criterion = NLPDLoss(device=DEVICE)
            elif config.loss_type == "MAE":
                criterion = nn.L1Loss()
            
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            
            # 3. Data (Subset for speed)
            # Train on 500, Validate on 100
            full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
            train_ds = Subset(full_ds, range(0, 500))   
            val_ds = Subset(full_ds, range(500, 600))   
            
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            
            # --- TRAIN ---
            model.train()
            for lr_img, hr_img in train_loader:
                lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
                optimizer.zero_grad()
                output = model(lr_img)
                loss = criterion(output, hr_img)
                loss.backward()
                optimizer.step()
            
            # --- VALIDATE ---
            model.eval()
            total_psnr = 0
            total_ssim = 0
            
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
                    val_output = model(lr_img)
                    val_output = torch.clamp(val_output, 0.0, 1.0)
                    
                    # Call your function from src.utils
                    batch_psnr, batch_ssim = calculate_metrics(val_output, hr_img)
                    
                    total_psnr += batch_psnr
                    total_ssim += batch_ssim
                    
            avg_psnr = total_psnr / len(val_loader)
            avg_ssim = total_ssim / len(val_loader)
            # ... inside validation loop ...
            
            # --- NEW: ROBUST VISUALIZATION ---
            # 1. Pick the first image set
            ex_lr = lr_img[0].cpu()
            ex_hr = hr_img[0].cpu()
            ex_sr = val_output[0].cpu()

            def unnormalize(tensor):
                """
                Guesses the normalization and fixes it for display.
                """
                # Case A: If tensor is mostly negative or has values < 0, 
                # it's likely normalized. Shift it back to 0-1.
                if tensor.min() < 0:
                    # Option 1: If using ImageNet stats (approximate un-norm)
                    # return tensor * 0.229 + 0.485 
                    
                    # Option 2: Simple min-max scaling to force it into 0-1 (Safest)
                    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
                
                # Case B: Already 0-1, just clamp to be safe
                return tensor.clamp(0, 1)

            wandb.log({
                "predictions": wandb.Image(
                    unnormalize(ex_sr), 
                    caption=f"SR (PSNR: {avg_psnr:.2f})"
                ),
                "ground_truth": wandb.Image(
                    unnormalize(ex_hr),
                    caption="High Res (Target)"
                ),
                "input_lr": wandb.Image(
                    unnormalize(ex_lr),
                    caption="Low Res (Input)"
                ),
                "val_psnr": avg_psnr, 
                "val_ssim": avg_ssim
            })
            
            print(f"Run: {config.model_type} | {config.loss_type} | LR {config.learning_rate:.4f} -> PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"⚠️ OOM: Batch Size {wandb.config.batch_size} too large for {wandb.config.model_type}")
            # Log 0 so Bayesian agent learns to avoid this configuration
            wandb.log({"val_psnr": 0, "val_ssim": 0}) 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            raise e

# --- BAYESIAN CONFIGURATION ---
sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'val_psnr', # Optimizing PSNR leads to more stable convergence than SSIM
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.002
        },
        'batch_size': {
            'values': [8, 16] # Conservative values for Mac M4
        },
        'model_type': {
            'values': ['SRCNN', 'ResNet', 'AttentionSR']
        },
        'loss_type': {
            'values': ['MSE', 'Paper_NLPD', 'MAE']
        }
    }
}

if __name__ == "__main__":
    print("🚀 Starting Bayesian Sweep...")
    # 1. Register the sweep
    sweep_id = wandb.sweep(sweep_config, project="food101-sr")
    
    # 2. Run the agent (Execute 20 experiments)
    wandb.agent(sweep_id, train_one_epoch, count=32)