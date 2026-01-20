import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import math

# Fix import path so we can run this file from root
sys.path.append(os.getcwd())
# Allow PyTorch to fallback to CPU for operations not yet supported by MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Local imports
from src.loss import NLPDLoss
from src.dataset import FoodSRDataset
from src.models import SRCNN, ResNetSR, AttentionSR 
from src.utils import calculate_metrics, save_comparison

# --- CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {DEVICE}")

EPOCHS = 10  # Desired epochs per run

# The 3 specific architecture/loss/LR combos you requested
EXPERIMENT_CONFIGS = [
    {"arch": "SRCNN", "loss_type": "Paper", "lr": 0.001},
]

# Run every config with both batch sizes
BATCH_SIZES = [8, 16]

def get_model(arch_name):
    if arch_name == "SRCNN":
        return SRCNN(scale_factor=2).to(DEVICE)
    elif arch_name == "ResNet":
        return ResNetSR(scale_factor=2).to(DEVICE)
    elif arch_name == "AttentionSR":
        return AttentionSR(scale_factor=2).to(DEVICE)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

def get_loss(loss_name):
    if loss_name == "MAE":
        return nn.L1Loss()
    elif loss_name == "Paper":
        return NLPDLoss(device=DEVICE)
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")

def run_experiment(config, batch_size):
    # Construct Unique Name
    arch = config['arch']
    loss_name = config['loss_type']
    lr = config['lr']
    
    run_name = f"{arch}_{loss_name}_LR{lr}_BS{batch_size}"
    print(f"\n==================================================")
    print(f"🎬 STARTING RUN: {run_name}")
    print(f"==================================================\n")

    # 1. Initialize WandB
    wandb.init(project="food101-sr", name=run_name, reinit=True, config={
        "architecture": arch,
        "loss": loss_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "split": "80/10/10"
    })
    
    # 2. Dataset & 3-Way Split (Train/Val/Test)
    # We load the 'train' split and partition it ourselves to ensure no leakage
    full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    
    total_len = len(full_ds)
    train_len = int(0.8 * total_len)
    val_len   = int(0.1 * total_len)
    test_len  = total_len - train_len - val_len # Remaining 10%
    
    print(f"📊 Splitting Data: Train={train_len}, Val={val_len}, Test={test_len}")
    
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    # num_workers=0 is safer on Mac M4 to prevent multiprocessing crashes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. Model, Loss, Optimizer
    model = get_model(arch)
    criterion = get_loss(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"[{run_name}] Ep {epoch+1}/{EPOCHS}")
        
        for lr_img, hr_img in loop:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(lr_img)
            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATE (Used for Scheduler) ---
        model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for i, (lr_img, hr_img) in enumerate(val_loader):
                lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
                output = model(lr_img)
                loss = criterion(output, hr_img)
                val_loss += loss.item()
                
                # Clamp for metrics
                output_clamped = torch.clamp(output, 0.0, 1.0)
                p, s = calculate_metrics(output_clamped, hr_img)
                val_psnr += p
                val_ssim += s

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        
        # Update Scheduler based on Val Loss
        scheduler.step(avg_val_loss)
        
        # Log Train/Val Metrics
        wandb.log({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save Checkpoint
        os.makedirs("weights", exist_ok=True)
        save_path = f"weights/{run_name}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
    
    # 5. FINAL TEST EVALUATION (The Gold Standard)
    print(f"🧪 Evaluation on HELD-OUT TEST SET ({test_len} images)...")
    model.eval()
    test_psnr = 0
    test_ssim = 0
    
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(test_loader):
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            output = model(lr_img)
            output_clamped = torch.clamp(output, 0.0, 1.0)
            
            p, s = calculate_metrics(output_clamped, hr_img)
            test_psnr += p
            test_ssim += s
            
            # Save one visual example from Test Set
            if i == 0:
                try:
                    save_comparison(lr_img, output_clamped, hr_img, f"{run_name}_TEST_FINAL")
                except:
                    pass

    final_test_psnr = test_psnr / len(test_loader)
    final_test_ssim = test_ssim / len(test_loader)
    
    print(f"🏆 FINAL TEST RESULTS: PSNR {final_test_psnr:.2f} dB | SSIM {final_test_ssim:.4f}")
    
    # Log Final Summary Metrics
    wandb.log({
        "test_psnr": final_test_psnr,
        "test_ssim": final_test_ssim
    })

    wandb.finish()
    print(f"✅ Finished Run: {run_name}\n")

if __name__ == "__main__":
    total_runs = len(EXPERIMENT_CONFIGS) * len(BATCH_SIZES)
    print(f"🛑 QUEUED {total_runs} EXPERIMENTS")
    
    run_count = 1
    for config in EXPERIMENT_CONFIGS:
        for batch_size in BATCH_SIZES:
            print(f"--- Processing {run_count}/{total_runs} ---")
            try:
                run_experiment(config, batch_size)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ OOM Error on {config['arch']} with BS={batch_size}. Skipping...")
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
            run_count += 1