import os
# --- FIX: Enable MPS Fallback for Mac M4 ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import json

# Local imports
from models import SRCNN
from dataset import FoodSRDataset
from loss import NLPDLoss 

# CONFIG
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
LRS_TO_TEST = [0.01, 0.001, 0.0001] 
STEPS_PER_RUN = 200 # Short run to test convergence speed
OUTPUT_DIR = "reports/estimation"

def run_sweep():
    print(f"🧪 Starting Hyperparameter Estimation on {DEVICE}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    criterion = NLPDLoss(device=DEVICE)

    # Dictionary to store results for plotting
    history = {} 

    for lr in LRS_TO_TEST:
        print(f"\n--- Testing LR: {lr} ---")
        # reinit=True is crucial for multiple runs in one script
        wandb.init(project="food101-sr", name=f"LR_Sweep_{lr}", reinit=True)
        
        model = SRCNN(scale_factor=2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        
        losses = []
        
        # Training Loop
        for i, (lr_img, hr_img) in enumerate(tqdm(loader, total=STEPS_PER_RUN)):
            if i >= STEPS_PER_RUN:
                break
            
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(lr_img)
            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            wandb.log({"train_loss": loss.item()})
            
        history[str(lr)] = losses
        wandb.finish()

    # --- AUTOMATIC REPORT GENERATION ---
    print("\n📊 Generating Report...")
    
    # 1. Generate Plot
    plt.figure(figsize=(10, 6))
    for lr, loss_curve in history.items():
        # Smooth the curve slightly for better visualization
        smoothed = [sum(loss_curve[i:i+10])/len(loss_curve[i:i+10]) for i in range(0, len(loss_curve), 10)]
        plt.plot(smoothed, label=f"LR = {lr}")
    
    plt.title("Hyperparameter Estimation: Learning Rate Convergence")
    plt.xlabel("Training Steps (x10)")
    plt.ylabel("Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "lr_sweep_plot.png")
    plt.savefig(plot_path)
    print(f"✅ Saved plot to {plot_path}")
    
    # 2. Generate Text Summary
    summary_path = os.path.join(OUTPUT_DIR, "lr_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HYPERPARAMETER ESTIMATION REPORT\n")
        f.write("==============================\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model: SRCNN\n")
        f.write(f"Steps per run: {STEPS_PER_RUN}\n\n")
        f.write("Results (Final Average Loss):\n")
        
        best_lr = None
        best_loss = float('inf')
        
        for lr, losses in history.items():
            final_avg = sum(losses[-50:]) / 50 # Avg of last 50 steps
            f.write(f"LR {lr}: {final_avg:.4f}\n")
            
            if final_avg < best_loss:
                best_loss = final_avg
                best_lr = lr
                
        f.write("\n------------------------------\n")
        f.write(f"RECOMMENDATION: Use Learning Rate = {best_lr}\n")
        
    print(f"✅ Saved summary to {summary_path}")

if __name__ == "__main__":
    run_sweep()