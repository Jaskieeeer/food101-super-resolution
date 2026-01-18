import sys
import os
sys.path.append(os.getcwd())
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb



from src.dataset import FoodSRDataset
from src.models import SRCNN, Discriminator 
from src.loss import NLPDLoss
from src.utils import calculate_metrics, save_comparison

# --- CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR_G = 0.001          
LR_D = 0.0001         
EPOCHS = 20           
BATCH_SIZE = 16       # Increased to 16 for speed (SRCNN is small)
RUN_NAME = "SRGAN_SRCNN_PaperLoss_BlindTrain"

def train_gan():
    # 1. Initialize WandB
    wandb.init(project="food101-sr", name=RUN_NAME, config={
        "type": "SRGAN",
        "generator": "SRCNN",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "split": "90_Train_10_Val" # Documenting the blind split
    })
    
    # 2. Init Models
    generator = SRCNN(scale_factor=2).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # 3. Init Optimizers & Schedulers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    
    # Decay LR by 50% every 5 epochs
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)
    
    # 4. Losses
    criterion_GAN = nn.BCELoss()
    criterion_content = NLPDLoss(device=DEVICE)
    
    # 5. Data Splitting (Train/Val Only)
    print("⏳ Loading Official Training Data...")
    full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    
    # OPTIONAL: Train on a subset for speed? (Remove these 3 lines to train on full 75k)
    # subset_indices = torch.randperm(len(full_ds))[:20000] # Train on 20k images
    # full_ds = torch.utils.data.Subset(full_ds, subset_indices)
    # print(f"⚡ Speed Mode: Reduced dataset to {len(full_ds)} images")

    # Split 90% Train / 10% Validation
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    
    print(f"📊 Blind Split: {train_size} Train | {val_size} Validation (Test Set Hidden)")
    
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"🚀 Starting Training on {DEVICE}...")
    
    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            
            valid = torch.ones(lr_imgs.size(0), 1, device=DEVICE)
            fake = torch.zeros(lr_imgs.size(0), 1, device=DEVICE)
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            gen_imgs = generator(lr_imgs)
            
            loss_adversarial = criterion_GAN(discriminator(gen_imgs), valid)
            loss_content = criterion_content(gen_imgs, hr_imgs)
            loss_G = loss_content + (1e-3 * loss_adversarial)
            
            loss_G.backward()
            optimizer_G.step()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_loss = criterion_GAN(discriminator(hr_imgs), valid)
            fake_loss = criterion_GAN(discriminator(gen_imgs.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
            
        # --- VALIDATION (Mini-Check) ---
        generator.eval()
        avg_psnr = 0
        avg_ssim = 0
        
        # Check first 20 batches only (approx 600 images) to save time
        check_batches = 20 
        
        with torch.no_grad():
            for i, (lr, hr) in enumerate(val_loader):
                if i >= check_batches: break
                
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = generator(lr).clamp(0, 1)
                
                p, s = calculate_metrics(sr, hr)
                avg_psnr += p
                avg_ssim += s
                
                if i == 0:
                    save_comparison(lr, sr, hr, f"GAN_SRCNN_ep{epoch+1}")

        avg_psnr /= check_batches
        avg_ssim /= check_batches
        
        # Step Schedulers
        scheduler_G.step()
        scheduler_D.step()
        current_lr = optimizer_G.param_groups[0]['lr']

        print(f"   -> Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | LR: {current_lr:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "lr_G": current_lr
        })
        
        # Save Checkpoint
        os.makedirs("weights", exist_ok=True)
        torch.save(generator.state_dict(), f"weights/SRGAN_SRCNN_ep{epoch+1}.pth")

    wandb.finish()
    print("✅ Training Complete. Run 'src/benchmark.py' to evaluate on the hidden Test Set.")

if __name__ == "__main__":
    train_gan()