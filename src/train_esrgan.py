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
import numpy as np



# Imports
from src.dataset import FoodSRDataset
from src.models import RRDBNet, Discriminator 
from src.loss import NLPDLoss
from src.utils import calculate_metrics, save_comparison

# --- CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ESRGAN Hyperparameters
LR_G = 0.0001           # Lower LR for fine-tuning
LR_D = 0.0001
EPOCHS = 20 # usually needs more, but 20 is fine for demo
BATCH_SIZE = 256         # RRDB is heavier than SRCNN, so reduced batch size
RUN_NAME = "ESRGAN_RRDB_Relativistic"

# Weights: 1.0 Content, 0.005 Adversarial, 0.01 Perceptual
LAMBDA_CONTENT = 1.0
LAMBDA_ADV = 5e-3
LAMBDA_PERCEP = 1e-2 

# Path to a pre-trained generator (trained with just L1 loss)
# If None, initializes randomly (harder to train)
PRETRAINED_GENERATOR_PATH = None 
# PRETRAINED_GENERATOR_PATH = "weights/RRDB_PSNR_best.pth" 

def train_esrgan():
    # 1. Initialize WandB
    wandb.init(project="food101-sr", name=RUN_NAME, config={
        "type": "ESRGAN",
        "generator": "RRDBNet",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lambda_adv": LAMBDA_ADV
    })
    
    # 2. Init Models
    # nb=3 blocks is lighter for laptop training. (Paper uses nb=23)
    generator = RRDBNet(in_channels=3, out_channels=3, nf=64, nb=3, scale_factor=2).to(DEVICE)
    discriminator = Discriminator(input_shape=(3, 200, 200)).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    # Load Pre-trained Weights if available (Warm Start)
    if PRETRAINED_GENERATOR_PATH and os.path.exists(PRETRAINED_GENERATOR_PATH):
        print(f"🔄 Loading pre-trained generator from {PRETRAINED_GENERATOR_PATH}...")
        generator.load_state_dict(torch.load(PRETRAINED_GENERATOR_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("⚠️ No pre-trained weights found. Training from scratch (Cold Start).")
    
    # 3. Optimizers & Schedulers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.9, 0.999))

    # Decay LR
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)
    
    # 4. Losses
    # BCEWithLogitsLoss is more stable than BCELoss + Sigmoid
    criterion_GAN = nn.BCEWithLogitsLoss() 
    criterion_content = nn.L1Loss() # Pixel loss (L1 is sharper than MSE)
    criterion_percep = NLPDLoss(device=DEVICE) # Texture loss
    
    # 5. Data Loading
    print("⏳ Loading Data...")
    full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    
    # Train/Val Split (90/10)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"🚀 Starting ESRGAN Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        if epoch < 5:
            print(f"🔥 Warmup Phase (Epoch {epoch+1}): Generator L1 Only")
            
            for lr, hr in train_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                
                optimizer_G.zero_grad()
                sr = generator(lr)
                loss_pixel = criterion_content(sr, hr) # L1 Loss
                loss_pixel.backward()
                optimizer_G.step()
                
            continue
        generator.train()
        discriminator.train()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            
            # -----------------------------------------
            #  1. Train Discriminator (Relativistic)
            # -----------------------------------------
            optimizer_D.zero_grad()
            
            # Generate Fake Images
            gen_imgs = generator(lr_imgs)
            
            # Get Raw Logits (No Sigmoid yet)
            real_logits = discriminator(hr_imgs)
            fake_logits = discriminator(gen_imgs.detach()) # Detach so G doesn't get grads
            
            # RaGAN Logic for Discriminator:
            # D tries to make Real > Fake
            # Loss 1: Real - mean(Fake) -> Label 1 (Real looks realer than fake average)
            d_loss_real = criterion_GAN(real_logits - torch.mean(fake_logits), torch.ones_like(real_logits))
            # Loss 2: Fake - mean(Real) -> Label 0 (Fake looks faker than real average)
            d_loss_fake = criterion_GAN(fake_logits - torch.mean(real_logits), torch.zeros_like(fake_logits))
            
            loss_D = (d_loss_real + d_loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            # -----------------------------------------
            #  2. Train Generator (Relativistic)
            # -----------------------------------------
            optimizer_G.zero_grad()
            
            # Re-calculate logits (Fake now keeps gradients for G)
            # We treat real_logits as constant here (detach)
            real_logits = real_logits.detach() 
            fake_logits = discriminator(gen_imgs) 
            
            # RaGAN Logic for Generator:
            # G tries to fool D. 
            # Loss 1: Real - mean(Fake) -> Label 0 (Real looks Fake compared to Fake??)
            # Ideally: Make Real look less real than Fake.
            g_loss_real = criterion_GAN(real_logits - torch.mean(fake_logits), torch.zeros_like(real_logits))
            # Loss 2: Fake - mean(Real) -> Label 1 (Fake looks Realer than Real avg)
            g_loss_fake = criterion_GAN(fake_logits - torch.mean(real_logits), torch.ones_like(fake_logits))
            
            loss_adversarial = (g_loss_real + g_loss_fake) / 2
            
            # Other Losses
            loss_pixel = criterion_content(gen_imgs, hr_imgs)
            loss_texture = criterion_percep(gen_imgs, hr_imgs)
            
            # Total G Loss
            loss_G = (LAMBDA_CONTENT * loss_pixel) + \
                     (LAMBDA_PERCEP * loss_texture) + \
                     (LAMBDA_ADV * loss_adversarial)
            
            loss_G.backward()
            optimizer_G.step()
            
            # Logging
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
            
        # --- VALIDATION (Mini Check) ---
        generator.eval()
        avg_psnr = 0
        limit = 50 
        
        with torch.no_grad():
            for i, (lr, hr) in enumerate(val_loader):
                if i >= limit: break
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = generator(lr).clamp(0, 1)
                p, s = calculate_metrics(sr, hr)
                avg_psnr += p
                
                if i == 0:
                    save_comparison(lr, sr, hr, f"ESRGAN_ep{epoch+1}")
        
        avg_psnr /= limit
        
        scheduler_G.step()
        scheduler_D.step()
        
        print(f"   -> Val PSNR: {avg_psnr:.2f} dB")
        wandb.log({
            "epoch": epoch,
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "val_psnr": avg_psnr
        })
        
        # Save Model
        os.makedirs("weights", exist_ok=True)
        torch.save(generator.state_dict(), f"weights/ESRGAN_RRDB_ep{epoch+1}.pth")

if __name__ == "__main__":
    train_esrgan()