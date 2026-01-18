import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

sys.path.append(os.getcwd())


from src.dataset import FoodSRDataset
# CHANGED: Imported SRCNN instead of ResNetSR
from src.models import SRCNN, Discriminator 
from src.loss import NLPDLoss
from src.utils import calculate_metrics, save_comparison

# --- CONFIGURATION (UPDATED) ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# YOUR SPECIFIC SETTINGS
LR_G = 0.001          # <--- High LR for SRCNN
LR_D = 0.0001         # <--- Keep Discriminator slow/stable
EPOCHS = 20           
BATCH_SIZE = 8        # <--- Small batch for generalization
RUN_NAME = "SRGAN_SRCNN_PaperLoss_LR0.001_BS8"

def train_gan():
    # 1. Initialize WandB
    wandb.init(project="food101-sr", name=RUN_NAME, config={
        "type": "SRGAN",
        "generator": "SRCNN",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate_G": LR_G,
        "learning_rate_D": LR_D,
        "split": "80/10/10"
    })
    
    # 2. Init Models
    # CHANGED: Using SRCNN as the Generator
    generator = SRCNN(scale_factor=2).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # 3. Init Optimizers
    # 3. Init Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    
    # --- NEW: SCHEDULERS ---
    # Decays LR by 50% (gamma=0.5) every 5 epochs
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)
    # 4. Losses
    criterion_GAN = nn.BCELoss() # Real vs Fake
    criterion_content = NLPDLoss(device=DEVICE) # Texture/Pixel match
    
    # 5. Data Splitting (80% Train / 10% Val / 10% Test)
    full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    total_len = len(full_ds)
    train_len = int(0.8 * total_len)
    val_len   = int(0.1 * total_len)
    test_len  = total_len - train_len - val_len 
    
    print(f"📊 Splitting Data: Train={train_len}, Val={val_len}, Test={test_len}")
    
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    # num_workers=0 is safer on Mac M4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"🚀 Starting SRGAN (SRCNN) Training on {DEVICE}...")
    
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
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_imgs = generator(lr_imgs)
            
            loss_adversarial = criterion_GAN(discriminator(gen_imgs), valid)
            loss_content = criterion_content(gen_imgs, hr_imgs)
            
            # Weighted Loss: Content (1.0) + Adversarial (0.001)
            loss_G = loss_content + (1e-3 * loss_adversarial)
            
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            real_loss = criterion_GAN(discriminator(hr_imgs), valid)
            fake_loss = criterion_GAN(discriminator(gen_imgs.detach()), fake)
            
            loss_D = (real_loss + fake_loss) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
            
        # --- VALIDATION LOOP ---
        generator.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for i, (lr, hr) in enumerate(val_loader):
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = generator(lr).clamp(0, 1)
                
                p, s = calculate_metrics(sr, hr)
                avg_psnr += p
                avg_ssim += s
                
                if i == 0:
                    save_comparison(lr, sr, hr, f"GAN_SRCNN_ep{epoch+1}")

        avg_psnr /= len(val_loader)
        avg_ssim /= len(val_loader)
        
        print(f"   -> Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f}")
        
        scheduler_G.step()
        scheduler_D.step()
        
        current_lr_G = optimizer_G.param_groups[0]['lr']
        print(f"   -> LR Adjusted to: {current_lr_G:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "lr_G": current_lr_G # Log this to see the stairs in WandB
        })
        
        # Save Checkpoint
        os.makedirs("weights", exist_ok=True)
        torch.save(generator.state_dict(), f"weights/SRGAN_SRCNN_ep{epoch+1}.pth")

    # --- FINAL TEST EVALUATION ---
    print(f"\n🧪 Evaluation on HELD-OUT TEST SET ({test_len} images)...")
    generator.eval()
    test_psnr = 0
    test_ssim = 0
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = generator(lr).clamp(0, 1)
            
            p, s = calculate_metrics(sr, hr)
            test_psnr += p
            test_ssim += s
            
            if i < 3:
                save_comparison(lr, sr, hr, f"GAN_SRCNN_TEST_sample_{i}")

    final_psnr = test_psnr / len(test_loader)
    final_ssim = test_ssim / len(test_loader)
    
    print(f"🏆 FINAL TEST RESULTS: PSNR {final_psnr:.2f} dB | SSIM {final_ssim:.4f}")
    
    wandb.log({
        "test_psnr": final_psnr,
        "test_ssim": final_ssim
    })
    wandb.finish()

if __name__ == "__main__":
    train_gan()