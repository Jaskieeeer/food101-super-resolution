import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

# Local Imports
from src.dataset import FoodSRDataset
from src.models import get_model, Discriminator
from src.loss import get_loss_function, TVLoss 
from src.metrics import MetricsCalculator
from src.utils import get_gradient_norm, get_layer_grad_ratio, save_checkpoint, get_weight_norm

# --- HELPER: NOISE INJECTION (For GAN Stability) ---
def add_noise(img, sigma=0.15):
    """Adds random Gaussian noise to blindfold the Discriminator"""
    if sigma <= 0: return img
    noise = torch.randn_like(img) * sigma
    return img + noise

# --- HELPER: ICNR SURGERY (The Checkerboard Fix) ---
def icnr_init(tensor, scale_factor=2):
    """
    Modifies a ConvTranspose or PixelShuffle weight tensor to behave 
    like a Nearest Neighbor resize (smooth) instead of random noise.
    """
    out_c, in_c, h, w = tensor.shape
    if out_c % (scale_factor**2) != 0: return tensor # Skip if shapes mismatch
    
    # 1. Create a smaller sub-kernel
    sub_kernel = torch.zeros(out_c // (scale_factor**2), in_c, h, w)
    nn.init.kaiming_normal_(sub_kernel)
    sub_kernel = sub_kernel.transpose(0, 1).contiguous()
    
    # 2. "Lock" the sub-pixels by repeating the kernel
    kernel = sub_kernel.view(in_c, sub_kernel.shape[1], -1)
    kernel = kernel.repeat(1, scale_factor**2, 1) 
    
    return kernel.view(in_c, out_c, h, w).transpose(0, 1)

def train(config=None):
    with wandb.init(config=config) as run:
        cfg = run.config
        
        # --- SETUP ---
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Running on {device} | Arch: {cfg.architecture} | Loss: {cfg.loss_function}")

        # --- DATASET ---
        full_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=4)
        
        if cfg.subset < 1.0:
            subset_len = int(len(full_ds) * cfg.subset)
            full_ds, _ = random_split(full_ds, [subset_len, len(full_ds) - subset_len])
            print(f"⚠️ PROXY MODE: Using {subset_len} images.")

        train_size = int(0.9 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # --- MODEL ---
        model = get_model(cfg.architecture, scale_factor=4, device=device)
        
        # Load Phase 1 weights if provided
        if cfg.pretrained_weights and os.path.exists(cfg.pretrained_weights):
            print(f"🔄 Loading weights from {cfg.pretrained_weights}")
            model.load_state_dict(torch.load(cfg.pretrained_weights, map_location=device), strict=False)

            # ============================================================
            # 💉 WEIGHT SURGERY (Remove Checkerboard Artifacts)
            # ============================================================
            print("💉 Performing ICNR surgery on upsampler to prevent checkerboard...")
            surgery_count = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # We look for the upsampling convolution weights inside the model
                    if "upsample" in name and "weight" in name and param.dim() == 4:
                        # Reset this specific layer to be "locked" (Nearest Neighbor)
                        # We assume scale_factor=2 for the individual upsample layers
                        param.data.copy_(icnr_init(param.data, scale_factor=2))
                        surgery_count += 1
            print(f"✅ Surgery complete. Fixed {surgery_count} layers. Sub-pixels are now correlated.")
            # ============================================================

        # --- OPTIMIZER ---
        # Initialize optimizer AFTER surgery so it sees the clean weights
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        # --- GAN SETUP ---
        is_gan = cfg.loss_function == 'gan'
        if is_gan:
            discriminator = Discriminator().to(device)
            # Discriminator learns slower (0.5x LR)
            optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.lr * 0.5, betas=(0.5, 0.999))
            
            criterion_gan_bce = nn.BCEWithLogitsLoss()
            criterion_content = get_loss_function('mae', device) 
            criterion_percep = get_loss_function('perceptual', device)
            # TV Loss helps smooth out any remaining noise
            criterion_tv = TVLoss(tv_loss_weight=1).to(device)

        # --- STANDARD SETUP ---
        else:
            criterion = get_loss_function(cfg.loss_function, device)

        metrics_calc = MetricsCalculator(device)
        best_psnr = 0.0
        patience_counter = 0

        # --- TRAINING LOOP ---
        for epoch in range(cfg.epochs):
            model.train()
            if is_gan: discriminator.train()
            
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.epochs}")
            
            # Tracker for D loss (avoids UnboundLocalError)
            loss_D_item = 0.0
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(loop):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                
                # --- A. GAN TRAINING PATH ---
                if is_gan:
                    # 1. Train Discriminator (Only every 2nd step)
                    if batch_idx % 2 == 0:
                        optimizer_D.zero_grad()
                        fake_imgs = model(lr_imgs).detach()
                        
                        # Add noise to inputs to stabilize D
                        noisy_hr = add_noise(hr_imgs, sigma=0.05)   # Add 5% noise to real
                        noisy_fake = add_noise(fake_imgs, sigma=0.05) # Add 5% noise to fake
                        
                        real_logits = discriminator(noisy_hr)
                        fake_logits = discriminator(noisy_fake)
                        
                        # Soft Labels (0.9 / 0.1)
                        valid = torch.tensor(0.9, device=device).expand_as(real_logits)
                        fake = torch.tensor(0.1, device=device).expand_as(fake_logits)
                        
                        # Relativistic Logic
                        d_loss_real = criterion_gan_bce(real_logits - fake_logits.mean(), valid)
                        d_loss_fake = criterion_gan_bce(fake_logits - real_logits.mean(), fake)
                        loss_D = (d_loss_real + d_loss_fake) / 2
                        
                        loss_D.backward()
                        optimizer_D.step()
                        
                        loss_D_item = loss_D.item()

                    # 2. Train Generator (Every step)
                    optimizer.zero_grad()
                    fake_imgs = model(lr_imgs) # Re-compute
                    
                    # Fool Discriminator (No noise here! G needs accurate gradients)
                    fake_logits_g = discriminator(fake_imgs)
                    real_logits_g = discriminator(hr_imgs).detach()
                    
                    # Generator Losses
                    loss_adv = criterion_gan_bce(fake_logits_g - real_logits_g.mean(), torch.ones_like(fake_logits_g))
                    loss_pixel = criterion_content(fake_imgs, hr_imgs)
                    loss_perc = criterion_percep(fake_imgs, hr_imgs)
                    loss_tv = criterion_tv(fake_imgs)
                    
                    # TOTAL LOSS (Tuned Weights)
                    # 1e-4 Adv Weight is safer to avoid mesh artifacts
                    loss = (1e-2 * loss_pixel) + (1.0 * loss_perc) + (1e-5 * loss_adv) + (2e-5 * loss_tv)
                    
                    loss.backward()
                    
                    if not torch.isfinite(loss):
                        print(f"💥 GENERATOR EXPLODED: {loss.item()}. Stopping run.")
                        wandb.log({"status": "exploded"})
                        run.finish()
                        return

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
                    
                    grad_norm = get_gradient_norm(model)
                    weight_norm = get_weight_norm(model)  
                    layer_ratio = get_layer_grad_ratio(model)
                    
                    optimizer.step()
                    
                    wandb.log({
                        "train_loss_G": loss.item(), 
                        "train_loss_D": loss_D_item,
                        "train_loss_TV": loss_tv.item(),
                        "dynamics/D_real": torch.sigmoid(real_logits_g).mean().item(),
                        "dynamics/D_fake": torch.sigmoid(fake_logits_g).mean().item(),
                        "dynamics/grad_norm": grad_norm,
                        "dynamics/weight_norm": weight_norm,
                        "dynamics/layer_ratio": layer_ratio
                    })

                # --- B. STANDARD TRAINING PATH ---
                else:
                    optimizer.zero_grad()
                    sr_imgs = model(lr_imgs)
                    loss = criterion(sr_imgs, hr_imgs)
                    
                    if not torch.isfinite(loss):
                        print(f"💥 LOSS EXPLOSION: {loss.item()}.")
                        run.finish()
                        return
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
                    
                    grad_norm = get_gradient_norm(model)
                    weight_norm = get_weight_norm(model)
                    layer_ratio = get_layer_grad_ratio(model)
                    
                    optimizer.step()
                    
                    wandb.log({
                        "train_loss": loss.item(),
                        "dynamics/grad_norm": grad_norm,
                        "dynamics/layer_ratio": layer_ratio,
                        "dynamics/weight_norm": weight_norm 
                    })

            # --- VALIDATION ---
            model.eval()
            avg_metrics = {"psnr": 0, "ssim": 0, "lpips": 0, "nlpd": 0}
            
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    batch_metrics = metrics_calc.compute(sr, hr)
                    for k in avg_metrics:
                        avg_metrics[k] += batch_metrics[k]
            
            for k in avg_metrics: avg_metrics[k] /= len(val_loader)
            
            scheduler.step(avg_metrics['psnr'])
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"   -> Val PSNR: {avg_metrics['psnr']:.2f} | SSIM: {avg_metrics['ssim']:.4f} | LR: {current_lr}")
            
            wandb.log({
                "epoch": epoch,
                "val_psnr": avg_metrics['psnr'],
                "val_ssim": avg_metrics['ssim'],
                "val_lpips": avg_metrics['lpips'],
                "val_nlpd": avg_metrics['nlpd'],
                "lr": current_lr
            })
            
            if avg_metrics['psnr'] > best_psnr or is_gan: 
                best_psnr = avg_metrics['psnr']
                patience_counter = 0
                
                save_path_G = f"weights/{cfg.save_name}.pth"
                save_checkpoint(model, epoch, save_path_G)
                wandb.save(save_path_G)
                
                if is_gan:
                    save_path_D = f"weights/{cfg.save_name}_D.pth"
                    torch.save(discriminator.state_dict(), save_path_D)
                    print(f"   🔥 Saved G and D models.")
                else:
                    print(f"   🔥 New Best PSNR: {best_psnr:.2f} (Saved)")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement. Patience: {patience_counter}/{cfg.patience}")

            if patience_counter >= cfg.patience:
                print(f"🛑 EARLY STOPPING at Epoch {epoch}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="SRCNN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss_function", type=str, default="mae") 
    parser.add_argument("--subset", type=float, default=1.0) 
    parser.add_argument("--pretrained_weights", type=str, default="")
    parser.add_argument("--patience", type=int, default=5) 
    parser.add_argument("--max_norm", type=float, default=1.0) 
    parser.add_argument("--save_name", type=str, default="model_best") 
    args = parser.parse_args()
    
    train(config=vars(args))