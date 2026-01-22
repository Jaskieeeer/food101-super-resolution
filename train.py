import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import os

# Local Imports
from src.dataset import FoodSRDataset
from src.models import get_model, Discriminator
from src.loss import get_loss_function
from src.metrics import MetricsCalculator
from src.utils import get_gradient_norm, get_layer_grad_ratio, save_checkpoint, get_weight_norm

def train(config=None):
    with wandb.init(config=config) as run:
        cfg = run.config
        
        # --- SETUP ---
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Running on {device} | Arch: {cfg.architecture} | Loss: {cfg.loss_function}")

        # --- DATASET (With Proxy/Subset Logic) ---
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
        
        # Optional: Load weights if warming up for GAN
        if cfg.pretrained_weights and os.path.exists(cfg.pretrained_weights):
            print(f"🔄 Loading weights from {cfg.pretrained_weights}")
            model.load_state_dict(torch.load(cfg.pretrained_weights, map_location=device))

        # --- OPTIMIZER ---
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        # --- GAN SETUP ---
        is_gan = cfg.loss_function == 'gan'
        if is_gan:
            discriminator = Discriminator().to(device)
            optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.lr)
            criterion_gan_bce = nn.BCEWithLogitsLoss()
            # For GAN, we combine pixel/perceptual + adv
            criterion_content = get_loss_function('mae', device) 
            criterion_percep = get_loss_function('perceptual', device)

        # --- STANDARD LOSS ---
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
            
            for lr_imgs, hr_imgs in loop:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                
                # --- A. GAN TRAINING PATH ---
                if is_gan:
                    # 1. Train Discriminator
                    optimizer_D.zero_grad()
                    fake_imgs = model(lr_imgs).detach()
                    
                    real_logits = discriminator(hr_imgs)
                    fake_logits = discriminator(fake_imgs)
                    
                    # Relativistic Logic
                    d_loss_real = criterion_gan_bce(real_logits - fake_logits.mean(), torch.ones_like(real_logits))
                    d_loss_fake = criterion_gan_bce(fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))
                    loss_D = (d_loss_real + d_loss_fake) / 2
                    
                    loss_D.backward()
                    optimizer_D.step()

                    # 2. Train Generator
                    optimizer.zero_grad()
                    fake_imgs = model(lr_imgs) # Re-compute for G gradients
                    
                    # Fool Discriminator
                    fake_logits_g = discriminator(fake_imgs)
                    real_logits_g = discriminator(hr_imgs).detach()
                    loss_adv = criterion_gan_bce(fake_logits_g - real_logits_g.mean(), torch.ones_like(fake_logits_g))
                    
                    loss_pixel = criterion_content(fake_imgs, hr_imgs)
                    loss_perc = criterion_percep(fake_imgs, hr_imgs)
                    
                    # Weighted Sum (ESRGAN Paper style)
                    loss = (1e-2 * loss_pixel) + (1.0 * loss_perc) + (5e-3 * loss_adv)
                    
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
                        "train_loss_D": loss_D.item(),
                        "dynamics/D_real": torch.sigmoid(real_logits).mean().item(),
                        "dynamics/D_fake": torch.sigmoid(fake_logits).mean().item(),
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
                        print(f"💥 LOSS EXPLOSION DETECTED: {loss.item()}. Stopping run to save time.")
                        wandb.log({"status": "exploded"})
                        run.finish() # Cleanly end the wandb run
                        return # Exit the function
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
                    # Dynamics
                    grad_norm = get_gradient_norm(model)
                    layer_ratio = get_layer_grad_ratio(model)
                    weight_norm = get_weight_norm(model)
                    
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
            
            # Average over batches
            for k in avg_metrics: avg_metrics[k] /= len(val_loader)
            
            # Update LR Scheduler
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
            
            # Save
            if avg_metrics['psnr'] > best_psnr:
                best_psnr = avg_metrics['psnr']
                patience_counter = 0
                # Save the winner
                save_path = f"weights/{cfg.save_name}.pth"
                save_checkpoint(model, epoch, save_path)
                wandb.save(save_path) # Cloud backup
                print(f"   🔥 New Best PSNR: {best_psnr:.2f} (Saved)")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement. Patience: {patience_counter}/{cfg.patience}")

            if patience_counter >= cfg.patience:
                print(f"🛑 EARLY STOPPING TRIGGERED at Epoch {epoch}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="SRCNN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss_function", type=str, default="mae") # mae, mse, perceptual, nlpd, gan
    parser.add_argument("--subset", type=float, default=1.0) # 0.1 for Proxy Sweep
    parser.add_argument("--pretrained_weights", type=str, default="")
    parser.add_argument("--patience", type=int, default=5) 
    parser.add_argument("--max_norm", type=float, default=1.0) 
    parser.add_argument("--save_name", type=str, default="model_best") 
    args = parser.parse_args()
    
    train(config=vars(args))