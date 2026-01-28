import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from src.dataset import FoodSRDataset
from src.models import get_model, Discriminator
from src.loss import get_loss_function, TVLoss 
from src.metrics import MetricsCalculator
from src.utils import get_gradient_norm, get_layer_grad_ratio, save_checkpoint, get_weight_norm, get_update_ratio

def add_noise(img, sigma=0.15):
    if sigma <= 0: return img
    return img + torch.randn_like(img) * sigma

def train(config=None):
    with wandb.init(config=config) as run:
        cfg = run.config
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Running on {device} | Arch: {cfg.architecture}")

        full_train_ds = FoodSRDataset(split='train', crop_size=200, scale_factor=4)
        
        if cfg.subset < 1.0:
            total_len = len(full_train_ds)
            subset_len = int(total_len * cfg.subset)
            full_train_ds, _ = random_split(full_train_ds, [subset_len, total_len - subset_len])
            print(f"PROXY MODE: Training on {subset_len} images.")

        train_len = int(0.9 * len(full_train_ds))
        val_len = len(full_train_ds) - train_len
        train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

        test_ds = FoodSRDataset(split='test', crop_size=200, scale_factor=4)
        
        if cfg.subset < 1.0:
             test_ds, _ = random_split(test_ds, [int(len(test_ds)*cfg.subset), len(test_ds)-int(len(test_ds)*cfg.subset)])

        print(f"Dataset: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = get_model(cfg.architecture, scale_factor=4, device=device)
        
        if cfg.pretrained_weights:
            model.load_state_dict(torch.load(cfg.pretrained_weights, map_location=device), strict=False)

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        is_gan = cfg.loss_function == 'gan'
        if is_gan:
            discriminator = Discriminator().to(device)
            optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.lr * 0.1, betas=(0.5, 0.999))
            criterion_gan_bce = nn.BCEWithLogitsLoss()
            criterion_content = get_loss_function('mae', device) 
            criterion_percep = get_loss_function('perceptual', device)
            criterion_tv = TVLoss(tv_loss_weight=1).to(device)
        else:
            criterion = get_loss_function(cfg.loss_function, device)

        metrics_calc = MetricsCalculator(device)
        best_psnr = 0.0
        patience_counter = 0

        for epoch in range(cfg.epochs):
            model.train()
            if is_gan: discriminator.train()
            
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.epochs}")
            
            loss_D_item = 0.0
            prob_real = 0.5
            prob_fake = 0.5
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(loop):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                
                if is_gan:
                    if batch_idx % 5 == 0:
                        optimizer_D.zero_grad()
                        fake_imgs = model(lr_imgs).detach()
                        
                        real_logits = discriminator(add_noise(hr_imgs, 0.2))
                        fake_logits = discriminator(add_noise(fake_imgs, 0.2))
                        
                        prob_real = torch.sigmoid(real_logits).mean().item()
                        prob_fake = torch.sigmoid(fake_logits).mean().item()

                        d_loss_real = criterion_gan_bce(real_logits - fake_logits.mean(), torch.tensor(0.9, device=device).expand_as(real_logits))
                        d_loss_fake = criterion_gan_bce(fake_logits - real_logits.mean(), torch.tensor(0.1, device=device).expand_as(fake_logits))
                        loss_D = (d_loss_real + d_loss_fake) / 2
                        loss_D.backward()
                        optimizer_D.step()
                        loss_D_item = loss_D.item()

                    optimizer.zero_grad()
                    fake_imgs = model(lr_imgs)
                    fake_logits_g = discriminator(fake_imgs)
                    real_logits_g = discriminator(hr_imgs).detach()
                    
                    loss_adv = criterion_gan_bce(fake_logits_g - real_logits_g.mean(), torch.ones_like(fake_logits_g))
                    loss_pixel = criterion_content(fake_imgs, hr_imgs)
                    loss = (1e-2 * loss_pixel) + (1.0 * criterion_percep(fake_imgs, hr_imgs)) + (1e-5 * loss_adv) + (2e-5 * criterion_tv(fake_imgs))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                else:
                    optimizer.zero_grad()
                    loss = criterion(model(lr_imgs), hr_imgs)
                    loss.backward()
                    optimizer.step()
                    
                if batch_idx % 100 == 0:
                     current_lr = optimizer.param_groups[0]['lr']
                     update_ratio = get_update_ratio(model, current_lr)
                     grad_norm = get_gradient_norm(model)
                     layer_ratio = get_layer_grad_ratio(model)

                     log_data = {
                        "train_loss": loss.item(),
                        "dynamics/grad_norm": grad_norm,
                        "dynamics/layer_ratio": layer_ratio,
                        "dynamics/update_ratio": update_ratio
                    }
                     
                     if is_gan:
                         log_data.update({
                             "train_loss_D": loss_D_item,
                             "gan_dynamics/prob_real": prob_real,
                             "gan_dynamics/prob_fake": prob_fake
                         })

                     wandb.log(log_data)

            model.eval()
            avg_psnr = 0
            avg_val_loss = 0.0
            
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    
                    avg_psnr += metrics_calc.compute(sr, hr)['psnr']
                    
                    if is_gan:
                        val_loss = criterion_content(sr, hr)
                    else:
                        val_loss = criterion(sr, hr)
                    avg_val_loss += val_loss.item()
                    
            avg_psnr /= len(val_loader)
            avg_val_loss /= len(val_loader)
            
            scheduler.step(avg_psnr)
            print(f"   -> Val PSNR: {avg_psnr:.2f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")
            
            wandb.log({
                "epoch": epoch,
                "val_psnr": avg_psnr,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                patience_counter = 0
                save_checkpoint(model, epoch, f"weights/{cfg.save_name}_best.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= cfg.patience:
                print("Early stopping triggered")
                break

        print("\n🔎 Running Final Test Set Evaluation...")
        model.load_state_dict(torch.load(f"weights/{cfg.save_name}_best.pth", map_location=device))
        model.eval()
        test_metrics = {"psnr": 0, "ssim": 0, "lpips": 0, "nlpd": 0}
        with torch.no_grad():
            for lr, hr in tqdm(test_loader, desc="Testing"):
                lr, hr = lr.to(device), hr.to(device)
                batch_res = metrics_calc.compute(model(lr), hr)
                for k in test_metrics: test_metrics[k] += batch_res[k]
        
        for k in test_metrics: test_metrics[k] /= len(test_loader)
        print(f"🏆 Final Test Results: {test_metrics}")
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="SRCNN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0004) 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss_function", type=str, default="nlpd") 
    parser.add_argument("--subset", type=float, default=1.0) 
    parser.add_argument("--pretrained_weights", type=str, default="")
    parser.add_argument("--patience", type=int, default=5) 
    parser.add_argument("--save_name", type=str, default="model_best") 
    args = parser.parse_args()
    
    train(config=vars(args))