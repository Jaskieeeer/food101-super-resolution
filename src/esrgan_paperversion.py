import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. Configuration & Hyperparameters
# ==============================================================================
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'hr_height': 200,          # User requested 200px
    'hr_width': 200,
    'scale_factor': 4,         # Standard scale x4
    'batch_size': 16,          # Mini-batch size 16
    'num_workers': 4,
    'lr_pretrain': 2e-4,       # Pre-training LR
    'lr_gan': 1e-4,            # GAN training LR
    'epochs_pretrain': 10,     # Adjust based on needs (Paper implies convergence)
    'epochs_gan': 20,          # Adjust based on needs
    'lambda_adv': 5e-3,        # Lambda for adversarial loss
    'eta_pixel': 1e-2,         # Eta for pixel loss (L1)
    'data_root': './data',     # Where Food-101 will download
}

# ==============================================================================
# 2. Dataset Preparation (Food-101 for Super-Resolution)
# ==============================================================================
class Food101SRDataset(Dataset):
    def __init__(self, root, split='train'):
        # We use torchvision's built-in Food101 handling for convenience
        self.base_dataset = datasets.Food101(root=root, split=split, download=True)
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop((CONFIG['hr_height'], CONFIG['hr_width'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90), # Augmentations mentioned in
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # We ignore the class label for SR tasks
        img, _ = self.base_dataset[idx]
        
        # Ensure image is large enough for cropping
        if img.size[0] < CONFIG['hr_width'] or img.size[1] < CONFIG['hr_height']:
            img = transforms.Resize((CONFIG['hr_height'], CONFIG['hr_width']))(img)

        # 1. Get High Resolution (HR) Crop
        hr_image = self.hr_transform(img)

        # 2. Generate Low Resolution (LR) via Bicubic Downsampling
        # Paper uses MATLAB bicubic, we use PyTorch's bicubic approximation
        lr_image = F.interpolate(
            hr_image.unsqueeze(0), 
            scale_factor=1/CONFIG['scale_factor'], 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)

        return lr_image, hr_image

# ==============================================================================
# 3. Model Architecture: Generator (RRDBNet)
# ==============================================================================
class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 convolution layers.
    Paper reference: RRDB uses dense connections.
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, nf: number of filters
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Initialization trick from paper: smaller initialization
        # (Implemented implicitly by default PyTorch init or can be refined)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling (usually 0.2)
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block.
    Reference: Fig 4 in paper.
    """
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Residual scaling
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """
    ESRGAN Generator.
    - Removes Batch Normalization.
    - Uses RRDB as basic block.
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        
        # Low-level feature extraction
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # RRDB Body (nb = 23 blocks typical for ESRGAN)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk # Global residual connection

        # Upsampling x2
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # Upsampling x2 (Total x4)
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

# ==============================================================================
# 4. Model Architecture: Discriminator (VGG-Style)
# ==============================================================================
class Discriminator(nn.Module):
    """
    Standard VGG-style discriminator structure. 
    Paper modifies the *loss* to be Relativistic, not necessarily the structure.
    """
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator, self).__init__()
        
        # Strided convolutions to downsample
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf, nf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf * 2, nf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nf * 8, nf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(nf * 8 * (CONFIG['hr_height'] // 16) * (CONFIG['hr_width'] // 16), 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        feat = self.net(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out

# ==============================================================================
# 5. Loss Functions (Perceptual & Relativistic)
# ==============================================================================
class FeatureExtractor(nn.Module):
    """
    Extracts features from VGG19.
    Paper Improvement: Use features *before* activation.
    Specifically VGG19 conv5_4 (34th layer in standard torchvision vgg19 features).
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        # Extract up to conv5_4 (before activation)
        # Index 34 is conv5_4 in torchvision VGG19 structure
        self.features = nn.Sequential(*list(vgg19.features.children())[:35]) 
        
        # No grad for VGG
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # VGG expects normalization
        return self.features(x)

class ESRGANLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(ESRGANLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.l1_loss = nn.L1Loss()
        
    def forward(self, hr_real, hr_fake, d_out_real, d_out_fake):
        # 1. Perceptual Loss (Feature space L1)
        # "Features before activation"
        real_feat = self.feature_extractor(hr_real).detach()
        fake_feat = self.feature_extractor(hr_fake)
        loss_percep = self.l1_loss(fake_feat, real_feat)

        # 2. Adversarial Loss (Relativistic Average GAN)
        # L_G^Ra = -E[log(1 - D_Ra(xr, xf))] - E[log(D_Ra(xf, xr))]
        
        # Calculate relativistic output
        real_mean = torch.mean(d_out_real)
        fake_mean = torch.mean(d_out_fake)
        
        # D_Ra(xf, xr) = sigmoid(C(xf) - mean(C(xr)))
        # D_Ra(xr, xf) = sigmoid(C(xr) - mean(C(xf)))
        
        pred_fake_relativistic = torch.sigmoid(d_out_fake - real_mean)
        pred_real_relativistic = torch.sigmoid(d_out_real - fake_mean)
        
        loss_adv = -torch.mean(torch.log(pred_fake_relativistic + 1e-10)) \
                   -torch.mean(torch.log(1 - pred_real_relativistic + 1e-10)) # Symmetrical form

        # 3. Content Loss (Pixel-wise L1)
        loss_content = self.l1_loss(hr_fake, hr_real)

        # Total Loss formulation
        # L_G = L_percep + lambda * L_adv + eta * L_1
        total_loss = loss_percep + (CONFIG['lambda_adv'] * loss_adv) + (CONFIG['eta_pixel'] * loss_content)
        
        return total_loss, loss_percep, loss_adv, loss_content

# ==============================================================================
# 6. Training Routine
# ==============================================================================
def train():
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Data Loading ---
    print(f"Loading Food-101 Dataset to {CONFIG['data_root']}...")
    dataset = Food101SRDataset(root=CONFIG['data_root'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=True, num_workers=CONFIG['num_workers'])

    # --- Model Initialization ---
    generator = RRDBNet().to(CONFIG['device'])
    discriminator = Discriminator().to(CONFIG['device'])
    feature_extractor = FeatureExtractor().to(CONFIG['device'])
    
    # --- Optimizers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['lr_pretrain'], betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['lr_gan'], betas=(0.9, 0.999))

    criterion_pixel = nn.L1Loss()
    criterion_gan = ESRGANLoss(feature_extractor)

    # ==========================================================================
    # Phase 1: PSNR-Oriented Pre-training (L1 Loss only)
    # Reference: Paper Section 4.1 "First, we train a PSNR-oriented model"
    # ==========================================================================
    print(">>> Starting Phase 1: PSNR-oriented Pre-training...")
    generator.train()
    
    for epoch in range(CONFIG['epochs_pretrain']):
        loop = tqdm(dataloader, desc=f"Pre-train Epoch {epoch+1}/{CONFIG['epochs_pretrain']}")
        for i, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(CONFIG['device']), hr.to(CONFIG['device'])

            optimizer_G.zero_grad()
            fake_hr = generator(lr)
            loss = criterion_pixel(fake_hr, hr)
            loss.backward()
            optimizer_G.step()
            
            loop.set_postfix(loss=loss.item())

    # Save pre-trained model (Optional: Network Interpolation source 1)
    torch.save(generator.state_dict(), "weights/psnr_pretrained.pth")
    
    # ==========================================================================
    # Phase 2: ESRGAN Training (Perceptual + Adversarial + Content)
    # Reference: Paper Section 4.1 "Use trained PSNR model as initialization"
    # ==========================================================================
    print(">>> Starting Phase 2: GAN Training (ESRGAN)...")
    
    # Update Learning Rate for GAN phase
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = CONFIG['lr_gan']

    for epoch in range(CONFIG['epochs_gan']):
        loop = tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{CONFIG['epochs_gan']}")
        for i, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(CONFIG['device']), hr.to(CONFIG['device'])

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            fake_hr = generator(lr).detach() # Detach to avoid G gradients
            
            d_out_real = discriminator(hr)
            d_out_fake = discriminator(fake_hr)
            
            # Relativistic Discriminator Loss
            # L_D^Ra = -E[log(D_Ra(xr, xf))] - E[log(1 - D_Ra(xf, xr))]
            real_mean = torch.mean(d_out_real)
            fake_mean = torch.mean(d_out_fake)
            
            # Real is "more realistic" than fake?
            d_ra_real = torch.sigmoid(d_out_real - fake_mean)
            # Fake is "less realistic" than real?
            d_ra_fake = torch.sigmoid(d_out_fake - real_mean)
            
            loss_D = -torch.mean(torch.log(d_ra_real + 1e-10)) \
                     -torch.mean(torch.log(1 - d_ra_fake + 1e-10))
            
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Re-generate fake images (with gradients for G this time)
            fake_hr = generator(lr)
            
            # Get discriminator outputs again
            d_out_real = discriminator(hr).detach() # Detach real, we only train G here
            d_out_fake = discriminator(fake_hr)
            
            total_loss, l_percep, l_adv, l_content = criterion_gan(hr, fake_hr, d_out_real, d_out_fake)
            
            total_loss.backward()
            optimizer_G.step()

            loop.set_postfix(
                D_loss=loss_D.item(), 
                G_loss=total_loss.item(), 
                Adv=l_adv.item()
            )

        # Save Checkpoint
        torch.save(generator.state_dict(), f"weights/esrgan_epoch_{epoch+1}.pth")
        
        # Save Sample
        with torch.no_grad():
            sample_lr = lr[0]
            sample_fake = fake_hr[0]
            sample_hr = hr[0]
            # Upsample LR simply for comparison visualization
            sample_lr_up = F.interpolate(sample_lr.unsqueeze(0), scale_factor=4).squeeze(0)
            img_grid = torch.cat((sample_lr_up, sample_fake, sample_hr), 2)
            save_image(img_grid, f"results/epoch_{epoch+1}.png", normalize=True)

if __name__ == "__main__":
    train()