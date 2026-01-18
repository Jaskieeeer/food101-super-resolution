import torch
import torch.nn as nn

# --- MODEL 1: SRCNN (The Baseline) ---
class SRCNN(nn.Module):
    def __init__(self, scale_factor=2):
        super(SRCNN, self).__init__()
        
        # FIXED: Add internal upsampling to match Target size
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        
        # Layer 1: Feature Extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Non-linear Mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        # 1. Upscale first (100x100 -> 200x200)
        x = self.upsample(x)
        
        # 2. Refine with CNN
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# --- MODEL 2: ResNetSR (The Advanced Model) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        residual = self.bn2(residual)
        return x + residual

class ResNetSR(nn.Module):
    def __init__(self, scale_factor=2, num_residuals=4):
        super(ResNetSR, self).__init__()
        
        self.input_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [ResidualBlock(64) for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upsampling (PixelShuffle)
        self.upsample_conv = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.output_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        initial = self.prelu(self.input_conv(x))
        residual = self.res_blocks(initial)
        residual = self.bn_mid(self.mid_conv(residual))
        x = initial + residual
        x = self.pixel_shuffle(self.upsample_conv(x))
        x = self.prelu(x)
        x = self.output_conv(x)
        return x
    
# --- MODEL 3: Attention-Augmented ResNet (A-ResNet) ---
# Adds "Squeeze-and-Excitation" (SE) blocks to focus on texture.
# Counts as: "Architecture Tuning" & "Original/Ambitious Architecture"

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    It learns which channels are important and scales them up.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionResidualBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # The Attention Mechanism
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        residual = self.bn2(residual)
        
        # Apply Attention to the residual
        residual = self.se(residual)
        
        return x + residual

class AttentionSR(nn.Module):
    def __init__(self, scale_factor=2, num_residuals=4):
        super(AttentionSR, self).__init__()
        
        self.input_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        # Use the new Attention Blocks
        res_blocks = [AttentionResidualBlock(64) for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upsampling
        self.upsample_conv = nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.output_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
    def forward(self, x):
        initial = self.prelu(self.input_conv(x))
        residual = self.res_blocks(initial)
        residual = self.bn_mid(self.mid_conv(residual))
        x = initial + residual
        x = self.pixel_shuffle(self.upsample_conv(x))
        x = self.prelu(x)
        x = self.output_conv(x)
        return x