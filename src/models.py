import torch
import torch.nn as nn
import torch.nn.functional as F

# --- COMMON BLOCKS ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for AttentionSR"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        residual = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        residual = self.bn2(residual)
        if self.use_se:
            residual = self.se(residual)
        return x + residual

# --- MODEL 1: SRCNN ---
class SRCNN(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Force CPU for bicubic resize to prevent Mac crash
        x = F.interpolate(x.cpu(), scale_factor=self.scale_factor, mode='bicubic', align_corners=False).to(x.device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# --- MODEL 2 & 3: ResNetSR & AttentionSR ---
class ResNetSR(nn.Module):
    def __init__(self, scale_factor=4, num_residuals=16, use_attention=False):
        super(ResNetSR, self).__init__()
        self.input_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [ResidualBlock(64, use_se=use_attention) for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.output_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        initial = self.prelu(self.input_conv(x))
        residual = self.res_blocks(initial)
        residual = self.bn_mid(self.mid_conv(residual))
        x = initial + residual
        x = self.upsample(x)
        x = self.output_conv(x)
        return x

# --- DISCRIMINATOR (Needed for Phase 2) ---
class Discriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator, self).__init__()
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.classifier(self.net(x))

def get_model(name, scale_factor=4, device='cpu'):
    if name == "SRCNN":
        return SRCNN(scale_factor=scale_factor).to(device)
    elif name == "RESNET":
        return ResNetSR(scale_factor=scale_factor).to(device)
    elif name == "AttentionSR":
        return ResNetSR(scale_factor=scale_factor, use_attention=True).to(device)
    else:
        raise ValueError(f"Unknown architecture: {name}")