import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def icnr_init(layer, scale_factor=2):
    tensor = layer.weight.data
    out_c, in_c, h, w = tensor.shape
    
    if out_c % (scale_factor**2) != 0: 
        return

    sub_kernel = torch.zeros(out_c // (scale_factor**2), in_c, h, w)
    nn.init.kaiming_normal_(sub_kernel)
    sub_kernel = sub_kernel.transpose(0, 1).contiguous()
    
    kernel = sub_kernel.view(in_c, sub_kernel.shape[1], -1)
    kernel = kernel.repeat(1, scale_factor**2, 1) 
    
    layer.weight.data.copy_(kernel.view(in_c, out_c, h, w).transpose(0, 1))
    
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
class SEBlock(nn.Module):
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

class AttentionResidualBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.se = SEBlock(channels)
        
        self.res_scale = 0.1 

    def forward(self, x):
        residual = self.conv2(self.prelu(self.conv1(x)))
        
        residual = self.se(residual)
        
        return x + (residual * self.res_scale)

class SRCNN(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4, hidden_dim=64):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, hidden_dim, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_dim, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.interpolate(x.cpu(), scale_factor=self.scale_factor, mode='bicubic', align_corners=False).to(x.device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ResNetSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=64, num_residuals=16):
        super(ResNetSR, self).__init__()
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [ResidualBlock(num_channels, use_se=False) for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.mid_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_channels)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.output_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        icnr_init(self.upsample[0], scale_factor=2)
        icnr_init(self.upsample[3], scale_factor=2)

    def forward(self, x):
        initial = self.prelu(self.input_conv(x))
        residual = self.res_blocks(initial)
        residual = self.bn_mid(self.mid_conv(residual))
        x = initial + residual
        x = self.upsample(x)
        x = self.output_conv(x)
        return x

class AttentionSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=64, num_residuals=32):
        super(AttentionSR, self).__init__()
        
        self.input_conv = nn.Conv2d(3, num_channels, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = [AttentionResidualBlock(num_channels) for _ in range(num_residuals)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.mid_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.output_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        icnr_init(self.upsample[0], scale_factor=2)
        icnr_init(self.upsample[3], scale_factor=2)

    def forward(self, x):
        initial = self.prelu(self.input_conv(x))
        residual = self.res_blocks(initial)
        residual = self.mid_conv(residual)
        
        x = initial + residual
        
        x = self.upsample(x)
        x = self.output_conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator, self).__init__()
        
        def sn_block(in_f, out_f, kernel, stride, padding, bias=True, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_f, out_f, kernel, stride, padding, bias=bias))]
            if bn: layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *sn_block(in_nc, nf, 3, 1, 1, bias=True, bn=False),
            *sn_block(nf, nf*2, 3, 2, 1, bias=False),
            *sn_block(nf*2, nf*4, 3, 2, 1, bias=False),
            *sn_block(nf*4, nf*8, 3, 2, 1, bias=False),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(nf * 8, 100)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(100, 1))
        )

    def forward(self, x):
        return self.classifier(self.net(x))

def get_model(name, scale_factor=4, device='cpu'):
    if name == "SRCNN":
        return SRCNN(scale_factor=scale_factor, hidden_dim=64).to(device)
    elif name == "RESNET":
        return ResNetSR(scale_factor=scale_factor, num_residuals=32, num_channels=96).to(device)
    elif name == "AttentionSR":
        return AttentionSR(scale_factor=scale_factor, num_residuals=32, num_channels=96).to(device)
    else:
        raise ValueError(f"Unknown architecture: {name}")