import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # VGG19 features up to layer 35
        self.vgg = vgg19(weights='DEFAULT').features[:35].eval().to(device)
        for p in self.vgg.parameters(): 
            p.requires_grad = False
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(self.vgg(input), self.vgg(target))

class NLPDLoss(nn.Module):
    def __init__(self, device='cpu', n_levels=4):
        super(NLPDLoss, self).__init__()
        self.n_levels = n_levels
        self.mae = nn.L1Loss()
        
        # 1. Create the kernel
        kernel = self._get_gaussian_kernel()
        
        # 2. CRITICAL FIX: Register as buffer
        # This creates 'self.kernel' and ensures it moves to GPU automatically
        self.register_buffer("kernel", kernel) 
        
    def _get_gaussian_kernel(self, size=5, sigma=1.0, channels=3):
        x_coord = torch.arange(size)
        x_grid = x_coord.repeat(size).view(size, size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (size - 1)/2.
        variance = sigma**2.
        
        gaussian_kernel = (1./(2.*3.14159*variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    def get_laplacian_pyramid(self, img):
        pyramid = []
        current = img
        for _ in range(self.n_levels):
            # self.kernel is now automatically on the correct device
            blurred = F.conv2d(current, self.kernel, padding=2, groups=3)
            down = blurred[:, :, ::2, ::2]
            up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            diff = current - up
            pyramid.append(diff)
            current = down
        return pyramid

    def forward(self, input, target):
        loss_mae = self.mae(input, target)
        
        pyr_input = self.get_laplacian_pyramid(input)
        pyr_target = self.get_laplacian_pyramid(target)
        
        loss_nlpd = 0
        for p_in, p_tgt in zip(pyr_input, pyr_target):
            loss_nlpd += torch.mean(torch.abs(p_in - p_tgt))
        
        return 0.7 * loss_mae + 0.3 * loss_nlpd

def get_loss_function(name, device):
    name = name.lower()
    if name == "mae":
        return nn.L1Loss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "perceptual":
        return PerceptualLoss(device)
    elif name == "nlpd":
        return NLPDLoss(device=device).to(device)
    else:
        raise ValueError(f"❌ Unknown loss function: {name}")