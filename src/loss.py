import torch
import torch.nn as nn
import torch.nn.functional as F

class NLPDLoss(nn.Module):
    """
    Implementation of the Loss Function from "Enhancing Image Perception Quality..."
    Formula: Loss = 0.7 * MAE + 0.3 * NLPD 
    """
    def __init__(self, device='cpu', n_levels=4):
        super(NLPDLoss, self).__init__()
        self.device = device
        self.n_levels = n_levels
        self.mae = nn.L1Loss()
        
        # 5x5 Gaussian Kernel standard for pyramids (Eq. 8 in paper implies a specific filter P)
        # We use a standard Gaussian as a robust approximation for the pyramid construction.
        self.kernel = self._get_gaussian_kernel().to(device)
        
    def _get_gaussian_kernel(self, size=5, sigma=1.0, channels=3):
        # Create a 2D Gaussian kernel
        x_coord = torch.arange(size)
        x_grid = x_coord.repeat(size).view(size, size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (size - 1)/2.
        variance = sigma**2.
        
        gaussian_kernel = (1./(2.*3.14159*variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance))
        
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    def get_laplacian_pyramid(self, img):
        # Implements the pyramid construction (Eq. 5 and 6) [cite: 164, 167]
        pyramid = []
        current = img
        for _ in range(self.n_levels):
            # 1. Blur (Gaussian Low Pass)
            blurred = F.conv2d(current, self.kernel, padding=2, groups=3)
            # 2. Downsample
            down = blurred[:, :, ::2, ::2]
            # 3. Upsample back
            up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            # 4. Laplacian = Original - Upsampled Blur (High Frequencies)
            diff = current - up
            pyramid.append(diff)
            current = down
        return pyramid

    def forward(self, input, target):
        # 1. Calculate Content Loss: MAE (Eq. 13) 
        loss_mae = self.mae(input, target)
        
        # 2. Calculate Perceptual Loss: NLPD (Eq. 11 & 12) [cite: 189]
        pyr_input = self.get_laplacian_pyramid(input)
        pyr_target = self.get_laplacian_pyramid(target)
        
        loss_nlpd = 0
        for p_in, p_tgt in zip(pyr_input, pyr_target):
            # We minimize the difference between the pyramids
            loss_nlpd += torch.mean(torch.abs(p_in - p_tgt))
        
        # 3. Combine them using the paper's specific weights (Eq. 14) 
        # "Loss = 0.7 * MAE + 0.3 * NLPD"
        total_loss = 0.7 * loss_mae + 0.3 * loss_nlpd
        
        return total_loss