import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from src.loss import NLPDLoss

class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net='alex', verbose=False).to(device)
        self.nlpd = NLPDLoss(device=device, channels=3).to(device)

    @torch.no_grad()
    def compute(self, sr, hr):
        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)
        
        score_psnr = self.psnr(sr, hr)
        score_ssim = self.ssim(sr, hr)
        
        score_lpips = self.lpips((sr * 2) - 1, (hr * 2) - 1).mean()
        
        score_nlpd = self.nlpd(sr, hr)

        return {
            "psnr": score_psnr.item(),
            "ssim": score_ssim.item(),
            "lpips": score_lpips.item(),
            "nlpd": score_nlpd.item()
        }