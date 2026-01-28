import torch
import torch.nn as nn
from torchinfo import summary
from models import SRCNN, ResNetSR, AttentionSR, Discriminator

def analyze_models():
    configs = [
        (
            "SRCNN", 
            SRCNN(scale_factor=4, hidden_dim=64), 
            (1, 3, 50, 50) 
        ),
        (
            "ResNetSR", 
            ResNetSR(scale_factor=4, num_channels=64, num_residuals=16), 
            (1, 3, 50, 50) 
        ),
        (
            "AttentionSR", 
            AttentionSR(scale_factor=4, num_channels=64, num_residuals=32), 
            (1, 3, 50, 50)
        ),
        (
            "Discriminator", 
            Discriminator(), 
            (1, 3, 200, 200)
        ),
    ]

    print(f"{'Model Name':<20} | {'Params':<12} | {'Size (MB)':<10}")
    print("-" * 46)

    for name, model, input_size in configs:
        model_stats = summary(model, input_size=input_size, verbose=0)
        
        param_count = model_stats.total_params
        size_mb = model_stats.total_param_bytes / (1024 ** 2)
        
        print(f"{name:<20} | {param_count:<12,} | {size_mb:<10.2f}")

if __name__ == "__main__":
    analyze_models()