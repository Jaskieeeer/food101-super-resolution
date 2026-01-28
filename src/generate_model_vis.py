import torch
from torch.utils.tensorboard import SummaryWriter

from models import SRCNN, ResNetSR, AttentionSR, Discriminator

def visualize_all_models():
    scale_factor = 4
    lr_size = 50       
    hr_size = lr_size * scale_factor

    models = {
        "SRCNN": SRCNN(scale_factor=scale_factor),
        "ResNetSR": ResNetSR(scale_factor=scale_factor, num_channels=64, num_residuals=16),
        "AttentionSR": AttentionSR(scale_factor=scale_factor, num_channels=64, num_residuals=32),
        "Discriminator": Discriminator() 
    }

    print(f"Starting visualization for {len(models)} models...")
    print(f"Logs will be saved to: runs/all_models_viz/")

    for name, model in models.items():
        print(f"Tracing {name}...")
        
        if name == "Discriminator":
            dummy_input = torch.randn(1, 3, hr_size, hr_size)
            print(f"   -> Input shape: (1, 3, {hr_size}, {hr_size}) [High Res]")
        else:
            dummy_input = torch.randn(1, 3, lr_size, lr_size)
            print(f"   -> Input shape: (1, 3, {lr_size}, {lr_size}) [Low Res]")

        writer = SummaryWriter(f"runs/all_models_viz/{name}")
        
        model.eval()
        try:
            writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"   Error tracing {name}: {e}")
        finally:
            writer.close()
    
    print("\nDone! Run the following command in your terminal to view:")
    print("tensorboard --logdir=runs/all_models_viz")

if __name__ == "__main__":
    visualize_all_models()