import torch
import os
import wandb

def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_layer_grad_ratio(model):
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    first = params[0].grad.data.norm(2).item()
    last = params[-1].grad.data.norm(2).item()
    return first / (last + 1e-8)

def save_checkpoint(model, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    wandb.save(path)