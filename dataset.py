import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

class FoodSRDataset(Dataset):
    def __init__(self, split='train', crop_size=200, scale_factor=4): # Changed default scale to 4 to match SRCNN/ResNet
        #  Requirement: Minimal input size 200x200px
        assert crop_size % scale_factor == 0, "Crop size must be divisible by scale factor to prevent aspect ratio distortion."
        
        self.base_dataset = datasets.Food101(root='./data', split=split, download=True)
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.split = split
        
        # [cite: 41] Requirement: Data augmentation (RandomCrop, Flip) for training
        if split == 'train':
            self.final_transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            # Validation/Test should be deterministic
            self.final_transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img, _ = self.base_dataset[index]
        
        # --- SAFETY FIX  ---
        # Ensure image is at least crop_size on smallest side
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            # Resize smallest side to crop_size, maintaining aspect ratio
            img = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.BICUBIC)(img)
        
        # 1. Get High Res (Target)
        hr_image = self.final_transform(img)
        
        # 2. Get Low Res (Input)
        # Calculate exact LR dimensions
        lr_size = (self.crop_size // self.scale_factor)
        
        # Standard SR degradation model: Bicubic downsampling
        resize_down = transforms.Resize((lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = resize_down(hr_image)
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.base_dataset)