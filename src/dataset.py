import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

class FoodSRDataset(Dataset):
    def __init__(self, split='train', crop_size=200, scale_factor=2):
        self.base_dataset = datasets.Food101(root='./data', split=split, download=True)
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.split = split
        
        # Base transforms that happen AFTER the safety resize
        if split == 'train':
            self.final_transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.final_transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        # 1. Get the original high-res image
        img, _ = self.base_dataset[index]
        
        # --- SAFETY FIX START ---
        # Check if the image is smaller than our required crop size
        # img.size is (width, height)
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            # Resize the smallest side to match crop_size
            # Example: 165x512 -> 200x620. This prevents the crash.
            img = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.BICUBIC)(img)
        # --- SAFETY FIX END ---
        
        # 2. Process High Res (Target)
        # Now we know for sure the image is big enough to crop
        hr_image = self.final_transform(img)
        
        # 3. Process Low Res (Input)
        lr_height = self.crop_size // self.scale_factor
        lr_width = self.crop_size // self.scale_factor
        
        # Downsample using Bicubic
        resize_down = transforms.Resize((lr_height, lr_width), interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = resize_down(hr_image)
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.base_dataset)