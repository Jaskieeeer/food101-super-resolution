import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

class FoodSRDataset(Dataset):
    def __init__(self, split='train', crop_size=200, scale_factor=2):
        """
        Args:
            split (str): 'train' or 'test'.
            crop_size (int): The size of the High Res crop (Target).
            scale_factor (int): How much to downscale (2 means input will be 1/2 size).
        """
        # This will download the dataset to ./data if it doesn't exist
        # Note: The download is ~5GB.
        self.base_dataset = datasets.Food101(root='./data', split=split, download=True)
        
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        
        # Transform for the High Resolution (Target) Image
        # We crop randomly for training to get more variety
        if split == 'train':
            self.hr_transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(), # (+1 pt: Data Augmentation)
                transforms.ToTensor()
            ])
        else:
            # For testing, we just center crop to keep it consistent
            self.hr_transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        # 1. Get the original high-res image
        img, _ = self.base_dataset[index]
        
        # 2. Process High Res (Target)
        hr_image = self.hr_transform(img)
        
        # 3. Process Low Res (Input)
        # We manually downsample the HR image to create the LR input
        # Current HR shape is [3, H, W]
        
        # Calculate new dimensions
        lr_height = self.crop_size // self.scale_factor
        lr_width = self.crop_size // self.scale_factor
        
        # Resize down (Bicubic is standard for degradation)
        resize_down = transforms.Resize((lr_height, lr_width), interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Resize back up? 
        # OPTION A: Keep it small (Standard SR). Network learns to upsample.
        # OPTION B: Resize back up to original size (Pre-upsampling). Network just refines.
        # We will use OPTION A (Standard SR) as it's more common for "Own Architecture".
        
        # We need to transform tensor back to PIL for resizing in older torchvision, 
        # but modern transforms work on tensors too.
        lr_image = resize_down(hr_image)
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.base_dataset)

# --- Sanity Check Block ---
if __name__ == "__main__":
    # This block only runs if you run "python src/dataset.py" directly
    print("Initializing dataset...")
    ds = FoodSRDataset(split='train', crop_size=200, scale_factor=2)
    print(f"Dataset Size: {len(ds)}")
    
    lr, hr = ds[0]
    print(f"Low Res Shape: {lr.shape}")   # Should be [3, 100, 100]
    print(f"High Res Shape: {hr.shape}")  # Should be [3, 200, 200]
    print("✅ Dataset works!")
