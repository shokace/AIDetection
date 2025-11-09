# pipeline/transforms.py
import torch
from torchvision import transforms

# This must be defined at the TOP LEVEL, not inside get_transforms
def add_gaussian_noise(x, std: float = 0.02):
    # x is a tensor in [0, 1] after ToTensor()
    return (x + std * torch.randn_like(x)).clamp(0.0, 1.0)


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),

            # Geometric augmentations
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),

            # Photometric augmentations
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=1.5)
            ], p=0.2),

            transforms.ToTensor(),

            # Noise injection – now using a picklable top-level function
            transforms.RandomApply([
                transforms.Lambda(add_gaussian_noise)
            ], p=0.1),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
