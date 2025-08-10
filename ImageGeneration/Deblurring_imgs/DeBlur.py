import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

# Hyperparams
NUM_WORKERS = 12
BATCH_SIZE = 256
PREFETCH_FACTOR = 8

# Load Dataset
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tfs = [tv.transforms.Resize((64, 64)), tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor(), tv.transforms.Normalize(*stats)]
train_dataset = tv.datasets.ImageFolder('./ImageGeneration/data', transform=tv.transforms.Compose(tfs))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

# Noise Layer
class RandomNoise(nn.Module):
    def __init__(self, size, noise_std=0.1, noise_channels=1):
        super().__init__()
        self.size = size
        self.noise_std = noise_std
        self.noise_channels = noise_channels
        self.enabled = True
        self.scaling_factor = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, X):
        if not self.enabled:
            return X
        
        B, _, H, W = X.shape
        noise = torch.randn(B, self.noise_channels, H, W, device=X.device) * self.noise_std

        return X + noise*self.scaling_factor[0]

# DeBlurrer
class DeBlurrer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

        )


    def forward(self, X):
        pass


#MAINLOOP


