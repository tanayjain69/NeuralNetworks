# Trying to find out how to make the model better -
# Learnt this new concept of hinge loss and implemented it

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tvt
import torchvision
import time


## HYPERPARAMS

#FIXED PARAMS
imagesize = 64
batch_size = 256
latent_size = 4*imagesize
DEVICE = 'cuda'
NUM_WORKERS = 12
PREFETCH_FACTOR = 4

# TUNABLE PARAMS
EPOCHS = 50
disc_lr = 0.0001
gen_lr = 0.0001
RATIO = 4.3
RATIO_INCREMENT = 0.11
DISC_LRS = 0.95
GEN_LRS = 0.95

NOISE_STD = 0.08
NOISE_CHANNELS = 1
NOISE_TYPE = 'gaussian'

LABEL_SMOOTHING = 0.05
GRADIENT_CLIP = 1.0



stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tfs = [tvt.Resize(imagesize), tvt.CenterCrop(imagesize), tvt.ToTensor(), tvt.Normalize(*stats)]

# Dataset
train_dataset = ImageFolder('./ImageGeneration/data', transform=tvt.Compose(tfs))
train_loader = DataLoader(train_dataset, batch_size, True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

#Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, inf, outf, k1, k2, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inf, outf, k1, stride, (k1-1)//2, bias=False),
            nn.BatchNorm2d(outf),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outf, outf, k2, 1, (k2-1)//2, bias=False),
            nn.BatchNorm2d(outf),
            nn.LeakyReLU(inplace=True)
        )
        if (inf != outf):
            self.skip_layer = nn.Sequential(nn.Conv2d(inf, outf, 1, stride, 0, bias=False), nn.BatchNorm2d(outf))
        else:
            self.skip_layer = nn.Identity()

    def forward(self, X):
        Y = self.layers(X)
        X = self.skip_layer(X)
        Z = X+Y
        return Z

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(3, 64, 5, 3, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(64, 128, 5, 3, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(128, 256, 3, 3, 2),
            nn.Dropout2d(0.2),

            ResidualBlock(256, 512, 3, 3, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, X):
        X = self.layers(X)
        return X

# Random Noise Adder
class RandomNoise(nn.Module):
    def __init__(self, size, noise_std=0.1, noise_channels=1, noise_type='gaussian'):
        super().__init__()
        self.size = size
        self.noise_std = noise_std
        self.noise_channels = noise_channels
        self.noise_type = noise_type
        self.enabled = True
        
        # Validate noise type
        if noise_type not in ['gaussian', 'uniform', 'laplace']:
            raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian', 'uniform', or 'laplace'")

    def forward(self, X):
        if not self.enabled:
            return X
            
        B, _, H, W = X.shape
        
        if self.noise_type == 'gaussian':
            noise = torch.randn(B, self.noise_channels, H, W, device=X.device) * self.noise_std
        elif self.noise_type == 'uniform':
            noise = (torch.rand(B, self.noise_channels, H, W, device=X.device) - 0.5) * 2 * self.noise_std
        elif self.noise_type == 'laplace':
            noise = torch.distributions.Laplace(0, self.noise_std).sample((B, self.noise_channels, H, W)).to(X.device)
        
        X = X + noise
        return X

# Generator
class Generator(nn.Module):
    def __init__(self, noise_std=NOISE_STD, noise_channels=NOISE_CHANNELS, noise_type=NOISE_TYPE, noise=True):
        super().__init__()
        
        layers = []
        self.noise_layers = []
        
        # First layer
        layers.extend([
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512)
        ])
        if noise:
            noise_layer = RandomNoise(4, noise_std, noise_channels, noise_type)
            layers.append(noise_layer)
            self.noise_layers.append(noise_layer)
        layers.append(nn.ReLU(True))

        # Second layer
        layers.extend([
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256)
        ])
        if noise:
            noise_layer = RandomNoise(8, noise_std, noise_channels, noise_type)
            layers.append(noise_layer)
            self.noise_layers.append(noise_layer)
        layers.append(nn.ReLU(True))

        # Third layer
        layers.extend([
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128)
        ])
        if noise:
            noise_layer = RandomNoise(16, noise_std, noise_channels, noise_type)
            layers.append(noise_layer)
            self.noise_layers.append(noise_layer)
        layers.append(nn.ReLU(True))

        # Fourth layer
        layers.extend([
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ])

        # Output layer
        layers.extend([
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def enable_noise(self):
        for noise_layer in self.noise_layers:
            noise_layer.enabled = True
    
    def disable_noise(self):
        for noise_layer in self.noise_layers:
            noise_layer.enabled = False
    
    def forward(self, X):
        X = self.layers(X)
        return X

# Training and loss functions
def hinge_discriminator_loss(real_preds, fake_preds):
    real_loss = torch.mean(nn.functional.relu(1. - real_preds))
    fake_loss = torch.mean(nn.functional.relu(1. + fake_preds))
    
    return real_loss + fake_loss

def hinge_generator_loss(fake_preds):
    return -torch.mean(fake_preds)

def train_discriminator(real_imgs, bs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    fake_imgs = gen(ppt).detach()

    preds1 = disc(fake_imgs)
    preds2 = disc(real_imgs)

    loss = disc_loss(preds2, preds1)
    
    loss.backward()
    
    # Apply gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(disc.parameters(), GRADIENT_CLIP)
    disc_optimiser.step()
    
    return loss.item()

def train_generator(bs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    gen_imgs = gen(ppt)
    disc_guess = disc(gen_imgs)
    # loss = gen_loss(disc_guess, torch.full_like(disc_guess, 1))
    loss = gen_loss(disc_guess)

    loss.backward()
    
    # Apply gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(gen.parameters(), GRADIENT_CLIP)
    gen_optimiser.step()

    return loss.item()


# MAINLOOP
if __name__ == '__main__':
    disc = Discriminator()
    disc_loss = hinge_discriminator_loss
    disc_optimiser = torch.optim.Adam(disc.parameters(), disc_lr, betas=(0.5, 0.999))
    disc.load_state_dict(torch.load('./ImageGeneration/models/disc-new.pth', weights_only=True))
    disc.to(DEVICE)

    gen = Generator(noise_std=NOISE_STD, noise_channels=NOISE_CHANNELS, noise_type=NOISE_TYPE, noise=True)
    gen_loss = hinge_generator_loss
    gen_optimiser = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    gen.load_state_dict(torch.load('./ImageGeneration/models/gen-new.pth', weights_only=True))
    gen.to(DEVICE)
    
    disc_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimiser, gamma=DISC_LRS)
    gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimiser, gamma=GEN_LRS)

    for epoch in range(EPOCHS):
        st = time.time()
        discloss = []
        genloss = []
        for image, _ in train_loader:
            image = image.to(DEVICE)
            disc.train()
            disc_optimiser.zero_grad()
            discloss.append(train_discriminator(image, len(image)))
            
            for i in range(int(RATIO)):
                gen.train()
                gen_optimiser.zero_grad()
                genloss.append(train_generator(len(image)))

        RATIO += RATIO_INCREMENT
        
        # Step the learning rate schedulers
        disc_scheduler.step()
        gen_scheduler.step()
        
        with torch.no_grad():
            gen.eval()
            gen.disable_noise()
            torchvision.utils.save_image(gen(torch.randn(64, latent_size, 1, 1, device=DEVICE)), f'./ImageGeneration/Saved_imgs/learntimg{30+epoch}.png', nrow=8, padding=2, normalize=True)
            gen.enable_noise()
        
        et = time.time()
        
        print(f"Discriminator Loss {epoch+1}: {torch.mean(torch.Tensor(discloss)):.2f}")
        print(f"Generator Loss {epoch+1}: {torch.mean(torch.Tensor(genloss)):.2f} in {(et-st):.2f}s")

        torch.save(disc.state_dict(), './ImageGeneration/models/disc-new.pth')
        torch.save(gen.state_dict(), './ImageGeneration/models/gen-new.pth')
