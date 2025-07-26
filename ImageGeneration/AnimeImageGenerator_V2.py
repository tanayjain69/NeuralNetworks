# Trying to find out how to make the model better -
# Learnt this new concept of hinge loss and implemented it

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tvt
import torchvision
import time
from torch.amp import autocast, GradScaler

# HyperParams
imagesize = 64
batch_size = 256
latent_size = 4*imagesize
EPOCHS = 50
DEVICE = 'cuda'
disc_lr = 0.00025
gen_lr = 0.0001
RATIO = 2   #Training per epoch (discriminator=1, generator=RATIO)
RATIO_INCREMENT = 0.12

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tfs = [tvt.Resize(imagesize), tvt.CenterCrop(imagesize), tvt.ToTensor(), tvt.Normalize(*stats)]
scaler = GradScaler('cuda')

# Dataset
train_dataset = ImageFolder('./ImageGeneration/data', transform=tvt.Compose(tfs))
train_loader = DataLoader(train_dataset, batch_size, True, num_workers=8, pin_memory=True)

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
            nn.Dropout2d(0.3),

            ResidualBlock(64, 128, 5, 3, 2),
            nn.Dropout2d(0.3),

            ResidualBlock(128, 256, 3, 3, 2),
            nn.Dropout2d(0.3),

            ResidualBlock(256, 512, 3, 3, 2),
            nn.Dropout2d(0.3),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, X):
        X = self.layers(X)
        return X

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, X):
        X = self.layers(X)
        return X

# Training and loss functions
def hinge_discriminator_loss(real_preds, fake_preds):
    return torch.mean(nn.functional.relu(1. - real_preds)) + torch.mean(nn.functional.relu(1. + fake_preds))

def hinge_generator_loss(fake_preds):
    return -torch.mean(fake_preds)

def train_discriminator(real_imgs, bs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    fake_imgs = gen(ppt).detach()

    with autocast('cuda'):
        preds1 = disc(fake_imgs)
        # loss1 = disc_loss(preds1, torch.full_like(preds1, 0.1))

        preds2 = disc(real_imgs)
        # loss2 = disc_loss(preds2, torch.full_like(preds2, 0.9))

        # loss = (loss1+loss2)/2
        loss = disc_loss(preds2, preds1)
    scaler.scale(loss).backward()
    scaler.step(disc_optimiser)
    scaler.update()
    
    return loss.item()

def train_generator(bs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    gen_imgs = gen(ppt)
    disc_guess = disc(gen_imgs)
    with autocast('cuda'):
        # loss = gen_loss(disc_guess, torch.full_like(disc_guess, 1))
        loss = gen_loss(disc_guess)

    scaler.scale(loss).backward()
    scaler.step(gen_optimiser)
    scaler.update()

    return loss.item()


# MAINLOOP
if __name__ == '__main__':
    disc = Discriminator()
    disc_loss = hinge_discriminator_loss
    disc_optimiser = torch.optim.Adam(disc.parameters(), disc_lr, betas=(0.5, 0.999))
    # disc.load_state_dict(torch.load('./ImageGeneration/models/disc-res64.pth', weights_only=True))
    disc.to(DEVICE)

    gen = Generator()
    gen_loss = hinge_generator_loss
    gen_optimiser = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    # gen.load_state_dict(torch.load('./ImageGeneration/models/gen-res64.pth', weights_only=True))
    gen.to(DEVICE)

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
        with torch.no_grad():
            gen.eval()
            torchvision.utils.save_image(gen(torch.randn(64, latent_size, 1, 1, device=DEVICE)), f'./ImageGeneration/Saved_imgs/learntimg{epoch}.png', nrow=8, padding=2, normalize=True)
        
        et = time.time()
        
        print(f"Discriminator Loss {epoch+1}: {torch.mean(torch.Tensor(discloss)):.2f}")
        print(f"Generator Loss {epoch+1}: {torch.mean(torch.Tensor(genloss)):.2f} in {(et-st):.2f}s")

        torch.save(disc.state_dict(), './ImageGeneration/models/disc-res-itnorm64.pth')
        torch.save(gen.state_dict(), './ImageGeneration/models/gen-res-itnorm64.pth')
