# Learnt about WGAN and WGAN-GP models which are pretty good for sharp image generation and applied it
# Studied about the fascinating math behind GradientPenalty(GP) in this model
# InstanceNorm and LayerNorm removed all variability from the images so explored all my normalization options and finally used GroupNorm

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import time

#Hyperparams
image_size = 64
latent_size = 2*image_size
GEN_LR = 2e-4
CTC_LR = 1e-4
GEN_LRS_GAMMA = 0.985
CTC_LRS_GAMMA = 0.985

EPOCHS = 200
BATCH_SIZE = 256
RATIO = 2
RATIO_INCREMENT = 0.1
DEVICE = 'cuda'

#Importing Training Data
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tf = [tv.transforms.Resize((64, 64)), tv.transforms.RandomHorizontalFlip(0.4), tv.transforms.ToTensor(), tv.transforms.Normalize(*stats)]
train_dataset = tv.datasets.ImageFolder('./ImageGeneration/data', transform=tv.transforms.Compose(tf))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

# Critic
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 1, 4, 1, 0),
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
            nn.GroupNorm(16, 512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, X):
        X = self.layers(X)
        return X

# Helper functions
def gradient_penalty(critic, real_data, fake_data):
    bs = real_data.size(0)
    epsilon = torch.rand(bs, 1, 1, 1, device=DEVICE)

    mixedup_img = epsilon*real_data + (1-epsilon)*fake_data
    mixedup_img.requires_grad_(True)

    critic_op = critic(mixedup_img)
    grads = torch.autograd.grad(critic_op, mixedup_img, torch.ones_like(critic_op), create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(bs, -1)

    grad_norm = grads.norm(2, 1)
    penalty = ((grad_norm-1)**2).mean()
    
    return penalty

def critic_loss(real_preds, fake_preds, gradient_penalty, penalty_lambda=10):
    return torch.mean(fake_preds)-torch.mean(real_preds) + penalty_lambda*gradient_penalty

def generator_loss(fake_preds):
    return -torch.mean(fake_preds)

def train_critic(real_data, bs, critic, gen, loss_fn, optimiser, lrs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    fake_imgs = gen(ppt)

    preds1 = critic(real_data)
    preds2 = critic(fake_imgs)
    loss = loss_fn(preds1, preds2, gradient_penalty(critic, real_data, fake_imgs))

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    lrs.step()

    return loss.item()

def train_generator(bs, critic, gen, loss_fn, optimiser, lrs):
    ppt = torch.randn(bs, latent_size, 1, 1, device=DEVICE)
    fake_imgs = gen(ppt)

    preds = critic(fake_imgs)
    loss = loss_fn(preds)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    lrs.step()

    return loss.item()


## MAINLOOP
if __name__ == '__main__':
    critic = Critic()
    ctc_loss = critic_loss
    ctc_opt = torch.optim.Adam(critic.parameters(), lr=CTC_LR, betas=(0.01, 0.9))
    critic.to(DEVICE)
    ctc_lrs = torch.optim.lr_scheduler.ExponentialLR(ctc_opt, gamma=CTC_LRS_GAMMA)

    gen = Generator()
    gen_loss = generator_loss
    gen_opt = torch.optim.Adam(gen.parameters(), lr=GEN_LR, betas=(0.01, 0.9))
    gen.to(DEVICE)
    gen_lrs = torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=GEN_LRS_GAMMA)

    for epoch in range(EPOCHS):
        st = time.time()
        ctc_losses = []
        generator_losses = []

        for image, _ in train_loader:
            image = image.to(DEVICE)

            x = 1 if RATIO>=1 else int(1/RATIO)
            for __ in range(x):
                critic.train()
                ctc_losses.append(train_critic(image, len(image), critic, gen, ctc_loss, ctc_opt, ctc_lrs))

            gen.train()
            for __ in range(int(RATIO) if RATIO>=1 else 1):
                generator_losses.append(train_generator(len(image), critic, gen, gen_loss, gen_opt, gen_lrs))
            
        RATIO += RATIO_INCREMENT
        with torch.no_grad():
            tv.utils.save_image(gen(torch.randn(64, latent_size, 1, 1, device=DEVICE)), f'./ImageGeneration/Saved_imgs/learntimg{epoch}.png', nrow=8)
        
        et = time.time()

        print(f"Critic Loss {epoch+1}: {torch.mean(torch.Tensor(ctc_losses)):.2f}")
        print(f"Generator Loss {epoch+1}: {torch.mean(torch.Tensor(generator_losses)):.2f}", f"in {(et-st):.2f}s")

        torch.save(gen.state_dict(), './ImageGeneration/models/gen-wgan64.pth')
        torch.save(critic.state_dict(), './ImageGeneration/models/ctc-wgan64.pth')
