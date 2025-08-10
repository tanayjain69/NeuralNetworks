# Learning StyleGAN Architecture from scratch

#imports
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import time


## Hyperparams
# FIXED
image_size = 64
BATCH_SIZE = 128
latent_size = 512
DEVICE='cuda'
NUM_WORKERS = 12
PREFETCH_FACTOR = 8
UPSAMPLE_MODE = 'bilinear'

# TUNABLE
MLP_LAYERS = 8
MLP_NEURONS = 512
EPOCHS = 100
RATIO = 1.1
RATIO_INCREMENT = 0.11
DISC_LR = 1e-4
GEN_LR = 2e-4
DISC_LRS = 0.9
GEN_LRS = 0.9
GRADIENT_CLIP = 1.
NOISE_REGULARIZATION_WEIGHTAGE = 1e-4
DIVERSITY_INVERSE = 4   # 4-8 is a good value for here
RATIO_LRS = 0.985

# Load Dataset
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tfs = [tv.transforms.Resize((64, 64)), tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor(), tv.transforms.Normalize(*stats)]
train_dataset = tv.datasets.ImageFolder('./ImageGeneration/data', transform=tv.transforms.Compose(tfs))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    
# Multilayer Perceptron
class MLP(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(num_neurons, num_neurons),
                nn.LeakyReLU(0.2),
            ])

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        X = self.layers(X)
        return X

# Adaptive Instance Normalization
class AdaIN(nn.Module):
    def __init__(self, channels, latent_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.style_scale = nn.Linear(latent_dim, channels)
        self.style_shift = nn.Linear(latent_dim, channels)

    def forward(self, x, z):
        # x: [B, C, H, W], z: [B, latent_dim]
        B, C, H, W = x.shape

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, self.eps)
        normX = (x - mean) / std

        style_scale = self.style_scale(z).view(B, C, 1, 1)
        style_shift = self.style_shift(z).view(B, C, 1, 1)

        return normX * style_scale + style_shift


# Random Noise
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

# MiniBatchStdDev (A new layer I found that improves diversity)
class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N)

        if N % G != 0:
            G = N

        x = x.view(G, -1, C, H, W) 
        mean = x.mean(dim=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0)
        var = torch.clamp(var, self.eps)
        std = torch.sqrt(var)
        mean_std = std.mean(dim=[1, 2, 3], keepdim=True)

        mean_std = mean_std.repeat(G, 1, H, W)
        return torch.cat([x.view(N, C, H, W), mean_std], dim=1)


# Discriminator
class InceptionLayer(nn.Module):
    def __init__(self, inf, outf, layer, stride=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inf, outf, 1, stride, 0)),
            nn.LeakyReLU(0.2),
        )
        self.layer1_3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inf, outf//2, 1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(outf//2, outf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        self.layer1_5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inf, outf//4, 1, 1, 0)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(outf//4, outf, 5, 1, 2)),
            nn.LeakyReLU(0.2),
        )
        self.layerMP = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.utils.spectral_norm(nn.Conv2d(inf, outf, 1, 1, 0)),
            nn.LeakyReLU(0.2),
        )
        if (layer == 'layerMP'): 
            self.layer = self.layerMP
        elif layer == 'layer1_5':
            self.layer = self.layer1_5
        elif layer == 'layer1_3':
            self.layer = self.layer1_3
        else:
            self.layer = self.layer1

    def forward(self, X):
        X = self.layer(X)
        return X

class InceptionBlock(nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.layers = nn.ModuleList([InceptionLayer(inf, outf//4, 'layer1', 1), InceptionLayer(inf, outf//2, 'layer1_3'), InceptionLayer(inf, outf//8, 'layer1_5'), InceptionLayer(inf, outf//8, 'layerMP')])

    def forward(self, X):
        feat = []
        for layer in self.layers:
            Y = layer(X)
            feat.append(Y)
        return torch.cat(feat, dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            InceptionBlock(64, 192),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.1),
            nn.utils.spectral_norm(nn.Conv2d(192, 96, 1, 1, 0)),
            nn.LeakyReLU(0.2),
            InceptionBlock(96, 288),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.1),
            nn.utils.spectral_norm(nn.Conv2d(288, 128, 1, 1, 0)),
            nn.LeakyReLU(0.2),
            InceptionBlock(128, 256),
            nn.AvgPool2d(2),
            MiniBatchStdDev(DIVERSITY_INVERSE),
            nn.utils.spectral_norm(nn.Conv2d(257, 512, 4, 1, 0)),
            nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.utils.spectral_norm(nn.Linear(512, 1))
        )

    def forward(self, X):
        X = self.layers(X)
        return X


# Generator
class Generator(nn.Module):
    def __init__(self, noise=True):
        super().__init__()
        self.latent_proj = nn.Linear(latent_size, MLP_NEURONS)
        self.MLP = MLP(MLP_LAYERS, MLP_NEURONS)
        self.constparam = nn.Parameter(torch.randn(1, MLP_NEURONS, 4, 4))
        self.noise_enabled = True
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
        ])
        self.upsample_layers = nn.ModuleList([nn.Upsample(scale_factor=2, mode=UPSAMPLE_MODE) for _ in range(4)])
        self.noise_layers = nn.ModuleList([
            RandomNoise(4),
            RandomNoise(4),
            RandomNoise(8),
            RandomNoise(8),
            RandomNoise(16),
            RandomNoise(16),
            RandomNoise(32),
            RandomNoise(32),
            RandomNoise(64),
            RandomNoise(64)
        ])
        self.AdaIN_channels = [512, 512, 512, 256, 256, 128, 128, 64, 64, 3]
        self.AdaIN_layers = nn.ModuleList([AdaIN(ch, MLP_NEURONS) for ch in self.AdaIN_channels])
    
    def layer_prop(self, ip, conv_idx, ups_idx, noise_idx, ada_idx, Z):
        res0 = self.noise_layers[noise_idx](self.conv_layers[conv_idx](self.upsample_layers[ups_idx](ip)))
        res1 = self.AdaIN_layers[ada_idx](res0, Z)
        res2 = self.AdaIN_layers[ada_idx+1](self.noise_layers[noise_idx+1](self.conv_layers[conv_idx+1](res1)), Z)
        return res2

    def forward(self, W):
        W = W.view(W.size(0), -1)
        Z = self.MLP(self.latent_proj(W))

        # Layer1 Propagation
        res0 = self.AdaIN_layers[0](self.noise_layers[0](self.constparam.repeat(W.size(0), 1, 1, 1)), Z)
        res1 = self.AdaIN_layers[1](self.noise_layers[1](self.conv_layers[0](res0)), Z)

        # Mid Layers
        res2 = self.layer_prop(res1, 1, 0, 2, 2, Z)
        res3 = self.layer_prop(res2, 3, 1, 4, 4, Z)
        res4 = self.layer_prop(res3, 5, 2, 6, 6, Z)
        res5 = self.layer_prop(res4, 7, 3, 8, 8, Z)

        return torch.tanh(res5)

    def enable_noise(self):
        for layer in self.noise_layers:
            layer.enabled = True
    
    def disable_noise(self):
        for layer in self.noise_layers:
            layer.enabled = False

def noise_regularization(noise_weights):
    return (noise_weights ** 2).mean()

def hinge_discriminator_loss(real_preds, fake_preds):
    real_loss = torch.mean(nn.functional.relu(1. - real_preds))
    fake_loss = torch.mean(nn.functional.relu(1. + fake_preds))
    
    return real_loss + fake_loss

def hinge_generator_loss(fake_preds):
    return -torch.mean(fake_preds) + (NOISE_REGULARIZATION_WEIGHTAGE)*noise_regularization(torch.cat([nl.scaling_factor.view(-1) for nl in gen.noise_layers]))


def train_disc(real_imgs, bs):
    disc.train()
    preds1 = disc(real_imgs)
    ppt = torch.randn(bs, latent_size, device=DEVICE)
    with torch.no_grad():
        fake_imgs = gen(ppt)
    preds2 = disc(fake_imgs)
    loss = dl(preds1, preds2)

    disc_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(disc.parameters(), GRADIENT_CLIP)
    disc_opt.step()

    return loss.item()

def train_gen(bs):
    disc.eval()
    ppt = torch.randn(bs, latent_size, device=DEVICE)
    preds = disc(gen(ppt))
    loss = gl(preds)

    gen_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), GRADIENT_CLIP)
    gen_opt.step()

    return loss.item()

# MAINLOOP
if __name__ == '__main__':
    disc = Discriminator()
    dl = hinge_discriminator_loss
    disc_opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
    disc.to(DEVICE)
    # disc.load_state_dict(torch.load('./ImageGeneration/models/disc-styleGAN-ganloss.pth', weights_only=True))
    disc_lrs = torch.optim.lr_scheduler.ExponentialLR(disc_opt, gamma=DISC_LRS)

    gen = Generator()
    gl = hinge_generator_loss
    gen_opt = torch.optim.Adam(gen.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    gen.to(DEVICE)
    # gen.load_state_dict(torch.load('./ImageGeneration/models/gen-styleGAN-ganloss.pth', weights_only=True))
    gen_lrs = torch.optim.lr_scheduler.ExponentialLR(gen_opt, gamma=GEN_LRS)

    for epoch in range(EPOCHS):
        st = time.time()
        disc_losses = []
        gen_losses = []
        for image, _ in train_loader:
            image = image.to(DEVICE)
            disc_opt.zero_grad()
            disc_losses.append(train_disc(image, len(image)))
            
            for i in range(int(RATIO)):
                gen_opt.zero_grad()
                gen_losses.append(train_gen(len(image)))

        RATIO += RATIO_INCREMENT
        RATIO_INCREMENT *= RATIO_LRS

        disc_lrs.step()
        gen_lrs.step()
        
        with torch.no_grad():
            gen.disable_noise()
            with torch.no_grad(): 
                tv.utils.save_image(gen(torch.randn(64, latent_size, device=DEVICE)), f'./ImageGeneration/Saved_imgs/learntimg{epoch}.png', nrow=8, padding=2, normalize=True)
            gen.enable_noise()
        
        et = time.time()
        
        print(f"Discriminator Loss {epoch+1}: {torch.mean(torch.Tensor(disc_losses)):.2f}")
        print(f"Generator Loss {epoch+1}: {torch.mean(torch.Tensor(gen_losses)):.2f} in {(et-st):.2f}s")

        torch.save(disc.state_dict(), './ImageGeneration/models/disc-styleGAN-inception.pth')
        torch.save(gen.state_dict(), './ImageGeneration/models/gen-styleGAN-inception.pth')
