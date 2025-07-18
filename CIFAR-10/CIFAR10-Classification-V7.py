# Started learning and implementing new types of NNs other than CNNs
# Created my own ResNet using custom created ResidualBlocks

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time

# Hyperparameters
DEVICE = 'cuda'
BATCH_SIZE = 160
NUM_WORKERS = 8
LR = 0.02
LR_SCHEDULER_GAMMA = 0.92
MOMENTUM = 0.9
EPOCHS = 70
SAVED_ACC = 85.71

tf1 = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

tf2 = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(0.37),
    transforms.RandomCrop(32),
    transforms.ToTensor()
]) 

datasets_train = datasets.ImageFolder("./CIFAR-10/data/train/", transform=tf2)
loader_train = DataLoader(datasets_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

datasets_test = datasets.ImageFolder("./CIFAR-10/data/test/", transform=tf1)
loader_test = DataLoader(datasets_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

class ResidualBlock(nn.Module):
    def __init__(self, inf, outf, k1, k2, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inf, outf, k1, stride, (k1-1)//2),
            nn.BatchNorm2d(outf),
            nn.ReLU(),
            nn.Conv2d(outf, outf, k2, 1, (k2-1)//2),
            nn.BatchNorm2d(outf),
            nn.ReLU()
        )
        if (inf != outf):
            self.skip_layer = nn.Sequential(nn.Conv2d(inf, outf, 1, stride, 0), nn.BatchNorm2d(outf))
        else:
            self.skip_layer = nn.Identity()

    def forward(self, X):
        Y = self.layers(X)
        X = self.skip_layer(X)
        Z = X+Y
        return Z

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            ResidualBlock(32, 64, 3, 3, 2),
            ResidualBlock(64, 64, 3, 3),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 128, 3, 3, 2),
            ResidualBlock(128, 128, 3, 3),
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            ResidualBlock(256, 256, 3, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 10)
        )
    
    def forward(self, X):
        X = self.layers(X)
        return X


model = Resnet()
lossfn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=LR_SCHEDULER_GAMMA)

# model.load_state_dict(torch.load("./CIFAR-10/CIFAR-10modelV7.pth", weights_only=True))


model.to(DEVICE)
lossfn.to(DEVICE)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        correct=0
        for X, Y in loader_train:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            ypreds = model(X)
            loss = lossfn(ypreds, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item()
        
        lr_scheduler.step()
        accuracy = round((100*correct/len(loader_train.dataset)),2)
        end_time = time.time()
        print(f"Train Accuracy {epoch+1}: {accuracy}% in {(end_time-start_time):.2f}s")
        if (epoch%2==1):
                model.eval()
                with torch.inference_mode():
                    correct=0
                    for X, Y in loader_test:
                        X, Y = X.to(DEVICE), Y.to(DEVICE)
                        ypreds = model(X)
                        correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item()

                    accuracy = round((100*correct/len(loader_test.dataset)),2)
                    print(f"Test Accuracy : {accuracy}%")
                    if (accuracy>SAVED_ACC): 
                        torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV7.pth")
                        SAVED_ACC=accuracy
                    if (accuracy==100): break

