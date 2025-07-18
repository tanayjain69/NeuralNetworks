# Trying custom Inception architecture after getting pretty good results with my custom ResNet and DenseNet Structures

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time

# Hyperparameters
DEVICE = 'cuda'
BATCH_SIZE = 192
NUM_WORKERS = 8
LR = 0.04
LR_SCHEDULER_GAMMA = 0.93 
MOMENTUM = 0.9
EPOCHS = 100
SAVED_ACC = 85.85


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

class InceptionLayer(nn.Module):
    def __init__(self, inf, outf, layer, stride=1):
        super(InceptionLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inf, outf, 1, stride, 0),
            nn.ReLU(),
            nn.BatchNorm2d(outf),
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv2d(inf, outf//2, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(outf//2),
            nn.Conv2d(outf//2, outf, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(outf),
        )
        self.layer1_5 = nn.Sequential(
            nn.Conv2d(inf, outf//4, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(outf//4),
            nn.Conv2d(outf//4, outf, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(outf),
        )
        self.layerMP = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(inf, outf, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(outf),
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
        super(InceptionBlock, self).__init__()
        self.layers = nn.ModuleList([InceptionLayer(inf, outf//4, 'layer1', 1), InceptionLayer(inf, outf//2, 'layer1_3'), InceptionLayer(inf, outf//8, 'layer1_5'), InceptionLayer(inf, outf//8, 'layerMP')])

    def forward(self, X):
        feat = []
        for layer in self.layers:
            Y = layer(X)
            feat.append(Y)
        return torch.cat(feat, dim=1)

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            InceptionBlock(64, 192),
            nn.Dropout2d(0.3),
            nn.Conv2d(192, 96, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            InceptionBlock(96, 288),
            nn.Dropout2d(0.3),
            nn.Conv2d(288, 128, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            InceptionBlock(128, 256),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, X):
        X = self.layers(X)
        return X
    

model = Inception()
model.to(DEVICE)
lossfn = nn.CrossEntropyLoss()
lossfn.to(DEVICE)
optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=LR_SCHEDULER_GAMMA)

# model.load_state_dict(torch.load("./CIFAR-10/CIFAR-10modelV9.pth", weights_only=True))


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
                        torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV9.pth")
                        SAVED_ACC=accuracy
                    if (accuracy==100): break

