# Trying custom DenseNets after getting pretty good results with my custom ResNet

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time

# Hyperparameters
DEVICE = 'cuda'
BATCH_SIZE = 160
NUM_WORKERS = 8
LR = 0.03
LR_SCHEDULER_GAMMA = 0.96
MOMENTUM = 0.92
EPOCHS = 100
SAVED_ACC = 75


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

class DenseLayer(nn.Module):
    def __init__(self, inf, growth_factor):
        super(DenseLayer, self).__init__()
        self.convlayer = nn.Conv2d(inf, growth_factor, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(growth_factor)

    def forward(self, X):
        X = self.BN(self.relu(self.convlayer(X)))
        return X

class DenseBlock(nn.Module):
    def __init__(self, inf, growth_factor, num_layers=1):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_factor = growth_factor
        self.inf = inf
        for i in range(num_layers):
            self.layers.append(DenseLayer(inf + i * growth_factor, growth_factor))

    def forward(self, X):
        features = [X]
        for layer in self.layers:
            Y = layer(torch.cat(features, dim=1))
            features.append(Y)
        return torch.cat(features, dim=1)

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            DenseBlock(32, 16, 4),
            nn.Dropout2d(0.3),
            nn.Conv2d(96, 64, 1, 1, 0),
            DenseBlock(64, 32, 4),
            nn.Dropout2d(0.3),
            nn.Conv2d(192, 128, 1, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            DenseBlock(128, 32, 4),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    
    def forward(self, X):
        X = self.layers(X)
        return X


model = DenseNet()
lossfn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=LR_SCHEDULER_GAMMA)

# model.load_state_dict(torch.load("./CIFAR-10/CIFAR-10modelV8_cuda.pth"))

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
                        torch.save(model.to('cpu').state_dict(), "./CIFAR-10/CIFAR-10modelV8_cuda.pth")
                        SAVED_ACC=accuracy
                        model.to(DEVICE)
                    if (accuracy==100): break

