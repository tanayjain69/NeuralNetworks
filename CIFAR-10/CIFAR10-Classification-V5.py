# To speed up training without losing accuracy (reducing params)
# Got down from 1.18M params to 0.175M params
## Realize time for training is independent of number of params but apparently convolutional layers are extremely computationally-heavy


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor()
]) 

datasets_train = torchvision.datasets.ImageFolder("./CIFAR-10/data/train/", transform=transform)
loader_train = DataLoader(datasets_train, batch_size=160, shuffle=True, num_workers=12)

datasets_test = torchvision.datasets.ImageFolder("./CIFAR-10/data/test/", transform=transform)
loader_test = DataLoader(datasets_test, batch_size=160, shuffle=True, num_workers=12)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers=nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linlayers=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, X):
        X = self.convlayers(X)
        X = self.linlayers(X)
        return X

model = NeuralNetwork()
# model.load_state_dict(torch.load("./CIFAR-10/CIFAR-10modelV4.pth", weights_only=True))
saved_accuracy=60


lossfn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.96)

model.to(DEVICE)
lossfn.to(DEVICE)

if __name__ == '__main__':
    epochs = 30
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        correct=0
        model.train()

        for X, Y in loader_train:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            ypreds = model(X)
            loss = lossfn(ypreds, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item()
        
        accuracy = round((100*correct/len(loader_train.dataset)),4)
        print(f"Train Accuracy {epoch+1}: {accuracy}%")
        lr_scheduler.step()
        end_time = time.time()  # End timing the epoch
        print("Time per epoch: ", (end_time-start_time))
        if (epoch%3==2):
            model.eval()
            with torch.inference_mode():
                correct=0
                for X, Y in loader_test:
                    X, Y = X.to(DEVICE), Y.to(DEVICE)
                    ypreds = model(X)
                    correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item()

                accuracy = round((100*correct/len(loader_test.dataset)),4)
                print(f"Test Accuracy : {accuracy}%")
                if (accuracy>saved_accuracy): 
                    torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV5.pth")
                    saved_accuracy=accuracy
                if (accuracy==100): break

