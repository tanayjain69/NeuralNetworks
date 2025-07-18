## 1. Tuning hyperparameters for SGD model - no better
## 2. Adding more convolution layers-more features and also adding batch norm layers (problem is too many params long time to train)

# 1. SGD (32x32)-256 -------- Accuracy: 73%
# 2. SGD (32x32)-128 +Layers- Accuracy: 78%
#Fine tuning convolutional layers to get accuracy boost and increasing one convolutional layer.
# Lesson Learned - Basic CNNs are too naive(for higher level datasets like CIFAR-10) and hit accuracy ceiling pretty fast. Even after a lot of data augmentation and hyperparameter tuning it doesnt get much better

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import time

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 160
NUM_WORKERS = 8
LR = 0.005
LR_SCHEDULER_GAMMA = 0.98
MOMENTUM = 0.92
EPOCHS = 70
SAVED_ACC = 78.35

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor()
]) 

datasets_train = torchvision.datasets.ImageFolder("./CIFAR-10/data/train/", transform=transform)
loader_train = DataLoader(datasets_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

datasets_test = torchvision.datasets.ImageFolder("./CIFAR-10/data/test/", transform=transform)
loader_test = DataLoader(datasets_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256)
        )
        self.linlayers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, X):
        X = self.convlayers(X)
        X = self.linlayers(X)
        return X

model = NeuralNetwork()
model.load_state_dict(torch.load("./CIFAR-10/CIFAR-10modelV6.pth", weights_only=True))

lossfn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=LR_SCHEDULER_GAMMA)

model.to(DEVICE)
lossfn.to(DEVICE)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        start_time = time.time()
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
        
        accuracy = round((100*correct/len(loader_train.dataset)),2)
        lr_scheduler.step()
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
                    torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV6.pth")
                    SAVED_ACC=accuracy
                if (accuracy==100): break

