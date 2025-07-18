#Both Adam and AdamW optimisers gave almost similar outputs (32x32)-192 -------- 71%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor()
])

datasets_train = torchvision.datasets.ImageFolder("./CIFAR-10/data/train/", transform=transform)
loader_train = DataLoader(datasets_train, batch_size=64, shuffle=True, num_workers=8)

datasets_test = torchvision.datasets.ImageFolder("./CIFAR-10/data/test/", transform=transform)
loader_test = DataLoader(datasets_test, batch_size=64, shuffle=True, num_workers=8)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers=nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.linlayers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 10)
        )
    def forward(self, X):
        X = self.convlayers(X)
        X = self.linlayers(X)
        return X

model = NeuralNetwork()
# model.load_state_dict(torch.load("./CIFAR-10modelV1.pth", weights_only=True))
saved_accuracy=71.29

lossfn = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.975)

if __name__ == '__main__':
    epochs = 250
    for epoch in range(epochs):
        correct=0
        model.train()
        for X, Y in loader_train:
            ypreds = model(X)
            loss = lossfn(ypreds, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            correct += (torch.argmax(ypreds, dim=1) == Y).sum().item()
        
        accuracy = round((100*correct/len(loader_train.dataset)),4)
        print(f"Train Accuracy {epoch+1}: {accuracy}%")
        lr_scheduler.step()
        if (epoch%2==1):
            model.eval()
            correct=0
            for X, Y in loader_test:
                ypreds = model(X)
                correct += (torch.argmax(ypreds, dim=1) == Y).sum().item()

            accuracy = round((100*correct/len(loader_test.dataset)),4)
            print(f"Test Accuracy : {accuracy}%")
            if (accuracy>saved_accuracy): 
                torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV3.pth")
                saved_accuracy=accuracy
            if (accuracy==100): break
        

