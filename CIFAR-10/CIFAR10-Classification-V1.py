#Both Adam and AdamW optimisers gave almost similar outputs (64x64)-256 --------- Accuracy: 65%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor()
])

datasets_train = torchvision.datasets.ImageFolder("./train/", transform=transform)
loader_train = DataLoader(datasets_train, batch_size=64, shuffle=True, num_workers=16)

datasets_test = torchvision.datasets.ImageFolder("./test/", transform=transform)
loader_test = DataLoader(datasets_test, batch_size=64, shuffle=True, num_workers=16)

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
            nn.Linear(64*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, X):
        X = self.convlayers(X)
        X = self.linlayers(X)
        return X

model = NeuralNetwork()
model.load_state_dict(torch.load("./CIFAR-10modelV1.pth", weights_only=True))
saved_accuracy=64.92

lossfn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    epochs = 15
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
        if (epoch%3==2):
            model.eval()
            correct=0
            for X, Y in loader_test:
                ypreds = model(X)
                correct += (torch.argmax(ypreds, dim=1) == Y).sum().item()

            accuracy = round((100*correct/len(loader_test.dataset)),4)
            print(f"Test Accuracy : {accuracy}%")
            if (accuracy>saved_accuracy): 
                torch.save(model.state_dict(), "./CIFAR-10/CIFAR-10modelV1.pth")
                saved_accuracy=accuracy
            if (accuracy==100): break

