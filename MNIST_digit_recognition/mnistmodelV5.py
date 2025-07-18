# Tuning hyperparameters of Convolutional layers (worked pretty good)

import torch
import time
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

#HyperParams
BATCH_SIZE = 32
NUM_WORKERS = 1
LR=0.001
MOMENTUM=0.9
SAVED_ACC = 99.49
EPOCHS = 100

datatrain = pd.read_csv('./MNIST_digit_recognition/mnist_train.csv').to_numpy()
train_dataset = TensorDataset(torch.Tensor(datatrain[:, 1:]), torch.Tensor(datatrain[:, 0]).long())
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

datatest = pd.read_csv('./MNIST_digit_recognition/mnist_test.csv').to_numpy()
test_dataset = TensorDataset(torch.Tensor(datatest[:, 1:]), torch.Tensor(datatest[:, 0]).long())
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

class CapsuleLayer(nn.Module):
    def __init__(self, numcaps, dim):
        super(CapsuleLayer, self).__init__()
        self.numcaps = numcaps
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(size=(dim, dim)))
        self.bias = nn.Parameter(torch.zeros(numcaps))

    def squash(self, x):
        norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = norm / (1 + norm) / torch.sqrt(norm + 1e-8)
        return scale * x

    def forward(self, X):
        print(X.size)
        X.view(-1, self.numcaps, self.dim)
        print(X.size)
        X = torch.matmul(X, self.weight.expand(X.size(0), -1, -1)) + self.bias
        X = self.squash(X)
        X = nn.Flatten(X)
        return X
    
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            CapsuleLayer(8*7*7, 8),
            nn.Linear(8*7*7, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
    
    def forward(self, X):
        X = self.layers(X)
        return X

model = ConvNeuralNet()
optimiser = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# model.load_state_dict(torch.load('./MNIST_digit_recognition/mnistV5.pth', weights_only=True))
model.to('xpu')
loss_fn.to('xpu')

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        st = time.time()
        model.train()
        correct=0
        for X, Y in trainloader:
            X = X.view(-1, 1, 28, 28)
            X, Y = X.to('xpu'), Y.to('xpu')
            ypreds = model(X)
            loss = loss_fn(ypreds, Y)
            correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item();

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        et = time.time()
        accuracy = round((100*correct/len(train_dataset)), 2)
        print(f"Train Accuracy: {accuracy}% in {(et-st):.1f}s")
        
        if (epoch%3==2):
            correct=0
            model.eval()
            for X, Y in testloader:
                X = X.view(-1, 1, 28, 28)
                X, Y = X.to('xpu'), Y.to('xpu')
                ypreds = model(X)
                correct += (torch.argmax(ypreds, dim=1).to('cpu') == Y.to('cpu')).sum().item();
            
            accuracy = round((100*correct/len(test_dataset)), 2)
            print(f"Test Accuracy: {accuracy}%")
            if (accuracy>SAVED_ACC):
                SAVED_ACC = accuracy
                torch.save(model.state_dict(), './MNIST_digit_recognition/mnistV4.pth')
