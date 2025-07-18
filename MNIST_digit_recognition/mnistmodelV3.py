# Tried something new (didn't work out)
import torch
import time
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

datatrain = pd.read_csv('./MNIST_digit_recognition/mnist_train.csv').to_numpy()
train_dataset = TensorDataset(torch.Tensor(datatrain[:, 1:]/255.0), torch.Tensor(datatrain[:, 0]).long())
trainloader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4)

datatest = pd.read_csv('./MNIST_digit_recognition/mnist_test.csv').to_numpy()
test_dataset = TensorDataset(torch.Tensor(datatest[:, 1:]/255.0), torch.Tensor(datatest[:, 0]).long())
testloader = DataLoader(test_dataset, batch_size=96, shuffle=True, num_workers=4)

class LinearNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.attentionlayer = nn.MultiheadAttention(64*7*7, 2)
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, X):
        X = self.layers(X)
        X = self.attentionlayer(X, X, X)[0].view(-1, 64*7*7)
        X = self.fc(X)
        return X

model = LinearNeuralNet()
optimiser = optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

# model.load_state_dict('./MNIST_digit_recognition/mnistV3.pth')
saved_accuracy = 98.5
model.to('xpu')
loss_fn.to('xpu')

if __name__ == '__main__':
    epochs = 100
    for epoch in range(epochs):
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
            if (accuracy>saved_accuracy):
                saved_accuracy = accuracy
                torch.save(model.state_dict(), './MNIST_digit_recognition/mnistV3.pth')
