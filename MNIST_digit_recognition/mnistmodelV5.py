# Trying out a completely new kind of model using Capsule Netoworks
# CapsNets are too hard to build from scratch (got destroyed so hard by trying)
# Give up on CapsNets

import torch
import time
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from capsule_network import CapsuleLayer

#Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS=1
LR = 0.001
MOMENTUM=0.9

datatrain = pd.read_csv('./MNIST_digit_recognition/mnist_train.csv').to_numpy()
train_dataset = TensorDataset(torch.Tensor(datatrain[:, 1:]), torch.Tensor(datatrain[:, 0]).long())
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

datatest = pd.read_csv('./MNIST_digit_recognition/mnist_test.csv').to_numpy()
test_dataset = TensorDataset(torch.Tensor(datatest[:, 1:]), torch.Tensor(datatest[:, 0]).long())
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

class CapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.capsules = nn.Sequential(
            CapsuleLayer(num_capsules=10, in_channels=12544, out_channels=16, num_iterations=3)
        )
        self.linlayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*16, 10)
        )

    def forward(self, X):
        X = self.convlayers(X)
        X = self.capsules(X)
        X = self.linlayer(X)
        return X

model = CapsNet()
optimiser = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# model.load_state_dict(torch.load('./MNIST_digit_recognition/mnistV4.pth', weights_only=True))
saved_accuracy = 99
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
                torch.save(model.state_dict(), './MNIST_digit_recognition/mnistV5.pth')
        torch.xpu.empty_cache()
