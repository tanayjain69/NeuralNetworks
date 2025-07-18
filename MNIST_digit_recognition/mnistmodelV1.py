import torch
import time
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

datatrain = pd.read_csv('./MNIST_digit_recognition/mnist_train.csv').to_numpy()
train_dataset = TensorDataset(torch.Tensor(datatrain[:, 1:]), torch.Tensor(datatrain[:, 0]).long())
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

datatest = pd.read_csv('./MNIST_digit_recognition/mnist_test.csv').to_numpy()
test_dataset = TensorDataset(torch.Tensor(datatest[:, 1:]), torch.Tensor(datatest[:, 0]).long())
testloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

class LinearNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, X):
        X = self.layers(X)
        return X

model = LinearNeuralNet()
optimiser = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('./MNIST_digit_recognition/mnistV1.pth', weights_only=True))
saved_accuracy = 97.85
# model.to('xpu')
# loss_fn.to('xpu')

if __name__ == '__main__':
    epochs = 100
    for epoch in range(epochs):
        st = time.time()
        model.train()
        correct=0
        for X, Y in trainloader:
            ypreds = model(X)
            loss = loss_fn(ypreds, Y)
            correct += (torch.argmax(ypreds, dim=1) == Y).sum().item();

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
                ypreds = model(X)
                correct += (torch.argmax(ypreds, dim=1) == Y).sum().item();
            
            accuracy = round((100*correct/len(test_dataset)), 2)
            print(f"Test Accuracy: {accuracy}%")
            if (accuracy>saved_accuracy):
                saved_accuracy = accuracy
                torch.save(model.state_dict(), './MNIST_digit_recognition/mnistV1.pth')
