#Created a Neural Network from scratch using Vanilla Python and Numpy also tried using it with the MNIST Dataset
# June-2025

import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv('mnist_train.csv'))
# test_data = np.array(pd.read_csv('mnist_test.csv'))

Y_train = train_data.T[0]
X_train = train_data.T[1:]
# Y_test = test_data.T[0]
# X_test = test_data.T[1:]

def initialize_params():
    W1 = np.random.rand(10, 784) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, W2, B1, B2

def ReLU(X):
    return np.maximum(0, X)

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions==Y)/Y.size

def deriv_ReLU(Z):
    return Z>0

def softmax(Z):
    return (np.exp(Z)/sum(np.exp(Z)))

def encode_Y(Y):
    encodedY=np.zeros((10, Y.size))
    encodedY[Y, np.arange(Y.size)] = 1
    return encodedY

def forward_prop(W1, W2, B1, B2, X):
    Z1 = W1.dot(X.T) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, Z2, A1, A2

def back_prop(Z1, Z2, W1, W2, X, Y, alpha, A1, A2):
    eY = encode_Y(Y)
    dB1 = (1/Y.size)*alpha*(deriv_ReLU(Z1)*W2.T.dot(A2-eY))
    dB2 = (1/Y.size)*alpha*(A2-eY)
    dW1 = (1/Y.size)*alpha*((deriv_ReLU(Z1)*W2.T.dot(A2-eY)).dot(X))
    dW2 = (1/Y.size)*alpha*((A2-eY).dot(A1.T))
    return dW1, dW2, dB1, dB2

def update_params(W1, W2, B1, B2, dW1, dW2, dB1, dB2):
    W1 = W1-dW1
    W2 = W2-dW2
    B1 = B1-dB1
    B2 = B2-dB2
    return W1, W2, B1, B2
    
def grad_descent(X, Y, iterations, alpha):
    W1, W2, B1, B2 = initialize_params()
    for i in range(iterations):
        Z1, Z2, A1, A2 = forward_prop(W1, W2, B1, B2, X)
        dW1, dW2, dB1, dB2 = back_prop(Z1, Z2, W1, W2, X, Y, alpha, A1, A2)
        W1, W2, B1, B2 = update_params(W1, W2, B1, B2, dW1, dW2, dB1, dB2)
        if i%5==0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2

W1, B1, W2, B2 = grad_descent(X_train.T, Y_train, 500, 0.01)