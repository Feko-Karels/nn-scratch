import numpy as np
import pandas as pd
import matplotlib as plt

train_data = pd.read_csv('mnist_train.csv')
train_data = np.array(train_data)
m, n = train_data.shape

class NN():
    def __init__(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        self.theta = W1, b1, W2, b2
        self.Z1 = np.zeros((10,1))
        self.A1 = np.zeros((10,1))
        self.Z2 = np.zeros((10,1))
        self.A2 = np.zeros((10,1))
        self.grads = None

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
        
    def forward(self, X):
        W1, b1, W2, b2 = self.theta
        self.Z1 = W1.dot(X) + b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = W2.dot(self.A1) + b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((10, Y.size))  # Ensure one_hot_Y has 10 rows for 10 classes
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    def backward_prop(self, X, Y):
        _, _, W2, _ = self.theta
        one_hot_Y = self.one_hot(Y)
        dZ2 = self.A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        self.grads = dW1, db1, dW2, db2

    def basic_sdg(self, alpha):
        W1, b1, W2, b2 = self.theta
        dW1, db1, dW2, db2 = self.grads
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        self.theta = W1, b1, W2, b2
    
    def cross_entropy_loss(self, A2, Y):
        m = Y.shape[0]
        loss = np.sum(-np.log(A2[Y, np.arange(m)])) / m
        return loss
    
    def get_predictions(self, X):
        out = self.forward(X)
        return np.argmax(out, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
batch_size=100
alpha=0.15
epochs=500
model = NN()

for epoch in range(epochs):
    np.random.shuffle(train_data)
    train_data = train_data[:].T
    Y = train_data[0]
    X = train_data[1:n]
    X = X / 255.

    loss = 0
    for i in range(0, X.shape[1], batch_size):
        X_batch = X[:,i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        out = model.forward(X_batch)
        model.backward_prop(X_batch, Y_batch)
        model.basic_sdg(alpha)
        loss += model.cross_entropy_loss(out, Y_batch)
    if epoch%10 == 0:
        print("epoch: ", epoch)
        predictions = model.get_predictions(X)
        print('Average loss is: ', loss/batch_size)
        print('Accuracy is: ', model.get_accuracy(predictions, Y))
