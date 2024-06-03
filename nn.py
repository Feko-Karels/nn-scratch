import numpy as np
import pandas as pd
import matplotlib as plt

train_data = pd.read_csv('mnist_train.csv')
train_data = np.array(train_data)
m, n = train_data.shape

class NN():
    def __init__(self):
        self.theta = self.init_params()

    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
        
    def forward_prop(self, theta, X):
        W1, b1, W2, b2 = theta
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, theta, dW1, db1, dW2, db2, alpha):
        W1, b1, W2, b2 = theta
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2
    
    def cross_entropy_loss(self, A2, Y):
        m = Y.shape[0]
        loss = np.sum(-np.log(A2[Y, range(m)])) / m
        return loss
    
    def get_predictions(self, X):
        _, _, _, out = self.forward_prop(self.theta, X)
        return np.argmax(out, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def sgd(self, data, batch_size=100, alpha=0.15, epochs=500):

        for epoch in range(epochs):
            np.random.shuffle(data)
            train_data = data[:].T
            Y = train_data[0]
            X = train_data[1:n]
            X = X / 255.

            loss = 0
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:,i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                Z1, A1, Z2, A2 = self.forward_prop(self.theta, X_batch)
                dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, self.theta[0], self.theta[2], X_batch, Y_batch)
                self.theta = self.update_params(self.theta, dW1, db1, dW2, db2, alpha)
                loss += self.cross_entropy_loss(A2, Y_batch)
            if epoch%10 == 0:
                print("epoch: ", epoch)
                predictions = self.get_predictions(X)
                print('Average loss is: ', loss/batch_size)
                print('Accuracy is: ', self.get_accuracy(predictions, Y))
        return self.theta

#W1, b1, W2, b2 = sdg(X_train,Y_train)
model = NN()
theta = model.sgd(train_data)
