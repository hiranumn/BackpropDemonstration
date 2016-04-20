from math import exp
import random
import numpy as np

# TODO: Calculate logistic
def logistic(x):
    pass

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    pass

class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
            
    # TODO: Calculate prediction based on model
    def predict(self, point):
        pass

    # TODO: Update model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        pass

    # TODO: Perfrom the forward step of backpropagation
    def feedforward(self, point):
        pass
    
    # TODO: Backpropagate errors
    def backpropagate(self, a, label):
        pass

    # TODO: Train your model
    def train(self, data, epochs, rate, lam):
        pass

def submission(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, 100, 0.05, lam)
    return m
    
