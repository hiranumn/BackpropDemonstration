from unittest import TestCase
import unittest
import numpy as np
from sgd import logistic, model, accuracy, submission 
from data import loadmnist

class SGDTest(unittest.TestCase):

    # unit test for the matricized logistic function 
    def test_logistic(self):
        input = logistic(np.matrix([1,2,-1]))
        output = np.matrix([0.7310585786300049, 0.8807970779778823, 0.2689414213699951])
        for i in range(len(input)):
            self.assertAlmostEqual(input[0, i], output[0, i])

    # unit test for the predict function
    def test_predict(self):
        m = model([5,1])
        m.weights = [np.matrix([1,2,1,0,1]).T]
        m.bias = [np.matrix([0])]
        point = {'label': 1, 'features':np.matrix([.4,1,3,.01,.1])}
        p = m.predict(point)
        self.assertAlmostEqual(p[0,0], 0.995929862284)
        
        m = model([5,1])
        m.weights = [np.matrix([3,5,-3,2,3]).T]
        m.bias = [np.matrix([0])]
        point = {'label': 1, 'features':np.matrix([.4,-0.2,3.1,.01,.1])}
        p = m.predict(point)
        self.assertAlmostEqual(p[0,0], 0.000153754441135)

    #unit test for the accuracy function
    def test_accuracy(self):
        data = [dict([('label',np.matrix([1]))]) for i in range(25)]+[dict([('label',np.matrix([0]))]) for i in range(75)]
        a = accuracy(data, [np.matrix([0]) for i in range(len(data))])
        self.assertAlmostEqual(a, 0.75)

    
    #training and running your model
    def test_submission(self):
        #getting data using 80% of data as training.
        #If you want to test soething other than 3 vs 5, just change the input to loadmnist.
        #For example, loadmnist(1, 7)
        data = loadmnist(3, 5)
        train_data = data[:int(len(data)*0.8)]
        validation_data =data[int(len(data)*0.8):]

        #train the model
        m = submission(train_data)

        #evaluate
        predictions = [m.predict(p) for p in train_data]
        print "Training Accuracy:", accuracy(train_data, predictions)
        predictions = [m.predict(p) for p in validation_data]
        print "Validation Accuracy:", accuracy(validation_data, predictions)
    
    #unit test for feedforward
    def test_feedforward(self):
        m = model([5,3,1])
        m.weights = [np.arange(5*3).reshape((5,3)), np.matrix([3,5,-3]).T]
        m.bias = [np.matrix([1,-1,0]), np.matrix([1])]
        point = {'label': 1, 'features':np.matrix([1,-0.1,3,-2,0.1])}
        
        a = m.feedforward(point)
        answer = [np.matrix([[ 1., -0.1, 3, -2, 0.1]]), np.matrix([[ 0.86989153, 0.86989153, 0.99260846]]), np.matrix([[0.99318172]])]
        for i in range(len(a)):
            for j in range(a[i].shape[1]):
                self.assertAlmostEqual(a[i][0,j], answer[i][0,j])

    #unit test for backprop
    def test_backpropagate(self):
        m = model([5,3,1])
        m.weights = [np.arange(5*3).reshape((5,3)), np.matrix([3,5,-3]).T]
        m.bias = [np.matrix([1,-1,0]), np.matrix([1])]
        point = {'label': 1, 'features':np.matrix([1,-0.1,3,-2,0.1])}
        
        a = m.feedforward(point)
        delta = m.backpropagate(a, np.matrix([1]))
        answer = [np.matrix([0.00231508, 0.00385847, -0.00015008]).T, np.matrix([0.00681828])]
        
        for i in range(len(delta)):
            for j in range(delta[i].shape[1]):
                self.assertAlmostEqual(delta[i][0,j], answer[i][0,j])
        
if __name__ == '__main__':
    unittest.main()

