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
        point = {'features':np.matrix([.4,1,3,.01,.1]), 'label': 1}
        p = m.predict(point)
        self.assertAlmostEqual(p[0,0], 0.995929862284)
        
        m = model([5,1])
        m.weights = [np.matrix([3,5,-3,2,3]).T]
        point = {'features':np.matrix([.4,-0.2,3.1,.01,.1]), 'label': 1}
        p = m.predict(point)
        self.assertAlmostEqual(p[0,0], 0.000153754441135)

    #unit test for the accuracy function
    def test_accuracy(self):
        data = [dict([('label',np.matrix([1]))]) for i in range(25)]+[dict([('label',np.matrix([0]))]) for i in range(75)]
        a = accuracy(data, [np.matrix([0]) for i in range(len(data))])
        self.assertAlmostEqual(a, 0.75)

    #training and running your model
    def test_submission(self):
        print "Testing neural net models"
        #getting data using 80% of data as training and 
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

    def test_feedforward(self):
        pass

    def test_backpropagate(self):
        pass
        
if __name__ == '__main__':
    unittest.main()

