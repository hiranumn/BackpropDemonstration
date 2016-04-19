# Logistic Regression and Neural Networks using SGD
In this assignment you are performing binary classification on handwritten digits from [MNIST](http://yann.lecun.com/exdb/mnist/) database using logisitc regression and neural nets. 

Here are some webpages that might help your understandings
- [Logistic regression (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_regression)
- [Neural Nets (Wikipedia)](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Back Propagation](http://neuralnetworksanddeeplearning.com/chap2.html)

First get the assignment by running:
```
git clone https://github.com/hiranumn/LR_NN_via_SGD
cd LR_NN_via_SGD
```
What you will see in the folder:
- ```test.py``` contains unit tests to check your implementation. Do **NOT** modify this file.
- ```sgd.py``` contains functions that train logsitic regressio and neural net models using SGD.
- ```load_data.py``` contains helper functions that load data. Do **NOT** modify this file.

You are required to use Python 2.7 and **strongly** recommended to use Numpy for vectorizing your calculations.  
[Annaconda](https://www.continuum.io/downloads) offers Python 2.7 distribution with various scientific packages including Numpy. 
# Step 1: Implmenting  ```logistic()``` and  ```logsitic_derivative()``` [10 pts]
The first step in this assignment is to fill out two simple functions ```logistic()``` and  ```logsitic_derivative()``` in ```sgd.py```.  
- ```logistic()``` takes a float *x* and outputs *f(x)*, where *f* is a [logistic function](https://en.wikipedia.org/wiki/Logistic_regression#Definition_of_the_logistic_function), ```f(x)=1/(1+e^-x)```.
- ```logistic_derivative()``` takes a float *x* and outputs *f(x)*, where *f* is the first derivative of the logsitic function.  

Fill in ```logistic()``` and  ```logsitic_derivative()```.

Todo: Tell them to vectorize 

# Step 2: Implementing ```accuracy()``` [10 pts]
```accuracy()``` function takes the follwoing arguments:
- ```y_hat```: &nbsp; a vector of predictions (floats from 0.0 to 1.0) made by your model. 
- ```y```: &nbsp; &nbsp;a vector of true labels (integers 0 or 1) 

Given *y_hat* and *y*, ```accuracy()``` simply computes ```(# correct predictions)/(# predictions made)``` . Fill in ```accuracy()```.

# Step 3: Implementing logsitic regression as a 1 layer NN
In this section, you will be implementing logistic regression as a single layer neural net with a single output.   

Upon instantiation, the ```model```class takes in a vector indicating the structure of a neural net. The first element of the vector indicates the number of inputs, and the last element of the vector indicates the number of nodes in the output layer. Any number in the middle part of the vector indicates the number of nodes in a hidden layer. Let me give you some examples.
- ```[13, 1]``` indicates that this is a single layer neural net with 13 inputs and 1 output. Given that we are using logistic funtion, **this is exactly equal to logistic regression model with 13 features.**
- ```[10, 4, 3, 3]``` indicates that the there are 10 inputs. This is followed by the first hidden layer with 4 nodes and the second hidden layer with 3 nodes. Finally there is an output layer with 3 nodes.

For the sake of simplicity, you can assume that the network is always a single layer NN with a single output. So the argument to the constructor is always ```[x,1]```, where *x* is the number of features in a data point. 

The ```model``` class stores a list of weight matrices, where the *i* th matrix corresponds to the weight matrix for the (*i+1*) th layer. For instance, if you have ```[M1, M2, M3]```, the *0*th matrix *M1* corresponds to the weight matrix for the 1st hidden layer. The last matrix is always the weight matrix for the output layer. In this section, we will assume that this is always a length 1 list with *d* by *1* matrix corresponding to the weights for the output layer.

# Step 3.1: Implementing ```predict()```  [15 pts]
The ```predict()``` function takes in a datapoint and uses the current weight matrices to make a prediction.  
Given a feature vector *x*, your prediction should be  
``` 
Prediction = P(y=True| W,x) = logistic(dot(W,x))
```
where *W* is the *d* by *1* matrix in the weight list of the current model. The output should be in the form of a list containing a length 1 vector; e.g. ```[np.Array(output)]```. It is rahter a strange way to output a value. However, this make it easier to extend your model to a general neural net later. Fill in ```predict()```. Your ```predict()``` function only has to work for the logistic regresion case.

# Step 3.2: Implementing ```update()``` [20 pts]
Stochastic gradient descent works by making a stochastic approximation of the true gradient of an objective function.
Our approximation for the loss function is as follows: 
```
Loss = -1*ln(P(y|W,x)) + lambda*||W||^2 
```
where lambda is a regularization constant.   
Then, the partial gradient with respect to W_i becomes
```
lambda * w_i - x_i * (y - P(y=True| W,x))
```
Thus the update rule for gradient **descent** becomes:
```
w_i <-- w_i - eta *( lambda * w_i - x_i * (y - P(y=True| W,x)) )
```
Fill in the ```update()``` function. This function takes in 4 arguments:
- ```eta```: &nbsp; a learning rate.
- ```lambda```: &nbsp; a regularization constant.
- ```delta```: &nbsp; ```(y - P(y=True| W,x))```.
- ```x```: &nbsp; a vector containing feature data x_1, ...., x_d.

Here is the derivation of the partial gradient with respect to W_i. This might become handy later when you implement backpropagaion. 

# TODO write the derivation

# Step 3.3: Implementing ```train()``` [25 pts]  
The ```train()``` function performs [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to optimize your model. For each iteration, it randomly samples one datapoint with replacement, approximates the true gradients, and updates the model weights via gradient descent. The function takes 4 arguments.
- ```data```: &nbsp; a list of datapoints
- ```epochs```: &nbsp; the number of epochs to run. An epoch refers to a full pass over the dataset. This means that if you run the algorithm for 5 epochs, you call the ```update()``` function ```5*N``` times, where *N* is the number of data points. 
- ```lambda```: a regularization constant.
- ```eta```: a learning rate.

Fill in the ```train()``` function.

# Step 3.4: Choosing appropriate ```lambda```, ```eta``` [20 pts]
```Submission1``` funtion trains your logistic regression model. ```lambda``` is currently at 0, and ```eta``` is at 0.05. This means that you are getting no regularzation in your model. Tweek ```lambda```, ```eta```, and ```epochs``` to obtain better validation accuracy. Your model will be trained on the full training data and run on test data you do not have access to. Your grade for this section will partially depend on the peformance of your model on the test data.  

# Step 4: Extending logistic regression to Neural Nets [Ex credit 50 pts]
So far, we have implemented logistic regression as a single layer neural net with one output with a logisitc activation function. In this section, we will extend our model to a more general multi-layer neural net. The basis of learning a neural net is the same as that of logsitic regression. We will simply caculate a partial gradient with repsect to each weight and optimize the weight via gradient descent. 

# Step 4.1: [Forward propagation] Modifying ```predict()```

# Step 4.2: [Back propagation] Modifying ```update()```

# Step 4.3: Evaulating a neural net model.

# Step 5: Submission
Submit your ```sgd.py``` to ______











