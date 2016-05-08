# Logistic Regression and Neural Networks using SGD
In this assignment you are performing binary classification on handwritten digits data from [MNIST](http://yann.lecun.com/exdb/mnist/) database using logisitc regression and neural nets. 

Here are some webpages that might help you understand the material, in addition to the lecture slides/notes you can find under our course website. 
- [Logistic regression (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_regression)
- [Neural Nets (Wikipedia)](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Back Propagation](http://neuralnetworksanddeeplearning.com/chap2.html)

First get the assignment by running:
```
git clone https://github.com/hiranumn/LR_NN_via_SGD
cd LR_NN_via_SGD
```
What you will see in the folder:
- `test.py` contains unit tests to check your implementation. 
- `sgd.py` contains functions that train logsitic regression and neural net models using SGD. You will be submitting this file.
- `data.py` contains helper functions that load the MNIST data. Do **NOT** modify this file.

Download mnist.zip from the [course website](https://courses.cs.washington.edu/courses/cse446/16sp/) and extract it to the same folder.

You are **required** to use Python 2.7 and Numpy for vectorizing your calculations. It is **strongly** recommended that you avoid using forloops as much as possible for matrix calculations. If you have not installed Numpy, [Annaconda](https://www.continuum.io/downloads) offers Python 2.7 distribution with various scientific packages including Numpy. 

# Step 1: Implementing  `logistic()` [5 pts]
In `sgd.py`, you will find an empty function called `logistic()`. The `logistic()` function should take in a Numpy matrix of floats and apply a logistic function `f(x)=1/(1+e^-x)` elementwise. The output should be a numpy matrix with size equal to that of the input.  [Here](https://en.wikipedia.org/wiki/Logistic_regression#Definition_of_the_logistic_function) is the Wikipedia page for logistic  functions if you are not clear on what they are. Fill in `logistic()`.

# Step 2: Implementing `accuracy()` [5 pts]
We need a way to measure the performance of your model. The `accuracy()` function will do this for you.  
It takes the following arguments:
- `data`: a Python list of datapoints. 
- `predictions`: a Python list of 1 by 1 Numpy matrices containing your predictions. 

Let us first start by explaining the structure of a datapoint. We represent a datapoint as a 2-element dictionary. The first key `label` is mapped to a matrix for the true label, and the second key `features` is mapped to a 1 by d matrix representing the features for the datapoint.  

Yes, it is a bit strange that `predictions` is a list of 1 by 1 matrices, but this will make it easier to extend logistic regression to neural nets later in this assignment. Given `data` and `predictions`, `accuracy()` simply computes `(# correct predictions)/(# predictions made)` and outputs it as a float. Fill in `accuracy()`.

# Step 3: Implementing logistic regression as a single layer NN
In this section, you will be implementing logistic regression as a single layer neural net with a single output.   

Let us start by explaining the `model` class. Upon instantiation, the `model` class takes in a Python list of integers indicating the structure of a neural net. The first element of the list indicates the number of inputs, and the last element of the list indicates the number of nodes in the output layer. Any number in the middle indicates the number of nodes in a hidden layer. Let me give you some examples.
- `[13, 1]` indicates that this is a single layer neural net with 13 inputs and 1 output. Given that we are using logistic activation function, **this is exactly equal to logistic regression model with 13 features.**
- `[10, 4, 3, 3]` indicates that there are 10 inputs. This is followed by the first hidden layer with 4 nodes and the second hidden layer with 3 nodes. Finally, there is an output layer with 3 nodes.

For now, you can assume that the model is always a single layer NN with a single output. So the argument to the constructor is always `[x,1]`, where *x* is the number of features in a data point. 

The `model` class stores a list of weight matrices, `weights`, where the *i* th matrix corresponds to the weight matrix for the (*i+1*) th layer. For instance, if you have `[M1, M2, M3]`, the *0*th matrix *M1* corresponds to the weight matrix for the 1st hidden layer. The last matrix is always the weight matrix for the output layer. Again, we assume a single layer neural network with a single output for now. Therefore,  `weights` is a length 1 Python list of *d* by *1* matrix, which corresponds to the weights for the output layer.

The `model` class also stores a list of bias terms, `bias`. This is structured very similar to `weights`. For instance, let us say that you have `bias = [M1, M2]`. Then, M1 corresponds to the bias terms for the first hidden layer M2 corresponds to the bias terms for the last layer. Again, since we are first implementing logistic regression, you can assume that this is always `[np.matrix(b)]` where b is a float.

The `model` constructor is already implemented for you. You do not have to change the constructor. 

# Step 3.1: Implementing `predict()`  [5 pts]
The `predict()` function takes in a datapoint and uses the current weight matrices and bias terms to make a prediction. 

Given a feature vector *x*, your prediction should be  
```
Prediction = P(y=True| W,x,b) = logistic(dot(x,W)+b)
```
where *W* is the *d* by *1* matrix in `weights` of the current model, *b* is the bias vector `bias`, and *x* is the 1 by d matrix for the features of a datapoint. 

The output should be in the form of a 1 by 1 matrix; e.g. `np.matrix(output)`. Again, this seems like a strange way to store an output. However, this make it easier to extend your model to a general neural net later. Fill in `predict()`. Your `predict()` function only needs to work for the logistic regression case for now.

# Step 3.2: Implementing `update()` [15 pts]
Stochastic gradient descent works by making a stochastic approximation of the true gradient of an objective function. 

Our approximation for the loss function is as follows: 
```
Loss = -1*ln(P(y|W,x,b)) + 0.5*lambda*||W||^2 
```
where lambda is a regularization constant.   
Then, the partial gradient with respect to W_i becomes
```
lambda * W_i - x_i * (y - P(y=True| W,x,b))
```
Thus, the update rule for gradient **descent** becomes:
```
w_i <-- w_i - eta *( lambda * w_i - x_i * (y - P(y=True| W,x,b)) )
```
The `update()` function takes 4 arguments:
- `eta`: a learning rate. This is a float
- `lam`: a regularization constant *lambda*. This is a float.
- `delta`: a length 1 Python list containing a 1 by 1 Numpy matrix of `(y - P(y=True| W,x,b))`.
- `a`: a length 1 Python list containing a 1 by d Numpy matrix of the feature values of a datapoint.

Fill in `update()`. Remember to update `self.bias` as well.

# Step 3.3: Implementing `train()` [10 pts]  
The `train()` function performs [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to optimize your model. For each iteration, it randomly samples one datapoint with replacement, approximates the true gradients, and updates `weights` via `update()`. The function takes 4 arguments.
- `data`: a list of data points
- `epochs`: &nbsp; the number of epochs to run. An epoch refers to a full pass over the dataset. This means that if you run the algorithm for 5 epochs, you call the `update()` function `5*N` times, where *N* is the number of data points. 
- `lam`: a regularization constant.
- `eta`: a learning rate.

Fill in the `train()` function.

# Step 3.4: Tuning your logistic regression model [20 pts]
Now that your have all functions implemented for logistic regression, you can train your model on the MNist data. The `logistic_regression()` function trains your logistic regression model on handwritten 3s and 5s. Do you get validation accuracy over 90%? If not, you might want to debug your implementation. Try other pairs of integers if you are interested.

The regularization constant `lam` is currently at 0.00001, and `eta` is at 0.05. Tweek `lam`, `eta`, and `epochs` to obtain better validation accuracy. Often it is a good idea to start with a high learning rate and decrease it over time.

Your model will be trained on the full training data (`3` vs `5`) and run on test data you do not have access to. Your grade for this section will partially depend on the performance of your model on the test data. **Make sure your code completes in under 5 minutes for the data you currently have.** You will receive 0 point for this section otherwise.

# Step 4: Extending logistic regression to Neural Nets
So far, we have implemented logistic regression as a single layer neural net with a single output and a logistic activation function. In this section, we will extend our model to a more general multi-layer neural net.  

The basis of learning a neural net is the same as that of logistic regression. We will simply calculate a partial gradient with respect to each weight and optimize the weight via gradient ascent (or descent). The calculation of the partial derivatives are performed by the algorithm called backpropagation. The details of the algorithm can be found in the [lecture notes](https://courses.cs.washington.edu/courses/cse446/16sp/) or [here](http://neuralnetworksanddeeplearning.com/chap2.html).

# Step 4.1: Implementing `feedforward()` [10 pts]
We will start by implementing the feedforward step of the backpropagation algorithm. The `feedforward()` function takes in a datapoint and propagates the input signals towards the output layer. The function outputs a Python list of Numpy matrices, which represent input values and postsynaptic activation values for nodes in non-input layers. We will call this output list `a`.

Let us give you an example. Suppose we have a 3-layer neural net, and our `a` is
```
a=[M0, M1, M2, Mn]
```
M0 is exactly equal to the features of an input datapoint. M1 stores the postsynaptic activation values for the first layer, which can be obtained by combining W and M0. Mn represents the postsynaptic activation for the output layer. This is equal to the prediction of your model given the input datapoint. Notice that these matrices can be calculated dynamically, and you are **strongly** recommended to do so. You are also advised to use Numpy matrix calculations whenever you can, since Python for-loops are sometimes too costly.

Fill in `feedforward()`. Also, modify `predict()` so that it uses the last matrix of `a` as a prediction. 

# Step 4.2: Implementing `backpropagate()` [10 pts]
In this section, we are implementing the backward step of the backpropagation function. The `backpropagate()` function takes 2 arguments:  
- `a`: the output of `feedforward()`.
- `label`: the true label for an input datapoint.

The objective of this functions is to calculate an error term for each node in non-input layer. The error term of a node *n* is the partial derivative of the likelihood function *L* with respect to its **PRE**-synaptic value. We are approximating *L* by a single point. Thus, 
```
L = 1*ln(P(y|W,x,b)) - 0.5*lambda*||W||^2 
```

The error values are stored in a form similar to `a`, and we will call this `delta`. Here is an example:
```
delta = [M1, M2, ... , Mn]
```
Mn is the error terms for the nodes in the output layer. Similarly, M1 is the error terms for the nodes in the 1st hidden layer.  
Fill in `backpropagate()`

# Step 4.3: Modifying `update()` [10 pts]
Your current `update()` function only works for a single layer case. 

The `update()` function takes 4 arguments:
- `a`: output of `feedforward()`
- `delta`: output of `backpropagate()`
- `lam`: a regularization constant lambda.
- `rate`: a learning rate

Modify the `update()` function so that it updates the weights for all layers and the bias terms. Also, change the `train()` function so that it uses `feedforward()`, `backpropagate()`, and the new `update()` function.

# Step 4.3: Evaluating your neural net model [10 pts]
Your neural network model is now ready for trained by the function `neural_net()`. For a starter, create a single layer neural net with a single output. Again, this should be exactly equal to logistic regression. Although the performance varies due to the stochastic nature of SGD, your accuracy should be very similar to that of your earlier implementation. If not, you might have a bug in your code.

Now let's add a single hidden layer with 15 nodes. Does it improve the performance?

Try different numbers of hidden nodes/layers to improve your model as much as you can. Your model will be trained on the full training data (`3` vs `5`) and run on test data you do not have access to. Your grade for this section will partially depend on the performance of your model on the test data. **Make sure your code completes in under 5 minutes for the data you currently have.** You will receive 0 point for this section otherwise.

# Step 5: Submission
Submit your `sgd.py` to [cataylst](https://catalyst.uw.edu/collectit/dropbox/summary/akshays/38074).











