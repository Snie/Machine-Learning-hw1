from activations import *
from mean_squared_error_functions import *
import plotters as plt
import numpy as np

def get_part1_data():
    """
    Returns the toy data for the first part.
    """
    X = np.array([[1, 8],[6,2],[3,6],[4,4],[3,1],[1, 6],
              [6,10],[7,7],[6,11],[10,5],[4,11]])
    T = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).reshape(-1, 1)
    return X, T

class Perceptron:
    """
    Keeps track of the variables of the Perceptron model. Can be used for predictoin and to compute the gradients.
    """
    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        self.var = {
         "W": np.array([[.8], [-.5]]),
         "b": 2
        }

    def forward(self, inputs):
        """
        Implements the forward pass of the perceptron model and returns the prediction y. We need to
        store the current input for the backward function.
        """
        x = self.x = inputs
        W = self.var['W']
        b = self.var['b']

        ## Implement
        y = np.vdot(W, x)+b


        ## End
        return y

    def backward(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.
        """
        x = self.x
        db = error

        updates = {"W": x*error,
                   "b": db}

        return updates


def train_one_step(model, learning_rate, inputs, targets):
    """
    Uses the forward and backward function of a model to compute the error and updates the model
    weights while overwritting model.var. Returns the cost.
    """
    # n is the number of inputs in the input set
    n = len(inputs)
    # These three vectors are respectively the outputs y, the error derivatives for each of the elements in the training set
    # predictions[Y] = [Y1,...,Yn]
    predictions = np.zeros([n])
    # the derivative of the error with respect to the bias
    # biases[[dB]] = [
    #               [dB1],
    #               [...],
    #               [dBn]
    #              ]
    biases = np.zeros([n, 1])
    # the derivative of the error with respect of each weight
    # weights[[dW1, dW2]] = [
    #               [dW1.1, dW1.2],
    #               [...],
    #               [dWn.1, dWn.2]
    #              ]
    weights = np.zeros([n, 2])
    meanErr = 0
    ##  loops over each element the input set
    for i in range(n):
        # computes the neuron output
        y = model.forward(inputs[i])
        # compute the derivative of the error between the input set result and the input set category (target)
        error = dMSE(y, targets[i])
        # gets the derivative of the error with respect to the weights (dW) and to the bias (b)
        dW, b = model.backward(error).values()
        # insert the output values in the matrices/vectors
        predictions[i] = y
        weights[i] = dW
        biases[i] = b
        # utility to see the mean derivative of the error
        meanErr += error
    # get the mean between all delta weights and biases to adjust weights values
    new_weights = sum(weights)/n
    new_bias = sum(biases)/n
    model.var['W'] = model.var['W'] - (learning_rate * new_weights.reshape(-1, 1))
    model.var['b'] = model.var['b'] - (learning_rate * new_bias)

    cost = MSE(predictions.reshape(-1,1), targets)
    print("Predictions:")
    print(predictions.reshape(-1,1))
    ## End
    return cost

def run_part1():
    """
    Train the perceptron according to the assignment.
    """
    LEARNING_RATE = 0.02
    ITERATIONS =15
    perceptron = Perceptron()

    X, T = get_part1_data()
    Xp = []
    for i in range(ITERATIONS):
        Xp.append(train_one_step(perceptron, LEARNING_RATE, X, T))
    # print(Xp)
    plt.simple_plot(Xp)
