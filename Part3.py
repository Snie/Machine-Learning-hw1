## Part 3
from activations import *
from mean_squared_error_functions import *
import plotters as plt
import numpy as np
import random

## Part 2
from activations import *
from mean_squared_error_functions import *
import plotters as plt
import numpy as np

def twospirals(n_points=240, noise=0.6, twist=600):
    """
     Returns a two spirals dataset.
    """
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))
    T = np.reshape(T, (T.shape[0], 1))
    return X, T


def compute_accuracy(model, X, T):
    """
    Computes the average accuracy over this data.
    """
    return np.mean(((model.forward(X) > 0.5) * 1 == T) * 1)


class NeuralNetwork:
    """
    Keeps track of the variables of the Multi Layer Perceptron model. Can be
    used for predictoin and to compute the gradients.
    """

    def __init__(self):
        """
        The variables are stored inside a dictonary to make them easy accessible.
        """
        ## Implement
        # pre-acrivation values
        self.preA = []
        # activations
        self.A = []

        self.old_momentum = {}
        self.momentum_parameter = 0.9

        self.var = {
            "W1": np.random.randn(2,20),
            "b1": np.random.random_sample([1,1]),
            "W2":  np.random.randn(20,30) ,
            "b2": np.random.random_sample([1,1]),
            "W3": np.random.randn(30, 15),
            "b3": np.random.random_sample([1, 1]),
            "W4": np.random.randn(15,5) ,
            "b4": np.random.random_sample([1,1]),
            "W5": np.random.randn(5, 1),
            "b5": np.random.random_sample([1, 1])
        }
        self.pre_var = self.var
        self.original = self.var.copy()

        ## End
    def reset(self):
        self.var = self.original.copy()


    def forward(self, inputs):
        """
        Implements the forward pass of the MLP model and returns the prediction y. We need to
        store the current input for the backward function.
        """
        x = self.x = inputs

        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']
        W4 = self.var['W4']
        b4 = self.var['b4']
        W5 = self.var['W5']
        b5 = self.var['b5']

        ## Implement
        a0 = inputs

        z1 = a0.dot(W1) + b1
        a1 = tanh(z1)

        z2 = a1.dot(W2) + b2
        a2 = tanh(z2)

        z3 = a2.dot(W3) + b3
        a3 = tanh(z3)

        z4 = a3.dot(W4) + b4
        a4 = sigmoid(z4)

        z5 = a4.dot(W5) + b5
        a5 = sigmoid(z5)

        self.preA = [z1, z2, z3, z4, z5]
        self.A = [a0, a1, a2, a3, a4, a5]

        y = a5
        ## End
        return y

    def backward(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.
        """
        x = self.x
        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']
        W4 = self.var['W4']
        b4 = self.var['b4']
        W5 = self.var['W5']
        b5 = self.var['b5']

        ## Implement
        # We backpropagate and we dynamically save the partial derivatives to later compute the
        # gradient with respect to every weight

        # delta at the last level is (y - Ti) activation'(Z)
        dW5 = error * (dsigmoid(self.preA[4]))
        db5 = np.mean(dW5)

        # for all the other levels delta is: (next_level_delta)X(next_level_weights)dot(derivative of activation func applied to this level pre activation values)
        dW4 = (dW5.dot(W5.T)) * dtanh(self.preA[3])
        db4 = np.mean(dW4)

        dW3 = (dW4.dot(W4.T)) * dtanh(self.preA[2])
        db3 = np.mean(dW3)

        dW2 = (dW3.dot(W3.T)) * dtanh(self.preA[1])
        db2 = np.mean(dW2)

        dW1 = (dW2.dot(W2.T)) * dtanh(self.preA[0])
        db1 = np.mean(dW1)

        # now we compute the gradient with respect to every weight with the following formula
        # (current level activations transposed)X(current level delta)
        dW5 = self.A[4].T.dot(dW5)
        dW4 = self.A[3].T.dot(dW4)
        dW3 = self.A[2].T.dot(dW3)
        dW2 = self.A[1].T.dot(dW2)
        dW1 = self.A[0].T.dot(dW1)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3,
                   "W4": dW4,
                   "b4": db4,
                   "W5": dW5,
                   "b5": db5
                   }
        return updates

    # same as backwards, but it subtracts the old momentum weights delta multiplied by the momentum parameter from the current weights
    # this accellerates the gradient descent used with momentum
    def backward_2(self, error):
        """
        Backpropagates through the model and computes the derivatives. The forward function must be
        run before hand for self.x to be defined. Returns the derivatives without applying them using
        a dictonary similar to self.var.
        """
        x = self.x
        W1 = self.var['W1'] - self.old_momentum['W1'] * self.momentum_parameter
        b1 = self.var['b1'] - self.old_momentum['b1'] * self.momentum_parameter
        W2 = self.var['W2'] - self.old_momentum['W2'] * self.momentum_parameter
        b2 = self.var['b2'] - self.old_momentum['b2'] * self.momentum_parameter
        W3 = self.var['W3'] - self.old_momentum['W3'] * self.momentum_parameter
        b3 = self.var['b3'] - self.old_momentum['b3'] * self.momentum_parameter
        W4 = self.var['W4'] - self.old_momentum['W4'] * self.momentum_parameter
        b4 = self.var['b4'] - self.old_momentum['b4'] * self.momentum_parameter
        W5 = self.var['W5'] - self.old_momentum['W5'] * self.momentum_parameter
        b5 = self.var['b5'] - self.old_momentum['b5'] * self.momentum_parameter

        ## Implement
        # We backpropagate and we dynamically save the partial derivatives to later compute the
        # gradient with respect to every weight

        # delta at the last level is (y - Ti) activation'(Z)
        dW5 = error * (dsigmoid(self.preA[4]))
        db5 = np.mean(dW5)

        # for all the other levels delta is: (next_level_delta)X(next_level_weights)dot(derivative of activation func applied to this level pre activation values)
        dW4 = (dW5.dot(W5.T)) * dtanh(self.preA[3])
        db4 = np.mean(dW4)

        dW3 = (dW4.dot(W4.T)) * dtanh(self.preA[2])
        db3 = np.mean(dW3)

        dW2 = (dW3.dot(W3.T)) * dtanh(self.preA[1])
        db2 = np.mean(dW2)

        dW1 = (dW2.dot(W2.T)) * dtanh(self.preA[0])
        db1 = np.mean(dW1)

        # now we compute the gradient with respect to every weight with the following formula
        # (current level activations transposed)X(current level delta)
        dW5 = self.A[4].T.dot(dW5)
        dW4 = self.A[3].T.dot(dW4)
        dW3 = self.A[2].T.dot(dW3)
        dW2 = self.A[1].T.dot(dW2)
        dW1 = self.A[0].T.dot(dW1)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3,
                   "W4": dW4,
                   "b4": db4,
                   "W5": dW5,
                   "b5": db5
                   }
        return updates
    # gradient descent
    def adjust_weights(self, data, learning_rate):
        self.var['W1'] = self.var['W1'] - (data['W1']*learning_rate)
        self.var['b1'] = self.var['b1'] - (data['b1']*learning_rate)
        self.var['W2'] = self.var['W2'] - (data['W2']*learning_rate)
        self.var['b2'] = self.var['b2'] - (data['b2']*learning_rate)
        self.var['W3'] = self.var['W3'] - (data['W3']*learning_rate)
        self.var['b3'] = self.var['b3'] - (data['b3']*learning_rate)
        self.var['W4'] = self.var['W4'] - (data['W4'] * learning_rate)
        self.var['b4'] = self.var['b4'] - (data['b4'] * learning_rate)
        self.var['W5'] = self.var['W5'] - (data['W5'] * learning_rate)
        self.var['b5'] = self.var['b5'] - (data['b5'] * learning_rate)

    # Momentum function for gradient descent
    def momentum(self, data, learning_rate):
        momentum_parameter = self.momentum_parameter
        updates = {
            "W1":(learning_rate * data['W1']) + (momentum_parameter*self.old_momentum['W1']),
            "b1":(learning_rate * data['b1']) + (momentum_parameter*self.old_momentum['b1']),
            "W2":(learning_rate * data['W2']) + (momentum_parameter*self.old_momentum['W2']),
            "b2":(learning_rate * data['b2']) + (momentum_parameter*self.old_momentum['b2']),
            "W3":(learning_rate * data['W3']) + (momentum_parameter*self.old_momentum['W3']),
            "b3":(learning_rate * data['b3']) + (momentum_parameter*self.old_momentum['b3']),
            "W4":(learning_rate * data['W4']) + (momentum_parameter * self.old_momentum['W4']),
            "b4":(learning_rate * data['b4']) + (momentum_parameter * self.old_momentum['b4']),
            "W5":(learning_rate * data['W5']) + (momentum_parameter * self.old_momentum['W5']),
            "b5":(learning_rate * data['b5']) + (momentum_parameter * self.old_momentum['b5'])
        }
        return updates

    def scale_weights(self, data, learning_rate):
        updates = {
            "W1": (learning_rate * data['W1']),
            "b1": (learning_rate * data['b1']),
            "W2": (learning_rate * data['W2']),
            "b2": (learning_rate * data['b2']),
            "W3": (learning_rate * data['W3']),
            "b3": (learning_rate * data['b3']),
            "W4": (learning_rate * data['W4']),
            "b4": (learning_rate * data['b4']),
            "W5": (learning_rate * data['W5']),
            "b5": (learning_rate * data['b5'])
            }

        return updates

    def adjust_momentum(self, data):
        self.var['W1'] = self.var['W1'] - (data['W1'])
        self.var['b1'] = self.var['b1'] - (data['b1'])
        self.var['W2'] = self.var['W2'] - (data['W2'])
        self.var['b2'] = self.var['b2'] - (data['b2'])
        self.var['W3'] = self.var['W3'] - (data['W3'])
        self.var['b3'] = self.var['b3'] - (data['b3'])
        self.var['W4'] = self.var['W4'] - (data['W4'])
        self.var['b4'] = self.var['b4'] - (data['b4'])
        self.var['W5'] = self.var['W5'] - (data['W5'])
        self.var['b5'] = self.var['b5'] - (data['b5'])

def gradient_check():
    """
    Computes the gradient numerically and analitically and compares them.
    """
    X, T = twospirals(n_points=10)
    NN = NeuralNetwork()
    eps = 0.0001

    for key, value in NN.var.items():
        row = np.random.randint(0, NN.var[key].shape[0])
        col = np.random.randint(0, NN.var[key].shape[1])
        print("Checking ", key, " at ", row, ",", col)

        ## Implement
        # analytic_grad = ...

        # x1 =  ...
        NN.var[key][row][col] += eps
        # x2 =  ...

        ## End
        numeric_grad = (x2 - x1) / eps
        print("numeric grad: ", numeric_grad)
        print("analytic grad: ", analytic_grad)
        if abs(numeric_grad - analytic_grad) < 0.00001:
            print("[OK]")
        else:
            print("[FAIL]")

def run_info(train_error, test_error, weights, learning_rate):
    return {
        "error_train" : train_error,
        "error_test" : test_error,
        "weights" : weights,
        "learning_rate" : learning_rate
    }

def run_part3():
    """
    Train the multi layer perceptron according to the assignment.
    """
    # GET THE INPUT SPIRAL DATA AND THE LEARNING RATES
    X, T = twospirals()
    print("WHOLE SET: ", len(X))
    print(X)
    print("WHOLE SET TARGETS: ", len(T))
    print(T)

    learning_rates = [0.01]
    # , 0.03, 0.07, 0.01, 0.1, 0.05]

    # GET THE TRAINING AND TEST DATA WITH SIZE N
    test_size = 96
    train_size = 384
    test_X, test_T, train_X, train_T = get_input_data(X, T, len(X), train_size, test_size)
    print("TEST SET")
    print(test_X)
    print("TEST SET TARGETS")
    print(test_T)
    print("TRAIN SET")
    print(train_X)
    print("TRAIN SET TARGETS")
    print(train_T)


    # some variables for plotting
    runs_info = []
    train_plot = []
    test_plot = []
    # CREATE THE NEURAL NETWORK AND TRAIN IT
    nn = NeuralNetwork()
    plt.plot_boundary(nn, test_X, test_T)
    plt.plot_boundary(nn, train_X, train_T)

    # variables containing errors the initial values are to test convergence
    train_mse, test_mse = 2, 2
    old_train_mse = 1
    momentum = {}
    # variable to test convergence
    landa = 0.00000001
    # counter for iterations
    n = 0
    # predictions
    for learning_rate in learning_rates:
        # convergence testing
        while abs(old_train_mse - train_mse) >= landa or n <= 1:
            y = nn.forward(train_X)
            error = dMSE(y, train_T)
            if n < 1:
                # first iteration its batch gradient descent to generate momentum pre values
                weight_adjustments = nn.backward(error)
                nn.old_momentum = nn.scale_weights(weight_adjustments, learning_rate)
                nn.adjust_weights(weight_adjustments, learning_rate)
            else:
                # from second iteration on the network uses momentum with nesterov accellerated descent
                weight_adjustments = nn.backward_2(error)
                momentum = nn.momentum(weight_adjustments, learning_rate)
                nn.adjust_momentum(momentum)
                nn.old_momentum = momentum.copy()
            old_train_mse = train_mse
            # calculates mean squared error
            train_mse = MSE(y, train_T)
            train_plot.append(train_mse)

            # Do a run with the test set without changing weights
            y = nn.forward(test_X)
            test_mse = MSE(y, test_T)
            test_plot.append(test_mse)
            n += 1
            if(n > 100):
                if(test_plot[-1] - test_plot[-2] > 0.01):
                    print("error increasing too much")
                    break

            print("Train MSE: ",train_mse," - Test MSE: ", test_mse)

        print("Iterations: ",n)
        plt.compare_plots(train_plot, test_plot, train_mse, test_mse, learning_rate)
        plt.plot_boundary(nn, test_X, test_T, 0.5)
        plt.plot_boundary(nn, train_X, train_T, 0.5)
        runs_info.append(run_info(train_plot, test_plot, nn.var.copy(), learning_rate))
        train_mse, test_mse = 1, 1
        n = 0
        train_plot = []
        test_plot = []
        nn.reset()








def get_input_data(X,T,len_X,len_train, len_test):
    a = len_train/2
    train_indexes = random.sample(range(int(len_X/2)), int(len_train/2))
    train_indexes += random.sample(range(int(len_X/2), int(len_X)), int(len_train/2))
    print("INDEXES: ",train_indexes )
    test_X = np.empty([len_test, 2])
    train_X = np.empty([len_train, 2])
    test_T = np.empty([len_test, 1])
    train_T = np.empty([len_train, 1])
    n = 0
    for i in train_indexes:
        train_X[n] = X[i]
        train_T[n] = T[i]
        n += 1
    n = 0
    tmp = range(len_X)
    test_indexes = []
    for i in tmp:
        if(i not in train_indexes):
            test_indexes.append(i)
    for i in test_indexes:
        test_X[n] = X[i]
        test_T[n] = T[i]
        n += 1
    return test_X, test_T, train_X, train_T


def competition_train_from_scratch(testX, testT):
    """
    Trains the BetterNeuralNet model from scratch using the twospirals data and calls the other
    competition funciton to check the accuracy.
    """
    trainX, trainT = twospirals(n_points=250, noise=0.6, twist=800)
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from scratch: ", compute_accuracy(NN, testX, testT))


def competition_load_weights_and_evaluate_X_and_T(testX, testT):
    """
    Loads the weight values from a file into the BetterNeuralNetwork class and computes the accuracy.
    """
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from trained model: ", compute_accuracy(NN, testX, testT))
