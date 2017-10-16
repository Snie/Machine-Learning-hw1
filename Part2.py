## Part 2
from activations import *
from mean_squared_error_functions import *
import plotters as plt
import numpy as np

def twospirals(n_points=120, noise=1.6, twist=420):
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
        self.levels = 3

        self.var = {
            "W1": np.random.randn(2,20),
            "b1": np.random.random_sample([1,1]),
            "W2":  np.random.randn(20,15) ,
            "b2": np.random.random_sample([1,1]),
            "W3": np.random.randn(15,1) ,
            "b3": np.random.random_sample([1,1])
        }

        ## End

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

        ## Implement
        a0 = np.array([inputs])

        z1 = a0.dot(W1) + b1
        a1 = tanh(z1)

        z2 = a1.dot(W2) + b2
        a2 = tanh(z2)

        z3 = a2.dot(W3) + b3
        a4 = sigmoid(z3)

        self.preA = [z1,z2,z3]
        self.A = [a0,a1,a2,a4]

        y = a4
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

        ## Implement
        level = self.levels
        # We backpropagate and we dynamically save the partial derivatives to later compute the
        # gradient with respect to every weight

        # delta at the last level is (y - Ti) activation'(Z)
        dW3 = error.dot(dsigmoid(self.preA[2]))
        db3 = np.mean(dW3)

        # for all the other levels delta is: (next_level_delta)X(next_level_weights)dot(derivative of activation func applied to this level pre activation values)
        dW2 = (dW3.dot(W3.T)) * dtanh(self.preA[1])
        db2 = np.mean(dW2)

        dW1 = (dW2.dot(W2.T)) * dtanh(self.preA[0])
        db1 = np.mean(dW1)

        # now we compute the gradient with respect to every weight with the following formula
        # (current level activations transposed)X(current level delta)
        dW3 = self.A[2].T.dot(dW3)
        dW2 = self.A[1].T.dot(dW2)
        dW1 = self.A[0].T.dot(dW1)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}
        return updates

    def adjust_weights(self, data, learning_rate):
        self.var['W1'] = self.var['W1'] - (data['W1']*learning_rate)
        self.var['b1'] = self.var['b1'] - (data['b1']*learning_rate)
        self.var['W2'] = self.var['W2'] - (data['W2']*learning_rate)
        self.var['b2'] = self.var['b2'] - (data['b2']*learning_rate)
        self.var['W3'] = self.var['W3'] - (data['W3']*learning_rate)
        self.var['b3'] = self.var['b3'] - (data['b3']*learning_rate)

    def momentum(self, data, learning_rate, momentum_parameter = 0.9):
        self.var['W1'] =  (data['W1'] * learning_rate) + (momentum_parameter*self.var['W1'])
        self.var['b1'] =  (data['b1'] * learning_rate) + (momentum_parameter*self.var['b1'])
        self.var['W2'] =  (data['W2'] * learning_rate) + (momentum_parameter*self.var['W2'])
        self.var['b2'] =  (data['b2'] * learning_rate) + (momentum_parameter*self.var['b2'])
        self.var['W3'] =  (data['W3'] * learning_rate) + (momentum_parameter*self.var['W3'])
        self.var['b3'] =  (data['b3'] * learning_rate) + (momentum_parameter*self.var['b3'])


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


def run_part2():
    """
    Train the multi layer perceptron according to the assignment.
    """
    # GET THE INPUT SPIRAL DATA AND THE LEARNING RATES
    X, T = twospirals()
    print("WHOLE SET: ", len(X))
    print(X)
    print("WHOLE SET TARGETS: ", len(T))
    print(T)

    learning_rates = [0.01 , 0.03, 0.04, 0.01, 0.1]

    # GET THE TRAINING AND TEST DATA WITH SIZE N
    test_size = 80
    train_size = 25
    test_X, test_T, train_X, train_T = get_input_data(X, T, len(X), test_size)
    print("TEST SET")
    print(test_X)
    print("TEST SET TARGETS")
    print(test_T)
    print("TRAIN SET")
    print(train_X)
    print("TRAIN SET TARGETS")
    print(train_T)
    to_plot = []
    # CREATE THE NEURAL NETWORK AND TRAIN IT
    nn = NeuralNetwork()
    for learning_rate in learning_rates:
        for j in range(3000):
            lr_predictions = np.empty([test_size])
            for i in range(test_size):
                y = nn.forward(train_X[i])
                lr_predictions[i] = y
                error = dMSE(y, train_T[i])
                weight_adjustments = nn.backward(error)
                nn.adjust_weights(weight_adjustments, learning_rate)
            mse = MSE(lr_predictions, train_T)[0]
            print(mse)
            to_plot.append(mse)
        plt.simple_plot(to_plot)
        plt.plot_boundary(nn,test_X, test_T)






def get_input_data(X,T,len_X,len):
    test_indexes = np.random.choice(len_X, len)
    print("INDEXES: ",test_indexes )
    test_X = np.empty([len, 2])
    train_X = np.empty([len, 2])
    test_T = np.empty([len, 1])
    train_T = np.empty([len, 1])
    n = 0
    for i in test_indexes:
        test_X[n] = X[i]
        test_T[n] = T[i]
        n += 1
    n = 0
    test_indexes = np.random.choice(len_X, len)
    for i in test_indexes:
        train_X[n] = X[i]
        train_T[n] = T[i]
        n += 1
    return test_X, test_T, train_X, train_T
