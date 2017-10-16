import numpy as np

def sigmoid(x):

    return 1/(1 + np.exp(-x))

def dsigmoid(x):

    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    """
    Implements the hyperbolic tangent activation function.
    """
    tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    # End

    return tanh



def dtanh(x):
    """
    Implements the derivative of the hyperbolic tangent activation function.
    """
    ## Implement

    return 4*np.exp(-2*x)/((np.exp(-2*x)+1)**2)