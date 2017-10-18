import numpy as np

def MSE(prediction, target):
    """
    Computes the Mean Squared Error of a prediction and its target
    """
    y = prediction
    t = target
    n = prediction.size

    ## Implement
    error = (t - y)**2
    meanCost = np.sum(error)

    meanCost /= 2*n



    ## End
    return meanCost

def dMSE(prediction, target):
    """
    Computes the derivative of the Mean Squared Error function.
    """
    y = prediction
    t = target
    n = prediction.size

    ## Implement

    error = (1/n)*(y-t)
    # error = y-t

    ## End
    return error