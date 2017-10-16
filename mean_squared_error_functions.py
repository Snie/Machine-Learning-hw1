import numpy as np

def MSE(prediction, target):
    """
    Computes the Mean Squared Error of a prediction and its target
    """
    y = prediction
    t = target
    n = prediction.size

    ## Implement
    error = np.absolute(t - y.reshape(-1,1))
    sum_error = sum(error)
    meanCost = sum_error**2
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

    #error = (1/n)*sum(y-t)
    error = y-t

    ## End
    return error