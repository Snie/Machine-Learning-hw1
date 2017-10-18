import matplotlib.pyplot as plt
import numpy as np
def simple_plot(X):
    plt.plot(X)
    plt.show()

def close_plot():
    plt.close()

def compare_plots(X, Y,errorain ,errorest , learning_rate):
    plt.close()
    plt.subplot(211)
    plt.title("Training set, learning rate: " + str(learning_rate) +" MSE: "+ str(errorain)[0:4])
    plt.plot(X)

    plt.subplot(212)
    plt.title("Test set MSE: "+str(errorest)[0:4])
    plt.plot(Y)
    plt.show()


def plot_data(X,T):
    """
    Plots the 2D data as a scatterplot
    """
    plt.scatter(X[:,0], X[:,1], s=40, c=T, cmap=plt.cm.Spectral)
    plt.show()

def plot_boundary(model, X, targets, threshold=0.0):
    """
    Plots the data and the boundary lane which seperates the input space into two classes.
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y = model.forward(X_grid)
    plt.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
    plot_data(X, targets)
    plt.ylim([y_min, y_max])
