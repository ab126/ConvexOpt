import numpy as np


def gradient_descent(gradient_func, n, x_0=None, line_search=None, tol=None ):
    """ Implements the gradient descent method """

    if x_0 is None:
        x_0 = np.zeros(n)

    if tol is None:
        tol = 10 ** -8


