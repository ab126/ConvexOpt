import numpy as np


def gradient_descent(gradient_caller, n, x_0=None, t_0=1, step_size_method=None, func_caller=None, alpha=0.1, beta=0.8,
                     tol=None, max_iter=100_000):
    """
    Implements the gradient descent method

    :param gradient_caller: Function caller returning the gradient at the input point
    :param n: Dimension of the vector space in the respective optimization problem
    :param x_0: Inital points for gradient descent
    :param t_0: Initial step size
    :param step_size_method: Predetermined step sizes OR method for computing step sizes\
        - 'Amijo': Amijo method for computing step sizes
    :param func_caller: Caller that evaluate the objective function at a given point
    :param alpha: Alpha parameter in Amijo method for step size computation
    :param beta: Beta parameter in Amijo method for step size computation
    :param tol: Tolerance condition on norm of gradient for termination
    :param max_iter: Maximum number of iteration before termination
    :return: Minimizer point
    """

    if x_0 is None:
        x_0 = np.zeros(n)

    if step_size_method is None:
        step_size_method = 'Amijo'

    if tol is None:
        tol = 10 ** -8

    i = 0
    x_k = x_0
    t_k = t_0
    gradient = gradient_caller(x_0)
    gradient_prev = gradient.copy()
    while np.norm(gradient - gradient_prev) > tol and i < max_iter:
        # FILL
        gradient_prev = gradient
        gradient = gradient_caller(x_k)
        x_k = x_k - t_k * gradient
        if step_size_method == 'Amijo':
            t_k = amijo_step(x_k, t_k, func_caller, gradient_caller, alpha, beta)
        else:
            raise NotImplementedError("Step size method {} is not implemented".format(step_size_method))
        i += 1

    if i == max_iter:
        raise RuntimeError("Maximum number of iterations reached ({}) for gradient descent".format(max_iter))

    return x_k


def amijo_step(x_k, t_curr, func_caller, gradient_caller, alpha, beta, max_iter=10_000):
    """ Return the step size for the current iteration"""

    assert func_caller is not None, "func_caller must be provided for 'Amijo' step size method"

    assert 0 < beta < 1, "beta ({}) must be in range (0,1) for Amijo step size method".format(beta)
    assert 0 < alpha <= 0.5, "alpha ({}) must be in range (0,0.5] for Amijo step size method".format(alpha)

    i = 0
    t_k = t_curr
    x_next = x_k - t_curr * gradient_caller(x_k)
    while not func_caller(x_next) < func_caller(x_k) - alpha * t_k * np.linalg.norm(gradient_caller(x_k)) ** 2 and \
            i < max_iter:
        t_k = beta * t_k
        i += 1

    if i == max_iter:
        raise RuntimeError("Maximum number of iterations reached ({}) for Amijo step size iteration".format(max_iter))

    return t_k
