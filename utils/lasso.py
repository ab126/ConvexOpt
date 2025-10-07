import numpy as np


def lasso_f(x, mat_a, b, tau):
    """ Objective function for Lasso problem"""
    return 0.5 * (np.linalg.norm(mat_a @ x - b.reshape((-1, 1))) ** 2) + tau * np.linalg.norm(x, ord=1)


def lasso_sub_gradient(x, mat_a, b, tau):
    """ A subgradient of the Lasso objective function"""
    return mat_a.T @ mat_a @ x - mat_a.T @ b.reshape((-1, 1)) + tau * np.sign(x)


def lasso_diffable_gradient(x, mat_a, b):
    """ Gradient of the differentiable part in lasso problem"""
    return mat_a.T @ mat_a @ x - mat_a.T @ b.reshape((-1, 1))


def proximal_l1(x, tau):
    """ Computes the proximal mapping for h(x) = tau * ||x||_1 """
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lipschitz_const_caller(mat_a):
    """ Returns the Lipschitz constant for the lasso problem"""
    return np.linalg.norm(mat_a.T @ mat_a, ord=2)

