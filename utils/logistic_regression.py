import numpy as np


def logistic_reg_neg_log_likelihood(x, mat_a, b):
    """ Computes the negative log likelihood at a given point x=(w, c)"""
    m = mat_a.shape[0]
    mat_k = np.concatenate((mat_a, np.ones((m, 1))), axis=1)
    return np.sum(np.log(1 + np.exp(- b.reshape((-1, 1)) * (mat_k @ x)))) / m


def logistic_reg_neg_log_likelihood_gradient(x, mat_a, b):
    """ Computes the gradient of negative log likelihood at a given point x=(w, c)"""
    m = mat_a.shape[0]
    mat_k = np.concatenate((mat_a, np.ones((m, 1))), axis=1)
    mat_p = 1 / (1 + np.exp(- b.reshape((-1, 1)) * (mat_k @ x)))

    gradient = - (mat_k.T @ (b.reshape((-1, 1)) * (1 - mat_p))) / m
    return gradient