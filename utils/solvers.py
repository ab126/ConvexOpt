import warnings

import numpy as np


def gradient_descent(gradient_caller, n, x_0=None, t_0=1, step_size_method=None, func_caller=None, alpha=0.1, beta=0.8,
                     tol=None, max_iter=1_000, debug=False):
    """
    Implements the gradient descent method with optional backtracking line searches

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
    :param debug: If True, prints  debug information
    :return: Minimizer point
    """

    if x_0 is None:
        x_0 = np.zeros((n + 1, 1))

    if step_size_method is None:
        step_size_method = 'Amijo'

    if tol is None:
        tol = 1e-8

    i = 0
    x_k = x_0
    t_k = t_0
    grad_norms = []

    for i in range(max_iter):
        grad_k = gradient_caller(x_k)
        grad_norm = np.linalg.norm(grad_k)
        grad_norms.append(grad_norm)

        # Convergence
        if grad_norm < tol:
            return x_k, grad_norms

        if step_size_method == 'Amijo':
            t_k = amijo_step(x_k, t_k, func_caller, gradient_caller, alpha, beta, debug=debug)
        else:
            raise NotImplementedError(f"Step size method {step_size_method} is not implemented")

        # Debug

        if i % 100 == 0 and debug:
            print(f"[Grad Descend] Iter {i}: ||grad_k||={grad_norm:.6g}, t={t_k:.3e}")

        # Update
        x_k = x_k - t_k * grad_k

    warnings.warn(f"Maximum number of iterations reached ({max_iter}) in gradient descent")
    return x_k, grad_norms


def bfgs(gradient_caller, n, x_0=None, mat_b_0=None, t_0=1, step_size_method=None, func_caller=None, alpha=0.1, beta=0.8,
         tol=None, max_iter=1_000, debug=False):
    """
    Implements the BFGS method with optional backtracking line searches

    :param gradient_caller: Function caller returning the gradient at the input point
    :param n: Dimension of the vector space in the respective optimization problem
    :param x_0: Inital points for gradient descent
    :param mat_b_0: Initial matrix B approximating Hessian of the objective function
    :param t_0: Initial step size
    :param step_size_method: Predetermined step sizes OR method for computing step sizes\
        - 'Amijo': Amijo method for computing step sizes
    :param func_caller: Caller that evaluate the objective function at a given point
    :param alpha: Alpha parameter in Amijo method for step size computation
    :param beta: Beta parameter in Amijo method for step size computation
    :param tol: Tolerance condition on norm of gradient for termination
    :param max_iter: Maximum number of iteration before termination
    :param debug: If True, prints  debug information
    :return: Minimizer point
    """

    if x_0 is None:
        x_0 = np.zeros((n + 1, 1))

    if mat_b_0 is None:
        mat_b_0 = np.eye(n + 1)

    if step_size_method is None:
        step_size_method = 'Amijo'

    if tol is None:
        tol = 1e-8

    x_k = x_0.copy()
    t_k = t_0
    mat_b_k = mat_b_0.copy()
    grad_norms = []

    for i in range(max_iter):
        grad_k = gradient_caller(x_k)
        p_k = - np.linalg.solve(mat_b_k, grad_k)
        grad_norm = np.linalg.norm(grad_k)
        grad_norms.append(grad_norm)

        # Convergence
        if grad_norm < tol:
            return x_k, grad_norms

        if step_size_method == 'Amijo':
            t_k = amijo_step(x_k, t_k, func_caller, gradient_caller, alpha, beta, debug=debug)
        else:
            raise NotImplementedError(f"Step size method {step_size_method} is not implemented")

        # Debug

        if i % 100 == 0 and debug:
            print(f"[BFGS] Iter {i}: ||grad_k||={grad_norm:.6g}, t={t_k:.3e}")

        # Update
        x_k_plus_1 = x_k + t_k * p_k
        grad_k_plus_1 = gradient_caller(x_k_plus_1)
        y_k = grad_k_plus_1 - grad_k
        s_k = x_k_plus_1 - x_k

        mat_b_k = mat_b_k - (mat_b_k @ s_k @ s_k.T @ mat_b_k.T) / (s_k.T @ mat_b_k @ s_k) + \
                  (y_k @ y_k.T) / (y_k.T @ s_k )
        x_k = x_k_plus_1

        # Force positive definiteness for B_k
        mat_b_k = (mat_b_k + mat_b_k.T) / 2

    warnings.warn(f"Maximum number of iterations reached ({max_iter}) in gradient descent")
    return x_k, grad_norms


def amijo_step(x_k, t_curr, func_caller, gradient_caller, alpha, beta, max_iter=10_000, debug=False):
    """ Return the step size for the current iteration"""

    assert func_caller is not None, "func_caller must be provided for 'Amijo' step size method"
    assert 0 < beta < 1, "beta ({}) must be in range (0,1) for Amijo step size method".format(beta)
    assert 0 < alpha <= 0.5, "alpha ({}) must be in range (0,0.5] for Amijo step size method".format(alpha)

    i = 0

    grad_k = gradient_caller(x_k)
    grad_norm_sq = np.linalg.norm(grad_k) ** 2
    f_k = func_caller(x_k)

    t_k = t_curr

    for i in range(max_iter):
        x_next = x_k - t_k * grad_k
        f_next = func_caller(x_next)

        if f_next <= f_k - alpha * t_k * grad_norm_sq:
            return t_k

        # Debug

        if i % 100 == 0 and debug:
            print(f"[Amijo] Iter {i}: f_k={f_k:.6f}, f_next={f_next:.6f}, t={t_k:.3e}")

        t_k *= beta

    if i == max_iter:
        raise RuntimeError("Maximum number of iterations reached ({}) for Amijo step size iteration".format(max_iter))

    return t_k


def sub_gradient_method(sub_gradient_caller, n, func_caller, xs=None, x_0=None, t_0=.01, tol=None, max_iter=5_000,
                        debug=False):
    """
    Implements the subgradient method with diminishing step size

    :param sub_gradient_caller: Function caller returning the sub-gradient at the input point
    :param n: Dimension of the vector space in the respective optimization problem
    :param x_0: Initial points for gradient descent
    :param t_0: Initial step size
    :param func_caller: Caller that evaluate the objective function at a given point
    :param xs: The optimal solution, if given it is used for convergence criteria in relative error
    :param tol: Tolerance condition on norm of gradient for termination
    :param max_iter: Maximum number of iteration before termination
    :param debug: If True, prints  debug information
    :return: Minimizer point
    """

    if x_0 is None:
        x_0 = np.zeros((n, 1))

    if tol is None:
        tol = 1e-3

    x_k = x_0.copy()
    rel_errors = []
    f_prev = np.inf
    if xs is not None:
        fs = func_caller(xs)
    f_best = func_caller(x_0)

    for i in range(max_iter):
        sub_grad_k = sub_gradient_caller(x_k)
        t_k = t_0 / (i+1)
        x_k_new = x_k - t_k * sub_grad_k
        f_new = func_caller(x_k_new)
        if f_new < f_best:
            f_best = f_new

        if xs is not None:
            rel_error = np.abs(f_best - fs) / fs
        else:
            rel_error = np.abs((f_best - f_prev) / f_prev)
        rel_errors.append(rel_error)

        # Convergence
        if rel_error < tol:
            return x_k, rel_errors

        # Debug
        if i % 100 == 0 and debug:
            print(f"[Sub Gradient] Iter {i}: rel_error={rel_error:.6g}, t={t_k:.3e}")

        # Update
        x_k = x_k_new
        f_prev = f_new

    warnings.warn(f"Maximum number of iterations reached ({max_iter}) in gradient descent")
    return x_k, rel_errors


def ista(gradient_caller, proximal_mapping, n, func_caller, lipschitz_const, xs=None, x_0=None, tol=None,
         max_iter=5_000, debug=False):
    """
    Implements the ISTA proximal gradient method with fixed step size

    :param gradient_caller: Function caller returning the gradient of the differentiable portion at the input
    :param proximal_mapping: Proximal mapping of the non-differentiable function at the input and given time step t
    :param n: Dimension of the vector space in the respective optimization problem
    :param func_caller: Caller that evaluate the objective function at a given point
    :param x_0: Initial points for gradient descent
    :param lipschitz_const: Lipschitz constant for fixed step size
    :param xs: The optimal solution, if given it is used for convergence criteria in relative error
    :param tol: Tolerance condition on norm of gradient for termination
    :param max_iter: Maximum number of iteration before termination
    :param debug: If True, prints  debug information
    :return: minimizer, function_errors
    """

    if x_0 is None:
        x_0 = np.zeros((n, 1))

    if tol is None:
        tol = 1e-3

    x_k = x_0.copy()
    t = 1 / lipschitz_const
    rel_errors = []
    f_prev = np.inf
    if xs is not None:
        fs = func_caller(xs)

    for i in range(max_iter):
        grad_k = gradient_caller(x_k)
        x_k_new = proximal_mapping(x_k - t * grad_k, t)
        f_new = func_caller(x_k_new)

        if xs is not None:
            rel_error = np.abs(f_new - fs) / fs
        else:
            rel_error = np.abs((f_new - f_prev) / f_prev)
        rel_errors.append(rel_error)

        # Convergence
        if rel_error < tol:
            return x_k, rel_errors

        # Debug
        if i % 100 == 0 and debug:
            print(f"[Sub Gradient] Iter {i}: rel_error={rel_error:.6g}, t={t:.3e}")

        # Update
        x_k = x_k_new
        f_prev = f_new

    warnings.warn(f"Maximum number of iterations reached ({max_iter}) in gradient descent")
    return x_k, rel_errors


def grad_admm_logreg(grad_caller, prox_mapping, n, func_caller, tau, t, x_0=None, lambda_0=None, tol=None,
                     max_iter=5_000, debug=False):
    """
    Implements gradient-based Alternating Directions Method of Multipliers for the sparse logistic regression
     problem: min tau*f(x)+g(y) where f has easy proximal mapping and g is differentiable

    :param grad_caller: Function caller returning the gradient of the differentiable portion at the input
    :param prox_mapping: Function caller for the easy proximal mapping of tau*f(x)
    :param n: Dimension of the vector space in the respective optimization problem
    :param func_caller: Caller that evaluates the objective function at a given point
    :param tau: Constant of f(x)
    :param t: Step size
    :param x_0: Initial point
    :param lambda_0: Initial multiplier
    :param tol: Tolerance condition on norm primal and dual residual for convergence
    :param max_iter: Maximum number of iteration before termination
    :param debug: If True, prints debug information
    :return: minimizer, function_errors
    """

    if x_0 is None:
        x_0 = np.zeros((n, 1))

    if lambda_0 is None:
        lambda_0 = np.zeros((n, 1))

    if tol is None:
        tol = 1e-3

    x_k = x_0.copy()
    y_k = x_0.copy()
    lambda_k = lambda_0.copy()
    rel_errors = []
    f_prev = np.inf

    for i in range(max_iter):

        # Sub-problem x
        x_k_new = prox_mapping(y_k + lambda_k / t, tau/t)

        # Sub-problem y
        y_k_new = y_k - t * grad_caller(y_k) - t * lambda_k - (t ** 2) * (y_k - x_k_new)

        # Multiplier
        lambda_k_new = lambda_k - t * (x_k_new - y_k_new)

        f_new = func_caller( x_k_new )

        rel_error = np.abs((f_new - f_prev) / f_prev)
        rel_errors.append(rel_error)

        # Convergence via primal and dual residual
        primal_res = np.linalg.norm(x_k_new - y_k_new)
        dual_res = np.linalg.norm(y_k_new - y_k)
        if primal_res < tol and dual_res < tol:
            return x_k, rel_errors

        # Debug
        if i % 100 == 0 and debug:
            print(f"[Grad-ADMM] Iter {i}: rel_error={rel_error:.6g}, t={t:.3e}")

        # Update
        x_k = x_k_new
        y_k = y_k_new
        lambda_k = lambda_k_new

    warnings.warn(f"Maximum number of iterations reached ({max_iter}) in gradient descent")
    return x_k, rel_errors


