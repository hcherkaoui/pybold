# coding: utf-8
""" This module gathers usefull optimisation functions.
"""
import numpy as np
from .utils import spectral_radius_est


def condatvu(grad, L, prox,  x0, z0, w=None, sigma=0.5, tau=None, #noqa
             rho=1.0, nb_iter=999, early_stopping=True, wind=4, tol=1.0e-6,
             verbose=0):
    """ Condat-Vu algorithm.

    Minimize on x:
        grad.cost(x) + prox.w * prox.f(L.op(x))
    e.g
        0.5 * || h*x - y ||_2^2 + lbda * || Lx ||_1

    Parameters
    ----------
    grad : object with op and adj method,
        Gradient operator.

    L : object with op and adj method,
        The dual regularization operator.

    prox: object with __call__ method
        Proximal operator.

    x0 : np.ndarray (default=None),
        Initial guess for primal iterate.

    z0 : np.ndarray (default=None),
        Initial guess for dual iterate.

    w : np.ndarray (default=None),
        Weights.

    sigma : float (default=0.5),
        Dual step.

    tau : float (default=None),
        Primal step.

    rho: float (default=1.0),
        Update ratio.

    nb_iter : int (default=999),
        Number of iterations.

    early_stopping : bool (default=True),
        Whether or not to activate early stopping.

    wind : int (default=4),
        Windows on which average the early-stopping criterion.

    tol : float (default=1.0e-6),
        Tolerance on the the early-stopping criterion.

    verbose : int (default=0),
        Verbose level.

    Results
    -------
    X_new : np.ndarray,
        Primal minimizer.

    Z_new : np.ndarray,
        Dual minimizer.

    J : np.ndarray,
        Cost function values.

    """
    # estimate op-norm of dual operators
    op_norm = spectral_radius_est(L, x0.shape)
    lipschitz_cst = grad.grad_lipschitz_cst

    eps_ = 1.0e-8
    if (sigma is None) and (tau is None):
        sigma = 0.5
        tau = 1.0 / (lipschitz_cst/2 + sigma * op_norm**2 + eps_)
    if (sigma is not None) and (tau is None):
        tau = 1.0 / (lipschitz_cst/2 + sigma * op_norm**2 + eps_)
    if (sigma is None) and (tau is not None):
        sigma = (1.0 / tau - lipschitz_cst/2) / op_norm**2

    tmp = 1.0 / tau - sigma * op_norm ** 2
    convergence_test = (tmp >= lipschitz_cst / 2.0)

    if not convergence_test:
        raise ValueError((
            "Convergence test failed: 1.0 / tau - sigma * op_norm ** 2 = {0} "
            "and lipschitz_cst / 2.0 = {1}").format(tmp, lipschitz_cst / 2.0))

    # initialisation of the iterates
    x = x0
    x_old = np.zeros_like(x)
    z_old = np.copy(z0)
    if w is None:
        w = np.ones_like(z0)

    # saving variables
    J = []

    # main loop
    if verbose > 1:
        print("running main loop...")

    for j in range(nb_iter):

        # gradient descent
        x_prox = x_old - tau * grad.op(x) - tau * L.adj(z_old)

        # sparsity prox
        z_tmp = z_old + sigma * L.op(2 * x_prox - x_old)
        z_prox = z_tmp - sigma * prox.op(z_tmp / sigma, w=(w / sigma))

        # update
        x_new = rho * x_prox + (1 - rho) * x_old
        z_new = rho * z_prox + (1 - rho) * z_old

        # update previous iterates
        np.copyto(x_old, x_new)
        np.copyto(z_old, z_new)

        # iterate update and saving
        # x_new keeps increasing it amplitude through iterations...
        cost_value = grad.cost(x_new) + prox.w * prox.cost(L.op(x_new))
        J.append(cost_value)

        if verbose > 2:
            print("Iteration {0} / {1}, "
                  "cost function = {2}".format(j+1, nb_iter, J[j]))

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                old_iter = np.vstack(J[:-sub_wind_len]).mean()
                new_iter = np.vstack(J[-sub_wind_len:]).mean()
                crit_num = prox.cost(new_iter - old_iter)
                crit_deno = prox.cost(new_iter)
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return x_new, np.array(J)


def fista(grad, prox, v0=None, w=None, nb_iter=9999, early_stopping=True, #noqa
          wind=8, tol=1.0e-8, fista_acceleration=True, verbose=0):
    """ FISTA algorithm.
    Minimize on x:
        grad.cost(x) - prox.w * prox.f(L.op(x))
    e.g
        0.5 * || L h conv alpha - y ||_2^2 + lbda * || alpha ||_1

    Parameters
    ----------
    grad : object with op and adj method,
        Gradient operator.

    prox: object with __call__ method
        Proximal operator.

    v0 : np.ndarray (default=None),
        Initial guess for the iterate.

    w : np.ndarray (default=None),
        Weights.

    nb_iter : int (default=999),
        Number of iterations.

    early_stopping : bool (default=True),
        Whether or not to activate early stopping.

    wind : int (default=4),
        Windows on which average the early-stopping criterion.

    tol : float (default=1.0e-6),
        Tolerance on the the early-stopping criterion.

    fista_acceleration : bool (default=True),
        Enable Nesterov's acceleration

    verbose : int (default=0),
        Verbose level.

    Results
    -------
    X_new : np.ndarray,
        Minimizer.

    J : np.ndarray,
        Cost function values.
    """
    if wind < 2:
        raise ValueError("wind should at least 2, got {0}".format(wind))

    v = v0

    if fista_acceleration:
        z_old = np.zeros_like(v0)

    if w is None:
        w = np.ones_like(v0)

    # saving variables
    xx = []
    J = []

    # prepare the iterate
    if fista_acceleration:
        t = t_old = 1

    # additional weights
    if w is None:
        w = np.ones_like(v)

    # saving variables
    J, R, G = [], [], []
    xx = []

    # main loop
    if verbose > 1:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        z = prox.op(v - grad.step * grad.op(v), w=grad.step)

        # fista acceleration
        if fista_acceleration:
            t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
            v = z + (t_old-1)/t * (z - z_old)
        else:
            v = z

        # update iterates
        if fista_acceleration:
            t_old = t
            z_old = z

        # iterate update and saving
        xx.append(v)
        if len(xx) > wind:  # only hold the 'wind' last iterates
            xx = xx[1:]
        r = grad.cost(v)
        g = prox.cost(v)
        R.append(r)
        G.append(g)
        J.append(r + prox.w * g)

        if verbose > 2:
            print("Iteration {0} / {1}, "
                  "cost function = {2}".format(j+1, nb_iter, J[j], g))

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                crit_num = prox.cost(new_iter - old_iter)
                crit_deno = prox.cost(new_iter)
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return v, np.array(J), np.array(R), np.array(G)
