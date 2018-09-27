# coding: utf-8
""" This module gathers usefull optimisation functions.
"""
import numpy as np


def nesterov_forward_backward(grad, prox, v0=None, nb_iter=9999,
                              early_stopping=True, wind=8, tol=1.0e-8,
                              verbose=0):
    """ Nesterov forward backward algorithm.
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
        Minimizer.

    J : np.ndarray,
        Cost function values.
    """
    if wind < 2:
        raise ValueError("wind should at least 2, got {0}".format(wind))

    v = v0
    z_old = np.zeros_like(v0)
    J, xx = [], []
    t = t_old = 1

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        z = prox.op(v - grad.step * grad.op(v), w=1.0/grad.grad_lipschitz_cst)

        # fista acceleration
        t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
        v = z + (t_old-1)/t * (z - z_old)

        # update iterates
        t_old = t
        z_old = z

        # iterate update and saving
        xx.append(v)
        if len(xx) > wind:  # only hold the 'wind' last iterates
            xx = xx[1:]

        J.append(grad.cost(v) + prox.w * prox.cost(v))

        if verbose > 2:
            print("Iteration {0} / {1}, "
                  "cost function = {2}".format(j+1, nb_iter, J[j]))

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                crit_num = np.linalg.norm(new_iter - old_iter)
                crit_deno = np.linalg.norm(new_iter)
                diff = crit_num / (crit_deno + 1.0e-10)
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return v, np.array(J)
