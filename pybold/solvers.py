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
    if verbose > 2:
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
                crit_num = np.linalg.norm(new_iter - old_iter)
                crit_deno = np.linalg.norm(new_iter)
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return x_new, np.array(J)


def forward_backward(grad, prox, v0=None, w=None, nb_iter=9999, #noqa
                     early_stopping=True, wind=8, tol=1.0e-8, verbose=0):
    """ Forward backward algorithm.
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

    # saving variables
    v = v0
    xx = []
    J = []

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        v = prox.op(v - grad.step * grad.op(v), w=1.0/grad.grad_lipschitz_cst)

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
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return v, np.array(J)



def nesterov_forward_backward(grad, prox, v0=None, nb_iter=9999, #noqa
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
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return v, np.array(J)


def fista(grad, prox, v0=None, nb_iter=9999, early_stopping=True, #noqa
          wind=8, tol=1.0e-8, verbose=0):
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
        Proximal operator, will not be called, only the attribute 'w' is used.

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

    # init values
    v = v0
    z_old = np.zeros_like(v0)
    t = t_old = 1

    # store usefull constant
    lbda = prox.w
    th = lbda * 1.0 / grad.grad_lipschitz_cst
    step = grad.step

    # gather iterates and cost function values
    xx = []

    # inline function
    grad_ = lambda x: grad.L.adj(grad.L.op(x)) - grad.L_adj_y

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        v = v - step * grad_(v)
        z = np.sign(v) * np.maximum(np.abs(v) - th, 0)

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

        if verbose > 2:
            print("Iteration {0} / {1}".format(j+1, nb_iter))

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                crit_num = np.linalg.norm(new_iter - old_iter)
                crit_deno = np.linalg.norm(new_iter)
                diff = crit_num / crit_deno
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}".format(j, nb_iter))
                    break

    return v, None


def admm_sparse_positif_ratio_hrf_encoding( #noqa
                y, L, v0=None, mu=1.0, nb_iter=9999, early_stopping=True,
                wind=8, tol=1.0e-8, verbose=0):
    """ ADMM for Sparse positif ratio HRF encoding algorithm.
    Minimize on x:
    || D_hrf alpha conv x - y ||_2^2 + I{alpha_i >= 0} + I{sum alpha_i = 1}

    Parameters
    ----------
    y : np.ndarray
        input 1d data.

    L : linear operator instance (see linear.py)
        Linear operator in the data fidelity term.

    v0 : np.ndarray (default=None),
        Initial guess for the iterate.

    mu : float (default=1.0)
        data fidelity term proximal operator parameter.

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

    mu = 1.0

    u_1, u_2, u_2 = np.copy(v0), np.copy(v0), np.copy(v0)
    d_1, d_2, d_3 = np.zeros_like(v0), np.zeros_like(v0), np.zeros_like(v0)
    p = len(v0)
    J = []

    D_hrf = L.M.M
    H = L.K
    A = H.dot(D_hrf)
    A_T = A.T
    B = np.linalg.pinv(np.eye(p) + mu * A_T.dot(A))

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # linear system of equations
        z = (1.0/3.0) * (u_1 + u_2 + u_2) + (1.0/3.0) * (d_1 + d_2 + d_3)

        # prox operator
        v_1 = z - d_1
        v_2 = z - d_2
        v_3 = z - d_3

        u_1 = B.dot(v_1 + mu * A_T.dot(y))  # data fidelity term

        u_2 = np.copy(v_2)  # positivity constraint
        u_2[u_2 < 0] = 0.0

        u_3 = 1.0 / p * np.ones(p)  # ratio to one constraint
        u_3 -= 1.0 / p * np.ones((p, p)).dot(v_3)
        u_3 += v_3

        # Lagrange multipliers
        d_1 = u_1 - v_1
        d_2 = u_2 - v_2
        d_3 = u_3 - v_3

        J.append(0.5 * np.sum(np.square(L.op(z) - y)))

        if verbose > 2:
            print("Iteration {0} / {1}, "
                  "cost function = {2}".format(j+1, nb_iter, J[j]))

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len], axis=0)
                new_j = np.mean(J[-sub_wind_len:], axis=0)
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 1:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}, "
                              "cost function = {2}".format(j, nb_iter, J[j]))
                    break

    return z, np.array(J)
