# coding: utf-8
""" This module gathers usefull optimisation functions.
"""
import matplotlib.pyplot as plt
import numpy as np
from .utils import spectral_radius_est, mad_daub_noise_est
from .linear import DiscretInteg, Conv


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
                              verbose=0, plotting=False):
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
    N = len(grad.y)

    if wind < 2:
        raise ValueError("wind should at least 2, got {0}".format(wind))

    v = v0
    z_old = np.zeros_like(v0)
    J, xx = [], []
    t = t_old = 1

    # main loop
    if verbose > 2:
        print("running main loop...")

    if plotting:
        fig = plt.figure(99, figsize=(15, 15))
        plt.ion()

    for j in range(nb_iter):

        # main update
        z = prox.op(v - grad.step * grad.op(v), w=1.0/grad.grad_lipschitz_cst)

        # fista acceleration
        t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
        v = z + (t_old-1)/t * (z - z_old)

        # update iterates
        t_old = t
        z_old = z

        i_s = v
        ai_s = grad.L.M.op(i_s)
        ar_s = Conv(grad.L.k, N).op(ai_s)

        # plotting
        if plotting and (j % int(nb_iter / 100) == 0):

            plt.gcf().clear()
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.plot(ar_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax1.set_title("Estimated activity relating signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax2 = fig.add_subplot(4, 1, 2)
            ax2.plot(i_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax2.set_title("Estimated innovation signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax3 = fig.add_subplot(4, 1, 3)
            ax3.plot(ai_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax3.set_title("Estimated activation inducing signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax4 = fig.add_subplot(4, 1, 4)
            ax4.plot(v, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax4.set_title("(Delta_L^T)^-1((x-y)/lambda) "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            plt.tight_layout()
            plt.pause(0.1)

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

    if plotting:
        plt.ioff()

    return v, np.array(J)


def fgp_dev(y, L, lbda=None, mu=1.0e-3, x0=None, v0=None, w=None, #noqa
            nb_iter=9999, early_stopping=True, wind=8, eps=1.0e-8,
            verbose=True, plotting=False):
    """ Fast Gradient Projection algorithm.

    Parameters
    ----------

    Results
    -------

    """
    # to use the heuristic lbda update
    adaptative_lbda = (lbda is None)
    if adaptative_lbda:
        sigma = mad_daub_noise_est(y)
        lbda_update_eps = sigma*np.sqrt(len(y)) * 1.0e-5
        lbda = sigma

    if wind < 2:
        raise ValueError("wind should at least 2, got {0}".format(wind))

    # prepare the iterate
    t = t_old = 1

    # iterates declarations
    if (v0 is not None) and (x0 is not None):
        print("Warning: Dual and primal init value given: "
              "will ignore the primal")
        v = v0
    elif v0 is not None:
        v = v0
    elif x0 is not None:
        v = L.op((1/lbda) * y - x0)
    else:
        v = np.zeros_like(y)
    z_old = np.zeros_like(v)

    # additional weights
    if w is None:
        w = np.ones_like(v)

    # saving variables
    J = np.zeros(nb_iter)
    R = np.zeros(nb_iter)
    lbdas = np.zeros(nb_iter)
    I_S = np.zeros((nb_iter, len(y)))

    # precompute L.op(y)
    L_y = L.op(y)

    if plotting:
        fig = plt.figure(99, figsize=(15, 15))
        plt.ion()

    # main loop
    for j in range(nb_iter):

        # main update
        z = (mu/lbda) * L_y + v - mu * L.op(L.adj(v))
        z = np.clip(z, -1.0/w, 1.0/w)  # w is additional weights

        # fista acceleration
        t = 0.5 * (1 + np.sqrt(1 + 4*t_old**2))
        v = z + (t_old-1)/t * (z - z_old)

        # iterate update and saving
        ar_s = y - lbda * L.adj(v)
        r = np.linalg.norm(ar_s-y)
        g = np.sum(np.abs(L.op(ar_s)))
        R[j] = r
        J[j] = 0.5 * r**2 + lbda * g

        if verbose:
            print("Iteration {0} / {1}, loss = {2}".format(j+1, nb_iter, J[j]))

        # clearly recover each signals to retrieve proper outputs
        i_s = L.op(ar_s)
        ai_s = np.cumsum(i_s)
        I_S[j, :] = i_s

        # lbda heuristic
        if adaptative_lbda:
            gap_noise_estim = np.abs(sigma*np.sqrt(len(y)) - r)
            if(gap_noise_estim > lbda_update_eps):
                lbda *= sigma*np.sqrt(len(y)) / r
        lbdas[j] = lbda

        # update iterates
        t_old = t
        z_old = z

        # plotting
        if plotting and (j % int(nb_iter / 100) == 0):

            plt.gcf().clear()
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.plot(ar_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax1.set_title("Estimated activity relating signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax2 = fig.add_subplot(4, 1, 2)
            ax2.plot(i_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax2.set_title("Estimated innovation signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax3 = fig.add_subplot(4, 1, 3)
            ax3.plot(ai_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax3.set_title("Estimated activation inducing signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax4 = fig.add_subplot(4, 1, 4)
            ax4.plot(v, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax4.set_title("(Delta_L^T)^-1((x-y)/lambda) "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            plt.tight_layout()
            plt.pause(0.1)

        # early stopping
        if early_stopping:
            if j > wind:
                sub_wind_len = int(wind/2)
                i_s_old = I_S[j-2*sub_wind_len:j-sub_wind_len, :].mean(axis=0)
                i_s_new = I_S[j-sub_wind_len:j, :].mean(axis=0)
                norm_1_num = np.sum(np.abs(i_s_new - i_s_old))
                norm_1_deno = np.sum(np.abs(i_s_old))
                i_s_diff = norm_1_num / norm_1_deno
                if i_s_diff < eps:
                    if verbose:
                        print("\n-----> early-stopping "
                              "done at {0}/{1}".format(j, nb_iter))
                    break

    if plotting:
        plt.ioff()

    # compute distance to minimum
    norm_2_i_s_final = np.linalg.norm(I_S[j, :])
    D = [np.linalg.norm(I_S[i, :] - I_S[j, :]) / norm_2_i_s_final
         for i in range(j)]
    D = np.array(D)

    return ar_s, ai_s, i_s, v, J[:j], R[:j], D, lbdas[:j]


def fgp(y, L, lbda=None, mu=1.0e-3, v0=None, nb_iter=9999, #noqa
        early_stopping=True, wind=8, tol=1.0e-8,
        plotting=False, verbose=0):
    """ Nesterov forward backward algorithm.
    Minimize on x:
        grad.cost(x) - prox.w * prox.f(L.op(x))
    e.g
        0.5 * || L h conv alpha - y ||_2^2 + lbda * || alpha ||_1

    Parameters
    ----------
    grad : object with op and adj method,
        Gradient operator.

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

    adaptative_lbda = lbda is None

    if adaptative_lbda:
        N = len(y)
        sigma = mad_daub_noise_est(y)
        lbda_update_eps = sigma*np.sqrt(N) * 1.0e-12
        lbda = sigma

    v = v0
    z_old = np.zeros_like(v0)
    J, xx = [], []
    t = t_old = 1

    L_y = L.op(y)

    integ = DiscretInteg()

    if plotting:
        fig = plt.figure(99, figsize=(15, 15))
        plt.ion()

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        z = (mu/lbda) * L_y + v - mu * L.op(L.adj(v))
        z = np.clip(z, -1.0, 1.0)

        # fista acceleration
        t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
        v = z + (t_old-1)/t * (z - z_old)

        # update iterates
        t_old = t
        z_old = z

        # iterate update and saving
        x = y - lbda * L.adj(v)
        ar_s = x
        i_s = L.adj(ar_s)
        ai_s = integ.op(i_s)

        r = np.linalg.norm(x-y)
        g = np.sum(np.abs(L.op(x)))
        J.append(0.5*r**2 + lbda * g)

        if plotting and (j % int(nb_iter / 100) == 0):

            plt.gcf().clear()
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.plot(ar_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax1.set_title("Estimated activity relating signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax2 = fig.add_subplot(4, 1, 2)
            ax2.plot(i_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax2.set_title("Estimated innovation signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax3 = fig.add_subplot(4, 1, 3)
            ax3.plot(ai_s, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax3.set_title("Estimated activation inducing signal "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            ax4 = fig.add_subplot(4, 1, 4)
            ax4.plot(v, '-b')
            plt.xlabel("n scans")
            plt.ylabel("amplitude")
            ax4.set_title("(Delta_L^T)^-1((x-y)/lambda) "
                          "(iter:{0}/{1})".format(j+1, nb_iter))

            plt.tight_layout()
            plt.pause(0.1)

        xx.append(v)
        if len(xx) > wind:  # only hold the 'wind' last iterates
            xx = xx[1:]

        if adaptative_lbda:
            gap_noise_estim = np.abs(sigma * np.sqrt(N) - r)
            if(gap_noise_estim > lbda_update_eps):
                lbda *= sigma * np.sqrt(N) / r

        if verbose > 2:
            print("iteration {0}/{1}, "
                  "cost function = {2:0.6f}".format(j+1, nb_iter, J[j]))

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

    if plotting:
        plt.ioff()

    return ar_s, ai_s, i_s, lbda, np.array(J)


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

    # main loop
    if verbose > 2:
        print("running main loop...")

    for j in range(nb_iter):

        # main update
        v = v - step * (grad.L.adj(grad.L.op(v)) - grad.L_adj_y)
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
