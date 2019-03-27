# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import numpy as np
import numba
from scipy.optimize import fmin_l_bfgs_b
from .hrf_model import spm_hrf, MIN_DELTA, MAX_DELTA
from .linear import DiscretInteg, ConvAndLinear
from .convolution import spectral_convolve, toeplitz_from_kernel
from .utils import Tracker, mad_daub_noise_est, spectral_radius_est


def deconv(y, t_r, hrf, lbda=None, early_stopping=True, tol=1.0e-6,  # noqa
           wind=6, nb_iter=1000, nb_sub_iter=1000, verbose=0):
    """ Deconvolve the given BOLD signal given an HRF convolution kernel.
    The source signal is supposed to be a bloc signal.

    Parameters:
    ----------
    y : 1d np.ndarray,
        the observed bold signal.

    t_r : float,
        the TR.

    hrf : 1d np.ndarray,
        the HRF.

    lbda : float (default=1.0),
        the regularization parameter.

    verbose : int (default=0),
        the verbosity level.

    Return:
    ------
    x : 1d np.ndarray,
        the estimated convolved signal.

    z : 1d np.ndarray,
        the estimated convolved signal.

    diff_z : 1d np.ndarray,
        the estimated convolved signal.

    J : 1d np.ndarray,
        the evolution of the cost-function.
    """
    diff_z = np.zeros_like(y)
    H = ConvAndLinear(DiscretInteg(), hrf, dim_in=len(y), dim_out=len(y))
    H_adj_y = H.adj(y)
    grad_lipschitz_cst = 0.9 * spectral_radius_est(H, diff_z.shape)
    step = 1.0 / grad_lipschitz_cst

    if lbda is not None:

        th = lbda / grad_lipschitz_cst
        diff_z_old = np.zeros_like(y)
        J, xx = [], []
        t = t_old = 1

        for idx in range(nb_iter):

            grad = H.adj(H.op(diff_z)) - H_adj_y
            diff_z -= step * grad
            diff_z = np.sign(diff_z) * np.maximum(np.abs(diff_z) - th, 0)

            t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
            diff_z = diff_z + (t_old-1)/t * (diff_z - diff_z_old)

            t_old = t
            diff_z_old = diff_z

            z = np.cumsum(diff_z)
            x = spectral_convolve(hrf, z)
            J.append(0.5 * np.sum(np.square(x - y)) +
                     lbda * np.sum(np.abs(diff_z)))

            print("Main loop: iteration {0:03d}, |grad| = {1:0.6f},"
                  " j = {2:0.6f}".format(idx, np.linalg.norm(grad), J[idx]))

            xx.append(diff_z_old)
            if len(xx) > wind:
                xx = xx[1:]

            if early_stopping:
                if idx > wind:
                    sub_wind_len = int(wind/2)
                    old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                    new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                    crit_num = np.linalg.norm(new_iter - old_iter)
                    crit_deno = np.linalg.norm(new_iter)
                    diff = crit_num / (crit_deno + 1.0e-10)
                    if diff < tol:
                        break

        return x, z, diff_z, np.array(J) / (J[0] + 1.0e-30), None, None

    else:
        l_alpha, J, R, G = [], [], [], []
        diff_z = np.zeros(len(y))
        diff_z_old = np.zeros(len(y))
        sigma = mad_daub_noise_est(y)
        alpha = 1.0
        lbda = 1.0 / (2.0 * alpha)
        mu = 1.0e-4
        for i in range(nb_iter):

            # deconvolution step
            th = lbda / grad_lipschitz_cst
            xx = []
            t = t_old = 1

            for j in range(nb_sub_iter):

                diff_z -= step * (H.adj(H.op(diff_z)) - H_adj_y)
                diff_z = np.sign(diff_z) * np.maximum(np.abs(diff_z) - th, 0)

                t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
                diff_z = diff_z + (t_old-1)/t * (diff_z - diff_z_old)

                t_old = t
                diff_z_old = diff_z

                xx.append(diff_z_old)
                if len(xx) > wind:
                    xx = xx[1:]

                if early_stopping:
                    if j > wind:
                        sub_wind_len = int(wind/2)
                        old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                        new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                        crit_num = np.linalg.norm(new_iter - old_iter)
                        crit_deno = np.linalg.norm(new_iter)
                        diff = crit_num / (crit_deno + 1.0e-10)
                        if diff < tol:
                            break

            # lambda optimization
            z = np.cumsum(diff_z)
            x = spectral_convolve(hrf, z)
            grad = np.sum(np.square(x - y)) - len(y) * sigma**2
            alpha += mu * grad
            lbda = 1.0 / (2.0 * alpha)

            # iterate update and saving
            l_alpha.append(alpha)
            if len(l_alpha) > wind:  # only hold the 'wind' last iterates
                l_alpha = l_alpha[1:]

            # metrics evolution
            r = np.sum(np.square(x - y))
            g = np.sum(np.abs(diff_z))
            R.append(r)
            G.append(g)
            J.append(0.5 * r + lbda * g)
            if verbose > 0:
                print("Main loop: iteration {0:03d},"
                      " |grad| = {1:0.6f},"
                      " lbda = {2:0.6f},".format(i+1, np.abs(grad), lbda))

            # early stopping
            if early_stopping:
                if i > wind:
                    sub_wind_len = int(wind/2)
                    old_iter = np.mean(l_alpha[:-sub_wind_len], axis=0)
                    new_iter = np.mean(l_alpha[-sub_wind_len:], axis=0)
                    crit_num = np.abs(new_iter - old_iter)
                    crit_deno = np.abs(new_iter)
                    diff = crit_num / crit_deno
                    if diff < tol:
                        if verbose > 1:
                            print("\n-----> early-stopping "
                                  "done at {0:03d}/{1:03d}, "
                                  "cost function = {2:.6f}".format(i, nb_iter,
                                                                   J[i]))
                        break

        # last deconvolution with larger number of iterations
        th = lbda / grad_lipschitz_cst
        xx = []
        t = t_old = 1

        for j in range(nb_sub_iter):

            diff_z -= step * (H.adj(H.op(diff_z)) - H_adj_y)
            diff_z = np.sign(diff_z) * np.maximum(np.abs(diff_z) - th, 0)

            t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
            diff_z = diff_z + (t_old-1)/t * (diff_z - diff_z_old)

            t_old = t
            diff_z_old = diff_z

            xx.append(diff_z_old)
            if len(xx) > wind:
                xx = xx[1:]

            if early_stopping:
                if j > wind:
                    sub_wind_len = int(wind/2)
                    old_iter = np.mean(xx[:-sub_wind_len], axis=0)
                    new_iter = np.mean(xx[-sub_wind_len:], axis=0)
                    crit_num = np.linalg.norm(new_iter - old_iter)
                    crit_deno = np.linalg.norm(new_iter)
                    diff = crit_num / (crit_deno + 1.0e-10)
                    if diff < tol:
                        break

        z = np.cumsum(diff_z)
        x = spectral_convolve(hrf, z)

        return x, z, diff_z, J, R, G


def hrf_fit_err(theta, z, y, t_r, hrf_dur):
    """ Cost function for the scaled-gamma HRF model.
    e.g. 0.5 * || h*x - y ||_2^2 with h an HRF model.
    """
    h, _ = spm_hrf(theta, t_r, hrf_dur, False)
    return 0.5 * np.sum(np.square(y - spectral_convolve(h, z)))


def hrf_estim(z, y, t_r, dur, verbose=0):
    """ Private function HRF estimation.
    """
    args = (z, y, t_r, dur)
    bounds = [(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)]
    f_cost = Tracker(hrf_fit_err, args, verbose)

    theta, _, _ = fmin_l_bfgs_b(
                        func=hrf_fit_err, x0=MAX_DELTA, args=args,
                        bounds=bounds, approx_grad=True, callback=f_cost,
                        maxiter=99999, pgtol=1.0e-12)
    J = f_cost.J
    h, _ = spm_hrf(theta, t_r, dur, False)

    return h, J


@numba.jit((numba.float64[:], numba.float64[:], numba.float64[:, :],
            numba.float64, numba.int64, numba.boolean, numba.int64,
            numba.float64),
           cache=True, nopython=True)
def _loops_deconv(y, diff_z, H, lbda, nb_iter, early_stopping, wind, tol):
    """ Main loop for deconvolution.
    """
    L = np.tril(np.ones((len(y), len(y))), 0)
    A = H.dot(L)
    A_t_A = A.T.dot(A)
    A_t_y = A.T.dot(y)
    grad_lipschitz_cst = np.linalg.norm(A_t_A)
    step = 1.0 / grad_lipschitz_cst
    th = lbda / grad_lipschitz_cst
    diff_z_old = np.zeros(len(y))
    t = t_old = 1

    for j in range(nb_iter):

        diff_z -= step * (A_t_A.dot(diff_z) - A_t_y)
        diff_z = np.sign(diff_z) * np.maximum(np.abs(diff_z) - th, 0)

        t = 0.5 * (1.0 + np.sqrt(1 + 4*t_old**2))
        diff_z = diff_z + (t_old-1)/t * (diff_z - diff_z_old)

        if early_stopping:
            if j > 2:
                crit_num = np.linalg.norm(diff_z - diff_z_old)
                crit_deno = np.linalg.norm(diff_z)
                diff = crit_num / (crit_deno + 1.0e-10)
                if diff < tol:
                    break

        t_old = t
        diff_z_old = diff_z

    return diff_z


def bd(y, t_r, lbda=1.0, theta_0=None, z_0=None, hrf_dur=20.0,  # noqa
       bounds=None, nb_iter=100, nb_sub_iter=1000, nb_last_iter=10000,
       print_period=50, early_stopping=False, wind=4, tol=1.0e-12, verbose=0):
    """ BOLD blind deconvolution function based on a scaled HRF model and an
    blocs BOLD model.
    """
    # force cast for Numba
    y = y.astype(np.float64)

    # initialization
    theta = MAX_DELTA if theta_0 is None else theta_0
    h, _ = spm_hrf(theta, t_r, hrf_dur, False)

    if z_0 is None:
        diff_z = np.zeros_like(y)
        z = np.zeros_like(y)
        x = np.zeros_like(y)
    else:
        diff_z = np.append(0, z_0[1:] - z_0[:-1])
        z = z_0
        x = spectral_convolve(h, z)

    if bounds is None:
        bounds = [(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)]

    d = {}
    r_0 = np.sum(np.square(x - y))
    d['r'] = [1.0]
    g_0 = np.sum(np.abs(diff_z))
    d['g'] = [g_0]
    j_0 = r_0 + lbda * g_0
    d['J'] = [1.0]
    d['l_alpha'] = []

    if (verbose > 0):
        print("normalized global cost-function "
              "(init): {0:.6f}".format(d['J'][-1]))

    # main loop
    for idx in range(nb_iter):

        # deconvolution
        H = toeplitz_from_kernel(h, dim_in=len(y), dim_out=len(y))
        diff_z = _loops_deconv(y, diff_z, H, lbda, nb_iter, early_stopping,
                               wind, tol)
        z = np.cumsum(diff_z)

        # hrf estimation
        args = (z, y, t_r, hrf_dur)
        theta, _, _ = fmin_l_bfgs_b(
                            func=hrf_fit_err, x0=theta, args=args,
                            bounds=bounds, approx_grad=True, maxiter=999,
                            pgtol=1.0e-12)
        h, _ = spm_hrf(theta, t_r, hrf_dur, False)
        x = spectral_convolve(h, z)

        # cost function
        r = np.sum(np.square(x - y))
        g = np.sum(np.abs(diff_z))
        d['J'].append((r + lbda * g) / j_0 + 1.0e-30)
        d['r'].append(r / r_0 + 1.0e-30)
        d['g'].append(g)

        if (verbose > 0) and ((idx+1) % print_period == 0):
            print("normalized global cost-function "
                  "({0:03d}/{1:03d}): {2:.6f}".format(idx+1, nb_iter,
                                                      d['J'][-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(d['J'][:-sub_wind_len])
                new_j = np.mean(d['J'][-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at "
                              "{0:03d}/{1:03d}, global"
                              " normalized cost-function = "
                              "{2:.6f}".format(idx, nb_iter, d['J'][idx]))
                    break

    # last (long) deconvolution
    H = toeplitz_from_kernel(h, dim_in=len(y), dim_out=len(y))
    diff_z = _loops_deconv(y, diff_z, H, lbda, nb_iter, early_stopping, wind,
                           tol)
    z = np.cumsum(diff_z)
    x = spectral_convolve(h, z)

    # cost function
    r = np.sum(np.square(x - y))
    g = np.sum(np.abs(diff_z))
    d['J'].append((r + lbda * g) / j_0)
    d['r'].append(r / r_0)
    d['g'].append(g)

    d['J'] = np.array(d['J'])
    d['r'] = np.array(d['r'])
    d['g'] = np.array(d['g'])

    return x, z, diff_z, h, d
