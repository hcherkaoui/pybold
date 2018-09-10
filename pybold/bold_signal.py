# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import numpy as np
from scipy.optimize import minimize
from .hrf_model import spm_hrf
from .linear import DiscretInteg, Conv, ConvAndLinear
from .gradient import L2ResidualLinear
from .solvers import nesterov_forward_backward
from .proximity import L1Norm
from .utils import Tracker, mad_daub_noise_est


def bold_bloc_deconvolution(noisy_ar_s, tr, hrf, lbda=None,
                            early_stopping=True, tol=1.0e-3, wind=2,
                            verbose=0):
    """ Deconvolve the given BOLD signal given an HRF convolution kernel.
    The source signal is supposed to be a bloc signal.

    Parameters:
    ----------
    noisy_ar_s : 1d np.ndarray,
        the observed bold signal.

    tr : float,
        the TR.

    hrf : 1d np.ndarray,
        the HRF.

    lbda : float (default=1.0),
        the regularization parameter.

    verbose : int (default=0),
        the verbosity level.

    Return:
    ------
    est_ar_s : 1d np.ndarray,
        the estimated convolved signal.

    est_ai_s : 1d np.ndarray,
        the estimated convolved signal.

    est_i_s : 1d np.ndarray,
        the estimated convolved signal.

    J : 1d np.ndarray,
        the evolution of the cost-function.
    """
    if wind < 2:
        raise ValueError("wind should at least 2, got {0}".format(wind))

    l_alpha, R, G = [], [], []
    N = len(noisy_ar_s)

    v0 = np.zeros(N)

    Integ = DiscretInteg()
    H = ConvAndLinear(Integ, hrf, dim_in=N, dim_out=N)
    grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)

    if lbda is not None:
        # solve 0.5 * || L h conv alpha - y ||_2^2 + lbda * || alpha ||_1
        prox = L1Norm(lbda)
        x, J = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose,
                          )

        est_i_s = x
        est_ai_s = Integ.op(x)
        est_ar_s = Conv(hrf, N).op(est_ai_s)

        return est_ar_s, est_ai_s, est_i_s, J, None

    else:
        # solve || x ||_1 sc  || L h conv alpha - y ||_2^2 < sigma
        sigma = mad_daub_noise_est(noisy_ar_s)  # estim. of the noise std
        nb_iter = 50  # nb iters for main loop
        alpha = 1.0  # init regularization parameter lbda = 1/(2*alpha)
        mu = 1.0e-2  # gradient step of the lbda optimization
        for i in range(nb_iter):
            # deconvolution step
            lbda = 1.0 / (2.0 * alpha)
            prox = L1Norm(lbda)
            x, _ = nesterov_forward_backward(
                            grad=grad, prox=prox, v0=v0, nb_iter=999,
                            early_stopping=True, verbose=verbose,
                              )
            # lambda optimization
            alpha += mu * (grad.residual(x) - N * sigma**2)

            # iterate update and saving
            l_alpha.append(alpha)
            if len(l_alpha) > wind:  # only hold the 'wind' last iterates
                l_alpha = l_alpha[1:]

            # metrics evolution
            r = grad.residual(x)
            g = np.sum(np.abs(x))
            R.append(r)
            G.append(g)
            if verbose > 0:
                print("Main loop: iteration {0:03d},"
                      " lbda = {1:0.6f},"
                      " l1-norm = {2:0.6f},"
                      " N*sigma**2 = {3:0.6f},"
                      " ||r-sigma||_2^2 = {4:0.6f}".format(i+1, lbda, g,
                                                           N*sigma**2, r))

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
        lbda = 1.0 / (2.0 * alpha)
        prox = L1Norm(lbda)
        x, _ = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose,
                          )

        est_i_s = x
        est_ai_s = Integ.op(x)
        est_ar_s = Conv(hrf, N).op(est_ai_s)

        return est_ar_s, est_ai_s, est_i_s, R, G


def scale_factor_fit_err(delta, ai_i_s, ar_s, tr, dur):
    """ Cost function for the scaled-gamma HRF model.
    e.g. 0.5 * || h*x - y ||_2^2 with h an scaled-gamma HRF model.

    Parameters:
    -----------
    delta : float
        the parameter of scaled-gamma HRF model.

    ai_i_s : 1d np.ndarray,
        the source signal.

    ar_s : 1d np.ndarray,
        the convolved signal.

    tr : float,
        the TR.

    dur : float,
        number of seconds on which represent the HRF.

    Return:
    -------
    cost : float,
        the cost-function for the given parameter.
    """
    if not isinstance(delta, float):
        delta = float(delta)
    hrf, _ = spm_hrf(delta=delta, dur=dur, tr=tr)
    H = Conv(hrf, len(ar_s))

    return np.sum(np.square(ar_s - H.op(ai_i_s)))


def hrf_scale_factor_estimation(ai_i_s, ar_s, tr=1.0, dur=60.0, verbose=0):
    """ HRF scaled Gamma function estimation.

    Parameters:
    -----------
    ai_i_s : 1d np.ndarray,
        the source signal.

    ar_s : 1d np.ndarray,
        the convolved signal.

    tr : float (default=1.0),
        the TR.

    dur : float (default=60.0),
        number of seconds on which represent the HRF.

    verbose : int (default=0)

    Return:
    ------
    hrf : 1d np.ndarray,
        the estimated HRF.

    J : 1d np.ndarray,
        the evolution of the cost-function.
    """
    params_init = 1.0

    cst_args = (ai_i_s, ar_s, tr, dur)
    f_cost = Tracker(scale_factor_fit_err, cst_args, verbose)

    res = minimize(fun=scale_factor_fit_err, x0=params_init,
                   args=cst_args, bounds=[(0.2 + 1.0e-1, 2.0 - 1.0e-1)],
                   callback=f_cost)
    delta = res.x
    J = f_cost.J

    hrf, _ = spm_hrf(delta=delta, tr=tr, dur=dur)

    return hrf, J


def scaled_hrf_blind_blocs_deconvolution(
                        noisy_ar_s, tr, lbda_bold=1.0, # noqa
                        init_delta=None, init_i_s=None, dur_hrf=60.0,
                        nb_iter=50, early_stopping=False, wind=24, tol=1.0e-24,
                        verbose=0):
    """ BOLD blind deconvolution function based on a scaled HRF model and an
    blocs BOLD model.
    """
    # initialization of the HRF
    est_delta = init_delta if init_delta is not None else 1.0
    est_hrf, _ = spm_hrf(tr=tr, delta=est_delta)

    N = len(noisy_ar_s)
    # definition of the usefull operator
    Integ = DiscretInteg()
    prox_bold = L1Norm(lbda_bold)

    # initialization of the source signal
    if init_i_s is None:
        est_i_s = np.zeros(N)  # init spiky signal
        est_ai_s = np.zeros(N)  # init spiky signal
        est_ar_s = np.zeros(N)  # thus.. init convolved signal
    else:
        est_i_s = init_i_s
        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

    # init cost function value
    J = []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    J.append(0.5 * r + lbda_bold * g_bold)

    for idx in range(nb_iter):

        # BOLD deconvolution
        H = ConvAndLinear(Integ, est_hrf, dim_in=N, dim_out=N)

        v0 = est_i_s
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _ = nesterov_forward_backward(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # HRF estimation
        cst_args = (est_ai_s, noisy_ar_s, tr, dur_hrf)
        res = minimize(fun=scale_factor_fit_err, x0=est_delta,
                       args=cst_args, bounds=[(0.2 + 1.0e-1, 2.0 - 1.0e-1)])
        est_hrf, _ = spm_hrf(delta=res.x, tr=tr, dur=dur_hrf)

        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        J.append(0.5 * r + lbda_bold * g_bold)

        if (verbose > 0):
            print("global cost-function "
                  "({0:03d}/{1:03d}): {2:.6f}".format(idx+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at "
                              "{0:03d}/{1:03d}, global"
                              " cost-function = {2:.6f}".format(idx, nb_iter,
                                                                J[idx]))
                    break
    J = np.array(J)

    return est_ar_s, est_ai_s, est_i_s, est_hrf, J


def scaled_hrf_blind_blocs_deconvolution_auto_lbda( # noqa
                        noisy_ar_s, tr, init_delta=None, init_i_s=None,
                        dur_hrf=60.0, nb_iter=50, early_stopping=False,
                        wind=24, tol=1.0e-24, verbose=0):
    """ BOLD blind deconvolution function based on a scaled HRF model and an
    blocs BOLD model.
    """
    # initialization of the HRF
    est_delta = init_delta if init_delta is not None else 1.0
    est_hrf, _ = spm_hrf(tr=tr, delta=est_delta)

    N = len(noisy_ar_s)
    # definition of the usefull operator
    Integ = DiscretInteg()

    sigma = mad_daub_noise_est(noisy_ar_s)  # estim. of the noise std
    nb_iter_deconv = 50  # nb iters for main loop
    alpha = 1.0  # init regularization parameter lbda = 1/(2*alpha)
    lbda = 1.0 / (2.0 * alpha)
    mu = 1.0e-2  # gradient step of the lbda optimization

    # initialization of the source signal
    if init_i_s is None:
        est_i_s = np.zeros(N)  # init spiky signal
        est_ai_s = np.zeros(N)  # init spiky signal
        est_ar_s = np.zeros(N)  # thus.. init convolved signal
    else:
        est_i_s = init_i_s
        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

    # init cost function value
    J, l_alpha = [], []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    J.append(0.5 * r + lbda * np.sum(np.abs(est_i_s)))

    for i in range(nb_iter):

        # BOLD DECONVOLUTION --------------------------------

        H = ConvAndLinear(Integ, est_hrf, dim_in=N, dim_out=N)
        v0 = np.zeros(N)
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)

        for j in range(nb_iter_deconv):
            # deconvolution step
            lbda = 1.0 / (2.0 * alpha)
            prox = L1Norm(lbda)
            x, _ = nesterov_forward_backward(
                            grad=grad, prox=prox, v0=v0, nb_iter=999,
                            early_stopping=True, verbose=verbose,
                              )
            # lambda optimization
            alpha += mu * (grad.residual(x) - N * sigma**2)

            # iterate update and saving
            l_alpha.append(alpha)
            if len(l_alpha) > wind:  # only hold the 'wind' last iterates
                l_alpha = l_alpha[1:]

            # early stopping
            if early_stopping:
                if j > 2:
                    sub_wind_len = int(wind/2)
                    old_iter = np.mean(l_alpha[:-sub_wind_len], axis=0)
                    new_iter = np.mean(l_alpha[-sub_wind_len:], axis=0)
                    crit_num = np.abs(new_iter - old_iter)
                    crit_deno = np.abs(new_iter)
                    diff = crit_num / crit_deno
                    if diff < 1.0e-2:
                        if verbose > 1:
                            print("\n-----> early-stopping "
                                  "done at {0:03d}/{1:03d}, "
                                  "cost function = {2:.6f}".format(j+1,
                                                                   nb_iter,
                                                                   J[j]))
                        break

        # deconvolution with larger number of iterations
        prox = L1Norm(1.0 / (2.0 * alpha))
        x, _ = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose,
                          )
        est_ai_s = Integ.op(x)

        # HRF ESTIMATION --------------------------------

        cst_args = (est_ai_s, noisy_ar_s, tr, dur_hrf)
        res = minimize(fun=scale_factor_fit_err, x0=est_delta,
                       args=cst_args, bounds=[(0.2 + 1.0e-1, 2.0 - 1.0e-1)])
        est_hrf, _ = spm_hrf(delta=res.x, tr=tr, dur=dur_hrf)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        J.append(0.5 * r + lbda * np.sum(np.abs(est_i_s)))

        if (verbose > 0):
            print("global cost-function "
                  "({0:03d}/{1:03d}): {2:.6f}".format(i+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if i > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at "
                              "{0:03d}/{1:03d}, global cost-function"
                              " = {2:.6f}".format(i, nb_iter, J[i]))
                    break
    J = np.array(J)

    return est_ar_s, est_ai_s, est_i_s, est_hrf, J
