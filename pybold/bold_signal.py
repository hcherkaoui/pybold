# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import numpy as np
from scipy.optimize import fmin_bfgs, minimize
from .hrf_model import spm_hrf, il_hrf
from .linear import (Matrix, Diff, DiscretInteg, Conv,
                     LinearAndDeconv, ConvAndLinear)
from .gradient import L2ResidualLinear
from .solvers import (nesterov_forward_backward, fista,
                      admm_sparse_positif_ratio_hrf_encoding, fgp)
from .proximity import L1Norm
from .convolution import toeplitz_from_kernel
from .utils import (fwhm, Tracker, grid_search, spectral_radius_est,
                    mad_daub_noise_est)


# Sparse model amplitude correction


def sparse_hrf_ampl_corr(sparse_hrf, ar_s, hrf_dico, ai_s, th=1.0e-2):
    """ Re-estimate the amplitude of the spike signal.

    Try to correct the amplitude of the sparse encoding by doing a least square
    regression on the support.

    Parameters:
    -----------
    sparse_hrf : 1d np.ndarray,
        the sparse encoding of the HRF.

    ar_s : 1d np.ndarray,
        the convolved signal.

    hrf_dico : 2d np.ndarray,
        the parsifying dictionary.

    ai_s : 1d np.ndarray,
        the spike signal

    th : float (default=1.0e-2),
        the threshold to define the support

    Return:
    -------
    corr_hrf : 1d np.ndarray,
        the re-estimated HRF.

    corr_sparse_hrf : 1d np.ndarray,
        the re-estimated sparse encoding of the HRF.
    """
    # re-define the support
    mask = (np.abs(sparse_hrf) > np.max(np.abs(sparse_hrf)) * th)

    len_hrf, _ = hrf_dico.shape
    H = toeplitz_from_kernel(ai_s, dim_in=len_hrf, dim_out=len(ai_s))
    M = np.diag(mask)
    A = H.dot(hrf_dico).dot(M)

    # use the closed form solution
    L = np.linalg.pinv(A.T.dot(A)).dot(A.T)

    # solve J(i_s) = 0.5 * || H*Integ*M*i_s - M*y ||_2^2
    corr_sparse_hrf = L.dot(ar_s)
    corr_hrf = hrf_dico.dot(corr_sparse_hrf)

    return corr_hrf, corr_sparse_hrf


def i_s_ampl_corr(est_i_s, noisy_ar_s, hrf, th=1.0e-2):
    """ Re-estimate the amplitude of the spike signal.

    Try to correct the amplitude of the source signal by doing a least square
    regression on the support.

    Parameters:
    -----------
    est_i_s : 1d np.ndarray,
        the sparse encoding of the HRF.

    noisy_ar_s : 1d np.ndarray,
        the observed signal.

    hrf : 1d np.ndarray,
        the HRF.

    th : float (default=1.0e-2),
        the threshold to define the support

    Return:
    -------
    rest_ar_s : 1d np.ndarray,
        the re-estimated convolved signal.

    rest_ai_s : 1d np.ndarray,
        the re-estimated bloc signal.

    rest_i_s : 1d np.ndarray,
        the re-estimated source signal.
    """
    # re-define the support
    mask = (np.abs(est_i_s) > np.max(np.abs(est_i_s)) * th)

    H = toeplitz_from_kernel(hrf, dim_in=len(noisy_ar_s),
                             dim_out=len(noisy_ar_s))
    Integ = np.tril(np.ones((len(noisy_ar_s), len(noisy_ar_s))))
    M = np.diag(mask)
    A = H.dot(Integ).dot(M)

    # use the closed form solution
    L = np.linalg.pinv(A.T.dot(A)).dot(A.T)

    # solve J(i_s) = 0.5 * || H*Integ*M*i_s - M*y ||_2^2
    rest_i_s = L.dot(noisy_ar_s)
    rest_ai_s = Integ.dot(rest_i_s)
    rest_ar_s = H.dot(rest_ai_s)

    return rest_ar_s, rest_ai_s, rest_i_s


# Deconvolution


def bold_bloc_deconvolution_analysis(noisy_ar_s, tr, hrf,
                                     lbda=None, verbose=0):
    """ Deconvolve the given BOLD signal given an HRF convolution kernel.
    The source signal is supposed to be a bloc signal. Note: used the synthesis
    formulation.

    Parameters:
    ----------
    noisy_ar_s : 1d np.ndarray,
        the observed bold signal.

    tr : float,
        the TR.

    hrf : 1d np.ndarray,
        the HRF.

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
    N = len(noisy_ar_s)
    H = LinearAndDeconv(Diff(), hrf, dim_in=N, dim_out=N)
    v0 = np.zeros(N)

    L = spectral_radius_est(H, v0.shape)  # approx of the Lipschitz cst

    ar_s, ai_s, i_s, _, J = fgp(noisy_ar_s, H, lbda=lbda, mu=1.0/L, v0=v0,
                                nb_iter=99999, verbose=False, plotting=True)

    return ar_s, ai_s, i_s, J


def bold_bloc_deconvolution_synthesis(noisy_ar_s, tr, hrf,
                                      lbda=1.0, verbose=0):
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
    R, G = [], []
    N = len(noisy_ar_s)
    Integ = DiscretInteg()
    H = ConvAndLinear(Integ, hrf, dim_in=N, dim_out=N)
    v0 = np.zeros(N)
    grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)

    if lbda is not None:
        # solve 0.5 * || L h conv alpha - y ||_2^2 + lbda * || alpha ||_1
        prox = L1Norm(lbda)
        x, J = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose, plotting=False,
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
        for idx in range(nb_iter):
            # deconvolution step
            lbda = 1.0 / (2.0 * alpha)
            prox = L1Norm(lbda)
            x, _ = nesterov_forward_backward(
                            grad=grad, prox=prox, v0=v0, nb_iter=999,
                            early_stopping=True, verbose=verbose,
                            plotting=False,
                              )
            # lambda optimization
            alpha += mu * (grad.residual(x) - N * sigma**2)

            # inspect convergence
            r = grad.residual(x)
            g = np.sum(np.abs(x))
            R.append(r)
            G.append(g)
            print("Main loop: iteration {0:03d},"
                  " lbda = {1:0.4f},"
                  " l1-norm = {2:0.4f},"
                  " N*sigma**2 = {3:0.4f},"
                  " ||r-sigma||_2^2 = {4:0.4f}".format(idx+1, lbda, g,
                                                       N*sigma**2, r))

        # last deconvolution with larger number of iterations
        lbda = 1.0 / (2.0 * alpha)
        prox = L1Norm(lbda)
        x, _ = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose,
                        plotting=False,
                          )

        est_i_s = x
        est_ai_s = Integ.op(x)
        est_ar_s = Conv(hrf, N).op(est_ai_s)

        return est_ar_s, est_ai_s, est_i_s, R, G


def bold_event_deconvolution_synthesis(noisy_ar_s, tr, hrf,
                                       lbda=1.0, verbose=0):
    """ Deconvolve the given BOLD signal given an HRF convolution kernel.
    The source signal is supposed to be a Dirac signal.

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
    H = Conv(hrf, len(noisy_ar_s))
    z0 = np.zeros(len(noisy_ar_s))

    prox = L1Norm(lbda)
    grad = L2ResidualLinear(H, noisy_ar_s, z0.shape)

    x, J = nesterov_forward_backward(
                    grad=grad, prox=prox, v0=z0, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                      )
    est_i_s = x
    est_ar_s = Conv(hrf, len(noisy_ar_s)).op(est_i_s)

    return est_ar_s, est_i_s, J


# HRF estimation


def hrf_sparse_encoding_estimation(ai_i_s, ar_s, tr, hrf_dico, lbda=1.0,
                                   verbose=0):
    """ HRF sparse-encoding estimation.

    Paramters:
    ----------
    ai_i_s : 1d np.ndarray,
        the source signal.

    ar_s : 1d np.ndarray,
        the convolved signal.

    tr : float,
        the TR.

    hrf_dico : 2d np.ndarray,
        the parsifying dictionary.

    lbda : float (default=1.0),
        the parameter of regularization.

    verbose : int (default=0),
        the verbosity level.

    Return:
    -------
    hrf : 1d np.ndarray,
        the estimated HRF.

    sparce_encoding_hrf : 1d np.ndarray,
        the estimated sparse encoding of the HRF.

    J : 1d np.ndarray,
        the evolution of the cost-function.
    """
    # ai_i_s: either ai_s or i_s signal
    if not isinstance(hrf_dico, Matrix):
        hrf_dico = Matrix(hrf_dico)
    len_hrf, nb_atoms_hrf = hrf_dico.shape

    H = ConvAndLinear(hrf_dico, ai_i_s, dim_in=len_hrf, dim_out=len(ai_i_s))
    z0 = np.zeros(nb_atoms_hrf)

    prox = L1Norm(lbda)
    grad = L2ResidualLinear(H, ar_s, z0.shape)
    sparce_encoding_hrf, J = nesterov_forward_backward(
                    grad=grad, prox=prox, v0=z0, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                      )
    hrf = hrf_dico.op(sparce_encoding_hrf)

    return hrf, sparce_encoding_hrf, J


def logit_fit_err(hrf_logit_params, ai_i_s, ar_s, tr, dur):
    """ Cost function for the IL model.
    e.g. 0.5 * || h*x - y ||_2^2 with h an IL model.

    Parameters:
    -----------
    hrf_logit_params : 1d np.ndarray,
        the parameter of the IL model ([alpha_1, alpha_2, alpha_3, T1, T2, T3,
        D1, D2, D3]).

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
    hrf, _, _ = il_hrf(hrf_logit_params=hrf_logit_params, dur=dur, tr=tr)
    H = Conv(hrf, len(ar_s))

    return np.sum(np.square(ar_s - H.op(ai_i_s)))


def hrf_il_estimation(ai_i_s, ar_s, tr=1.0, dur=60.0, verbose=0):
    """ HRF three inverse-logit estimation.

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

    ils: list of 1d np.ndarray,
        the three IL that produce the estimated HRF.
    """
    # params = [alpha_1, alpha_2, alpha_3, T1, T2, T3, D1, D2, D3]
    params_init = np.array([1.0, -1.2, 0.2, 5.0, 10.0, 15.0, 1.33, 2.5, 2.0])

    cst_args = (ai_i_s, ar_s, tr, dur)
    f_cost = Tracker(logit_fit_err, cst_args, verbose)
    verbose = verbose > 0

    res = fmin_bfgs(f=logit_fit_err, x0=params_init,
                    args=cst_args, gtol=1.0e-12, maxiter=99999,
                    full_output=True, callback=f_cost, retall=True,
                    disp=verbose)
    xopt, fopt, gopt, Bopt, fun_calls, grad_calls, warnflag, allvecs = res

    hrf_logit_params = xopt
    J = f_cost.J
    hrf, _, il_s = il_hrf(hrf_logit_params, tr=tr, dur=dur)

    return hrf, J, il_s


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
                   args=cst_args, bounds=[(0.201, 1.99)],
                   callback=f_cost)
    delta = res.x
    J = f_cost.J

    hrf, _ = spm_hrf(delta=delta, tr=tr, dur=dur)

    return hrf, J


# Blind deconvolution


def sparse_encoding_hrf_blind_blocs_deconvolution( # noqa
                        noisy_ar_s, tr, hrf_dico, lbda_bold=1.0,
                        lbda_hrf=1.0, init_hrf=None, init_i_s=None,
                        hrf_fixed_ampl=False, nb_iter=50, early_stopping=False,
                        wind=24, tol=1.0e-24, verbose=0):
    """ BOLD blind deconvolution function based on a sparse encoding HRF model
    and an blocs BOLD model.
    """
    # cast hrf dictionnary/basis to the Matrix class used
    if not isinstance(hrf_dico, Matrix):
        hrf_dico = Matrix(hrf_dico)

    # initialization of the HRF
    if init_hrf is None:
        est_hrf, _, _ = spm_hrf(tr=tr, delta=1.0)  # init hrf
    else:
        est_hrf = init_hrf
    est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)  # sparse encoding init hrf

    # get the usefull dimension encounter in the problem
    len_hrf, nb_atoms_hrf = hrf_dico.shape
    N = len(noisy_ar_s)
    # definition of the usefull operator
    Integ = DiscretInteg()
    prox_bold = L1Norm(lbda_bold)

    # initialization of the source signal
    if init_i_s is None:
        est_i_s = np.zeros(N)  # init spiky signal
        est_ai_s = np.zeros(N)
        est_ar_s = np.zeros(N)  # thus.. init convolved signal
    else:
        est_i_s = init_i_s
        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

    # definition of the usefull operator
    Integ = DiscretInteg()
    prox_bold = L1Norm(lbda_bold)
    prox_hrf = L1Norm(lbda_hrf)

    # init cost function value
    J = []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    if not hrf_fixed_ampl:
        g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
        J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)
    else:
        J.append(0.5 * r + lbda_bold * g_bold)

    for idx in range(nb_iter):

        # BOLD deconvolution
        H = ConvAndLinear(Integ, est_hrf, dim_in=N, dim_out=N)

        v0 = est_i_s
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # HRF estimation
        H = ConvAndLinear(hrf_dico, est_ai_s, dim_in=len_hrf, dim_out=N)
        est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)
        v0 = est_sparse_encoding_hrf
        if hrf_fixed_ampl:
            est_sparse_encoding_hrf, _ = \
                admm_sparse_positif_ratio_hrf_encoding(
                     y=noisy_ar_s, L=H, v0=v0, mu=lbda_hrf, nb_iter=10000,
                     early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )
        else:
            grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
            est_sparse_encoding_hrf, _ = nesterov_forward_backward(
                    grad=grad, prox=prox_hrf, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )
        est_hrf = hrf_dico.op(est_sparse_encoding_hrf)

        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        if not hrf_fixed_ampl:
            g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
            J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)
        else:
            J.append(0.5 * r + lbda_bold * g_bold)

        if (verbose > 0):
            print("global cost-function "
                  "({0}/{1}): {2:.6f}".format(idx+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at {0}/{1}, global"
                              " cost-function = {2:.6f}".format(idx, nb_iter,
                                                                J[idx]))
                    break
    J = np.array(J)

    return est_ar_s, est_ai_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J


def sparse_encoding_hrf_blind_events_deconvolution( # noqa
                        noisy_ar_s, tr, hrf_dico, lbda_bold=1.0,
                        lbda_hrf=1.0, init_hrf=None, init_i_s=None,
                        hrf_fixed_ampl=False, nb_iter=50, early_stopping=False,
                        wind=24, tol=1.0e-24, verbose=0):
    """ BOLD blind deconvolution function based on a sparse encoding HRF model
    and an events BOLD model.
    """
    # cast hrf dictionnary/basis to the Matrix class used
    if not isinstance(hrf_dico, Matrix):
        hrf_dico = Matrix(hrf_dico)

    # initialization of the HRF
    if init_hrf is None:
        est_hrf, _, _ = spm_hrf(tr=tr, delta=1.0)  # init hrf
    else:
        est_hrf = init_hrf
    est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)  # sparse encoding init hrf

    # get the usefull dimension encounter in the problem
    len_hrf, nb_atoms_hrf = hrf_dico.shape
    N = len(noisy_ar_s)

    # initialization of the source signal
    if init_i_s is None:
        est_i_s = np.zeros(N)  # init spiky signal
        est_ar_s = np.zeros(N)  # thus.. init convolved signal
    else:
        est_i_s = init_i_s
        est_ar_s = Conv(est_hrf, N).op(est_i_s)

    # definition of the usefull operator
    prox_bold = L1Norm(lbda_bold)
    prox_hrf = L1Norm(lbda_hrf)

    # init cost function value
    J = []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    if not hrf_fixed_ampl:
        g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
        J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)
    else:
        J.append(0.5 * r + lbda_bold * g_bold)

    for idx in range(nb_iter):

        # BOLD deconvolution
        H = Conv(est_hrf, dim_in=N)

        v0 = est_i_s
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # HRF estimation
        H = ConvAndLinear(hrf_dico, est_i_s, dim_in=len_hrf, dim_out=N)
        est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)
        v0 = est_sparse_encoding_hrf
        if hrf_fixed_ampl:
            est_sparse_encoding_hrf, _ = \
                admm_sparse_positif_ratio_hrf_encoding(
                     y=noisy_ar_s, L=H, v0=v0, mu=lbda_hrf, nb_iter=10000,
                     early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )
        else:
            grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
            est_sparse_encoding_hrf, _ = nesterov_forward_backward(
                    grad=grad, prox=prox_hrf, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )
        est_hrf = hrf_dico.op(est_sparse_encoding_hrf)

        est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        if not hrf_fixed_ampl:
            g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
            J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)
        else:
            J.append(0.5 * r + lbda_bold * g_bold)

        if (verbose > 0):
            print("global cost-function "
                  "({0}/{1}): {2:.6f}".format(idx+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at {0}/{1}, global"
                              " cost-function = {2:.6f}".format(idx, nb_iter,
                                                                J[idx]))
                    break
    J = np.array(J)

    return est_ar_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J


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
        est_i_s, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # HRF estimation
        cst_args = (est_ai_s, noisy_ar_s, tr, dur_hrf)
        res = minimize(fun=scale_factor_fit_err, x0=est_delta,
                       args=cst_args, bounds=[(0.201, 1.99)])
        est_hrf, _ = spm_hrf(delta=res.x, tr=tr, dur=dur_hrf)

        est_ar_s = Conv(est_hrf, N).op(est_ai_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        J.append(0.5 * r + lbda_bold * g_bold)

        if (verbose > 0):
            print("global cost-function "
                  "({0}/{1}): {2:.6f}".format(idx+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at {0}/{1}, global"
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
    J = []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    J.append(0.5 * r + lbda * np.sum(np.abs(est_i_s)))

    for i in range(nb_iter):

        # BOLD DECONVOLUTION ###
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
                            plotting=False,
                              )
            # lambda optimization
            alpha += mu * (grad.residual(x) - N * sigma**2)

        # last deconvolution with larger number of iterations
        prox = L1Norm(1.0 / (2.0 * alpha))
        x, _ = nesterov_forward_backward(
                        grad=grad, prox=prox, v0=v0, nb_iter=9999,
                        early_stopping=True, verbose=verbose,
                        plotting=False,
                          )
        est_ai_s = Integ.op(x)

        # HRF ESTIMATION ###
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
                  "({0}/{1}): {2:.6f}".format(i+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if i > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at {0}/{1},"
                              " global cost-function"
                              " = {2:.6f}".format(i, nb_iter, J[i]))
                    break
    J = np.array(J)

    return est_ar_s, est_ai_s, est_i_s, est_hrf, J


def scaled_hrf_blind_events_deconvolution(
                        noisy_ar_s, tr, lbda_bold=1.0, # noqa
                        init_delta=None, init_i_s=None, dur_hrf=60.0,
                        nb_iter=50, early_stopping=False, wind=24, tol=1.0e-24,
                        verbose=0):
    """ BOLD blind deconvolution function based on a scaled HRF model and an
    events BOLD model.
    """
    # initialization of the HRF
    est_delta = init_delta if init_delta is not None else 1.0
    est_hrf, _ = spm_hrf(tr=tr, delta=est_delta)

    N = len(noisy_ar_s)

    # initialization of the source signal
    if init_i_s is None:
        est_i_s = np.zeros(N)  # init spiky signal
        est_ar_s = np.zeros(N)  # thus.. init convolved signal
    else:
        est_i_s = init_i_s
        est_ar_s = Conv(est_hrf, len(est_i_s)).op(est_i_s)

    # definition of the usefull operator
    prox_bold = L1Norm(lbda_bold)

    # init cost function value
    J = []
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    J.append(0.5 * r + lbda_bold * g_bold)

    for idx in range(nb_iter):

        # BOLD deconvolution
        H = Conv(est_hrf, dim_in=N)

        v0 = est_i_s
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        # HRF estimation
        cst_args = (est_i_s, noisy_ar_s, tr, dur_hrf)
        res = minimize(fun=scale_factor_fit_err, x0=est_delta,
                       args=cst_args, bounds=[(0.201, 1.99)])
        est_hrf, _ = spm_hrf(delta=res.x, tr=tr, dur=dur_hrf)

        est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # cost function
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        J.append(0.5 * r + lbda_bold * g_bold)

        if (verbose > 0):
            print("global cost-function "
                  "({0}/{1}): {2:.6f}".format(idx+1, nb_iter, J[-1]))

        # early stopping
        if early_stopping:
            if idx > wind:
                sub_wind_len = int(wind/2)
                old_j = np.mean(J[:-sub_wind_len])
                new_j = np.mean(J[-sub_wind_len:])
                diff = (new_j - old_j) / new_j
                if diff < tol:
                    if verbose > 0:
                        print("\n-----> early-stopping done at {0}/{1}, global"
                              " cost-function = {2:.6f}".format(idx, nb_iter,
                                                                J[idx]))
                    break
    J = np.array(J)

    return est_ar_s, est_i_s, est_hrf, J


# Cross-validation blind deconvolution


def sparse_encoding_hrf_blind_events_deconvolution_cv( # noqa
                    noisy_ar_s, tr, hrf_dico, t_hrf, orig_hrf,
                    lbda_bold=list(np.linspace(5.0e-2, 5.0, 5)),
                    lbda_hrf=list(np.linspace(5.0e-2, 5.0, 5)),
                    init_hrf=None, init_i_s=None, hrf_fixed_ampl=False,
                    nb_iter=50, early_stopping=False, wind=24, tol=1.0e-24,
                    n_jobs=1, verbose=0):
    """ Cross-validated BOLD blind deconvolution function based on a sparse
    encoding HRF model and an events BOLD model.
    """
    param_grid = {'noisy_ar_s': noisy_ar_s, 'tr': tr, 'hrf_dico': hrf_dico,
                  'lbda_bold': lbda_bold, 'lbda_hrf': lbda_hrf,
                  'init_hrf': init_hrf, 'init_i_s': init_i_s,
                  'hrf_fixed_ampl': hrf_fixed_ampl, 'nb_iter': nb_iter,
                  'early_stopping': early_stopping, 'wind': wind, 'tol': tol,
                  'verbose': 0}

    list_kwargs, res = grid_search(
                        sparse_encoding_hrf_blind_events_deconvolution,
                        param_grid, n_jobs=n_jobs, verbose=verbose
                                  )

    true_fwhm = fwhm(t_hrf, orig_hrf)
    errs_fwhm = []
    for (est_ar_s, est_i_s, est_hrf, sparse_encoding_hrf, J) in res:
        curr_fwhm = fwhm(t_hrf, est_hrf)
        errs_fwhm.append(np.abs(curr_fwhm - true_fwhm))
    idx_best = np.argmin(np.array(errs_fwhm))

    return list_kwargs[idx_best], res[idx_best]


def sparse_encoding_hrf_blind_blocs_deconvolution_cv( # noqa
                    noisy_ar_s, tr, hrf_dico, t_hrf, orig_hrf,
                    lbda_bold=list(np.linspace(5.0e-2, 5.0, 5)),
                    lbda_hrf=list(np.linspace(5.0e-2, 5.0, 5)),
                    init_hrf=None, init_i_s=None, hrf_fixed_ampl=False,
                    nb_iter=50, early_stopping=False, wind=24, tol=1.0e-24,
                    n_jobs=1, verbose=0):
    """ Cross-validated BOLD blind deconvolution function based on a sparse
    encoding HRF model and an blocs BOLD model.
    """
    param_grid = {'noisy_ar_s': noisy_ar_s, 'tr': tr, 'hrf_dico': hrf_dico,
                  'lbda_bold': lbda_bold, 'lbda_hrf': lbda_hrf,
                  'init_hrf': init_hrf, 'init_i_s': init_i_s,
                  'hrf_fixed_ampl': hrf_fixed_ampl, 'nb_iter': nb_iter,
                  'early_stopping': early_stopping, 'wind': wind, 'tol': tol,
                  'verbose': 0}

    list_kwargs, res = grid_search(
                        sparse_encoding_hrf_blind_blocs_deconvolution,
                        param_grid, n_jobs=n_jobs, verbose=verbose
                                  )

    true_fwhm = fwhm(t_hrf, orig_hrf)
    errs_fwhm = []
    for (est_ar_s, est_i_s, est_hrf, sparse_encoding_hrf, J) in res:
        curr_fwhm = fwhm(t_hrf, est_hrf)
        errs_fwhm.append(np.abs(curr_fwhm - true_fwhm))
    idx_best = np.argmin(np.array(errs_fwhm))

    return list_kwargs[idx_best], res[idx_best]


def scaled_hrf_blind_blocs_deconvolution_cv( # noqa
                        noisy_ar_s, tr, t_hrf, orig_hrf,
                        lbda_bold=list(np.linspace(5.0e-2, 5.0, 4)), # noqa
                        init_delta=None, init_i_s=None, dur_hrf=60.0,
                        nb_iter=50, early_stopping=False, wind=24, tol=1.0e-24,
                        n_jobs=1, verbose=0):

    """ Cross-validated BOLD blind deconvolution function based on a scaled
    HRF model and an blocs BOLD model.
    """
    param_grid = {'noisy_ar_s': noisy_ar_s, 'tr': tr, 'lbda_bold': lbda_bold,
                  'init_delta': init_delta, 'init_i_s': init_i_s,
                  'dur_hrf': dur_hrf, 'nb_iter': nb_iter,
                  'early_stopping': early_stopping, 'wind': wind, 'tol': tol,
                  'verbose': 0}

    list_kwargs, res = grid_search(
                        scaled_hrf_blind_blocs_deconvolution,
                        param_grid, n_jobs=n_jobs, verbose=verbose
                                  )

    true_fwhm = fwhm(t_hrf, orig_hrf)
    errs_fwhm = []
    for (est_ar_s, est_ai_s, est_i_s, est_hrf, J) in res:
        curr_fwhm = fwhm(t_hrf, est_hrf)
        errs_fwhm.append(np.abs(curr_fwhm - true_fwhm))
    idx_best = np.argmin(np.array(errs_fwhm))

    return list_kwargs[idx_best], res[idx_best]
