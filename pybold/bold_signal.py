# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import itertools
import psutil
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from .hrf_model import spm_hrf
from .linear import Matrix, DiscretInteg, Conv, ConvAndLinear
from .gradient import L2ResidualLinear
from .solvers import (nesterov_forward_backward, fista,
                      admm_sparse_positif_ratio_hrf_encoding)
from .proximity import L1Norm
from .convolution import toeplitz_from_kernel
from .utils import fwhm


def sparse_hrf_ampl_corr(sparse_hrf, ar_s, hrf_dico, ai_s, th=1.0e-2):
    """ Re-estimate the amplitude of the spike signal.
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


def bold_deconvolution(noisy_ar_s, tr, hrf, lbda=1.0, model_type='bloc',
                       verbose=0):
    """ Deconvolve the given BOLD signal.
    """
    if model_type == 'bloc':
        Integ = DiscretInteg()
        H = ConvAndLinear(Integ, hrf, dim_in=len(noisy_ar_s),
                          dim_out=len(noisy_ar_s))
    elif model_type == 'event':
        H = Conv(hrf, dim_in=len(noisy_ar_s))
    else:
        raise ValueError("model_type should be in ['bloc', 'event']")

    z0 = np.zeros(len(noisy_ar_s))

    prox = L1Norm(lbda)
    grad = L2ResidualLinear(H, noisy_ar_s, z0.shape)

    x, J = nesterov_forward_backward(
                    grad=grad, prox=prox, v0=z0, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                      )
    est_i_s = x

    if model_type == 'bloc':
        est_ai_s = Integ.op(x)
        est_ar_s = Conv(hrf, len(noisy_ar_s)).op(est_ai_s)

        return est_ar_s, est_ai_s, est_i_s, J

    elif model_type == 'event':
        est_ar_s = Conv(hrf, len(noisy_ar_s)).op(est_i_s)

        return est_ar_s, est_i_s, J


def hrf_sparse_encoding_estimation(ai_i_s, ar_s, tr, hrf_dico, lbda=None,
                                   verbose=0):
    """ HRF sparse-encoding estimation.
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


def _inv_logit(x):
    """ Inverse logit function
    """
    return np.exp(x - np.log(1 + np.exp(x)))  # avoid overflows


def _hrf_from_logit_params(hrf_logit_params, dur, tr):
    """ Return the HRF from the specified inv. logit parameters.
    """
    alpha_1, T1, T2, T3, D1, D2, D3 = hrf_logit_params

    num_alpha_2 = _inv_logit(-T1)/D1 - _inv_logit(-T3)/D3
    den_alpha_2 = _inv_logit(-T3)/D3 + _inv_logit(-T2)/D2
    tmp = np.exp(np.log(num_alpha_2) - np.log(den_alpha_2))  # avoid overflows
    alpha_2 = - alpha_1 * tmp

    alpha_3 = np.abs(alpha_2) - np.abs(alpha_1)

    t_hrf = np.linspace(0, dur, float(dur)/tr)
    il_1 = _inv_logit((t_hrf-T1)/D1)
    il_2 = _inv_logit((t_hrf-T2)/D2)
    il_3 = _inv_logit((t_hrf-T3)/D3)
    hrf = alpha_1 * il_1 + alpha_2 * il_2 + alpha_3 * il_3

    return hrf, [il_1, il_2, il_3], t_hrf


def _se_logit_fit(hrf_logit_params, ai_i_s, ar_s, tr, dur):
    """ 0.5 * || h*x - y ||_2^2 with h being a three IL hrf model.
    """
    hrf, _, _ = _hrf_from_logit_params(hrf_logit_params, dur, tr)
    H = Conv(hrf, len(ar_s))
    est_ar_s = H.op(ai_i_s)

    return np.sum(np.square(ar_s - est_ar_s))


def hrf_il_estimation(ai_i_s, ar_s, tr=1.0, dur=25.0):
    """ HRF three inverse-logit estimation.
    """
    # params = [alpha_1, T1, T2, T3, D1, D2, D3]
    params_init = np.array([1.0, 6.0, 10.0, 15.0, 1.0, 2.0, 1.0])

    hrf, _, _ = _hrf_from_logit_params(params_init, dur, tr)
    import matplotlib.pyplot as plt
    plt.plot(hrf)
    plt.show()

    cst_args = (ai_i_s, ar_s, tr, dur)
    res = minimize(fun=_se_logit_fit, x0=params_init,
                   args=cst_args, method='BFGS')
    hrf_logit_params = res.x
    hrf, _, _ = _hrf_from_logit_params(hrf_logit_params, dur, tr)

    return hrf


def bold_blind_deconvolution(noisy_ar_s, tr, hrf_dico, lbda_bold=1.0, # noqa
                             lbda_hrf=1.0, init_hrf=None, hrf_fixed_ampl=False,
                             model_type='bloc', nb_iter=50,
                             early_stopping=False, wind=24, tol=1.0e-24,
                             verbose=0):
    """ Blind deconvolution of the BOLD signal.
    """
    # cast hrf dictionnary/basis to the Matrix class used
    if not isinstance(hrf_dico, Matrix):
        hrf_dico = Matrix(hrf_dico)

    # initialization of the HRF
    if init_hrf is None:
        est_hrf, _, _ = spm_hrf(tr=tr, time_length=30.0)  # init hrf
    else:
        est_hrf = init_hrf

    # get the usefull dimension encounter in the problem
    len_hrf, nb_atoms_hrf = hrf_dico.shape
    N = len(noisy_ar_s)

    J = []

    # definition of the usefull operator
    Integ = DiscretInteg()
    prox_bold = L1Norm(lbda_bold)
    prox_hrf = L1Norm(lbda_hrf)

    # initialization of the signal of interess
    est_i_s = np.zeros(N)  # init spiky signal
    est_ar_s = np.zeros(N)  # thus.. init convolved signal
    est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)  # sparse encoding init hrf

    # init cost function value
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    if not hrf_fixed_ampl:
        g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
        J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)
    else:
        J.append(0.5 * r + lbda_bold * g_bold)

    for idx in range(nb_iter):

        # BOLD deconvolution
        if model_type == 'bloc':
            Integ = DiscretInteg()
            H = ConvAndLinear(Integ, est_hrf, dim_in=N, dim_out=N)
        elif model_type == 'event':
            H = Conv(est_hrf, dim_in=N)
        else:
            raise ValueError("model_type should be in ['bloc', 'event']")

        v0 = est_i_s
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, nb_iter=10000,
                    early_stopping=True, wind=8, tol=1.0e-12, verbose=verbose,
                        )

        if model_type == 'bloc':
            est_ai_s = Integ.op(est_i_s)
            est_ar_s = Conv(est_hrf, N).op(est_ai_s)
        elif model_type == 'event':
            est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # HRF estimation
        if model_type == 'bloc':
            H = ConvAndLinear(hrf_dico, est_ai_s, dim_in=len_hrf, dim_out=N)
        elif model_type == 'event':
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

        if model_type == 'bloc':
            est_ar_s = Conv(est_hrf, N).op(est_ai_s)
        elif model_type == 'event':
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
                  "({0}/{1}): {2:.4f}".format(idx+1, nb_iter, J[-1]))

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

    if model_type == 'bloc':
        return est_ar_s, est_ai_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J

    elif model_type == 'event':
        return est_ar_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J


def _default_wrapper(recons_func, **kwargs):
    """ Default wrapper to parallelize the image reconstruction.
    """
    return recons_func(**kwargs)


def grid_search(func, param_grid, wrapper=None, n_jobs=1, verbose=0):
    """ Run `func` on the carthesian product of `param_grid`.

        Results:
        --------
        list_kwargs: dict,
            the list of the params used for each reconstruction.
        res: list,
            the list of result for each reconstruction.
    """
    if wrapper is None:
        wrapper = _default_wrapper
    # sanitize value to list type
    for key, value in param_grid.iteritems():
        if not isinstance(value, list):
            param_grid[key] = [value]
    list_kwargs = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]
    # Run the reconstruction
    if verbose > 0:
        if n_jobs == -1:
            n_jobs_used = psutil.cpu_count()
        elif n_jobs == -2:
            n_jobs_used = psutil.cpu_count() - 1
        else:
            n_jobs_used = n_jobs
        print(("Running grid_search for {0} candidates"
               " on {1} jobs").format(len(list_kwargs), n_jobs_used))
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(wrapper)(func, **kwargs)
                   for kwargs in list_kwargs)
    return list_kwargs, res


def bold_blind_deconvolution_cv( # noqa
                    noisy_ar_s, tr, hrf_dico, t_hrf, orig_hrf,
                    lbda_bold=list(np.linspace(5.0e-2, 5.0, 5)),
                    lbda_hrf=list(np.linspace(5.0e-2, 5.0, 5)),
                    init_hrf=None, hrf_fixed_ampl=False, nb_iter=50,
                    model_type='bloc', early_stopping=False, wind=24,
                    tol=1.0e-24, n_jobs=1, verbose=0):
    """ Blind deconvolution of the BOLD signal.
    """
    param_grid = {'noisy_ar_s': noisy_ar_s, 'tr': tr, 'hrf_dico': hrf_dico,
                  'lbda_bold': lbda_bold, 'lbda_hrf': lbda_hrf,
                  'init_hrf': init_hrf, 'hrf_fixed_ampl': hrf_fixed_ampl,
                  'nb_iter': nb_iter, 'model_type': model_type,
                  'early_stopping': early_stopping, 'wind': wind, 'tol': tol,
                  'verbose': 0}
    list_kwargs, res = grid_search(bold_blind_deconvolution, param_grid,
                                   n_jobs=n_jobs, verbose=verbose)

    true_fwhm = fwhm(t_hrf, orig_hrf)
    errs_fwhm = []
    for (est_ar_s, est_i_s, est_hrf, sparse_encoding_hrf, J) in res:
        curr_fwhm = fwhm(t_hrf, est_hrf)
        errs_fwhm.append(np.abs(curr_fwhm - true_fwhm))
    idx_best = np.argmin(np.array(errs_fwhm))

    return list_kwargs[idx_best], res[idx_best]
