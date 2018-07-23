# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import numpy as np
from .data import spm_hrf
from .linear import Matrix, DiscretInteg, Conv, ConvAndLinear
from .gradient import L2ResidualLinear
from .solvers import fista
from .proximity import L1Norm
from .convolution import toeplitz_from_kernel


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

    x, J, _, _ = fista(
                    grad=grad, prox=prox, v0=z0, w=None, nb_iter=10000,
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

    sparce_encoding_hrf, J, _, _ = fista(
                    grad=grad, prox=prox, v0=z0, w=None, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                      )
    hrf = hrf_dico.op(sparce_encoding_hrf)

    return hrf, sparce_encoding_hrf, J


def bold_blind_deconvolution(noisy_ar_s, tr, hrf_dico, lbda_bold=1.0e-2, # noqa
                             lbda_hrf=1.0e-2, init_hrf=None, nb_iter=50,
                             model_type='bloc', verbose=0):
    """ Blind deconvolution of the BOLD signal.
    """
    if init_hrf is None:
        est_hrf, _, _ = spm_hrf(tr=tr, time_length=30.0)  # init hrf
    else:
        est_hrf = init_hrf
    len_hrf, nb_atoms_hrf = hrf_dico.shape

    N = len(noisy_ar_s)
    J = []

    if not isinstance(hrf_dico, Matrix):
        hrf_dico = Matrix(hrf_dico)

    Integ = DiscretInteg()
    prox_bold = L1Norm(lbda_bold)
    prox_hrf = L1Norm(lbda_hrf)

    est_i_s = np.zeros(N)  # init spiky signal
    est_ar_s = np.zeros(N)  # thus.. init convolved signal
    est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)  # sparse encoding init hrf

    # init cost function value
    r = np.sum(np.square(est_ar_s - noisy_ar_s))
    g_bold = np.sum(np.abs(est_i_s))
    g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
    J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)

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
        est_i_s, _, _, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, w=None, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                        )

        if model_type == 'bloc':
            est_ai_s = Integ.op(est_i_s)
            est_ar_s = Conv(est_hrf, N).op(est_ai_s)
        elif model_type == 'event':
            est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # cost function after deconvolution step
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
        J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)

        if (verbose > 0):
            print("cost function after deconvolution at iter "
                  "{0} / {1} : {2}".format(idx+1, nb_iter, J[-1]))

        # HRF estimation
        if model_type == 'bloc':
            H = ConvAndLinear(hrf_dico, est_ai_s, dim_in=len_hrf, dim_out=N)
        elif model_type == 'event':
            H = ConvAndLinear(hrf_dico, est_i_s, dim_in=len_hrf, dim_out=N)
        est_sparse_encoding_hrf = hrf_dico.adj(est_hrf)
        v0 = est_sparse_encoding_hrf
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_sparse_encoding_hrf, _, _, _ = fista(
                    grad=grad, prox=prox_hrf, v0=v0, w=None, nb_iter=10000,
                    early_stopping=True, verbose=verbose,
                        )
        est_hrf = hrf_dico.op(est_sparse_encoding_hrf)
        if model_type == 'bloc':
            est_ar_s = Conv(est_hrf, N).op(est_ai_s)
        elif model_type == 'event':
            est_ar_s = Conv(est_hrf, N).op(est_i_s)

        # cost function after hrf step
        r = np.sum(np.square(est_ar_s - noisy_ar_s))
        g_bold = np.sum(np.abs(est_i_s))
        g_hrf = np.sum(np.abs(est_sparse_encoding_hrf))
        J.append(0.5 * r + lbda_bold * g_bold + lbda_hrf * g_hrf)

        if (verbose > 0):
            print("cost function after HRF estimation at iter "
                  "{0} / {1} : {2}".format(idx+1, nb_iter, J[-1]))

    J = np.array(J)

    if model_type == 'bloc':
        return est_ar_s, est_ai_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J

    elif model_type == 'event':
        return est_ar_s, est_i_s, est_hrf, est_sparse_encoding_hrf, J
