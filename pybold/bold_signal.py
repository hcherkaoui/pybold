# coding: utf-8
""" Main module that provide the blind deconvolution function.
"""
import numpy as np
from .data import spm_hrf
from .utils import NoProgressBar
from .linear import Matrix, DiscretInteg, Conv, ConvAndLinear
from .gradient import L2ResidualLinear
from .solvers import condatvu, fista
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


def bold_deconvolution(noisy_ar_s, tr, hrf, lbda=1.0, verbose=0):
    """ Deconvolve the given BOLD signal.
    """
    hrf, _, _ = spm_hrf(tr=tr, time_length=hrf_time_length)
    Integ = DiscretInteg()
    H = ConvAndLinear(Integ, hrf, dim_in=len(noisy_ar_s),
                      dim_out=len(noisy_ar_s))
    z0 = np.zeros(len(noisy_ar_s))

    prox = L1Norm(lbda)
    grad = L2ResidualLinear(H, noisy_ar_s, z0.shape)

    x, _, _, _ = fista(
                    grad=grad, prox=prox, v0=z0, w=None, nb_iter=999,
                    early_stopping=True, verbose=verbose,
                      )
    est_i_s = x
    est_ai_s = Integ.op(x)
    est_ar_s = Conv(hrf, len(noisy_ar_s)).op(est_ai_s)

    return est_ar_s, est_ai_s, est_i_s


def hrf_sparse_encoding_estimation(ai_s, ar_s, tr, nb_time_deltas=50,
                                   nb_onsets=20, lbda=1.0e-4, verbose=0):
    """ HRF sparse-encoding estimation.
    """
    hrf_dico, _, _, _ = gen_hrf_spm_dict(tr=tr, nb_time_deltas=nb_time_deltas,
                                         nb_onsets=nb_onsets)
    hrf_dico = Matrix(hrf_dico)
    len_hrf, nb_atoms_hrf = hrf_dico.shape

    H = ConvAndLinear(hrf_dico, ai_s, dim_in=len_hrf, dim_out=len(ai_s))
    z0 = np.zeros(nb_atoms_hrf)

    prox = L1Norm(lbda)
    grad = L2ResidualLinear(H, ar_s, z0.shape)

    sparce_encoding_hrf, _, _, _ = fista(
                    grad=grad, prox=prox, v0=z0, w=None, nb_iter=9999,
                    early_stopping=True, verbose=verbose,
                      )
    hrf = hrf_dico.op(sparce_encoding_hrf)

    return hrf, sparce_encoding_hrf


def bold_blind_deconvolution(noisy_ar_s, tr, nb_time_deltas=50, nb_onsets=20,
                             lbda_bold=1.0, lbda_hrf=1.0e-4, verbose=0):
    """ Blind deconvolution of the BOLD signal.
    """
    est_hrf, _, _ = spm_hrf(tr=tr, time_length=32.0)

    hrf_dico, _, _, _ = gen_hrf_spm_dict(tr=tr, nb_time_deltas=nb_time_deltas,
                                         nb_onsets=nb_onsets)
    _, nb_atoms_hrf = hrf_dico.shape

    hrf_dico = Matrix(hrf_dico)

    D = Diff()
    Integ = DiscretInteg()

    prox_bold = L1Norm(lbda_bold)
    prox_hrf = L1Norm(lbda_hrf)

    nb_iter = 20
    for idx in range(nb_iter):

        # BOLD deconvolution
        H = ConvAndLinear(Integ, est_hrf, dim_in=len(noisy_ar_s),
                          dim_out=len(noisy_ar_s))
        v0 = np.zeros(len(noisy_ar_s))
        grad = L2ResidualLinear(H, noisy_ar_s, v0.shape)
        est_i_s, _, _, _ = fista(
                    grad=grad, prox=prox_bold, v0=v0, w=None, nb_iter=999,
                    early_stopping=True, verbose=verbose,
                      )

        # mid post-processing
        est_ai_s = Integ.op(est_i_s)
        est_ar_s = Conv(est_hrf, len(noisy_ar_s)).op(est_ai_s)

        # HRF estimation
        H = ConvAndLinear(hrf_dico, est_ai_s,
                          dim_in=len(est_hrf), dim_out=len(noisy_ar_s))
        v0 = np.zeros(nb_atoms_hrf)
        grad = L2ResidualLinear(H, est_ar_s, v0.shape)
        est_sparse_encoding_hrf, _, _, _ = fista(
                 prox=prox_hrf, w=None, nb_iter=999,
                 grad=grad, v0=v0, early_stopping=False,
                 verbose=verbose,
                                                )
        est_hrf = hrf_dico.op(est_sparse_encoding_hrf)

    est_i_s = D.op(est_ai_s)

    return est_ar_s, est_ai_s, est_i_s, est_hrf, est_sparse_encoding_hrf
