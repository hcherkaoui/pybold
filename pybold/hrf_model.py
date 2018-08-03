# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma


def spm_hrf(tr, time_length=32.0, normalized_hrf=True):
    """ Custom HRF.
    """
    if (time_length < 10.0) or (time_length > 50.0):
        raise ValueError("time_length can only belong to [10.0, 50.0], "
                         "got {0}".format(time_length))
    # fixed: from the literature
    dt = 0.001
    delay = 6
    undershoot = 16.
    disp = 1.
    u_disp = 1.
    ratio_gamma = 0.167

    # time_stamp_hrf: the (continious) time segment on which we represent all
    # the HRF. Can cut the signal too early. The time scale is second.
    time_stamp_hrf = 60.  # secondes

    # scale in time the HRF
    time_stamps = np.linspace(0, time_stamp_hrf, float(time_stamp_hrf) / dt)
    time_scale = time_stamp_hrf / time_length
    scaled_time_stamps = time_scale * time_stamps

    gamma_1 = gamma.pdf(scaled_time_stamps, delay / disp, dt / disp)
    gamma_2 = ratio_gamma * gamma.pdf(scaled_time_stamps,
                                      undershoot / u_disp,
                                      dt / u_disp)

    hrf = gamma_1 - gamma_2

    if normalized_hrf:
        # l2-unitary HRF
        hrf /= np.linalg.norm(hrf)
        # to produce convolved ~ unitary block
        hrf *= 10.0

    # subsample HRF to tr
    hrf = hrf[::int(tr/dt)]

    # return the HRF associated time stamp (sampled as the HRF)
    time_stamps = time_stamps[::int(tr/dt)]

    # by default HRF is output with a 'time_stamp_hrf / tr' length
    # returning 'right_zero_padding' allows the user to work only on the
    # pattern of interest
    right_zero_padding = (np.abs(hrf) >= (1.0e-3 * np.max(hrf)))

    return hrf, time_stamps, right_zero_padding


def gen_hrf_spm_dict(tr, nb_time_deltas=500, max_delta=50.0, min_delta=10.0):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    time_lengths = np.linspace(min_delta, max_delta, nb_time_deltas)

    for time_length in time_lengths:
        hrf, _, _ = spm_hrf(tr=tr, time_length=time_length,
                            normalized_hrf=False)
        hrf_dico.append(hrf)

    return np.vstack(hrf_dico).T


def gen_hrf_spm_dict_normalized(tr, nb_time_deltas=500, max_delta=50.0,
                                min_delta=10.0):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    time_lengths = np.linspace(min_delta, max_delta, nb_time_deltas)

    for time_length in time_lengths:
        hrf, _, _ = spm_hrf(tr=tr, time_length=time_length,
                            normalized_hrf=True)
        hrf_dico.append(hrf)

    return np.vstack(hrf_dico).T


def gen_hrf_spm_basis(tr, nb_time_deltas=500, max_delta=50.0, min_delta=10.0,
                      full_matrices=False, th=1.0e-3):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    time_lengths = np.linspace(min_delta, max_delta, nb_time_deltas)

    for time_length in time_lengths:
        hrf, _, _ = spm_hrf(tr=tr, time_length=time_length,
                            normalized_hrf=True)
        hrf_dico.append(hrf)
    hrf_dico = np.vstack(hrf_dico).T

    if full_matrices:
        U, s, V_T = np.linalg.svd(hrf_dico.T, full_matrices=True)
        # remove insignificant eigen values
        s[s <= (np.max(np.abs(s)) * th)] = 0.0
        S = np.zeros((U.shape[0], V_T.shape[0]))
        S[:len(s), :len(s)] = s
        hrf_dico = U.dot(S).dot(V_T).T
    else:
        hrf_dico, _, _ = np.linalg.svd(hrf_dico, full_matrices=False)

    return hrf_dico
