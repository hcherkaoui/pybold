# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma
from scipy.special import expit
import matplotlib.pyplot as plt


def il_hrf(hrf_logit_params, tr=1.0, dur=60.0):
    """ Return the HRF from the specified inv. logit parameters.
    """
    alpha_1, alpha_2, alpha_3, T1, T2, T3, D1, D2, D3 = hrf_logit_params

    t_hrf = np.linspace(0, dur, float(dur)/tr)
    il_1 = alpha_1 * expit((t_hrf-T1)/D1)
    il_2 = alpha_2 * expit((t_hrf-T2)/D2)
    il_3 = alpha_3 * expit((t_hrf-T3)/D3)
    hrf = il_1 + il_2 + il_3

    return hrf, t_hrf, [il_1, il_2, il_3]


def spm_hrf(delta, tr=1.0, dur=60.0, normalized_hrf=True):
    """ Custom HRF.
    """
    if (delta < 0.2) or (delta > 2.0):
        raise ValueError("delta should belong in [0.2, 2.0]"
                         ", got {0}".format(delta))

    # fixed: from the literature
    dt = 0.001
    delay = 6
    undershoot = 16.
    disp = 1.
    u_disp = 1.
    ratio_gamma = 0.167

    # time_stamp_hrf: the (continious) time segment on which we represent all
    # the HRF. Can cut the signal too early. The time scale is second.
    time_stamp_hrf = dur  # secondes

    # scale in time the HRF
    time_stamps = np.linspace(0, time_stamp_hrf, float(time_stamp_hrf) / dt)
    scaled_time_stamps = delta * time_stamps

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
    t_hrf = time_stamps[::int(tr/dt)]

    return hrf, t_hrf


def gen_hrf_spm_dict(tr, nb_time_deltas=50, max_delta=2.0, min_delta=0.2):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    deltas = np.linspace(min_delta, max_delta, nb_time_deltas)

    for delta in deltas:
        hrf, _ = spm_hrf(tr=tr, delta=delta, normalized_hrf=False)
        hrf_dico.append(hrf)

    return np.vstack(hrf_dico).T


def gen_hrf_spm_dict_normalized(tr, nb_time_deltas=50, max_delta=2.0,
                                min_delta=0.2):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    deltas = np.linspace(min_delta, max_delta, nb_time_deltas)

    for delta in deltas:
        hrf, _ = spm_hrf(tr=tr, delta=delta, normalized_hrf=True)
        hrf_dico.append(hrf)

    return np.vstack(hrf_dico).T


def gen_hrf_spm_basis(tr, nb_time_deltas=50, max_delta=2.0, min_delta=0.2,
                      full_matrices=False, th=1.0e-3):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    deltas = np.linspace(min_delta, max_delta, nb_time_deltas)

    for delta in deltas:
        hrf, _ = spm_hrf(tr=tr, delta=delta, normalized_hrf=True)
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
