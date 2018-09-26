# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma
from scipy.special import expit


MIN_DELTA = 0.5
MAX_DELTA = 2.0


def spm_hrf(delta, tr=1.0, dur=60.0, normalized_hrf=True, dt=0.001, p_delay=6,
            undershoot=16.0, p_disp=1.0, u_disp=1.0, p_u_ratio=0.167):
    """ SPM canonical HRF with a time scaling parameter.
    """
    if (delta < MIN_DELTA) or (delta > MAX_DELTA):
        raise ValueError("delta should belong in [{0}, {1}]; wich correspond"
                         " to a max FWHM of 10.52s and a min FWHM of 2.80s"
                         ", got delta = {2}".format(MIN_DELTA, MAX_DELTA,
                                                    delta))

    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, float(dur) / dt)
    scaled_time_stamps = delta * t

    peak = gamma.pdf(scaled_time_stamps, p_delay/p_disp, loc=dt/p_disp)
    undershoot = gamma.pdf(scaled_time_stamps, undershoot/u_disp,
                           loc=dt/u_disp)
    hrf = peak - p_u_ratio * undershoot

    if normalized_hrf:
        hrf /= (np.linalg.norm(hrf) + 1.0e-30)
        hrf *= 10.0

    hrf = hrf[::int(tr/dt)]
    t_hrf = t[::int(tr/dt)]

    return hrf, t_hrf


def il_hrf(hrf_logit_params, tr=1.0, dur=60.0, normalized_hrf=True):
    """ Return the HRF from the specified inv. logit parameters.
    """
    alpha_1, alpha_2, alpha_3, T1, T2, T3, D1, D2, D3 = hrf_logit_params

    t_hrf = np.linspace(0, dur, float(dur)/tr)
    il_1 = alpha_1 * expit((t_hrf-T1)/D1)
    il_2 = alpha_2 * expit((t_hrf-T2)/D2)
    il_3 = alpha_3 * expit((t_hrf-T3)/D3)
    hrf = il_1 + il_2 + il_3

    if normalized_hrf:
        hrf /= (np.linalg.norm(hrf) + 1.0e-30)
        hrf *= 10.0

    return hrf, t_hrf, [il_1, il_2, il_3]


def basis3_hrf(hrf_basis3_params, tr=1.0, dur=60.0, normalized_hrf=True):
    """ Return the HRF as a linear combinaison of the canonical HRF from SPM,
    its temporal derivative and the
    """
    alpha_1, alpha_2, alpha_3 = hrf_basis3_params

    hrf_1, t_hrf = spm_hrf(delta=1.0, tr=tr, dur=dur)
    hrf_2 = np.append(0, hrf_1[1:] - hrf_1[:-1])
    hrf_3 = (hrf_1 - spm_hrf(delta=1.0, tr=tr, dur=dur, p_disp=1.01)[0]) / 0.01

    hrf = alpha_1*hrf_1 + alpha_2*hrf_2 + alpha_3*hrf_3

    if normalized_hrf:
        hrf /= (np.linalg.norm(hrf) + 1.0e-30)
        hrf *= 10.0

    return hrf, t_hrf
