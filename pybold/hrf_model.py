# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma
from hrf_estimation.hrf import spmt, dspmt, ddspmt


MIN_DELTA = 0.5
MAX_DELTA = 2.0


def spm_hrf(delta, t_r=1.0, dur=60.0, normalized_hrf=True, dt=0.001, p_delay=6,
            undershoot=16.0, p_disp=1.0, u_disp=1.0, p_u_ratio=0.167,
            onset=0.0):
    """ SPM canonical HRF with a time scaling parameter.
    """
    if (delta < MIN_DELTA) or (delta > MAX_DELTA):
        raise ValueError("delta should belong in [{0}, {1}]; wich correspond"
                         " to a max FWHM of 10.52s and a min FWHM of 2.80s"
                         ", got delta = {2}".format(MIN_DELTA, MAX_DELTA,
                                                    delta))

    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, float(dur) / dt) - float(onset) / dt
    scaled_time_stamps = delta * t

    peak = gamma.pdf(scaled_time_stamps, p_delay/p_disp, loc=dt/p_disp)
    undershoot = gamma.pdf(scaled_time_stamps, undershoot/u_disp,
                           loc=dt/u_disp)
    hrf = peak - p_u_ratio * undershoot

    if normalized_hrf:
        hrf /= np.max(hrf + 1.0e-30)

    hrf = hrf[::int(t_r/dt)]
    t_hrf = t[::int(t_r/dt)]

    return hrf, t_hrf


def basis3_hrf(hrf_basis3_params, t_r=1.0, dur=60.0, normalized_hrf=True,
               pedregosa_hrf=True):
    """ Return the HRF as a linear combinaison of the canonical HRF from SPM,
    its temporal derivative and the width parameter derivative.
    """
    alpha_1, alpha_2, alpha_3 = hrf_basis3_params

    if pedregosa_hrf:
        t_hrf = np.linspace(0, dur, int(dur/t_r))
        b_1 = spmt(t_hrf)
        b_2 = dspmt(t_hrf)
        b_3 = ddspmt(t_hrf)
    else:
        b_1 = spm_hrf(delta=1.0, t_r=t_r, dur=dur, normalized_hrf=False)[0]
        b_2_ = spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=0.0,
                       normalized_hrf=False)[0]
        b_2__ = spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=t_r,
                        normalized_hrf=False)[0]
        b_2 = b_2_ - b_2__
        b_3_ = spm_hrf(delta=1.0, t_r=t_r, dur=dur, p_disp=1.001,
                       normalized_hrf=False)[0]
        b_3 = (b_1 - b_3_) / 0.001

    hrf = alpha_1*b_1 + alpha_2*b_2 + alpha_3*b_3
    t_hrf = np.linspace(0.0, len(b_1)*t_r, len(b_1))

    if normalized_hrf:
        hrf /= np.max(hrf + 1.0e-30)

    return hrf, t_hrf


def basis2_hrf(hrf_basis3_params, t_r=1.0, dur=60.0, normalized_hrf=True):
    """ Return the HRF as a linear combinaison of the canonical HRF from SPM
    and its temporal derivative.
    """
    alpha_1, alpha_2 = hrf_basis3_params

    b_1 = spm_hrf(delta=1.0, t_r=t_r, dur=dur, normalized_hrf=False)[0]
    b_2_ = spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=0.0, normalized_hrf=False)[0]
    b_2__ = spm_hrf(delta=1.0, t_r=t_r, dur=dur, onset=t_r, normalized_hrf=False)[0]
    b_2 = b_2_ - b_2__

    hrf = alpha_1*b_1 + alpha_2*b_2
    t_hrf = np.linspace(0.0, len(b_1)*t_r, len(b_1))

    if normalized_hrf:
        hrf /= np.max(hrf + 1.0e-30)

    return hrf, t_hrf
