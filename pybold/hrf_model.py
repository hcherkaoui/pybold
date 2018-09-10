# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma


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
