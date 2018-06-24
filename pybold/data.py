# coding: utf-8
""" This module gathers usefull data generation.
"""
import numpy as np
from scipy.stats import gamma
from .utils import random_generator
from .convolution import simple_convolve


def gen_hrf_spm_dict(tr, nb_time_deltas, max_delta=50.0, min_delta=10.0,
                     nb_onsets=0, max_onset=1.0, min_onset=0.0):
    """ Return a HRF dictionary based of the SPM model (difference of two
    gamma functions)
    """
    hrf_dico = []
    onsets = np.linspace(0.0, 1.0, nb_onsets)
    time_lengths = np.linspace(10.0, 50.0, nb_time_deltas)
    if nb_onsets != 0:
        for onset in onsets:
            for time_length in time_lengths:
                hrf, t_hrf, _ = spm_hrf(tr=tr, time_length=time_length,
                                        onset=onset)
                hrf_dico.append(hrf)
    else:
        for time_length in time_lengths:
            hrf, t_hrf, _ = spm_hrf(tr=tr, time_length=time_length)
            hrf_dico.append(hrf)
    return np.vstack(hrf_dico).T, t_hrf, time_lengths, onsets


def gen_ai_s(dur=3, tr=1.0, nb_events=4, avg_dur=5, std_dur=1,
             middle_spike=False, overlapping=False, random_state=None,
             nb_try=100000, nb_try_duration=100000):
    """ Generate a Activity inducing signal.
    dur : int (default=5),
        The length of the BOLD signal (in minutes).

    tr : float (default=1.0),
        Repetition time

    nb_events : int (default=4),
        Number of neural activity on-sets.

    avg_dur : int (default=5),
        The average duration (in second) of neural activity on-sets.

    std_dur : int (default=1),
        The standard deviation (in second) on the duration of neural
        activity on-sets.

    middle_spike : bool (default=False),
        Whether to force a dirac on the middle on the BOLD signal.

    overlapping : bool (default=False),
        Whether to authorize overlapping between on-sets.

    nb_try : int (default=10000),
        Number of try to generate the BOLD signal.

    nb_try_duration : int (default=10000),
        Number of try to generate the each neural activity on-set.

    random_state : int or None (default=None),
        Whether to impose a seed on the random generation or not (for
        reproductability).

    Return
    ------

    ai_s : np.ndarray,
        Activity inducing signal.

    i_s : np.ndarray,
        Innovation signal.

    t : np.ndarray,
        time scale signal.

    """
    dt = 0.001  # to similate continious signal generation
    N = int((dur * 60) / dt)
    avg_dur /= dt
    var_dur = (std_dur)**2
    var_dur /= dt
    center_neighbors = [int(N/2)-1, int(N/2)+1]

    # for reproductibility
    r = random_generator(random_state)

    # generate the block
    for _ in range(nb_try):
        offsets = r.randint(0, N, nb_events)
        for _ in range(nb_try_duration):
            durations = avg_dur + (var_dur)*r.randn(nb_events)
            if any(durations < 1):
                continue  # null or negative duration events: retry
            else:
                break

        # place the block
        durations = durations.astype(int)
        ai_s = np.zeros(N)
        for offset, duration in zip(offsets, durations):
            try:
                ai_s[offset:offset+duration] += 1
            except IndexError:
                break

        # check optional conditions
        if middle_spike:
            ai_s[int(N/2)] += 1
        if (not overlapping) and any(ai_s > 1):
            continue  # overlapping events: retry
        if overlapping:
            ai_s[ai_s > 1] = 1
        if middle_spike and any(ai_s[center_neighbors] > 0):
            continue  # middle-spike not isolated: retry

        # subsample signal at 1/TR
        ai_s = ai_s[::int(tr/dt)]

        # generate innovation signal and the time scale on 'dur' duration with
        # '1/TR' sample rate
        i_s = np.append(0, ai_s[1:] - ai_s[:-1])
        t = np.linspace(0, dur*60, len(ai_s))

        return ai_s, i_s, t

    raise RuntimeError("[Failure] Failed to produce an "
                       "activity-inducing signal, please re-run gen_ai_s "
                       "function with possibly new arguments.")


def gen_random_events(dur=5, tr=1.0, hrf_onset=0.0, hrf_time_length=32.0,
                      nb_events=4, avg_dur=5, std_dur=1, middle_spike=False,
                      overlapping=False, normalized_output=False, snr=1.0,
                      nb_try=10000, nb_try_duration=10000,
                      random_state=None):
    """ Generate a 1d-array that represents an activity-inducing signal with
    random events.

    Parameters
    ----------
    dur : int (default=5),
        The length of the BOLD signal (in minutes).

    tr : float (default=1.0),
        Repetition time

    nb_events : int (default=4),
        Number of neural activity on-sets.

    avg_dur : int (default=5),
        The average duration (in second) of neural activity on-sets.

    std_dur : int (default=1),
        The standard deviation (in second) on the duration of neural
        activity on-sets.

    middle_spike : bool (default=False),
        Whether to force a dirac on the middle on the BOLD signal.

    overlapping : bool (default=False),
        Whether to authorize overlapping between on-sets.

    normalized_output : bool (default=False),
        Whether to max-min normalize or not.

    snr: float (default=1.0),
        SNR of the noisy BOLD signal.

    nb_try : int (default=10000),
        Number of try to generate the BOLD signal.

    nb_try_duration : int (default=10000),
        Number of try to generate the each neural activity on-set.

    random_state : int or None (default=None),
        Whether to impose a seed on the random generation or not (for
        reproductability).

    Return
    ------

    noisy_ar_s : np.ndarray,
        Noisy activity related signal.

    ar_s : np.ndarray,
        Activity related signal.

    ai_s : np.ndarray,
        Activity inducing signal.

    i_s : np.ndarray,
        Innovation signal.

    t : np.ndarray,
        time scale signal.

    hrf : np.ndarray,
        HRF.

    noise : np.ndarray,
        Noise.

    """
    ai_s, i_s, t = gen_ai_s(dur=dur, tr=tr, nb_events=nb_events,
                            avg_dur=avg_dur, std_dur=std_dur,
                            middle_spike=middle_spike, overlapping=overlapping,
                            random_state=random_state)

    hrf_coef, t_hrf, right_zero_padding = spm_hrf(tr=tr,
                                                  time_length=hrf_time_length)
    ar_s = simple_convolve(hrf_coef, ai_s)

    noisy_ar_s, noise = add_gaussian_noise(ar_s, snr=snr,
                                           random_state=random_state)

    return noisy_ar_s, ar_s, ai_s, i_s, t, hrf_coef, t_hrf, noise


def add_gaussian_noise(signal, snr, random_state=None):
    """ Add a Gaussian noise to signal to ouput a signal with the targeted snr.

    Parameters
    ----------
    signal : np.ndarray,
        The given signal on which add a Guassian noise.

    snr : float,
        The expected SNR for the output signal.

    random_state :  int or None (default=None),
        Whether to impose a seed on the random generation or not (for
        reproductability).

    Return
    ------
    noisy_signal: np.ndarray,
        the noisy produced signal.

    noise:  np.ndarray,
        the additif produced noise.

    """

    if isinstance(random_state, int):
        r = np.random.RandomState(random_state)
    elif random_state is None:
        r = np.random
    else:
        raise ValueError("random_state could only be seed-int or None, "
                         "got {0}".format(type(random_state)))

    s_shape = signal.shape
    noise = r.randn(*s_shape)

    true_snr_num = np.linalg.norm(signal)
    true_snr_deno = np.linalg.norm(noise)
    true_snr = true_snr_num / (true_snr_deno + np.finfo(np.float).eps)
    std_dev = (1.0 / np.sqrt(10**(snr/10.0))) * true_snr
    noise = std_dev * noise
    noisy_signal = signal + noise

    return noisy_signal, noise


def spm_hrf(tr, time_length=32.0, onset=0.0):
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
    scaled_time_stamps -= onset

    gamma_1 = gamma.pdf(scaled_time_stamps, delay / disp, dt / disp)
    gamma_2 = ratio_gamma * gamma.pdf(scaled_time_stamps,
                                      undershoot / u_disp,
                                      dt / u_disp)

    hrf = gamma_1 - gamma_2
    # normalize and resample the HRF
    hrf /= np.linalg.norm(hrf)
    hrf = hrf[::int(tr/dt)]

    # return the HRF associated time stamp (sampled as the HRF)
    time_stamps = time_stamps[::int(tr/dt)]

    # by default HRF is output with a 'time_stamp_hrf / tr' length
    # returning 'right_zero_padding' allows the user to work only on the pattern
    # of interest
    right_zero_padding = (np.abs(hrf) >= (1.0e-3 * np.max(hrf)))

    return hrf, time_stamps, right_zero_padding
