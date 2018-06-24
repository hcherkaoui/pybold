# coding: utf-8
""" This module gathers usefull functions for testing.
"""
import itertools
import numpy as np
from ..data import spm_hrf, gen_ai_s
from ..data import gen_hrf_spm_dict


class YieldData():
    def yield_diracs_signal(self):
        """ Yield diracs signals on which to test the convolution functions.
        """
        dur = 1  # minutes
        random_state_s = [0]
        tr_s = [0.1, 2.0]
        hrf_time_length_s = [10.0, 50.0]
        onset_s = [0.0, 10.0]
        listparams = [tr_s, hrf_time_length_s, onset_s]
        for params in itertools.product(*listparams):
            tr, hrf_time_length, onset = params
            t = np.linspace(0, dur*60, int(dur*60 / tr))
            onset_event = int(0.20 * dur * 60 / tr)  # place Dirac at 20% of dur
            i_s = np.zeros_like(t)
            i_s[onset_event] = 1
            hrf_params = {'tr': tr,
                          'time_length': hrf_time_length,
                          'onset': onset,
                          }
            hrf, _, _ = spm_hrf(**hrf_params)
            nb_atoms = 20
            D_hrf, _, _, _ = gen_hrf_spm_dict(tr=1.0, nb_time_deltas=nb_atoms)
            alpha = np.zeros(nb_atoms)
            alpha[10] = 1
            yield i_s, hrf, D_hrf, alpha, tr

    def yield_blocks_signal(self):
        """ Yield blocks signals on which to test the convolution functions.
        """
        random_state_s = [0]
        tr_s = [0.1, 2.0]
        dur_orig_s = [3, 10]  # minutes
        hrf_time_length_s = [10.0, 50.0]
        onset_s = [0.0, 10.0]
        listparams = [random_state_s, tr_s, dur_orig_s,
                      hrf_time_length_s, onset_s]
        for params in itertools.product(*listparams):
            random_state, tr, dur_orig, hrf_time_length, onset = params
            ai_s_params = {'dur': dur_orig,
                           'tr': tr,
                           # nb_events should be adapted to
                           # the length of the signal
                           'nb_events': int(dur_orig/2),
                           'avg_dur': 1,
                           'std_dur': 5,
                           'overlapping': False,
                           'random_state': random_state,
                           }
            hrf_params = {'tr': tr,
                          'time_length': hrf_time_length,
                          'onset': onset,
                          }
            ai_s, _, _ = gen_ai_s(**ai_s_params)
            hrf, _, _ = spm_hrf(**hrf_params)
            nb_atoms = 20
            D_hrf, _, _, _ = gen_hrf_spm_dict(tr=1.0, nb_time_deltas=nb_atoms)
            alpha = np.zeros(nb_atoms)
            alpha[10] = 1
            yield ai_s, hrf, D_hrf, alpha, tr

