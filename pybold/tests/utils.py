# coding: utf-8
""" This module gathers usefull functions for testing.
"""
import itertools
import numpy as np
from ..data import gen_rnd_ai_s
from ..hrf_model import spm_hrf


class YieldData():
    def yield_diracs_signal(self):
        """ Yield diracs signals on which to test the convolution functions.
        """
        dur = 1  # minutes
        tr_s = [0.1, 2.0]
        delta_s = [0.5, 1.0]
        listparams = [tr_s, delta_s]
        for params in itertools.product(*listparams):
            tr, delta = params
            t = np.linspace(0, dur*60, int(dur*60 / tr))
            # place Dirac at 20% of dur
            onset_event = int(0.20 * dur * 60 / tr)
            i_s = np.zeros_like(t)
            i_s[onset_event] = 1
            hrf_params = {'tr': tr,
                          'delta': delta,
                          }
            hrf, _ = spm_hrf(**hrf_params)
            nb_atoms = 20
            alpha = np.zeros(nb_atoms)
            alpha[10] = 1
            yield i_s, hrf, alpha, tr

    def yield_blocks_signal(self):
        """ Yield blocks signals on which to test the convolution functions.
        """
        random_state_s = [0]
        tr_s = [0.1, 2.0]
        dur_orig_s = [3, 10]  # minutes
        delta_s = [0.5, 1.0]
        listparams = [random_state_s, tr_s, dur_orig_s,
                      delta_s]
        for params in itertools.product(*listparams):
            random_state, tr, dur_orig, delta = params
            ai_s_params = {'dur': dur_orig,
                           'tr': tr,
                           # nb_events should be adapted to
                           # the length of the signal
                           'nb_events': int(dur_orig/2),
                           'avg_dur': 1,
                           'std_dur': 5,
                           'overlapping': True,
                           'random_state': random_state,
                           }
            hrf_params = {'tr': tr,
                          'delta': delta,
                          }
            ai_s, _, _ = gen_rnd_ai_s(**ai_s_params)
            hrf, _ = spm_hrf(**hrf_params)
            nb_atoms = 20
            alpha = np.zeros(nb_atoms)
            alpha[10] = 1
            yield ai_s, hrf, alpha, tr
