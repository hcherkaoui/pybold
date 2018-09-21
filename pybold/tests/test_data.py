""" Test the data module.
"""
import unittest
import itertools
from joblib import Parallel, delayed
import numpy as np
from numpy.linalg import norm as norm_2
from pybold.convolution import simple_convolve
from pybold.data import gen_rnd_bloc_bold
from pybold.hrf_model import spm_hrf


# Here we mainly test the coherence of the API of the data generation function


def _test_data(res, snr, nb_events):
    """ Helper to test:
    - Manually reconvolve the block signal
      and compare it to the BOLD signal
    - test the produce signal SNR
    - test if noise == (noisy_ar_s - ar_s)
    """
    noisy_ar_s, ar_s, ai_s, i_s, _, hrf, noise = res

    # test if the 'ar_s = (hrf * ai_s)'
    ar_s_test = simple_convolve(hrf, ai_s)
    msg = ("test 'np.allclose(ar_s, ar_s_test)' "
           "with snr={0}, nb_events={1}".format(snr, nb_events))
    np.testing.assert_almost_equal(ar_s_test, ar_s, err_msg=msg)

    # test if expect snr == data snr
    true_snr = 20.0 * np.log10(norm_2(ar_s) / norm_2(noise))
    msg = ("test 'isclose(snr, true_snr)' "
           "with snr={0}, nb_events={1}".format(snr, nb_events))
    np.testing.assert_almost_equal(snr, true_snr, err_msg=msg)

    # test if expect noise == (noisy_ar_s - ar_s)
    residual = noisy_ar_s - ar_s
    msg = ("test 'allclose(residual, noise)' "
           "with snr={0}, nb_events={1}".format(snr, nb_events))
    np.testing.assert_almost_equal(residual, noise, err_msg=msg)

    # test if nb_events is respected
    true_nb_events = nb_events
    nb_events = np.sum(i_s > 0.5)
    msg = ("test 'allclose(true_nb_events, nb_events)' "
           "with snr={0}, nb_events={1}".format(snr, true_nb_events))
    np.testing.assert_almost_equal(nb_events, true_nb_events, err_msg=msg)


class TestDataGeneration(unittest.TestCase):

    def _yield_data(self):
        """ Yield signals for unittests.
        """
        # caution random_state should be fixed but it could also systematically
        # lead to failing signal generation, so this arg is carefully set
        random_state_s = [99]
        snr_s = [1.0, 10.0, 20.0]
        tr_s = [0.1, 2.0]
        dur_orig_s = [3, 5, 10]  # minutes
        delta_s = [0.5, 1.0]
        listparams = [random_state_s, snr_s, tr_s, dur_orig_s,
                      delta_s]
        for params in itertools.product(*listparams):
            random_state, snr, tr, dur_orig, delta = params
            nb_events = int(dur_orig/2)
            hrf, _ = spm_hrf(delta=delta, tr=tr)
            params = {'dur': dur_orig,
                      'tr': tr,
                      'hrf': hrf,
                      # if nb_events should be adapted to
                      # the length of the signal
                      'nb_events': nb_events,
                      'avg_dur': 1,
                      'std_dur': 5,
                      'overlapping': False,
                      'snr': snr,
                      'random_state': random_state,
                      }
            res = gen_rnd_bloc_bold(**params)
            yield res, snr, nb_events

    def test_data_gen(self):
        """ Test the data generation function:
            - Manually reconvolve the block signal
              and compare it to the BOLD signal
            - test the produce signal SNR
            - test if noise == (noisy_ar_s - ar_s)
        """
        Parallel(n_jobs=-1)(delayed(_test_data)(res, snr, nb_events)
                            for res, snr, nb_events in self._yield_data())


if __name__ == '__main__':
    unittest.main()
