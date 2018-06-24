""" Test the data module.
"""
import unittest
import itertools
from joblib import Parallel, delayed
import numpy as np
from numpy.linalg import norm as norm_2
from pybold.convolution import simple_convolve
from pybold.data import gen_random_events


# Here we mainly test the coherence of the API of the data generation function


def _test_data(res, snr):
    """ Helper to test:
    - Manually reconvolve the block signal
      and compare it to the BOLD signal
    - test the produce signal SNR
    - test if noise == (noisy_ar_s - ar_s)
    """
    noisy_ar_s, ar_s, ai_s, _, _, hrf, _, noise = res

    # test if the 'ar_s = (hrf * ai_s)'
    ar_s_test = simple_convolve(hrf, ai_s)
    if not np.allclose(ar_s, ar_s_test):
        msg = ("test 'np.allclose(ar_s, ar_s_test)' "
               "with snr={0}, tr={1}, dur={2}, "
               "hrf_time={3}, random_state={4}".format(
                                snr, tr, dur_orig, hrf_time_length,
                                random_state))
        raise(AssertionError(msg))

    # test if expect snr == data snr
    true_snr = 20.0 * np.log10(norm_2(ar_s) / norm_2(noise))
    if not np.isclose(snr, true_snr):
        msg = ("test 'isclose(snr, true_snr)' "
               "with snr={0}, tr={1}, dur={2}, "
               "hrf_time={3}, random_state={4}".format(
                                snr, tr, dur_orig, hrf_time_length,
                                random_state))
        raise(AssertionError(msg))

    # test if expect noise == (noisy_ar_s - ar_s)
    residual = noisy_ar_s - ar_s
    if not np.allclose(residual, noise):
        msg = ("test 'allclose(residual, noise)' "
               "with snr={0}, tr={1}, dur={2}, "
               "hrf_time={3}, random_state={4}".format(
                                snr, tr, dur_orig, hrf_time_length,
                                random_state))
        raise(AssertionError(msg))


class TestDataGeneration(unittest.TestCase):

    def _yield_data(self):
        """ Yield signals for unittests.
        """
        random_state_s = [0]
        snr_s = [1.0, 10.0, 20.0]
        tr_s = [0.1, 2.0]
        dur_orig_s = [3, 5, 10]  # minutes
        hrf_time_length_s = [10.0, 50.0]
        listparams = [random_state_s, snr_s, tr_s, dur_orig_s,
                      hrf_time_length_s]
        for params in itertools.product(*listparams):
            random_state, snr, tr, dur_orig, hrf_time_length = params
            params = {'dur': dur_orig,
                      'tr': tr,
                      'hrf_time_length': hrf_time_length,
                      # if nb_events should be adapted to
                      # the length of the signal
                      'nb_events': int(dur_orig/2),
                      'avg_dur': 1,
                      'std_dur': 5,
                      'overlapping': False,
                      'snr': snr,
                      'random_state': random_state,
                      }
            res = gen_random_events(**params)
            yield res, snr

    def test_data_gen(self):
        """ Test the data generation function:
            - Manually reconvolve the block signal
              and compare it to the BOLD signal
            - test the produce signal SNR
            - test if noise == (noisy_ar_s - ar_s)
        """
        Parallel(n_jobs=-1)(delayed(_test_data)(res, snr)
                            for res, snr in self._yield_data())


if __name__ == '__main__':
    unittest.main()
