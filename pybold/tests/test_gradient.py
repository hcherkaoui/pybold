""" Test the gradient module.
"""
import unittest
import itertools
import numpy as np
from scipy.optimize import check_grad
from pybold.gradient import L2ResidualLinear
from pybold.convolution import simple_convolve
from pybold.linear import Conv
from pybold.data import gen_ai_s, spm_hrf, add_gaussian_noise


def _test_gradient(ar_s, ai_s, hrf, random_state):
    """ Helper to parallelize the gradient testing.
    """
    x0_shape = (len(ar_s), )
    H = Conv(hrf, len(ar_s))
    grad = L2ResidualLinear(H, ar_s, x0_shape)
    noisy_ai_s, _ = add_gaussian_noise(ai_s, 10.0,
                                       random_state=random_state)

    x_s = [ai_s, noisy_ai_s, np.zeros_like(ai_s), np.ones_like(ai_s)]
    for x in x_s:
        err_grad = check_grad(grad.cost, grad.op, x)
        np.testing.assert_almost_equal(err_grad, 0.0, decimal=5)


class TestGradient(unittest.TestCase):
    @unittest.skip("deactivated till the tr problem is not fixed")
    def _yield_data(self):
        """ Yield data test case.
        """
        # caution random_state should be fixed but it could also systematically
        # lead to failing signal generation, so this arg is carefully set
        random_state_s = [6]
        dur = 5  # minutes
        tr_s = [0.1, 0.5, 2.0]
        hrf_time_length_s = [10.0, 50.0]
        listparams = [random_state_s, tr_s, hrf_time_length_s]
        for params in itertools.product(*listparams):
            random_state, tr, hrf_time_length = params
            ai_s_params = {'dur': dur,
                           'tr': tr,
                           'nb_events': 4,
                           'avg_dur': 1,
                           'std_dur': 5,
                           'overlapping': False,
                           'random_state': random_state,
                           }
            hrf_params = {'tr': tr,
                          'time_length': hrf_time_length,
                          }

            ai_s, _, _ = gen_ai_s(**ai_s_params)
            hrf, _, _ = spm_hrf(**hrf_params)
            ar_s = simple_convolve(hrf, ai_s)
            yield ar_s, ai_s, hrf, random_state

    def test_gradient(self):
        """ Test the gradient operator.
        """
        for ar_s, ai_s, hrf, random_state in self._yield_data():
            _test_gradient(ar_s, ai_s, hrf, random_state)


if __name__ == '__main__':
    unittest.main()
