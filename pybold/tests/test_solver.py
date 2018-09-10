""" Test the solver module.
"""
import unittest
import numpy as np
from pybold.linear import Identity
from pybold.proximity import SquareSum
from pybold.gradient import L2ResidualLinear
from pybold.solvers import nesterov_forward_backward
from pybold.data import gen_rnd_ai_s


class TestSolver(unittest.TestCase):
    def test_nesterov_forward_backward(self):
        """ Test the Nesterov forward backward algorithm.
        """
        ai_s, _, _ = gen_rnd_ai_s(dur=5, tr=1.0, nb_events=4,
                              avg_dur=5, std_dur=1, random_state=0)

        Id = Identity()
        z0 = np.zeros(len(ai_s))
        lbda = 1.0
        prox = SquareSum(lbda)
        grad = L2ResidualLinear(Id, ai_s, z0.shape)

        est_ai_s, _ = nesterov_forward_backward(
                    grad=grad, prox=prox, v0=z0, nb_iter=9999,
                    wind=24, tol=1.0e-30, early_stopping=True, verbose=0,
                          )
        true_est_ai_s = (1.0 / (2 * lbda + 1)) * ai_s
        np.testing.assert_almost_equal(true_est_ai_s, est_ai_s, decimal=7)


if __name__ == '__main__':
    unittest.main()
