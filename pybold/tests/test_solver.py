""" Test the solver module.
"""
import unittest
import numpy as np
from pybold.linear import Identity
from pybold.proximity import SquareSum
from pybold.gradient import L2ResidualLinear
from pybold.solvers import condatvu, nesterov_forward_backward
from pybold.data import gen_ai_s


# Here we test two solvers implementation:
# - Condat-Vu
# - FISTA
#
# The test is very dummy: a l2 norm residual and a l2 norm regularization with
# a very long number of iterations.

# TODO: make the Condat-vu pass the test... there is amplitude issue.


class TestCondat(unittest.TestCase):
    def test_condat(self):
        """ Test the Condat-Vu algorithm.
        """
        ai_s, _, _ = gen_ai_s(dur=5, tr=1.0, nb_events=4,
                              avg_dur=5, std_dur=1, random_state=0)

        Id = Identity()
        x0_shape = z0_shape = (len(ai_s), )
        x0 = np.zeros(x0_shape)
        z0 = np.zeros(z0_shape)
        lbda = 1.0e-3
        prox = SquareSum(lbda)
        grad = L2ResidualLinear(Id, ai_s, x0_shape)

        est_ai_s, _ = condatvu(
                grad=grad, L=Id, prox=prox, x0=x0, z0=z0, sigma=0.5, tau=None,
                rho=1.0, nb_iter=9999, early_stopping=True, wind=24,
                tol=1.0e-30, verbose=0,
                                    )
        true_est_ai_s = (1.0 / (2 * lbda + 1)) * ai_s
        np.testing.assert_almost_equal(true_est_ai_s, est_ai_s, decimal=7)


class TestFISTA(unittest.TestCase):
    def test_fista(self):
        """ Test the FISTA algorithm.
        """
        ai_s, _, _ = gen_ai_s(dur=5, tr=1.0, nb_events=4,
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
