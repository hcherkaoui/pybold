""" Test the proximity module.
"""
import unittest
import numpy as np
from pybold.proximity import LInfinityBallIndic, SquareSum, L2Norm, L1Norm


class TestLInfinityBallIndic(unittest.TestCase):
    def test_l1_ball_indic(self):
        """ Test the unit ball indic-func proximity operator.
        """
        prox = LInfinityBallIndic(0.5)
        x = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        prox_x = prox.op(x)
        prox_x_ref = np.array([-0.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.5])
        assert(np.allclose(prox_x_ref, prox_x))


class TestSquareSum(unittest.TestCase):
    def test_square_sum(self):
        """ Test the square sum proximity operator.
        """
        prox = SquareSum(0.5)
        x = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        prox_x = prox.op(x)
        prox_x_ref = np.array([-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5])
        assert(np.allclose(prox_x_ref, prox_x))


class TestL2Norm(unittest.TestCase):
    def test_l2_norm(self):
        """ Test the l2 norm proximity operator.
        """
        prox = L2Norm(0.5)
        x = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        prox_x = prox.op(x)
        prox_x_ref = x - 0.5 * x / np.linalg.norm(x)
        assert(np.allclose(prox_x_ref, prox_x))


class TestL1Norm(unittest.TestCase):
    def test_l1_norm(self):
        """ Test the soft-thresholding operator.
        """
        prox = L1Norm(0.5)
        x = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        prox_x = prox.op(x)
        prox_x_ref = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        assert(np.allclose(prox_x_ref, prox_x))


if __name__ == '__main__':
    unittest.main()
