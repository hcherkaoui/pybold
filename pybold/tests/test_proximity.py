""" Test the proximity module.
"""
import unittest
import numpy as np
from pybold.proximity import L1Norm


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
