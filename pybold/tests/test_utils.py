""" Test the utils module.
"""
import unittest
import numpy as np
from pybold.utils import inf_norm


class TestMinMaxNorm(unittest.TestCase):
    def test_inf_norm(self):
        """ Test (max - min) of the min-max-normalized array is 1.0.
        """
        for N in [100, 128, 200, 256, 250, 300, 500, 600, 1000]:
            signal = np.random.randn(N)
            signal = inf_norm(signal)
            np.testing.assert_almost_equal(np.max(np.abs(signal)), 1.0)


if __name__ == '__main__':
    unittest.main()
