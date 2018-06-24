""" Test the utils module.
"""
import unittest
import numpy as np
from pybold.utils import max_min_norm


class TestMinMaxNorm(unittest.TestCase):
    def test_max_min_of_minmaxnorm(self):
        """ Test (max - min) of the min-max-normalized array is 1.0.
        """
        for N in [100, 128, 200, 256, 250, 300, 500, 600, 1000]:
            signal = np.random.randn(N)
            signal = max_min_norm(signal)
            np.testing.assert_almost_equal(signal.max() - signal.min(), 1.0)


if __name__ == '__main__':
    unittest.main()
