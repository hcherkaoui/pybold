""" Test the linear module.
"""
import unittest
import numpy as np
from joblib import Parallel, delayed
from pybold.linear import DiscretInteg
from pybold.tests.utils import YieldData
from pybold.convolution import simple_convolve, simple_retro_convolve


class TestInteg(unittest.TestCase):
    def test_integ_op(self):
        """ Test the  operator.
        """
        integ_op = DiscretInteg()
        signal = np.random.randn(500)
        test_integ_signal = integ_op.op(signal)
        ref_integ_signal = np.cumsum(signal)

        assert(np.allclose(test_integ_signal, ref_integ_signal))

    def test_integ_adj(self):
        """ Test the integ adj operator.
        """
        integ_op = DiscretInteg()
        signal = np.random.randn(500)
        test_integ_signal = integ_op.adj(signal)
        ref_integ_signal = np.flipud(np.cumsum(np.flipud(signal)))



if __name__ == '__main__':
    unittest.main()
