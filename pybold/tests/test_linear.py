""" Test the linear module.
"""
import unittest
import numpy as np
from joblib import Parallel, delayed
from pybold.linear import Diff, Conv, DiscretInteg
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

        assert(np.allclose(test_integ_signal, ref_integ_signal))


class TestDiff(unittest.TestCase):
    def test_diff_op(self):
        """ Test the diff operator.
        """
        diff_op = Diff()
        signal = np.random.randn(500)
        test_diff_signal = diff_op.op(signal)
        ref_diff_signal = np.append(0, signal[1:] - signal[:-1])

        assert(np.allclose(test_diff_signal, ref_diff_signal))

    def test_diff_adj(self):
        """ Test the diff adj operator.
        """
        diff_op = Diff()
        signal = np.random.randn(500)
        test_diff_signal = diff_op.adj(signal)
        ref_diff_signal = np.append(signal[:-1] - signal[1:], 0)

        assert(np.allclose(test_diff_signal, ref_diff_signal))


class TestDiffInteg(unittest.TestCase):
    def test_diff_integ_op(self):
        """ Test the diff integ operator.
        """
        diff_op = Diff()
        integ_op = DiscretInteg()

        # border effecct CAN'T be avoid unless the signal is padded
        signal = np.hstack([0, np.cos(np.linspace(-5, 5, 500)), 0])

        test_diff_signal_1 = diff_op.op(integ_op.op(signal))
        test_diff_signal_2 = integ_op.op(diff_op.op(signal))
        ref_diff_signal = signal

        assert(np.allclose(test_diff_signal_1, test_diff_signal_2))
        assert(np.allclose(test_diff_signal_1, ref_diff_signal))

    def test_diff_integ_adj(self):
        """ Test the diff integ adjoperator.
        """
        diff_op = Diff()
        integ_op = DiscretInteg()

        # border effecct CAN'T be avoid unless the signal is padded
        signal = np.hstack([0, np.cos(np.linspace(-5, 5, 500)), 0])

        test_diff_signal_1 = diff_op.adj(integ_op.adj(signal))
        test_diff_signal_2 = integ_op.adj(diff_op.adj(signal))
        ref_diff_signal = signal

        assert(np.allclose(test_diff_signal_1, test_diff_signal_2))
        assert(np.allclose(test_diff_signal_1, ref_diff_signal))


def _test_conv(ai_s, hrf):
    """ Helper to test conv operator.
    """
    ar_s_ref = simple_convolve(hrf, ai_s)
    ar_s_test = Conv(hrf, len(ai_s)).op(ai_s)

    assert(np.allclose(ar_s_ref, ar_s_test))


class TestConv(unittest.TestCase, YieldData):
    def test_conv_op(self):
        """ Test the conv operator.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv)(ai_s, hrf)
                            for ai_s, hrf, _, _
                            in self.yield_blocks_signal())


def _test_adj_conv(ai_s, hrf):
    """ Helper to test conv adj operator.
    """
    ar_s_ref = simple_retro_convolve(hrf, ai_s)
    ar_s_test = Conv(hrf, len(ai_s)).adj(ai_s)

    assert(np.allclose(ar_s_ref, ar_s_test))


class TestAdjConv(unittest.TestCase, YieldData):
    def test_conv_adj_op(self):
        """ Test the conv adj operator.
        """
        Parallel(n_jobs=-1)(delayed(_test_adj_conv)(ai_s, hrf)
                            for ai_s, hrf, _, _
                            in self.yield_blocks_signal())


if __name__ == '__main__':
    unittest.main()
