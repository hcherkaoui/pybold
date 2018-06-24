""" Test the convolution module.
"""
import unittest
from joblib import Parallel, delayed
import numpy as np
from pybold.tests.utils import YieldData
from pybold.convolution import (simple_convolve, simple_retro_convolve,
                                spectral_convolve, spectral_retro_convolve,
                                toeplitz_from_kernel)
from pybold.data import spm_hrf, gen_ai_s


# Here we test three implementations for the convolution:
# - the mathematical definition (two for loops)
# - the Toeplitz matrix product
# - the Fourier (spectral multiplication)
#
# We test the direct convolution as long as the adjoint operator in the case
# where kernel = HRF and kernel = block signal (for different HRF and block
# signal)


### case kernel = HRF

## direct op

# Toeplitz


def _test_conv_toeplitz_kernel_hrf(ai_s, hrf):
    """ Helper to test the conv with a Toeplitz implementation,
    the kernel being the HRF.
    """
    H = toeplitz_from_kernel(hrf, len(ai_s), len(ai_s))
    ar_s_ref = simple_convolve(hrf, ai_s)
    ar_s_test = H.dot(ai_s)
    assert(np.allclose(ar_s_ref, ar_s_test, atol=1.0e-7))


class TestToeplitzConvolutionKernelHRF(unittest.TestCase, YieldData):

    def test_toeplitz_convolution_kernel_hrf_on_dirac(self):
        """ Test Toeplitz implementation of the convolution on a single dirac,
        the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_toeplitz_convolution_kernel_hrf_on_ai_s(self):
        """ Test Toeplitz implementation of the convolution on a block signal,
        the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_blocks_signal())


# Fourier


def _test_conv_fourier_kernel_hrf(ai_s, hrf):
    """ Helper to test the conv with a Fourier implementation,
    the kernel being the HRF.
    """
    ar_s_ref = simple_convolve(hrf, ai_s)
    ar_s_test = spectral_convolve(hrf, ai_s)
    assert(np.allclose(ar_s_ref, ar_s_test, atol=1.0e-7))


class TestFourierConvolutionKernelHRF(unittest.TestCase, YieldData):

    def test_fourier_convolution_kernel_hrf_on_dirac(self):
        """ Test Fourier implementation of the convolution on a single dirac,
        the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_fourier_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_fourier_convolution_kernel_hrf_on_ai_s(self):
        """ Test Fourier implementation of the convolution on a block signal,
        the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_fourier_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_blocks_signal())


## adj op


# Toeplitz


def _test_conv_adj_toeplitz_kernel_hrf(ai_s, hrf):
    """ Helper to test the adj conv with a Toeplitz implementation,
    the kernel being the HRF.
    """
    H = toeplitz_from_kernel(hrf, len(ai_s), len(ai_s))
    ar_s_ref = simple_retro_convolve(hrf, ai_s)
    adj_ar_s_test = H.T.dot(ai_s)
    assert(np.allclose(ar_s_ref, ar_s_test, atol=1.0e-7))


class TestToeplitzAdjConvolutionKernelHRF(unittest.TestCase, YieldData):
    def test_toeplitz_adj_convolution_kernel_hrf_on_dirac(self):
        """ Test Toeplitz implementation of the adj convolution on a single
        dirac, the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_adj_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_toeplitz_adj_convolution_kernel_hrf_on_ai_s(self):
        """ Test Toeplitz implementation of the adj convolution on a block
        signal, the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_adj_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_blocks_signal())


# Fourier


def _test_conv_adj_toeplitz_kernel_hrf(ai_s, hrf):
    """ Helper to test the adj conv with a Fourier implementation,
    the kernel being the HRF.
    """
    adj_ar_s_ref = simple_retro_convolve(hrf, ai_s)
    adj_ar_s_test = spectral_retro_convolve(hrf, ai_s)
    assert(np.allclose(adj_ar_s_ref, adj_ar_s_test, atol=1.0e-7))


class TestFourierAdjConvolutionKernelHRF(unittest.TestCase, YieldData):
    def test_fourier_adj_convolution_kernel_hrf_on_dirac(self):
        """ Test Fourier implementation of the adj convolution on a single
        dirac, the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_adj_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_fourier_adj_convolution_kernel_hrf_on_ai_s(self):
        """ Test Fourier implementation of the adj convolution on a block
        signal, the kernel being the HRF.
        """
        Parallel(n_jobs=-1)(delayed(_test_conv_adj_toeplitz_kernel_hrf)(ai_s, hrf)
                            for ai_s, hrf, _, _, _ in self.yield_blocks_signal())


### case kernel = block signal

## direct op

# Toeplitz

def _test_conv_toeplitz_kernel_signal(ai_s, hrf):
    """ Helper to test the conv with a Toeplitz implementation,
    the kernel being the block signal.
    """
    H = toeplitz_from_kernel(ai_s, len(hrf), len(ai_s))
    ar_s_ref = simple_convolve(hrf, ai_s)
    ar_s_sim = simple_convolve(ai_s, hrf, len(ai_s))
    ar_s_test = H.dot(hrf)
    assert(np.allclose(ar_s_ref, ar_s_test, atol=1.0e-7))


class TestToeplitzConvolutionKernelSignal(unittest.TestCase, YieldData):

    def test_toeplitz_convolution_kernel_signal_on_dirac(self):
        """ Test Toeplitz implementation of the convolution on a single dirac,
        the kernel being the block signal.
        """
        Parallel(n_jobs=-1)(
                        delayed(_test_conv_toeplitz_kernel_signal)(ai_s, hrf)
                        for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_toeplitz_convolution_kernel_signal_on_ai_s(self):
        """ Test Toeplitz implementation of the convolution on a block signal,
        the kernel being the block signal.
        """
        Parallel(n_jobs=-1)(
                        delayed(_test_conv_toeplitz_kernel_signal)(ai_s, hrf)
                        for ai_s, hrf, _, _, _ in self.yield_blocks_signal())

## adj op

# Toeplitz

def _test_conv_adj_toeplitz_kernel_signal(ai_s, hrf):
    """ Helper to test the adj conv with a Toeplitz implementation,
    the kernel being the block signal.
    """
    H = toeplitz_from_kernel(ai_s, len(hrf), len(ai_s))
    adj_ar_s_ref = simple_retro_convolve(ai_s, H.dot(hrf), len(hrf))
    adj_ar_s_test = H.T.dot(H.dot(hrf))
    assert(np.allclose(adj_ar_s_ref, adj_ar_s_test, atol=1.0e-7))


class TestToeplitzAdjConvolutionKernelSignal(unittest.TestCase, YieldData):
    def test_toeplitz_adj_convolution_kernel_signal_on_dirac(self):
        """ Test Toeplitz implementation of the adj convolution on a single
        dirac, the kernel being the block signal.
        """
        Parallel(n_jobs=-1)(
                    delayed(_test_conv_adj_toeplitz_kernel_signal)(ai_s, hrf)
                    for ai_s, hrf, _, _, _ in self.yield_diracs_signal())

    def test_toeplitz_adj_convolution_kernel_signal_on_ai_s(self):
        """ Test Toeplitz implementation of the adj convolution on a block
        signal, the kernel being the block signal.
        """
        Parallel(n_jobs=-1)(
                    delayed(_test_conv_adj_toeplitz_kernel_signal)(ai_s, hrf)
                    for ai_s, hrf, _, _, _ in self.yield_blocks_signal())


if __name__ == '__main__':
    unittest.main()
