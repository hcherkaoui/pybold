# coding: utf-8
""" This module gathers the definition of the HRF operator.
"""
import numpy as np
from .convolution import (toeplitz_from_kernel, spectral_convolve,
                          spectral_retro_convolve)


class DiscretInteg:
    """ Intergrator operator.
    """
    def __init__(self):
        pass

    def op(self, x):
        """ Return time integration of x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        integ_x : np.ndarray,
            the integrated 1d vector.
        """
        return np.cumsum(x)

    def adj(self, x):
        """ Return adj time integrated of x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        adj_integ_x : np.ndarray,
            the adj integretated 1d vector.
        """
        return np.flipud(np.cumsum(np.flipud(x)))


class ConvAndLinear:
    """ Linear (Matrix) operator followed by a convolution.
    """
    def __init__(self, M, kernel, dim_in, dim_out=None, spectral_conv=False):
        """ ConvAndLinear linear operator class.
        Parameters:
        -----------
        M : Matrix class,
            the first linear operator.

        kernel : 1d np.ndarray,
            kernel.

        dim_in : int,
            chosen convolution input dimension.

        dim_out : int (default None),
            chosen convolution ouput dimension.
        """
        self.M = M
        self.k = kernel
        self.spectral_conv = spectral_conv
        if not self.spectral_conv:
            self.K = toeplitz_from_kernel(self.k, dim_in=dim_in,
                                          dim_out=dim_out)
            self.K_T = self.K.T

    def op(self, x):
        """ Return k.convolve(D_hrf.dot(x)).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        img_x : np.ndarray,
            the resulting 1d vector.
        """
        bloc_signal = self.M.op(x)

        if self.spectral_conv:
            convolved_signal = spectral_convolve(self.k, bloc_signal)
        else:
            convolved_signal = self.K.dot(bloc_signal)

        return convolved_signal

    def adj(self, x):
        """ Return k.T.convolve(D_hrf.T.dot(x)).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        img_x : np.ndarray,
            the resulting 1d vector.
        """
        if self.spectral_conv:
            retro_convolved_signal = spectral_retro_convolve(self.k, x)
        else:
            retro_convolved_signal = self.K_T.dot(x)

        return self.M.adj(retro_convolved_signal)
