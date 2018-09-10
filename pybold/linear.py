# coding: utf-8
""" This module gathers the definition of the HRF operator.
"""
import numpy as np
from .convolution import spectral_convolve, spectral_retro_convolve


class Identity():
    """ Identity operator.
    """
    def __init__(self):
        pass

    def op(self, x):
        """ Return x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        x : np.ndarray,
            the signal.
        """
        return x

    def adj(self, x):
        """ Return x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        x : np.ndarray,
            the signal.
        """
        return x


class Matrix():
    """ Linear operator based on a matrix operator.
    """
    def __init__(self, M, M_T=None):
        if M_T is None:
            self.M = M
            self.M_T = M.T
        else:
            self.M = M
            self.M_T = M_T
        self.shape = (self.M_T.shape[1], self.M.shape[1])

    def op(self, x):
        """ Return M.dot(x).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        M_x : np.ndarray,
            M_x.
        """
        return self.M.dot(x)

    def adj(self, x):
        """ Return M.T.dot(x).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        M_T_x : np.ndarray,
            MT_x.
        """
        return self.M_T.dot(x)


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


class Diff:
    """ Diff operator.
    """
    def __init__(self):
        pass

    def op(self, x):
        """ Return time differentiate of x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        diff_x : np.ndarray,
            the differentiated 1d vector.
        """
        return np.append(0, x[1:] - x[:-1])

    def adj(self, x):
        """ Return adj time differentiate of x.

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        adj_diff_x : np.ndarray,
            the adj integretated 1d vector.
        """
        return np.append(x[:-1] - x[1:], 0)


class Conv:
    """ Convolutional operator.
    """
    def __init__(self, kernel, dim_in=None):
        """ Conv linear operator class.
        Parameters:
        -----------
        kernel : 1d np.ndarray,
            kernel.

        dim_in : int,
            chosen dimension image dimension.
        """
        self.k = kernel

    def op(self, x):
        """ Return k.convolve(x).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        k_conv_x : np.ndarray,
            the convolved 1d vector.
        """
        return spectral_convolve(self.k, x)

    def adj(self, x):
        """ Return k.T.convolve(x).

        Parameters:
        -----------
        x : 1d np.ndarray,
            signal.

        Results:
        --------
        k_conv_x : np.ndarray,
            the adj-convolved 1d vector.
        """
        return spectral_retro_convolve(self.k, x)


class ConvAndLinear:
    """ Linear (Matrix) operator followed by a convolution.
    """
    def __init__(self, M, kernel, dim_in, dim_out=None):
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
        convolved_signal = spectral_convolve(self.k, bloc_signal)

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
        retro_convolved_signal = spectral_retro_convolve(self.k, x)

        return self.M.adj(retro_convolved_signal)
