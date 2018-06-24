# coding: utf-8
""" This module gathers convolution functions.
"""
import numpy as np
from numpy.fft import rfft, irfft
from .padding import custom_padd, unpadd


def spectral_convolve(k, x):
    """ Return k.conv(x).

    Parameters:
    -----------
    k : 1d np.ndarray,
        kernel.
    x : 1d np.ndarray,
        signal.

    Results:
    --------
    k_conv_x : 1d np.ndarray,
        the convolved signal.
    """
    x, p = custom_padd(x)
    N = len(x)
    fft_k = rfft(k, n=N, norm=None)
    padded_h_conv_x = irfft(fft_k * rfft(x,  n=N, norm=None), norm=None)
    k_conv_x = unpadd(padded_h_conv_x, p)

    return k_conv_x


def spectral_retro_convolve(k, x):
    """ Return k_t.conv(x).

    Parameters:
    -----------
    k : 1d np.ndarray,
        kernel.
    x : 1d np.ndarray,
        signal.

    Results:
    --------
    k_conv_x : 1d np.ndarray,
        the convolved signal.
    """
    x, p = custom_padd(x)
    N = len(x)
    fft_k = rfft(k, n=N, norm=None).conj()
    padded_k_conv_x = irfft(fft_k * rfft(x,  n=N, norm=None), norm=None)
    k_conv_x = unpadd(padded_k_conv_x, p)

    return k_conv_x


def toeplitz_from_kernel(k, dim_in, dim_out=None):
    """ Return the Toeplitz matrix that correspond to k.conv(.).

    Parameters:
    -----------
    k : 1d np.ndarray,
        kernel.
    dim_in : int,
        dimension of the input vector (dim of x in y = k.conv(x)).
    dim_out : int (default None),
        dimension of the ouput vector (dim of y in y = in k.conv(x)). If None
        dim_out = dim_in.

    Results:
    --------
    K : 2d np.ndarray,
        the Toeplitz matrix corresponding to the convolution specified.
    """
    if dim_out is None:
        dim_out = dim_in

    padded_k = np.hstack([np.zeros(dim_in), np.flipud(k), np.zeros(dim_in)])
    K = np.empty((dim_out, dim_in))
    for i in range(dim_out):
        start_idx = (dim_in + len(k) - 1) - i
        K[i, :] = padded_k[start_idx: start_idx + dim_in]

    return K


def simple_convolve(k, x, dim_out=None):
    """ Return k.conv(x).

    Parameters:
    -----------
    k : 1d np.ndarray,
        kernel.
    x : 1d np.ndarray,
        signal.
    dim_out : int (default None),
        dimension of the ouput vector (dim of y in y = in k.conv(x)). If None
        d = len(x).

    Results:
    --------
    k_conv_x : 1d np.ndarray,
        the convolved signal.
    """
    if dim_out is None:
        dim_out = len(x)
    dim_in = len(x)

    padded_k = np.hstack([np.zeros(dim_in), np.flipud(k), np.zeros(dim_in)])
    k_conv_x = np.empty(dim_out)

    for i in range(dim_out):
        start_idx = (dim_in + len(k) - 1) - i
        k_conv_x[i] = (padded_k[start_idx: start_idx + dim_in] * x).sum()

    return k_conv_x


def simple_retro_convolve(k, x, dim_out=None):
    """ Return k_t.conv(x).

    Parameters:
    -----------
    k : 1d np.ndarray,
        kernel
    x : 1d np.ndarray,
        signal.
    dim_out : int (default None),
        dimension of the ouput vector (dim of y in y = in k.conv(x)). If None
        d = len(x).

    Results:
    --------
    k_conv_x : 1d np.ndarray,
        the convolved signal.
    """
    if dim_out is None:
        dim_out = len(x)
    dim_in = len(x)

    padded_k = np.hstack([np.zeros(dim_out - 1), k, np.zeros(dim_out - 1)])
    k_conv_x = np.empty(dim_out)

    for i in range(dim_out):
        start_idx = dim_out - 1 - i
        k_conv_x[i] = (padded_k[start_idx: start_idx + dim_in] * x).sum()

    return k_conv_x
