# coding: utf-8
""" This module gathers usefull usefull functions.
"""
import numpy as np
from numpy.linalg import norm as norm_2
from scipy.interpolate import splrep, sproot


class Tracker:
    """ Callback class to be used with fmin_bfgs from Scipy.
    """
    def __init__(self, f, args, verbose=0):
        self.J = []
        self.f = f
        self.args = list(args)
        self.verbose = verbose
        self.idx = 0

    def __call__(self, x):
        self.idx += 1
        args = [x] + self.args
        j = self.f(*args)
        if self.verbose > 2:
            print("At iter {0}, tracked function = "
                  "{1:.4f}".format(self.idx, j))
        self.J.append(j)


def fwhm(hrf_t, hrf, k=3):
    """Return the full width at half maximum.
    """
    half_max = np.amax(hrf) / 2.0
    s = splrep(hrf_t, hrf - half_max, k=k)
    roots = sproot(s)
    try:
        fwhm_ = np.abs(roots[1] - roots[0])
    except IndexError:
        fwhm_ = -1
    return fwhm_


def random_generator(random_state):
    """ Return a random instance with a fix seed if random_state is a int.
    """
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif random_state is None:
        return np.random  # tweak to call directly the np.random module
    else:
        raise ValueError("random_state could only be seed-int or None, "
                         "got {0}".format(type(random_state)))


def spectral_radius_est(L, x_shape, nb_iter=30, tol=1.0e-6, verbose=False):
    """ EStimation of the spectral radius of the operator L.
    """
    x_old = np.random.randn(*x_shape)

    stopped = False
    for i in range(nb_iter):
        x_new = L.adj(L.op(x_old)) / norm_2(x_old)
        if(np.abs(norm_2(x_new) - norm_2(x_old)) < tol):
            stopped = True
            break
        x_old = x_new
    if not stopped and verbose:
        print("Spectral radius estimation did not converge")

    return norm_2(x_new)


def __inf_norm(x):
    """ Private helper for inf-norm normalization a list of arrays.
    """
    return x / (np.max(np.abs(x)) + 1.0e-12)


def _inf_norm(arr, axis=1):
    """ Private helper of inf-norm normalization a list of arrays.
    """
    if arr.ndim == 2:
        arr = np.apply_along_axis(func1d=__inf_norm,
                                  axis=axis, arr=arr)
        return np.vstack(arr)
    elif arr.ndim in [1, 3]:
        return __inf_norm(arr)
    else:
        raise ValueError("inf-norm normalization only handle "
                         "1D, 2D or 3D arrays")


def inf_norm(arrays, axis=1):
    """ Inf-norm normalization a list of arrays.
    """
    if isinstance(arrays, list):
        return [_inf_norm(a, axis=axis) for a in arrays]
    else:
        return _inf_norm(arrays, axis=axis)
