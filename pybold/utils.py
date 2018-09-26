# coding: utf-8
""" This module gathers usefull usefull functions.
"""
import itertools
import psutil
from joblib import Parallel, delayed
import numpy as np
from numpy.linalg import norm as norm_2
from scipy.interpolate import splrep, sproot
import pywt


def grid_search(func, param_grid, wrapper=None, n_jobs=1, verbose=0):
    """ Run `func` on the carthesian product of `param_grid`.

    Parameters:
    -----------
    func : func,
        function on which apply a grid-search.

    param_grid : dict of list,
        dict of params specify by list of values.

    wrapper : fun (default=None),
        optinal function that will be called with 'func' as first argument.

    n_jobs : int (default=1),
        number of CPU used.

    verbose : int (default=0),
        verbosity level.

    Results:
    --------
    list_kwargs: dict,
        the list of the params used for each reconstruction.
    res: list,
        the list of result for each reconstruction.
    """
    if wrapper is None:
        def _default_wrapper(recons_func, **kwargs):
            return recons_func(**kwargs)
        wrapper = _default_wrapper
    # sanitize value to list type
    for key, value in param_grid.iteritems():
        if not isinstance(value, list):
            param_grid[key] = [value]
    list_kwargs = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]
    # Run the reconstruction
    if verbose > 0:
        if n_jobs == -1:
            n_jobs_used = psutil.cpu_count()
        elif n_jobs == -2:
            n_jobs_used = psutil.cpu_count() - 1
        else:
            n_jobs_used = n_jobs
        print(("Running grid_search for {0} candidates"
               " on {1} jobs").format(len(list_kwargs), n_jobs_used))
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(wrapper)(func, **kwargs)
                   for kwargs in list_kwargs)
    return list_kwargs, res


def mad(x, c=0.6744):
    """ Median absolute deviation.
    """
    return np.median(np.abs(x - np.median(x))) / c


def mad_daub_noise_est(x, c=0.6744):
    """ Estimate the statistical dispersion of the noise with Median Absolute
    Deviation on the first order detail coefficients of the 1d-Daubechies
    wavelets transform.
    """
    try:
        _, cD = pywt.wavedec(x, pywt.Wavelet('db3'), level=1)
    except ValueError:
        cD = pywt.wavedec(x, pywt.Wavelet('db3'), level=0)
    return mad(cD, c=c)


class Tracker:
    """ Callback class to be used with optimization function from Scipy.
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
            print("At iterate {0}, tracked function = "
                  "{1:.6f}".format(self.idx, j))
        self.J.append(j)


def fwhm(t_hrf, hrf, k=3):
    """Return the full width at half maximum.

    Parameters:
    -----------
    t_hrf : 1d np.ndarray,
        the sampling od time.

    hrf : 1d np.ndarray,
        the HRF.

    k : int (default=3),
        the degree of spline to fit the HRF.

    Return:
    -------
    fwhm : float,
        the FWHM
    """
    half_max = np.amax(hrf) / 2.0
    s = splrep(t_hrf, hrf - half_max, k=k)
    roots = sproot(s)
    try:
        return np.abs(roots[1] - roots[0])
    except IndexError:
        return -1


def tp(t_hrf, hrf):
    """ Return time to peak oh the signal.
    """
    return t_hrf[np.argmax(hrf)]


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
