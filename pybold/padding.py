# coding: utf-8
""" This module provide the padding functions used with the Fourier
implementation for the HRF operator.
"""
import numpy as np


def _padd_symetric(arrays, p, c, paddtype):
    """ Helper function to c-padd in a symetric way arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  int,
        length of padding.

    c : float,
        value to padd with.

    paddtype : ['left', 'right'],
        where to place the padding.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the padded array or list of arrays
    """
    # case list of arrays (symm padd for list of arr)
    if isinstance(arrays, list):
        if paddtype == "center":
            p_left = int(p / 2)
            p_right = int(p / 2) + (p % 2)
            return [np.hstack([c*np.ones(p_left), a, c*np.ones(p_right)])
                    for a in arrays]
        elif paddtype == "left":
            return [np.hstack([c*np.ones(p), a]) for a in arrays]
        elif paddtype == "right":
            return [np.hstack([a, c*np.ones(p)]) for a in arrays]
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")

    # case one arrays (symm padd for one of arr)
    else:
        if paddtype == "center":
            p_left = int(p / 2)
            p_right = int(p / 2) + (p % 2)
            return np.hstack([c*np.ones(p_left), arrays,
                              c*np.ones(p_right)])
        elif paddtype == "left":
            return np.hstack([c*np.ones(p), arrays])
        elif paddtype == "right":
            return np.hstack([arrays, c*np.ones(p)])
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")


def _padd_assymetric(arrays, p, c, paddtype):
    """ Helper function to c-padd in an assymetric way arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  tuple of int,
        length of padding.

    c : float,
        value to padd with.

    paddtype : ['center'],
        where to place the padding.

    Note:
    -----
    Will raise a ValueError if paddtype is not 'center'.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the padded array or list of arrays
    """
    # case list of arrays (assymm padd for one of arr)
    if isinstance(arrays, list):
        if paddtype == "center":
            return [np.hstack([c*np.ones(p[0]), a, c*np.ones(p[1])])
                    for a in arrays]
        elif paddtype == "left":
            raise ValueError("Can't have 'left' paddtype with a tuple padd "
                             " provided")
        elif paddtype == "right":
            raise ValueError("Can't have 'right' paddtype with a tuple padd "
                             " provided")
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")

    # case one arrays (assymm padd for one of arr)
    else:
        if paddtype == "center":
            p_a = np.hstack([c*np.ones(p[0]), arrays, c*np.ones(p[1])])
            return p_a
        elif paddtype == "left":
            raise ValueError("Can't have 'left' paddtype with a tuple padd "
                             " provided")
        elif paddtype == "right":
            raise ValueError("Can't have 'right' paddtype with a tuple padd "
                             " provided")
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")


def padd(arrays, p, c=0.0, paddtype="center"):
    """ Padd a list of arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  int or tuple of int,
        length of padding.

    c : float,
        value to padd with.

    paddtype : ['left', 'right' or 'center'],
        where to place the padding.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array or list of arrays.
    """
    # case of assymetric padding
    if isinstance(p, int):
        if p < 1:  # no padding array
            return arrays
        else:
            return _padd_symetric(arrays, p=p, c=c, paddtype=paddtype)

    # case of symetric padding
    else:
        return _padd_assymetric(arrays, p=p, c=c, paddtype=paddtype)


def _unpadd_symetric(arrays, p, paddtype):
    """ Helper function to unpadd in an assymetric way arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  int,
        length of padding.

    paddtype : ['left', 'right'],
        where to place the padding.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array or list of arrays.
    """
    # case list of arrays (symm padd for list of arr)
    if isinstance(arrays, list):
            if paddtype == "center":
                p_left = int(p / 2)
                p_right = int(p / 2) + (p % 2)
                return [a[p_left:-p_right] for a in arrays]
            elif paddtype == "left":
                return [a[p:] for a in arrays]
            elif paddtype == "right":
                return [a[:-p] for a in arrays]
            else:
                raise ValueError("paddtype should be ['left', "
                                 "'center', 'right']")

    # case one array (symm padd for one of arr)
    else:
        if paddtype == "center":
            p_left = int(p / 2)
            p_right = int(p / 2) + (p % 2)
            return arrays[p_left:-p_right]
        elif paddtype == "left":
            return arrays[p:]
        elif paddtype == "right":
            return arrays[:-p]
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")


def _unpadd_assymetric(arrays, p, paddtype):
    """ Helper function to unpadd in a symetric way arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  tuple of int,
        length of padding.

    paddtype : ['center'],
        where to place the padding.

    Note:
    -----
    Will raise a ValueError if paddtype is not 'center'.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array or list of arrays.
    """
    # case list of arrays (assymm padd for list of arr)
    if isinstance(arrays, list):
        if paddtype == "center":
            return [a[p[0]:-p[1]] for a in arrays]
        elif paddtype == "left":
            raise ValueError("Can't have 'left' paddtype with a tuple padd "
                             "provided")
        elif paddtype == "right":
            raise ValueError("Can't have 'right' paddtype with  a tuple padd "
                             " provided")
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")

    # case one array (assymm padd for one of arr)
    else:
        if paddtype == "center":
            return arrays[p[0]:-p[1]]
        elif paddtype == "left":
            raise ValueError("Can't have 'left' paddtype with a tuple padd "
                             "provided")
        elif paddtype == "right":
            raise ValueError("Can't have 'right' paddtype with a tuple "
                             "provided")
        else:
            raise ValueError("paddtype should be ['left', "
                             "'center', 'right']")


def unpadd(arrays, p, paddtype="center"):
    """ Unpadd a list of arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray
        array or list of arrays to padd.

    p :  int or tuple of int,
        length of padding.

    paddtype : ['left', 'right' or 'center'],
        where to place the padding.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array or list of arrays.
    """
    # case of assymetric padding
    if isinstance(p, int):
        if p < 1:  # no padding case
            return arrays
        else:
            return _unpadd_symetric(arrays, p, paddtype)

    # case of symetric padding
    else:
        return _unpadd_assymetric(arrays, p, paddtype)


def _custom_padd(a, min_power_of_2=1024, min_zero_padd=50,
                 zero_padd_ratio=0.5):
    """ Private helper to make a zeros-mirror-zeros padd to the next power of
    two of a.

    Parameters
    ----------
    arrays : np.ndarray,
        array to padd.

    min_power_of_2 : int (default=512),
        min length (power of two) for the padded array.

    zero_padd_ratio : float (default=0.5),
        determine the ratio of the length of zero padds (either for the first
        or the second zero-padd) w.r.t the array length.

    min_zero_padd : int (default=50)
        min zero padd, either for the first or the second zero-padd.

    Note:
    -----
    Having a signal close to ~200 can make trouble.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array.

    p : tuple of int,
        the applied padd.
    """
    if not np.log2(min_power_of_2).is_integer():
        raise ValueError("min_power_of_2 should be a power of two, "
                         "got {0}".format(min_power_of_2))

    nextpow2 = int(np.power(2, np.ceil(np.log2(len(a)))))
    nextpow2 = min_power_of_2 if nextpow2 < min_power_of_2 else nextpow2

    diff = nextpow2 - len(a)

    # define the three possible padding
    zero_padd_len = int(zero_padd_ratio * len(a))

    too_short = zero_padd_len < min_zero_padd
    zero_padd_len = min_zero_padd if too_short else zero_padd_len
    p_zeros = (zero_padd_len, zero_padd_len)

    len_padd_left = int(diff / 2)
    len_padd_right = int(diff / 2) + (len(a) % 2)
    p_total = (len_padd_left, len_padd_right)

    if diff == 0:
        # [ s ]

        p_total = 0

        return a, p_total

    elif (0 < diff) and (diff < 2 * zero_padd_len):
        # [ /zeros | s | zeros/ ]

        a = padd(a, p_total)

        return a, p_total

    elif (2 * zero_padd_len < diff) and (diff < 4 * zero_padd_len):
        # [ zeros | mirror-signal | s | mirror-signal | zeros ]

        len_reflect_padd_left = len_padd_left - zero_padd_len
        len_reflect_padd_right = len_padd_right - zero_padd_len
        p_reflect = (len_reflect_padd_left, len_reflect_padd_right)

        # padding
        a = np.pad(a, p_reflect, mode='reflect')
        a = padd(a, p_zeros)

        return a, p_total

    else:
        # [ zeros | mirror-signal | zeros | s | zeros | mirror-signal | zeros ]

        len_reflect_padd_left = len_padd_left - 2 * zero_padd_len
        len_reflect_padd_right = len_padd_right - 2 * zero_padd_len
        p_reflect = (len_reflect_padd_left, len_reflect_padd_right)

        # padding
        a = padd(a, p_zeros)
        a = np.pad(a, p_reflect, mode='reflect')
        a = padd(a, p_zeros)

        return a, p_total


def custom_padd(arrays, min_power_of_2=1024, min_zero_padd=50,
                zero_padd_ratio=0.5):
    """ Zeros-mirror-zeros padding function to the next power of two of arrays.

    Parameters
    ----------
    arrays : np.ndarray or list of np.ndarray,
        array or list of arrays to padd.

    min_power_of_2 : int (default=512),
        min length (power of two) for the padded array.

    zero_padd_ratio : float (default=0.5),
        determine the ratio of the length of zero padds (either for the first
        or the second zero-padd) w.r.t the array length.

    min_zero_padd : int (default=50)
        min zero padd, either for the first or the second zero-padd.

    Note:
    -----
    Having a signal close to ~200 can make trouble.

    Results
    -------
    arrays : np.ndarray or list of np.ndarray
        the unpadded array or list of arrays.

    p : tuple of int,
        the applied padd (might not be the same for all the arrays).
    """
    if isinstance(arrays, list):
        _, padd = _custom_padd(arrays[0],
                               min_power_of_2=min_power_of_2,
                               min_zero_padd=min_zero_padd,
                               zero_padd_ratio=zero_padd_ratio)
        padd_arrays = [_custom_padd(a,
                                    min_power_of_2=min_power_of_2,
                                    min_zero_padd=min_zero_padd,
                                    zero_padd_ratio=zero_padd_ratio)[0]
                       for a in arrays]
        return padd_arrays, padd
    else:
        return _custom_padd(arrays,
                            min_power_of_2=min_power_of_2,
                            min_zero_padd=min_zero_padd,
                            zero_padd_ratio=zero_padd_ratio)
