# coding: utf-8
""" This module gathers usefull proximal operators.
"""
import numpy as np


class LInfinityBallIndic:
    """ L_infinity-ball indicator proximity operator.
    """
    def __init__(self, w):
        self.w = w

    def cost(self, x):
        """ Return sum(x**2).
        """
        return np.inf if np.any(np.abs(x) > 1.0) else 0.0

    def op(self, x, w=1.0):
        """ Return the l1-ball indicator prox of x.
        """
        th = self.w
        return np.clip(x * w, -th, th)


class SquareSum:
    """ Sum of square proximity operator.
    """
    def __init__(self, w):
        self.w = w

    def cost(self, x):
        """ Return sum(x**2).
        """
        return np.sum(np.square(x))

    def op(self, x, w=1.0):
        """ Return the sum of square prox of x.
        """
        th = self.w * w
        return 1 / (2 * th + 1) * x


class L2Norm:
    """ L2 norm proximity operator.
    """
    def __init__(self, w):
        self.w = w

    def cost(self, x):
        """ Return the l2 norm of x.
        """
        return np.linalg.norm(x)

    def op(self, x, w=1.0):
        """ Return the l2 norm prox x.
        """
        th = self.w * w
        return np.maximum(1 - th / np.linalg.norm(x), 0) * x


class L1Norm:
    """ L1 norm proximity operator.
    """
    def __init__(self, w):
        self.w = w

    def cost(self, x):
        """ Return the l1 norm of x.
        """
        return np.sum(np.abs(x))

    def op(self, x, w=1.0):
        """  Return the l1 norm prox x.
        """
        return np.sign(x) * np.maximum(np.abs(x) - self.w * w, 0)
