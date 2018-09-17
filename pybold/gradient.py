# coding: utf-8
""" This module gathers the definition of the discret diff operator.
"""
import numpy as np
from .utils import spectral_radius_est


class L2ResidualLinear:
    """ Gradient of 1/2 || L(x) - y ||_2^2 w.r.t to x
    """
    def __init__(self, L, y, x_shape, step=None):
        # L being the linear operator
        self.L = L

        # y being the observed vector
        self.y = y

        # precomputation
        self.L_adj_y = self.L.adj(y)

        # x_shape should give the length of the main iterate (1d vector)
        if type(x_shape) is not tuple or not len(x_shape) == 1:
            raise ValueError("x_shape should be a tuple of length 1, "
                             "got {0}".format(x_shape))
        self.x_shape = x_shape

        # estimation of the Lipschitz constant of the linear op
        self.op_norm_L = spectral_radius_est(self.L, self.x_shape)
        self.grad_lipschitz_cst = self.op_norm_L  # majoration
        if step is None:
            self.step = 1.0 / self.grad_lipschitz_cst
        else:
            self.step = step

    def residual(self, x):
        """ Return || h.convolve(x) - y ||_2.
        """
        return np.sum(np.square(self.L.op(x) - self.y))

    def cost(self, x):
        """ Return 1/2 || h.convolve(x) - y ||_2^2.
        """
        return 0.5 * np.sum(np.square(self.L.op(x) - self.y))

    def op(self, x):
        """ Return h.adj_convolve( h.convolve(x) - y )
        """
        return self.L.adj(self.L.op(x)) - self.L_adj_y


class SquaredL2ResidualLinear:
    """ Gradient of 1/2 || L(x) - y ||_2 w.r.t to x
    """
    def __init__(self, L, y, x_shape, step=None):
        # L being the linear operator
        self.L = L

        # y being the observed vector
        self.y = y

        # precomputation
        self.L_adj_y = self.L.adj(y)

        # x_shape should give the length of the main iterate (1d vector)
        if type(x_shape) is not tuple or not len(x_shape) == 1:
            raise ValueError("x_shape should be a tuple of length 1, "
                             "got {0}".format(x_shape))
        self.x_shape = x_shape

        # estimation of the Lipschitz constant of the linear op
        self.op_norm_L = spectral_radius_est(self.L, self.x_shape)
        self.grad_lipschitz_cst = self.op_norm_L  # majoration
        if step is None:
            self.step = 1.0 / self.grad_lipschitz_cst
        else:
            self.step = step

    def residual(self, x):
        """ Return || h.convolve(x) - y ||_2.
        """
        return np.sum(np.square(self.L.op(x) - self.y))

    def cost(self, x):
        """ Return 1/2 || h.convolve(x) - y ||_2^2.
        """
        return np.linalg.norm(self.L.op(x) - self.y)

    def op(self, x):
        """ Return h.adj_convolve( h.convolve(x) - y )
        """
        deno = self.cost(x) + 1.0e-10
        return (self.L.adj(self.L.op(x)) - self.L_adj_y) / deno
