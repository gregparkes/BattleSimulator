#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:13:53 2019

@author: gparkes
"""
import numpy as np
from scipy import stats

class Distribution(object):
    """
    Here we provide a class that handles the creation, manipulation and deletion of
    Distributions which we can pass to manipulate the physical location of units
    in a given Battle.
    """

    def __init__(self, distribution="normal", **parameters):
        """
        Initialise the continuous distribution object with a name mapping to a
        scipy.stats distribution.

        Parameters
        --------
        distribution : str
            One of ['uniform','gamma','beta','normal','laplace','exp','chi','gaussian'], default 'normal'
        parameters : dict
            Parameters of the associated scipy.stats.<distribution>, if wanted.
        """
        self.dist_ = distribution
        self.x_offset_ = 0
        self.y_offset_ = 0
        if len(parameters) > 0:
            self._x_param = parameters
            self._y_param = parameters
        else:
            self._pmap = dict(zip(self.scipy_dists_, self._defaults))
            self.x_param_ = self._pmap[distribution]
            self.y_param_ = self._pmap[distribution]
        self._fmap = dict(zip(self.scipy_dists_, self._scipy_funcs))


    def _get_dists(self):
        return ["uniform", 'gaussian', 'gamma', 'beta', 'normal', 'laplace', 'exp', 'chi']
    def _get_dist_funcs(self):
        return [stats.uniform, stats.norm, stats.gamma, stats.beta, stats.norm, stats.laplace,
                      stats.expon, stats.chi]
    def _get_default_params(self):
        return [
            {'loc': 0., 'scale': 1.}, {"loc": 0., "scale": 1.},
            {"a": 2.}, {"a": .5, "b": .5},
            {"loc": 0., "scale": 1.}, {}, {}, {"df": 1}
        ]
    def _get_x_param(self):
        return self._x_param
    def _get_y_param(self):
        return self._y_param
    def _set_x_param(self,x):
        self._x_param = x
    def _set_y_param(self,y):
        self._y_param = y
    def _get_dist(self):
        return self._dist
    def _set_dist(self, dist):
        if dist not in self.scipy_dists_:
            raise ValueError("distribution '{}' not found in {}".format(dist,self.scipy_dists_))
        self._dist = dist


    scipy_dists_ = property(_get_dists, doc="list of accepted distributions")
    dist_ = property(_get_dist, _set_dist, doc="the selected distribution to use")
    x_param_ = property(_get_x_param, _set_x_param, doc="x-parameters directly passed to scipy.stats.<distribution>")
    y_param_ = property(_get_y_param, _set_y_param, doc="y-parameters directly passed to scipy.stats.<distribution>")

    _scipy_funcs = property(_get_dist_funcs, doc="list of accepted scipy function distributions")
    _defaults = property(_get_default_params, doc="list of default dictionary parameters for scipy.stats.<distribution>")

    def setx(self, **x):
        """
        Set the x parameter if need be.
        """
        self.x_param_ = x
        return self


    def sety(self, **y):
        """
        Set the u parameter if need be.
        """
        self.y_param_ = y
        return self

    def xoff(self, off):
        self.x_offset_ = off
        return self

    def yoff(self, off):
        self.y_offset_ = off
        return self


    def sample(self, size):
        """
        Sample 'size' points from the x and y distribution, parameters.

        Parameters
        -------
        size : int
            N number of samples.

        Returns
        -------
        S : np.ndarray(size,2)
            S represents np.ndnarray matrix of positions.
        """
        S = np.zeros((size, 2), dtype=np.float_)
        S[:, 0] = self._fmap[self.dist_](**self.x_param_).rvs(size=size) + self.x_offset_
        S[:, 1] = self._fmap[self.dist_](**self.y_param_).rvs(size=size) + self.y_offset_
        return S


    def __repr__(self):
        return "{}: x={}, y={}".format(self.dist_, self.x_param_, self.y_param_)
