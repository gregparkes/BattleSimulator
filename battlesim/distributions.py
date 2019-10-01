#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:13:53 2019

@author: gparkes
"""
import numpy as np
from scipy import stats


__all__ = ["Distribution"]


def get_options():
    return {
        "names": ["uniform", 'gaussian', 'gamma', 'beta', 'normal', 'laplace', 'exp', 'chi'],
        "funcs": [stats.uniform, stats.norm, stats.gamma, stats.beta, stats.norm, stats.laplace,
                      stats.expon, stats.chi],
        # default params
        "def_params": [
            {'loc': 0., 'scale': 1.}, {"loc": 0., "scale": 1.},
            {"a": 2.}, {"a": .5, "b": .5},
            {"loc": 0., "scale": 1.}, {}, {}, {"df": 1}
        ],
        # accepted params
        "acc_params": [("loc", "scale"), ("loc", "scale"), ("a", "loc","scale"),
            ("a", "b", "loc", "scale"), ("loc","scale"), ("loc", "scale"),
            ("loc","scale"), ("df", "loc", "scale")
        ]
    }


def dist(**kws):
    """
    Creates a distribution using keywords.

    Parameters
    -------
    kws : dict
        A dictionary corresponding to keywords found in the corresponding scipy.stats function
        Allows [name, id, dist, distribution] for type of dist
        Allows [mean, var, loc, scale, sd, std, a, b, df] for dist arguments
        Allows prefixes [x_, y_] or suffixes [_x, _y] to denote differences for x-coords and y-coords
            For example x_loc refers to loc on the x-direction only.

    Returns
    -------
    d : bsm.Distribution
        Corresponding distribution object
    """
    # the name of the distribution
    dist_keys = ["name", "id", "dist", "distribution"]
    # arguments accepted
    dist_args = ["mean","var","loc","scale","sd","std","a","b","df"]
    dist_map = dict(zip(dist_args, ["loc","scale","loc","scale","scale","scale","a","b","df"]))

    d = Distribution("normal")
    d_xk, d_yk = {}, {}

    # iterate over them all
    for keyword, argument in kws.items():
        dim = "none"
        # get x_ y_ _x _y dims
        if keyword.startswith("x_"):
            keyword = keyword[2:]
            dim = "x"
        elif keyword.startswith("y_"):
            keyword = keyword[2:]
            dim = "y"
        elif keyword.endswith("_x"):
            keyword = keyword[:-2]
            dim = "x"
        elif keyword.endswith("_y"):
            keyword = keyword[:-2]
            dim = "y"

        if keyword in dist_keys:
            # test argument
            if not isinstance(argument, str):
                raise TypeError("argument '{}' must be of type [str] for distribution".format(argument))
            # set the distribution object
            d.dist_ = argument
        elif keyword in dist_args:
            # test argument
            if not isinstance(argument, (float, int, np.int, np.float, np.int64, np.float64)):
                raise TypeError("argument '{}' must be of type [float, int] for distribution parameter".format(argument))

            k_m = dist_map[keyword]
            if dim == "x":
                d_xk[k_m] = argument
            elif dim == "y":
                d_yk[k_m] = argument
            else:
                d_xk[k_m] = argument
                d_yk[k_m] = argument
        else:
            raise ValueError("keyword '{}' not found in {}".format(keyword, dist_keys+dist_args))

    # assign d_k to d.x_param_ - these throw errors if incorrect format
    d.x_param_ = d_xk
    d.y_param_ = d_yk
    return d


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
        # set dist, also checking using the _set_dist function
        self.dist_ = distribution
        # extract option settings
        self._options = get_options()
        self._option_i = self._options["names"].index(self.dist_)

        # select accepted parameters for dist
        self._accepted_params = self._options["acc_params"][self._option_i]
        # select distribution func from scipy.stats
        self._dist_func = self._options["funcs"][self._option_i]

        if len(parameters) > 0:
            # check that parameters are actually found in the parameter list
            self._check_parameters(parameters)
            self.x_param_ = self.y_param_ = parameters
        else:
            # default parameters
            self.x_param_ = self.y_param_ = get_options()["def_params"][self._option_i]

    def _get_x_param(self):
        return self._x_param
    def _get_y_param(self):
        return self._y_param
    def _set_x_param(self, x):
        if not isinstance(x, dict):
            raise TypeError("'x' must be of type [dict]")
        if len(x) == 0:
            self._x_param = {}
            return
        for keyword, argument in x.items():
            if keyword not in get_options()["acc_params"][self._option_i]:
                raise ValueError("keyword '{}' does not belong in distribution '{}', which has {}"
                                 .format(keyword, self.dist_, self._options["acc_params"][self._option_i]))
        self._x_param = x

    def _set_y_param(self, y):
        if not isinstance(y, dict):
            raise TypeError("'x' must be of type [dict]")
        if len(y) == 0:
            self._y_param = {}
            return
        for keyword, argument in y.items():
            if keyword not in get_options()["acc_params"][self._option_i]:
                raise ValueError("keyword '{}' does not belong in distribution '{}', which has {}"
                                 .format(keyword, self.dist_, self._options["acc_params"][self._option_i]))
        self._y_param = y

    def _get_dist_names(self):
        return get_options()["names"]
    def _get_dist(self):
        return self._dist
    def _set_dist(self, dist):
        if not isinstance(dist, str):
            raise TypeError("dist must be of type [str]")
        if dist not in self.scipy_dists_:
            raise ValueError("distribution '{}' not found in {}".format(dist,self.scipy_dists_))
        self._dist = dist

    def _check_parameters(self, params):
        for p, v in params.items():
            if p not in self._accepted_params:
                raise ValueError("parameter '{}' not found in {}".format(p, self._accepted_params))
            if not isinstance(v, (float, np.float, np.float64, int, np.int, np.int64)):
                raise TypeError("parameter item must be of type [int, float]")


    scipy_dists_ = property(_get_dist_names, doc="list of accepted distributions")
    dist_ = property(_get_dist, _set_dist, doc="the selected distribution to use")
    x_param_ = property(_get_x_param, _set_x_param, doc="x-parameters directly passed to scipy.stats.<distribution>")
    y_param_ = property(_get_y_param, _set_y_param, doc="y-parameters directly passed to scipy.stats.<distribution>")


    def setx(self, **x):
        """
        Set the x parameter if need be.
        """
        self._check_parameters(x)
        self.x_param_ = x
        return self


    def sety(self, **y):
        """
        Set the u parameter if need be.
        """
        self._check_parameters(y)
        self.y_param_ = y
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
        S[:, 0] = self._dist_func(**self.x_param_).rvs(size=size)
        S[:, 1] = self._dist_func(**self.y_param_).rvs(size=size)
        return S


    def __repr__(self):
        return "{}: x={}, y={}".format(self.dist_, self.x_param_, self.y_param_)
