#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:13:53 2019

@author: gparkes
"""
import numpy as np

__all__ = ["Distribution"]


class Distribution(object):
    """
    Here we provide a class that handles the creation, manipulation and deletion of
    Distributions which we can pass to manipulate the physical location of units
    in a given Battle.
    """

    def __init__(self, *args: str, **parameters):
        """
        Initialise the continuous distribution object with a name mapping to a
        scipy.stats distribution.

        Parameters
        --------
        args : list or Distribution
            Argument (in order) of the associated scipy.stats.<distribution>.
            Accepted parameters are the names of distribution ONLY.
            ['norm', 'uniform', 'gamma', 'beta', 'norm', 'laplace', 'exp', 'chi']
            Overrides if another distribution is in the parameters dict.
        parameters : dict
            Parameters of the associated scipy.stats.<distribution>, if wanted.
            Accepted keys: ['name', 'id', 'dist', 'distribution', 'mean', 'var',
                'loc', 'scale', 'a', 'b', 'df', 'sd', 'std']
        """
        self._dist = None

        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], Distribution):
                # copy
                self._dist = args[0]._dist
                self._x_param = args[0]._x_param
                self._y_param = args[0]._y_param
                self._option_i = args[0]._option_i
                self._dist_func = self.options_["funcs"][self._option_i]
                return
            # assuming we are using the standard arguments for scipy.stats
            self._get_distribution_name_from_args(*args)
        else:
            self._get_distribution_name_from_kwargs(**parameters)

        self._from_dict(**parameters)
        # get mapper ready
        self._dist_func = self.options_["funcs"][self._option_i]

    def _get_distribution_name_from_args(self, *args):
        if args[0] in self.options_["names"]:
            self.dist_ = args[0]
        elif args[0] in self.options_["name_maps"]:
            self.dist_ = self.options_["name_maps"][args[0]]
        else:
            raise ValueError("argument {} not found in options {}".format(args[0], self.options_["names"]))
        self._option_i = self.options_["names"].index(self.dist_)

    def _get_distribution_name_from_kwargs(self, **kws):
        # get the name
        # accepted keys
        if self.dist_ is not None:
            return
        else:
            dist_keys = ["name", "id", "dist", "distribution"]

            for k, arg in kws.items():
                if k in dist_keys:
                    if arg in self.options_["names"]:
                        self.dist_ = arg
                    elif arg in self.options_["name_maps"]:
                        self.dist_ = self.options_["name_maps"][arg]
                    else:
                        raise ValueError("argument '{}' not recognised in {}".format(arg, self.options_["names"]))

        if self.dist_ is not None:
            self._option_i = self.options_["names"].index(self.dist_)
        else:
            self.dist_ = "normal"
            self._option_i = 3

    def _from_dict(self, **kws):
        # arguments accepted
        dist_args = ["mean", "var", "loc", "scale", "sd", "std", "a", "b", "df"]
        dist_map = dict(zip(dist_args, ["loc", "scale", "loc", "scale", "scale", "scale", "a", "b", "df"]))
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

            if keyword in dist_args:
                # test argument
                if not isinstance(argument, (float, int, np.int, np.float, np.int64, np.float64)):
                    raise TypeError(
                        "argument '{}' must be of type [float, int] for distribution parameter".format(argument))

                k_m = dist_map[keyword]
                if dim == "x":
                    d_xk[k_m] = argument
                elif dim == "y":
                    d_yk[k_m] = argument
                else:
                    d_xk[k_m] = argument
                    d_yk[k_m] = argument

        # search to fill any missing gaps with default parameters
        must_params = self.options_["acc_params"][self._option_i]
        for p in must_params:
            if p not in d_xk:
                d_xk[p] = self.options_["def_params"][self._option_i][p]
            if p not in d_yk:
                d_yk[p] = self.options_["def_params"][self._option_i][p]

        # assign d_k to d.x_param_ - these throw errors if incorrect format
        self.x_param_ = d_xk
        self.y_param_ = d_yk

    @property
    def x_param_(self):
        """Returns all the x-parameters."""
        return self._x_param

    @x_param_.setter
    def x_param_(self, x):
        if not isinstance(x, dict):
            raise TypeError("'x' must be of type [dict]")
        if len(x) == 0:
            self._x_param = {}
            return
        for keyword, argument in x.items():
            if keyword not in self.options_["acc_params"][self._option_i]:
                raise ValueError("keyword '{}' does not belong in distribution '{}', which has {}"
                                 .format(keyword, self.dist_, self.options_["acc_params"][self._option_i]))
        self._x_param = x

    @property
    def y_param_(self):
        """Returns all of the y-parameters."""
        return self._y_param

    @y_param_.setter
    def y_param_(self, y):
        if not isinstance(y, dict):
            raise TypeError("'x' must be of type [dict]")
        if len(y) == 0:
            self._y_param = {}
            return
        for keyword, argument in y.items():
            if keyword not in self.options_["acc_params"][self._option_i]:
                raise ValueError("keyword '{}' does not belong in distribution '{}', which has {}"
                                 .format(keyword, self.dist_, self._options["acc_params"][self._option_i]))
        self._y_param = y

    def _get_dist_names(self):
        return self.options_["names"]

    @property
    def dist_(self):
        """The distribution type."""
        return self._dist

    @dist_.setter
    def dist_(self, dist):
        if not isinstance(dist, str):
            raise TypeError("dist must be of type [str]")
        if dist not in self.options_["names"]:
            raise ValueError("distribution '{}' not found in {}".format(dist, self.options_["names"]))
        self._dist = dist

    def _check_parameters(self, params):
        for p, v in params.items():
            if p not in self.options_["acc_params"][self._option_i]:
                raise ValueError(
                    "parameter '{}' not found in {}".format(p, self.options_["acc_params"][self._option_i]))
            if not isinstance(v, (float, np.float, np.float64, int, np.int, np.int64)):
                raise TypeError("parameter item must be of type [int, float]")

    @property
    def mean_(self):
        """The location parameters."""
        return np.asarray([self.x_param_["loc"], self.y_param_["loc"]])

    @property
    def options_(self):
        """A clunky dictionary with the total options."""
        return {
            "names": ["uniform", 'gamma', 'beta', 'normal', 'laplace', 'exponential', 'chisquare'],
            "funcs": [np.random.uniform, np.random.gamma, np.random.beta, np.random.normal, np.random.laplace,
                      np.random.exponential, np.random.chisquare],
            # a dictionary to map alternative names used.
            "name_maps": {"gaussian": "normal", "norm": "normal", "expon": "exponential", "exp": "exponential"},
            # default params
            "def_params": [
                # uniform
                {'low': 0., 'high': 1.},
                # gamma
                {"k": 2., "scale": 1.},
                # beta
                {"a": .5, "b": .5},
                # normal
                {"loc": 0., "scale": 1.},
                # laplace
                {"loc": 0., "scale": 1.},
                # exp
                {"scale": 1.},
                # chisq
                {"df": 1}
            ],
            # accepted params
            "acc_params": [("low", "high"), ("k", "scale"),
                           ("a", "b"), ("loc", "scale"), ("loc", "scale"),
                           ("scale"), ("df")]
        }

    def setx(self, **x):
        """
        Set the x parameter if need be.
        """
        self._check_parameters(x)
        self.x_param_ = x
        return self

    def sety(self, **y):
        """
        Set the y parameter if need be.
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
        S[:, 0] = self._dist_func(**self.x_param_, size=size)
        S[:, 1] = self._dist_func(**self.y_param_, size=size)
        return S

    def sample_xy(self, size):
        """Sample `size` points from x and y distribution and return separately."""
        return self._dist_func(**self.x_param_, size=size), self._dist_func(**self.y_param_, size=size)

    def __repr__(self):
        return "{}: x={}, y={}".format(self.dist_, self.x_param_, self.y_param_)
