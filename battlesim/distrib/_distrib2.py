import numpy as np


class Sampling:
    """ Wrapper class for handling numpy.random distributions. """

    def __init__(self, name, *args):
        """name must be one of []"""
        self.__accepted_dists = {'beta', 'binomial', 'chisquare', 'exponential',
                                 'laplace', 'lognormal', 'normal',
                                 'uniform'}
        self.name = name
        self.args = args

    @property
    def name(self):
        """ The name of the numpy distribution to call. """
        return self._name

    @name.setter
    def name(self, name):
        if name not in self.__accepted_dists:
            raise ValueError(f"name {name} must be in {self.__accepted_dists}")
        self._name = name

    @property
    def f(self):
        """Returns the np.random function associated with the name"""
        return getattr(np.random, self.name)

    def sample(self, n):
        """ Samples a 1d from random. """
        return self.f(*self.args, size=(n,))

    def __repr__(self):
        return f"Sampling('{self.name}', {self.args})"
