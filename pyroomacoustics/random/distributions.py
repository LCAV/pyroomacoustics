import abc

import numpy as np
import scipy

from .spherical import power_spherical, uniform_spherical


class Distribution(abc.ABC):
    """
    Abstract base class for distributions. Derived classes
    should implement a ``pdf`` method that provides the pdf value for samples
    and a ``sample`` method that allows to sample from the distribution
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @abc.abstractmethod
    def pdf(self):
        pass

    @abc.abstractmethod
    def sample(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class AdHoc(Distribution):
    """
    A helper class to construct distribution from separate pdf/sample
    functions
    """

    def __init__(self, pdf, sampler, dim):
        super().__init__(dim)

        self._pdf = pdf
        self._sampler = sampler

    def pdf(self, *args, **kwargs):
        return self._pdf(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._sampler(*args, **kwargs)


class UniformSpherical(Distribution):
    """
    Uniform distribution on the n-sphere
    """

    def __init__(self, dim=3):
        super().__init__(dim)
        self._area = (
            self._dim
            * np.pi ** (self._dim / 2)
            / scipy.special.gamma(self._dim / 2 + 1)
        )

    @property
    def _n_sphere_area(self):
        return self._area

    def pdf(self, x):
        return 1.0 / self._n_sphere_area

    def sample(self, size=None):
        return uniform_spherical(dim=self.dim, size=size)


class PowerSpherical(Distribution):
    def __init__(self, dim=3, loc=None, scale=None):
        super().__init__(dim)

        # first canonical basis vector
        if loc is None:
            self._loc = np.zeros(self.dim)
            self._loc[0] = 1.0
        else:
            self._loc = loc

        if not scale:
            self._scale = 1.0
        else:
            self._scale = scale

        # computation the normalizing constant
        a = (self.dim - 1) / 2 + self.scale
        b = (self.dim - 1) / 2
        self._Z_inv = 1.0 / (
            2 ** (a + b)
            * np.pi**b
            * scipy.special.gamma(a)
            / scipy.special.gamma(a + b)
        )

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    def pdf(self, x):
        assert (
            x.shape[-1] == self.dim
        ), "Input dimension does not match distribution dimension"
        return self._Z_inv * (1.0 + np.matmul(x, self.loc)) ** self.scale
