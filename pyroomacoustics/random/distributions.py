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
    def sample(self, shape=None, rng=None):
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

    Parameters
    ----------
    dim: The number of dimensions of the random vectors.
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
        scale = 1.0 / self._n_sphere_area
        return np.ones_like(x, shape=x.shape[:-1]) * scale

    def sample(self, size=None, rng=None):
        return uniform_spherical(dim=self.dim, size=size, rng=rng)


class UnnormalizedUniformSpherical(UniformSpherical):
    """A convenience class to use for rejection sampling."""

    @property
    def _n_sphere_area(self):
        return 1.0


class PowerSpherical(Distribution):
    def __init__(self, loc=None, scale=None):
        if loc is None:
            loc = np.array([1.0, 0.0, 0.0])
        else:
            loc = np.array(loc)

            if loc.ndim != 1:
                raise ValueError(
                    f"The location should be a 1d array (got {loc.shape=})"
                )

        if scale is None:
            self._scale = 1.0
        elif scale > 0.0:
            self._scale = scale
        else:
            raise ValueError(f"Scale should be a positive number (got {scale}).")

        super().__init__(len(loc))

        loc_norm = np.linalg.norm(loc)
        if abs(loc_norm - 1.0) > 1e-5:
            raise ValueError(
                "The location parameter should be a unit "
                f"norm vector (got norm={loc_norm})."
            )

        self._loc = loc

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

    def sample(self, size=None, rng=None):
        return power_spherical(loc=self._loc, scale=self._scale, size=size, rng=rng)

    def pdf(self, x):
        assert (
            x.shape[-1] == self.dim
        ), "Input dimension does not match distribution dimension"
        return self._Z_inv * (1.0 + np.matmul(x, self.loc)) ** self.scale
