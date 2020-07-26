import abc

import numpy as np
from numpy.random import default_rng

from . import distributions
from .spherical import uniform_spherical


class RejectionSampler:
    def __init__(self, desired_func, proposal_dist, scale=None):

        if not scale:
            self._scale = 1.0
        else:
            self._scale = scale

        self._desired_func = desired_func
        self._proposal_dist = proposal_dist

        self._dim = self._proposal_dist.dim

        # get a random number generator
        self._rng = default_rng()

        # keep track of the efficiency for analysis purpose
        self._n_proposed = 0
        self._n_accepted = 0

    @property
    def efficiency(self):
        if self._n_proposed == 0:
            # before starting to sample, efficiency is maximum for consistency
            # i.e., we haven't rejected anything
            return 1.0
        else:
            return self._n_accepted / self._n_proposed

    @property
    def dim(self):
        return self._dim

    def __call__(self, size=None):

        if not size:
            size = 1

        flat_size = int(np.prod(size))  # flat size

        offset = 0
        samples = np.zeros((flat_size, self.dim))

        while offset < flat_size:

            n_propose = flat_size - offset

            proposal = self._proposal_dist.sample(size=n_propose)
            proposal_pdf_value = self._proposal_dist.pdf(proposal)
            desired_pdf_value = self._desired_func(proposal)
            u = (
                self._rng.uniform(size=proposal_pdf_value.shape)
                * proposal_pdf_value
                * self._scale
            )
            accept = np.where(u < desired_pdf_value)[0]

            n_accept = len(accept)
            samples[offset : offset + n_accept, :] = proposal[accept, :]

            offset += n_accept

            # accounting
            self._n_proposed += n_propose
            self._n_accepted += n_accept

        if isinstance(size, int):
            final_shape = [size, self.dim]
        else:
            final_shape = list(size) + [self.dim]
        return samples.reshape(final_shape)


class DirectionalSampler(RejectionSampler):
    def __init__(self, loc=None):

        if loc is None:
            self._dim = 3
            self._loc = np.zeros(self._dim, dtype=np.float)
            self._loc[0] = 1.0
        else:
            self._loc = np.array(loc, dtype=np.float)
            assert self._loc.ndim == 1
            self._dim = len(self._loc)

        self._loc /= np.linalg.norm(self._loc)

        # proposal distribution is the unnormalized uniform spherical distribution
        unnormalized_uniform = distributions.AdHoc(
            lambda x: np.ones_like(x.shape[-1]), uniform_spherical, self._dim
        )
        super().__init__(self._pattern, unnormalized_uniform)

    @abc.abstractmethod
    def _pattern(self, x):
        pass


class CardioidFamilySampler(DirectionalSampler):
    """
    This object draws sample from a cardioid shaped distribution on the sphere

    Parameters
    ----------
    loc: array_like
        The unit vector pointing in the main direction of the cardioid
    coeff: float
        The shape coefficient (default 0.5 for a regular cardioid)
    """

    def __init__(self, loc=None, coeff=0.5):
        super().__init__(loc=loc)
        self._coeff = coeff

    def _pattern(self, x):
        return np.abs(self._coeff + (1 - self._coeff) * np.matmul(x, self._loc))
