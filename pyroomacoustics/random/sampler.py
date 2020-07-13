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

            proposal = self._proposal_dist.sample(size=flat_size - offset)
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

        if isinstance(size, int):
            final_shape = [size, self.dim]
        else:
            final_shape = list(size) + [self.dim]
        return samples.reshape(final_shape)


class CardioidSampler(RejectionSampler):
    def __init__(self, loc=None):

        if loc is None:
            self._dim = 3
            self._loc = np.zeros(self._dim)
            self._loc[0] = 1.0
        else:
            self._loc = np.array(loc)
            assert self._loc.ndim == 1
            self._dim = len(self._loc)

        # proposal distribution is the unnormalized uniform spherical distribution
        unnormalized_uniform = distributions.AdHoc(
            lambda x: np.ones_like(x.shape[-1]), uniform_spherical, self._dim
        )
        super().__init__(self._pattern, unnormalized_uniform)

    def _pattern(self, x):
        return 0.5 + 0.5 * np.matmul(x, self._loc)
