"""
Tests the distribution objects for the pdf area and the sampler functions.
"""

import pytest
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities.integration import spherical_integral


def test_uniform_spherical_pdf():

    uniform_spherical = pra.random.distributions.UniformSpherical(dim=3)

    def pdf(x):
        return uniform_spherical.pdf(x.T)

    area = spherical_integral(pdf, 1000)

    assert abs(area - 1.0) < 1e-7


@pytest.mark.parametrize("rng", (None, np.random.default_rng(0)))
@pytest.mark.parametrize("dim", (2, 3, 4))
@pytest.mark.parametrize("size", (None, 10, (10,), (10, 3)))
def test_uniform_spherical_sampler(rng, dim, size):

    if size is None:
        shape_expected = (dim,)
    elif isinstance(size, int):
        shape_expected = (size, dim)
    else:
        shape_expected = size + (dim,)

    uniform_spherical = pra.random.distributions.UniformSpherical(dim=dim)

    samples = uniform_spherical.sample(size=size, rng=rng)

    assert samples.shape == shape_expected


def test_power_spherical_pdf():

    loc = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    scale = 0.5

    power_spherical = pra.random.distributions.PowerSpherical(loc=loc, scale=scale)

    def pdf(x):
        return power_spherical.pdf(x.T)

    area = spherical_integral(pdf, 1000)

    assert abs(area - 1.0) < 1e-6


@pytest.mark.parametrize("rng", (None, np.random.default_rng(0)))
@pytest.mark.parametrize("dim", (2, 3, 4))
@pytest.mark.parametrize("size", (None, 10, (10,), (10, 3)))
def test_power_spherical_sampler(rng, dim, size):

    if size is None:
        shape_expected = (dim,)
    elif isinstance(size, int):
        shape_expected = (size, dim)
    else:
        shape_expected = size + (dim,)

    loc = np.ones(shape=dim) / np.sqrt(dim)
    scale = 0.5

    power_spherical = pra.random.distributions.PowerSpherical(loc=loc, scale=scale)

    samples = power_spherical.sample(size=size, rng=rng)

    assert samples.shape == shape_expected
