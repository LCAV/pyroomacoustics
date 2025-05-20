import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_pdf_integral(p):

    loc = np.ones(3) / np.sqrt(3.0)

    distribution = pra.directivities.CardioidEnergyDistribution(loc=loc, p=p)

    def pdf(x):
        return distribution.pdf(x.T)

    area = pra.directivities.spherical_integral(pdf, 1000)

    assert abs(area - 1.0) < 1e-5


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_rejection_sampler_cardioid(p):

    loc = np.ones(3) / np.sqrt(3.0)

    cardioid_energy_distribution = pra.directivities.CardioidEnergyDistribution(
        loc=loc, p=p
    )

    rng = np.random.default_rng(94877675)

    hist = pra.directivities.SphericalHistogram(n_bins=36)

    random_points = cardioid_energy_distribution.sample(size=(100000,), rng=rng)
    hist.push(random_points.T)

    values_expected = cardioid_energy_distribution.pdf(hist.grid.cartesian.T)
    values_obtained = hist.histogram

    idx = np.argsort(values_expected)[::-1]
    cdf = np.cumsum(values_expected[idx] * hist._areas[idx])
    select = idx[cdf <= 1.0]

    np.testing.assert_allclose(
        values_expected[select], values_obtained[select], atol=0.02, rtol=0.1
    )


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_rejection_sampler_cardioid_repeatability(p):

    loc = np.ones(3) / np.sqrt(3.0)

    cardioid_energy_distribution = pra.directivities.CardioidEnergyDistribution(
        loc=loc, p=p
    )

    rng = np.random.default_rng(94877675)
    random_points1 = cardioid_energy_distribution.sample(size=(100000,), rng=rng)

    rng = np.random.default_rng(94877675)
    random_points2 = cardioid_energy_distribution.sample(size=(100000,), rng=rng)

    np.testing.assert_allclose(random_points1, random_points2)
