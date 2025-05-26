import numpy as np
import pyroomacoustics as pra
import pytest


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
@pytest.mark.parametrize("gain", (0.5, 1.0))
def test_pdf_integral(p, gain):

    loc = np.ones(3) / np.sqrt(3.0)

    distribution = pra.directivities.CardioidEnergyDistribution(loc=loc, p=p, gain=gain)

    def pdf(x):
        return distribution.pdf(x.T)

    area = pra.directivities.spherical_integral(pdf, 1000)

    assert abs(area - 1.0) < 1e-5


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
@pytest.mark.parametrize("gain", (0.5, 1.0))
def test_rejection_sampler_cardioid(p, gain):

    loc = np.ones(3) / np.sqrt(3.0)

    cardioid_energy_distribution = pra.directivities.CardioidEnergyDistribution(
        loc=loc, p=p, gain=gain
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


@pytest.mark.parametrize("p", (0.0, 0.25, 0.5, 0.75, 1.0))
@pytest.mark.parametrize("gain", (1.0, 0.5))
@pytest.mark.parametrize(
    "rng,rtol", [(None, 0.02), (np.random.default_rng(94877675), 0.08)]
)
def test_sample_rays(p, gain, rng, rtol):
    loc = np.ones(3) / np.sqrt(3.0)

    dir_obj = pra.directivities.CardioidFamily(loc, p, gain=gain)

    rays, energies = dir_obj.sample_rays(100_000, rng=rng)

    measured_total_energy = np.mean(energies)

    # Expected quantities.
    expected_total_energy = dir_obj.energy_distribution.total_energy

    np.testing.assert_allclose(expected_total_energy, measured_total_energy, atol=0.01)
