import numpy as np
import pyroomacoustics as pra


def test_pdf_integral():

    # Measured eigenmike response
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity", fs=16000)
    rot_54_73 = pra.Rotation3D([73, 54], "yz", degrees=True)
    dir_obj_eigenmic = eigenmike.get_mic_directivity("EM_32_9", orientation=rot_54_73)

    kdtree = dir_obj_eigenmic._kdtree
    ir_energy = np.square(dir_obj_eigenmic._irs).sum(axis=-1)
    distribution = pra.directivities.MeasuredDirectivityEnergyDistribution(
        kdtree, ir_energy
    )

    def pdf(x):
        return distribution.pdf(x.T)

    area = pra.directivities.spherical_integral(pdf, 1000)

    assert abs(area - 1.0) < 1e-3


def test_rejection_sampler_measured():

    # Measured eigenmike response
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity", fs=16000)
    rot_54_73 = pra.Rotation3D([73, 54], "yz", degrees=True)
    dir_obj_eigenmic = eigenmike.get_mic_directivity("EM_32_9", orientation=rot_54_73)

    rng = np.random.default_rng(94877675)

    hist = pra.directivities.SphericalHistogram(n_bins=30)

    random_points = dir_obj_eigenmic.sample_rays(100000, rng=rng)
    hist.push(random_points)

    values_expected = dir_obj_eigenmic.energy_distribution.pdf(hist.grid.cartesian.T)
    values_obtained = hist.histogram

    idx = np.argsort(values_expected)[::-1]
    cdf = np.cumsum(values_expected[idx] * hist._areas[idx])
    select = idx[cdf <= 1.0]

    np.testing.assert_allclose(
        values_expected[select], values_obtained[select], atol=0.02, rtol=0.1
    )


def test_rejection_sampler_measured_repeatability():

    # Measured eigenmike response
    eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity", fs=16000)
    rot_54_73 = pra.Rotation3D([73, 54], "yz", degrees=True)
    dir_obj_eigenmic = eigenmike.get_mic_directivity("EM_32_9", orientation=rot_54_73)

    kdtree = dir_obj_eigenmic._kdtree
    ir_energy = np.square(dir_obj_eigenmic._irs).sum(axis=-1)
    distribution = pra.directivities.MeasuredDirectivityEnergyDistribution(
        kdtree, ir_energy
    )

    rng = np.random.default_rng(94877675)
    random_points1 = distribution.sample(size=(100000,), rng=rng)

    rng = np.random.default_rng(94877675)
    random_points2 = distribution.sample(size=(100000,), rng=rng)

    np.testing.assert_allclose(random_points1, random_points2)
