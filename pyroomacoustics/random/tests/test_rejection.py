import pyroomacoustics as pra
import numpy as np
from pyroomacoustics.directivities import CardioidEnergyDistribution, SphericalHistogram
import pytest


def test_rejection_sampler_power_spherical():

    loc = np.ones(3) / np.sqrt(3.0)
    scale = 0.5
    power_spherical = pra.random.distributions.PowerSpherical(loc=loc, scale=scale)

    rng = np.random.default_rng(94877675)

    # Create a sampler with unnormalized power spherical distribution
    # and unnormalized uniform distribution.
    max_unnormalized = 2**scale
    sampler = pra.random.sampler.RejectionSampler(
        desired_func=lambda x: (1.0 + np.matmul(x, loc)) ** scale,
        proposal_dist=pra.random.distributions.UnnormalizedUniformSpherical(dim=3),
        scale=max_unnormalized,
    )

    hist = SphericalHistogram(n_bins=30)

    random_points = sampler(size=(100000,), rng=rng)
    hist.push(random_points.T)

    values_expected = power_spherical.pdf(hist.grid.cartesian.T)
    values_obtained = hist.histogram

    np.testing.assert_almost_equal(values_expected, values_obtained, decimal=2)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # let's visualize samples in the sphere

    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)
    X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")
    """

    for loc, scale in zip([[1, 1, 1]], [10, 1, 0.1]):
        loc = np.array(loc) / np.linalg.norm(loc)
        print(loc, scale)
        # Figure-of-eight
        cardioid_energy = CardioidEnergyDistribution(loc=loc, p=0.0)
        sampler = CardioidEnergyDistribution(loc=loc, p=0).sample

        # Measured eigenmike response
        eigenmike = pra.MeasuredDirectivityFile("EM32_Directivity", fs=16000)
        rot_54_73 = pra.Rotation3D([73, 54], "yz", degrees=True)
        dir_obj_Emic = eigenmike.get_mic_directivity("EM_32_9", orientation=rot_54_73)
        sampler = dir_obj_Emic.energy_distribution._sampler

        points = sampler(100000).T

        # Create a spherical histogram
        hist = SphericalHistogram(n_bins=500)
        hist.push(points)
        hist.plot()

        r"""
        ax.scatter(X, Y, Z, s=50)
        ax.plot(
            *np.stack((torch.zeros_like(loc), loc)).T,
            linewidth=4,
            label="$\kappa={}$".format(scale)
        )
        """

        print("Sampler's efficiency:", sampler.efficiency)

    """
    ax.view_init(30, 45)
    ax.tick_params(axis="both")
    plt.legend()
    """

    plt.show()
