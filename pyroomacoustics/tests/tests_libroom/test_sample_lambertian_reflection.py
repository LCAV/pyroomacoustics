import numpy as np
import pyroomacoustics as pra
from scipy.spatial import cKDTree, SphericalVoronoi


class SphericalHistogram:
    def __init__(self, n_bins):
        self.grid = pra.doa.fibonacci_spherical_sampling(n_bins).T
        self._kdtree = cKDTree(self.grid)

        # the counter variables for every bin
        self._bins = np.zeros(n_bins, dtype=int)

        # Compute the bin areas.
        sv = SphericalVoronoi(self.grid)
        self.areas = sv.calculate_areas()

    def get_values(self):
        return self._bins / np.sum(self._bins)

    def __call__(self, points):
        _, matches = self._kdtree.query(points)
        bin_indices, counts = np.unique(matches, return_counts=True)
        self._bins[bin_indices] += counts


def plot_spherical_grid(grid, values):
    grid = pra.doa.GridSphere(cartesian_points=grid.T)
    grid.set_values(values)
    grid.plot_old()


def compute_lambertian_distribution(grid, areas, normal):
    normal = normal / np.linalg.norm(normal)
    cos = grid @ normal
    up = cos > 0
    value = np.zeros(grid.shape[0])
    value[up] = areas[up] * cos[up] / np.pi
    return value


def compute_reflections_histogram(wall, n_points=100_000):
    pra.libroom.set_rng_seed(0)
    samples = np.array([wall.sample_lambertian_reflection() for _ in range(n_points)])
    hist = SphericalHistogram(1_000)
    hist(samples)
    return hist


def test_sample_lambertian_reflection():
    corners = np.eye(3)[::-1]
    wall = pra.libroom.Wall(corners)
    hist = compute_reflections_histogram(wall)
    est = hist.get_values()
    ref = compute_lambertian_distribution(hist.grid, hist.areas, wall.normal)
    np.testing.assert_allclose(est, ref, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    corners = np.eye(3)[::-1]
    wall = pra.libroom.Wall(corners)

    # Empirical histogram.
    hist = compute_reflections_histogram(wall)
    est = hist.get_values()

    # Analytical histogram.
    ref = compute_lambertian_distribution(hist.grid, hist.areas, wall.normal)

    error = est - ref
    rms = np.sqrt(np.mean(error**2.0))
    rel_error = np.mean(error**2) / np.mean(ref**2)
    print(rms)
    print(rel_error)
    print(abs(error).max() / abs(ref).mean())

    plot_spherical_grid(hist.grid, ref)
    plot_spherical_grid(hist.grid, est)
    plot_spherical_grid(hist.grid, error)
    plt.show()
