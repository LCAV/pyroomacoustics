import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

import pyroomacoustics as pra


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
        return self._bins / (self.areas * np.sum(self._bins))

    def __call__(self, points):
        _, matches = self._kdtree.query(points)
        bin_indices, counts = np.unique(matches, return_counts=True)
        self._bins[bin_indices] += counts


def plot_spherical_grid(grid, values):
    grid = pra.doa.GridSphere(cartesian_points=grid.T)
    grid.set_values(values)
    grid.plot_old()


def compute_lambertian_distribution(grid, normal):
    normal = normal / np.linalg.norm(normal)
    cos = grid @ normal
    up = cos > 0
    value = np.zeros(grid.shape[0])
    value[up] = cos[up] / np.pi
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
    hist = compute_reflections_histogram(wall, n_points=1_000_000)
    est = hist.get_values()
    ref = compute_lambertian_distribution(hist.grid, wall.normal)

    # Only check where the signal is large enough.
    mask = ref >= 0.1 * ref.max()
    print(abs(est[mask] - ref[mask]).max())
    np.testing.assert_allclose(est[mask], ref[mask], rtol=0.1, atol=1e-4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    corners = np.eye(3)
    corners[2, 2] = 0.0
    wall = pra.libroom.Wall(corners)

    # Empirical histogram.
    hist = compute_reflections_histogram(wall, n_points=1_000_000)
    est = hist.get_values()

    # Analytical histogram.
    ref = compute_lambertian_distribution(hist.grid, wall.normal)

    error = est - ref
    rms = np.sqrt(np.mean(error**2.0))
    rel_error = np.mean(error**2) / np.mean(ref**2)
    print(rms)
    print(rel_error)
    print(abs(error).max())
    print(np.mean(ref), ref.max(), ref[ref > 0.0].min())
    print(np.std(error))

    plot_spherical_grid(hist.grid, ref)
    plot_spherical_grid(hist.grid, est)
    plot_spherical_grid(hist.grid, error)

    # Now observe a slice.
    beta = np.linspace(0.0, np.pi, 32)
    p = np.array([np.cos(beta), np.zeros_like(beta), np.sin(beta)]).T
    ref_flat = compute_lambertian_distribution(p, wall.normal)

    _, matches = hist._kdtree.query(p)
    est_flat = est[matches]

    fig, ax = plt.subplots(1, 1)
    ax.plot(beta, ref_flat, label="analytical")
    ax.plot(beta, est_flat, label="empirical")
    ax.set_title("Slice")
    ax.legend()
    plt.show()
