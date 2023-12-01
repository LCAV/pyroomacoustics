import numpy as np
import pytest
from scipy.spatial import SphericalVoronoi

from pyroomacoustics.doa import fibonacci_spherical_sampling


@pytest.mark.parametrize("n", [20, 100, 200, 500, 1000, 2000, 5000, 10000])
@pytest.mark.parametrize("tol", [0.06])
def test_voronoi_area(n, tol):
    """
    check the implementation of Fibonacci spherical sampling

    The idea is that the area of the voronoi regions generated
    by the sampling should be very close to each other, i.e.,
    for ``n_points`` they should be approx. ``4.0 * np.pi / n_points``.

    We observed empirically that the relative max error is
    around 6% so we set that as the threshold for the test
    """
    points = fibonacci_spherical_sampling(n_points=n)
    sphere_area = 4.0 * np.pi
    area_one_pt = sphere_area / n

    sv = SphericalVoronoi(points=points.T)
    areas_voronoi = sv.calculate_areas()

    err = abs(areas_voronoi - area_one_pt) / area_one_pt
    max_err = err.max()
    avg_err = err.mean()
    min_err = err.min()

    print(f"{n=} {max_err=} {avg_err=} {min_err=}")
    assert max_err < tol


if __name__ == "__main__":
    test_voronoi_area(20, 0.01)
    test_voronoi_area(100, 0.01)
    test_voronoi_area(200, 0.01)
    test_voronoi_area(500, 0.01)
    test_voronoi_area(1000, 0.001)
    test_voronoi_area(2000, 0.001)
    test_voronoi_area(5000, 0.001)
    test_voronoi_area(10000, 0.001)
