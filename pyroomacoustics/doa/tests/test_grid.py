import numpy as np
import pytest
from scipy.spatial import SphericalVoronoi

import pyroomacoustics as pra


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
    points = pra.doa.fibonacci_spherical_sampling(n_points=n)
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


def _check_grid_consistency(grid):

    cart = pra.doa.spher2cart(grid.azimuth, grid.colatitude)
    assert np.allclose(grid.cartesian, np.array([grid.x, grid.y, grid.z]))
    assert np.allclose(grid.cartesian, cart)

    az, co, _ = pra.doa.cart2spher(grid.cartesian)
    assert np.allclose(grid.spherical, np.array([grid.azimuth, grid.colatitude]))
    assert np.allclose(grid.spherical, np.array([az, co]))


@pytest.mark.parametrize("n_points", [20, 100, 200, 500, 1000])
def test_grid_sphere_from_spherical(n_points):

    x, y, z = pra.doa.fibonacci_spherical_sampling(n_points)
    az, co, _ = pra.doa.cart2spher(np.array([x, y, z]))

    grid = pra.doa.GridSphere(spherical_points=np.array([az, co]))
    _check_grid_consistency(grid)


@pytest.mark.parametrize("n_points", [20, 100, 200, 500, 1000])
def test_grid_sphere_from_cartesian(n_points):

    x, y, z = pra.doa.fibonacci_spherical_sampling(n_points)

    grid = pra.doa.GridSphere(cartesian_points=np.array([x, y, z]))
    _check_grid_consistency(grid)


@pytest.mark.parametrize("n_points", [20, 100, 200, 500, 1000])
def test_grid_sphere_from_fibonacci(n_points):

    grid = pra.doa.GridSphere(n_points=n_points)
    _check_grid_consistency(grid)


if __name__ == "__main__":
    test_voronoi_area(20, 0.01)
    test_voronoi_area(100, 0.01)
    test_voronoi_area(200, 0.01)
    test_voronoi_area(500, 0.01)
    test_voronoi_area(1000, 0.001)
    test_voronoi_area(2000, 0.001)
    test_voronoi_area(5000, 0.001)
    test_voronoi_area(10000, 0.001)
