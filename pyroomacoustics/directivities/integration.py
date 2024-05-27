import numpy as np
from scipy.spatial import SphericalVoronoi

from pyroomacoustics.doa import fibonacci_spherical_sampling


def spherical_integral(func, n_points):
    """
    Numerically integrate a function over the sphere.

    Parameters
    -----------
    func: (callable)
        The function to integrate. It should take an array of shape (3, n_points)
        and return an array of shape (n_points,)
    n_points: int
        The number of points to use for integration

    Returns:
    -------------------------------
    value: (np.ndarray)
        The value of the integral
    """

    points = fibonacci_spherical_sampling(n_points).T  # shape (n_points, 3)

    # The weights are the areas of the voronoi cells
    sv = SphericalVoronoi(points)
    w_ = sv.calculate_areas()

    f = func(points.T)

    return np.sum(w_ * f)
