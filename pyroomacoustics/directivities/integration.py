# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2024  Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.
r"""
Provides a function to numerically integrate a function over the 3D unit-sphere (:math:`\mathbb{S}^2`).

.. math::

    \iint_{\mathbb{S}^2} f(\mathbf{x})\, d\mathbf{x}

"""
import numpy as np
from scipy.spatial import SphericalVoronoi

from pyroomacoustics.doa import fibonacci_spherical_sampling


def robust_spherical_voronoi_areas(data):
    """
    Computes areas of the spherical voronoi diagram even when the data points
    span less than 3 dimensions.
    The input data has shape (n_points, 3) and the output (n_points,).
    """
    if data.shape[1] != 3:
        raise ValueError(
            f"Only measurement points in 3D are handled (got {data.shape[1]})."
        )

    if data.shape[0] == 1:
        # The whole energy is on a single point.
        return np.array([4.0 * np.pi])

    elif data.shape[0] == 2:
        # Each point gets half the area of the sphere.
        return np.ones(2) * 2.0 * np.pi

    elif data.shape[0] == 3:
        # Make sure the lengths are normalized.
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        # The angle between all the points.
        angles = np.arccos(np.clip(data @ data.T, -1.0, 1.0))
        # The angle with itself is zero and a bunch of constants simplify so
        # that the ratio of the sphere area is just the sum.
        return np.sum(abs(angles), axis=1)

    else:
        try:
            # This is the normal case we expect to hit most often.
            sv = SphericalVoronoi(data)
            return sv.calculate_areas()
        except ValueError:
            # Make sure the lengths are normalized.
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

            # Find a basis and project in the 2D space.
            u, s, v = np.linalg.svd(data)

            # Check the second singular value to make sure the data is not 1D.
            # If the rank is 1 and data.shape[0] != 1, there are necessarily
            # duplicate points.
            if s[1] < 1e-6:
                raise ValueError(
                    "There are duplicate points in the measurement locations."
                )

            # Work in the basis of the 2D subspace.
            data = data @ v[:2, :].T

            # Find all the angles of the vectors in the 2D plane.
            angles = np.arctan2(data[:, 1], data[:, 0])

            # We need to sort the angles in increasing order and keep track of
            # the reverse permutation.
            sorted_indices = np.argsort(angles)
            reverse = np.empty_like(sorted_indices)
            reverse[sorted_indices] = np.arange(len(sorted_indices))

            # We use inner products to get the distance between adjacent points.
            sorted_data = data[sorted_indices, :]
            inner_product = np.sum(
                sorted_data * np.roll(sorted_data, 1, axis=0), axis=1
            )
            delta = np.arccos(np.clip(inner_product, -1.0, 1.0))

            if np.any(delta < 1e-6):
                # We found some duplicates.
                raise ValueError(
                    "There are duplicate points in the measurement locations."
                )

            areas = 0.5 * (delta + np.roll(delta, -1))
            # Now we need to divide by 2 * pi to get the ratio for each point
            # and then multiply by 4 * pi to get the volume of the sphere.
            # Thus, we only need to multply by 2.
            return 2.0 * areas[reverse]


def spherical_integral(func, n_points):
    """
    Numerically integrate a function over the sphere.

    Parameters
    -----------
    func: callable
        The function to integrate. It should take an array of shape (3, n_points)
        and return an array of shape (n_points,)
    n_points: int
        The number of points to use for integration

    Returns
    -------
    value: np.ndarray
        The value of the integral
    """

    points = fibonacci_spherical_sampling(n_points).T  # shape (n_points, 3)

    # The weights are the areas of the voronoi cells
    w_ = robust_spherical_voronoi_areas(points)

    f = func(points.T)

    return np.sum(w_ * f)
