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
    sv = SphericalVoronoi(points)
    w_ = sv.calculate_areas()

    f = func(points.T)

    return np.sum(w_ * f)
