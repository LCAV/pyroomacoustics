# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2022-2024  Prerak Srivastava, Robin Scheibler
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
This module provides functions to interpolate impulse responses on a sphere.
The interpolation is done in the spherical harmonics domain.
"""
import numpy as np
import scipy
from scipy.spatial import SphericalVoronoi

from ..doa import detect_regular_grid


def cal_sph_basis(azimuth, colatitude, degree):  # theta_target,phi_target
    """
    Calculate a spherical basis matrix

    Parameters
    -----------
    azimuth: array_like
       Azimuth of the spherical coordinates of the grid points
    phi: array_like
       Colatitude spherical coordinates of the grid points
    degree:(int)
       spherical harmonic degree

    Return
    ------
    Ysh (np.array) shape: (no_of_nodes, (degree + 1)**2)
        Spherical harmonics basis matrix

    """
    # build linear array of indices
    #        0
    #     -1 0 1
    #  -2 -1 0 1 2
    # ...
    ms = []
    ns = []
    for i in range(degree + 1):
        for order in range(-i, i + 1):
            ms.append(order)
            ns.append(i)
    m, n = np.array([ms]), np.array([ns])

    # compute all the spherical harmonics at once
    Ysh = scipy.special.sph_harm(m, n, azimuth[:, None], colatitude[:, None])

    return Ysh


def _weighted_pinv(weights, Y, rcond=1e-2):
    return np.linalg.pinv(
        weights[:, None] * Y, rcond=rcond
    )  # rcond is inverse of the condition number


def calculation_pinv_voronoi_cells(Ysh, colatitude, colatitude_grid, len_azimuth_grid):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for regular sampling in spherical coordinates

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    colatitude: (np.ndarray)
        The colatitudes of the measurements
    colatitude_grid: (int)
        the colatitudes of the grid lines
    len_azimuth_grid:
        The number of distinct azimuth values in the grid

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted psuedo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid
    """
    # compute the areas of the voronoi regions of the grid analytically
    # assuming that the grid is regular in azimuth/colatitude
    res = (colatitude_grid[:-1] + colatitude_grid[1:]) / 2
    res = np.insert(res, len(res), np.pi)
    res = np.insert(res, 0, 0)
    w = -np.diff(np.cos(res)) * (2 * np.pi / len_azimuth_grid)
    w_dict = {t: ww for t, ww in zip(colatitude_grid, w)}

    # select the weights
    w_ = np.array([w_dict[col] for col in colatitude])
    w_ /= 4 * np.pi  # normalizing by unit sphere area

    return _weighted_pinv(w_, Ysh), w_


def calculation_pinv_voronoi_cells_general(Ysh, points):
    """
    Weighted least square solution "Analysis and Synthesis of Sound-Radiation with
    Spherical Arrays: Franz Zotter Page 76"

    Calculation of pseudo inverse and voronoi cells for arbitrary sampling of the sphere.

    Parameters
    -----------
    Ysh: (np.ndarray)
        Spherical harmonic basis matrix
    points: numpy.ndarray, (n_points, 3)
        The sampling points on the sphere

    Returns:
    -------------------------------
    Ysh_tilda_inv : (np.ndarray)
        Weighted pseudo inverse of spherical harmonic basis matrix Ysh
    w_ : (np.ndarray)
        Weight on the original grid
    """

    # The weights are the areas of the voronoi cells
    sv = SphericalVoronoi(points)
    w_ = sv.calculate_areas()
    w_ /= 4 * np.pi  # normalizing by unit sphere area

    Ysh_tilda_inv = np.linalg.pinv(
        w_[:, None] * Ysh, rcond=1e-2
    )  # rcond is inverse of the condition number

    return _weighted_pinv(w_, Ysh), w_


def spherical_interpolation(
    grid,
    impulse_responses,
    new_grid,
    spherical_harmonics_order=12,
    axis=-2,
    nfft=None,
):
    """
    Parameters
    ----------
    grid: pyroomacoustics.doa.GridSphere
        The grid of the measurements
    impulse_responses: numpy.ndarray, (..., n_measurements, ..., n_samples)
        The impulse responses to interpolate, the last axis is time and one other
        axis should have dimension matching the length of the grid. By default,
        it is assumed to be second from the end, but can be specified with the
        `axis` argument.
    new_grid: pyroomacoustics.doa.GridSphere
        Grid of points at which to interpolate
    spherical_harmonics_order: int
        The order of spherical harmonics to use for interpolation
    axis: int
        The axis of the grid in the impulse responses array
    nfft: int
        The length of the FFT to use for the interpolation (default ``n_samples``)
    """
    ir = np.swapaxes(impulse_responses, axis, -2)
    if nfft is None:
        nfft = ir.shape[-1]

    if len(grid) != ir.shape[-2]:
        raise ValueError(
            "The length of the grid should be the same as the number of impulse"
            f"responses provide (grid={len(grid)}, impulse response={ir.shape[-2]})"
        )

    # Calculate spherical basis for the original grid
    Ysh = cal_sph_basis(grid.azimuth, grid.colatitude, spherical_harmonics_order)

    # Calculate spherical basis for the target grid (fibonacci grid)
    Ysh_fibo = cal_sph_basis(
        new_grid.azimuth,
        new_grid.colatitude,
        spherical_harmonics_order,
    )

    # this will check if the points are on a regular grid.
    # If they are, then the azimuths and colatitudes of the grid
    # are returned
    regular_grid = detect_regular_grid(grid.azimuth, grid.colatitude)

    # calculate pinv and voronoi cells for least square solution for the whole grid
    if regular_grid is not None:
        Ysh_tilda_inv, w_ = calculation_pinv_voronoi_cells(
            Ysh,
            grid.colatitude,
            regular_grid.colatitude,
            len(regular_grid.azimuth),
        )
    else:
        Ysh_tilda_inv, w_ = calculation_pinv_voronoi_cells_general(
            Ysh, grid.cartesian.T
        )

    # Do the interpolation in the frequency domain

    # shape: (..., n_measurements, n_samples // 2 + 1)
    tf = np.fft.rfft(ir, axis=-1, n=nfft)

    g_tilda = w_[:, None] * tf

    gamma_full_scale = np.matmul(Ysh_tilda_inv, g_tilda)

    interpolated_original_grid = np.fft.irfft(
        np.matmul(Ysh, gamma_full_scale), n=nfft, axis=-1
    )
    interpolated_target_grid = np.fft.irfft(
        np.matmul(Ysh_fibo, gamma_full_scale), n=nfft, axis=-1
    )

    # restore the order of the axes
    interpolated_target_grid = np.swapaxes(interpolated_target_grid, -2, axis)
    interpolated_original_grid = np.swapaxes(interpolated_original_grid, -2, axis)

    return interpolated_target_grid, interpolated_original_grid
