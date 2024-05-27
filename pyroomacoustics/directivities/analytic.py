# Directivity module that provides routines to use analytic and mesured directional
# responses for sources and microphones.
# Copyright (C) 2020-2024  Robin Scheibler, Satvik Dixit, Eric Bezzam
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
## Analytic Directional Responses

A class of directional responses can be defined analytically.
Such respones include in particular the cardioid family of patterns
that describes cardioid, super-cardioid, and figure-of-eight microphones.

### Cardioid Family

Cardioid directional patterns are defined by a unit-length direction vector and a parameter :math:`p`

.. math::

    r = a (1 + p \cos \theta)

where :math:`a` is the gain and :math:`\theta` is the angle between the direction vector and the source.
"""
from enum import Enum

import numpy as np

from ..doa import spher2cart
from ..utilities import all_combinations, requires_matplotlib
from .base import Directivity
from .direction import DirectionVector


class DirectivityPattern(Enum):
    """
    Common cardioid patterns and their corresponding coefficient for the expression:

    .. math::

        r = a (1 + \\cos \\theta),

    where :math:`a` is the coefficient that determines the cardioid pattern and :math:`r` yields
    the gain at angle :math:`\\theta`.

    """

    FIGURE_EIGHT = 0
    HYPERCARDIOID = 0.25
    CARDIOID = 0.5
    SUBCARDIOID = 0.75
    OMNI = 1.0


class CardioidFamily(Directivity):
    """
    Object for directivities coming from the
    `cardioid family <https://en.wikipedia.org/wiki/Microphone#Cardioid,_hypercardioid,_supercardioid,_subcardioid>`_.

    Parameters
    ----------
    orientation : DirectionVector
        Indicates direction of the pattern.
    pattern_enum : DirectivityPattern
        Desired pattern for the cardioid.
    gain: float
        The linear gain of the directivity pattern.
    """

    def __init__(self, orientation, pattern_enum, gain=1.0):
        Directivity.__init__(self, orientation)
        self._p = pattern_enum.value
        self._gain = gain
        self._pattern_name = pattern_enum.name
        self.filter_len_ir = 1

    @property
    def directivity_pattern(self):
        """Name of cardioid directivity pattern."""
        return self._pattern_name

    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles.

        Parameters
        ----------
        azimuth : array_like
            Azimuth in degrees
        colatitude : array_like, optional
            Colatitude in degrees. Default is to be on XY plane.
        magnitude : bool, optional
            Whether to return magnitude of response.
        frequency : float, optional
            For which frequency to compute the response. Cardioid are frequency-independent so this
            value has no effect.
        degrees : bool, optional
            Whether provided angles are in degrees.


        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Response at provided angles.
        """

        if colatitude is not None:
            assert len(azimuth) == len(colatitude)
        if self._p == DirectivityPattern.OMNI:
            return np.ones(len(azimuth))
        else:
            coord = spher2cart(azimuth=azimuth, colatitude=colatitude, degrees=degrees)

            resp = self._gain * (
                self._p
                + (1 - self._p) * np.matmul(self._orientation.unit_vector, coord)
            )

            if magnitude:
                return np.abs(resp)
            else:
                return resp

    @requires_matplotlib
    def plot_response(
        self, azimuth, colatitude=None, degrees=True, ax=None, offset=None
    ):
        """
        Plot directivity response at specified angles.

        Parameters
        ----------
        azimuth : array_like
            Azimuth values for plotting.
        colatitude : array_like, optional
            Colatitude values for plotting. If not provided, 2D plot.
        degrees : bool
            Whether provided values are in degrees (True) or radians (False).
        ax : axes object
        offset : list
            3-D coordinates of the point where the response needs to be plotted.

        Return
        ------
        ax : :py:class:`~matplotlib.axes.Axes`
        """
        import matplotlib.pyplot as plt

        if offset is not None:
            x_offset = offset[0]
            y_offset = offset[1]
        else:
            x_offset = 0
            y_offset = 0

        if degrees:
            azimuth = np.radians(azimuth)

        if colatitude is not None:
            if degrees:
                colatitude = np.radians(colatitude)

            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection="3d")

            if offset is not None:
                z_offset = offset[2]
            else:
                z_offset = 0

            spher_coord = all_combinations(azimuth, colatitude)
            azi_flat = spher_coord[:, 0]
            col_flat = spher_coord[:, 1]

            # compute response
            resp = self.get_response(
                azimuth=azi_flat, colatitude=col_flat, magnitude=True, degrees=False
            )
            RESP = resp.reshape(len(azimuth), len(colatitude))

            # create surface plot, need cartesian coordinates
            AZI, COL = np.meshgrid(azimuth, colatitude)
            X = RESP.T * np.sin(COL) * np.cos(AZI) + x_offset
            Y = RESP.T * np.sin(COL) * np.sin(AZI) + y_offset
            Z = RESP.T * np.cos(COL) + z_offset

            ax.plot_surface(X, Y, Z)

            if ax is None:
                ax.set_title(
                    "{}, azimuth={}, colatitude={}".format(
                        self.directivity_pattern,
                        self.get_azimuth(),
                        self.get_colatitude(),
                    )
                )
            else:
                ax.set_title("Directivity Plot")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        else:
            if ax is None:
                fig = plt.figure()
                ax = plt.subplot(111)

            # compute response
            resp = self.get_response(azimuth=azimuth, magnitude=True, degrees=False)
            RESP = resp

            # create surface plot, need cartesian coordinates
            X = RESP.T * np.cos(azimuth) + x_offset
            Y = RESP.T * np.sin(azimuth) + y_offset
            ax.plot(X, Y)

        return ax


def cardioid_func(x, direction, coef, gain=1.0, normalize=True, magnitude=False):
    """
    One-shot function for computing cardioid response.

    Parameters
    -----------
    x: array_like, shape (n_dim, ...)
         Cartesian coordinates
    direction: array_like, shape (n_dim)
         Direction vector, should be normalized.
    coef: float
         Parameter for the cardioid function.
    gain: float
         The gain.
    normalize : bool
        Whether to normalize coordinates and direction vector.
    magnitude : bool
        Whether to return magnitude, default is False.

    Returns
    -------
    resp : :py:class:`~numpy.ndarray`
        Response at provided angles for the speficied cardioid function.
    """
    assert coef >= 0.0
    assert coef <= 1.0

    # normalize positions
    if normalize:
        x /= np.linalg.norm(x, axis=0)
        direction /= np.linalg.norm(direction)

    # compute response
    resp = gain * (coef + (1 - coef) * np.matmul(direction, x))
    if magnitude:
        return np.abs(resp)
    else:
        return resp


def cardioid_energy(coef, gain=1.0):
    return gain**2 * (4.0 * np.pi / 3.0) * (4 * coef**2 - 2 * coef + 1)
