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
A class of directional responses can be defined analytically.
Such respones include in particular the cardioid family of patterns
that describes cardioid, super-cardioid, and figure-of-eight microphones, (see
`cardioid family <https://en.wikipedia.org/wiki/Microphone>`_, under Polar patterns, with cardioid, hypercardioid, cardioid, subcardioid, figure-eight, and omnidirectional).
In three dimensions, for an orientation given by unit vector :math:`\boldsymbol{u}`, a parameter :math:`p \in [0, 1]`,
and a gain :math:`G`, the response to direction :math:`\boldsymbol{r}` (also a unit vector) is given by the following equation.

.. math::
    f(\boldsymbol{r}\,;\,\boldsymbol{d}, p, G) = G (p + (1 - p) \boldsymbol{d}^\top \boldsymbol{r}),

Note that :math:`\boldsymbol{d}^\top \boldsymbol{r}` is the inner product of two unit
vectors, that is, the cosine of the angle between them.

Different values of :math:`p` correspond to different patterns: 0 for
figure-eight, 0.25 for hyper-cardioid, 0.5 for cardioid, 0.75 for
sub-cardioid, and 1 for omni.

Specialized objects
:py:class:`~pyroomacoustics.directivities.analytic.Cardioid`,
:py:class:`~pyroomacoustics.directivities.analytic.FigureEight`,
:py:class:`~pyroomacoustics.directivities.analytic.SubCardioid`,
:py:class:`~pyroomacoustics.directivities.analytic.HyperCardioid`,
and :py:class:`~pyroomacoustics.directivities.analytic.Omnidirectional` are provided
for the different patterns.
The class :py:class:`~pyroomacoustics.directivities.analytic.CardioidFamily` can be used to make
a pattern with arbitrary parameter :math:`p`.

.. code-block:: python

    # a cardioid pointing toward the ``z`` direction
    from pyroomacoustics.directivities import CardioidFamily

    dir = Cardioid([0, 0, 1], gain=1.0)
"""
import numpy as np

from ..doa import spher2cart
from ..utilities import all_combinations, requires_matplotlib
from .base import Directivity
from .direction import DirectionVector

_FIGURE_EIGHT = 0
_HYPERCARDIOID = 0.25
_CARDIOID = 0.5
_SUBCARDIOID = 0.75
_OMNI = 1.0


class CardioidFamily(Directivity):
    r"""
    Object for directivities coming from the
    `cardioid family`_.
    In three dimensions, for an orientation given by unit vector :math:`\\boldsymbol{u}`, a parameter :math:`p \in [0, 1]`,
    and a gain :math:`G`, the pattern is given by the following equation.

    .. math::
        f(\boldsymbol{r}\,;\,\boldsymbol{d}, p, G) = G (p + (1 - p) \boldsymbol{d}^\top \boldsymbol{r}),

    Different values of :math:`p` correspond to different patterns: 0 for
    figure-eight, 0.25 for hyper-cardioid, 0.5 for cardioid, 0.75 for
    sub-cardioid, and 1 for omni.

    Note that all the patterns are cylindrically symmetric around the
    orientation vector.

    Parameters
    ----------
    orientation: DirectionVector or numpy.ndarray
        Indicates direction of the pattern.
    p: float
        Parameter of the cardioid pattern. A value of 0 corresponds to a
        figure-eight pattern, 0.5 to a cardioid pattern, and 1 to an omni
        pattern
        The parameter must be between 0 and 1
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, orientation, p, gain=1.0):
        if isinstance(orientation, list) or hasattr(orientation, "__array__"):
            orientation = np.array(orientation)
            # check if valid direction vector, normalize, make object
            if orientation.shape != (3,):
                raise ValueError("Orientation must be a 3D vector.")
            orientation = orientation / np.linalg.norm(orientation)
            azimuth, colatitude = spher2cart(orientation)  # returns radians
            self._orientation = DirectionVector(azimuth, colatitude, degrees=False)
        elif isinstance(orientation, DirectionVector):
            self._orientation = orientation
        else:
            raise ValueError("Orientation must be a DirectionVector or a 3D vector.")

        self._p = p
        if not 0 <= self._p <= 1:
            raise ValueError("The parameter p must be between 0 and 1.")

        self._gain = gain
        self._pattern_name = f"cardioid family, p={self._p}"

    @property
    def is_impulse_response(self):
        # this is not an impulse response, do not make docstring to avoid clutter in the
        # documentation
        return False

    @property
    def filter_len_ir(self):
        # no impulse response means length 1
        return 1

    @property
    def directivity_pattern(self):
        """Name of cardioid directivity pattern."""
        return self._pattern_name

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees, degrees=degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees, degrees=degrees)

    def set_orientation(self, orientation):
        """
        Set orientation of directivity pattern.

        Parameters
        ----------
        orientation : DirectionVector
            New direction for the directivity pattern.
        """
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles.

        Parameters
        ----------
        azimuth : array_like
            Azimuth
        colatitude : array_like, optional
            Colatitude. Default is to be on XY plane.
        magnitude : bool, optional
            Whether to return magnitude of response.
        frequency : float, optional
            For which frequency to compute the response. Cardioid are frequency-independent so this
            value has no effect.
        degrees : bool, optional
            If ``True``, ``azimuth`` and ``colatitude`` are in degrees.
            Otherwise, they are in radians.

        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Response at provided angles.
        """

        if colatitude is not None:
            assert len(azimuth) == len(colatitude)
        if self._p == 1.0:
            return self._gain * np.ones(len(azimuth))
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


class Cardioid(CardioidFamily):
    """
    Cardioid directivity pattern.

    Parameters
    ----------
    orientation: DirectionVector
        Indicates direction of the pattern.
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, orientation, gain=1.0):
        super().__init__(orientation, p=_CARDIOID, gain=gain)
        self._pattern_name = "cardioid"


class FigureEight(CardioidFamily):
    """
    Figure-of-eight directivity pattern.

    Parameters
    ----------
    orientation: DirectionVector
        Indicates direction of the pattern.
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, orientation, gain=1.0):
        super().__init__(orientation, p=_FIGURE_EIGHT, gain=gain)
        self._pattern_name = "figure-eight"


class SubCardioid(CardioidFamily):
    """
    Sub-cardioid directivity pattern.

    Parameters
    ----------
    orientation: DirectionVector
        Indicates direction of the pattern.
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, orientation, gain=1.0):
        super().__init__(orientation, p=_SUBCARDIOID, gain=gain)
        self._pattern_name = "sub-cardioid"


class HyperCardioid(CardioidFamily):
    """
    Hyper-cardioid directivity pattern.

    Parameters
    ----------
    orientation: DirectionVector
        Indicates direction of the pattern.
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, orientation, gain=1.0):
        CardioidFamily.__init__(self, orientation, p=_HYPERCARDIOID, gain=gain)
        self._pattern_name = "hyper-cardioid"


class Omnidirectional(CardioidFamily):
    """
    Hyper-cardioid directivity pattern.

    Parameters
    ----------
    orientation: DirectionVector
        Indicates direction of the pattern.
    gain: float
        The linear gain of the directivity pattern (default is 1.0)
    """

    def __init__(self, gain=1.0):
        CardioidFamily.__init__(self, DirectionVector(0.0, 0.0), p=_OMNI, gain=gain)
        self._pattern_name = "omni"


def cardioid_func(x, direction, p, gain=1.0, normalize=True, magnitude=False):
    """
    One-shot function for computing cardioid response.

    Parameters
    -----------
    x: array_like, shape (n_dim, ...)
         Cartesian coordinates
    direction: array_like, shape (n_dim)
         Direction vector, should be normalized
    p: float
         Parameter for the cardioid function (between 0 and 1)
    gain: float
         The gain
    normalize : bool
        Whether to normalize coordinates and direction vector
    magnitude : bool
        Whether to return magnitude, default is False

    Returns
    -------
    resp : :py:class:`~numpy.ndarray`
        Response at provided angles for the speficied cardioid function.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("The parameter p must be between 0 and 1.")

    # normalize positions
    if normalize:
        x /= np.linalg.norm(x, axis=0)
        direction /= np.linalg.norm(direction)

    # compute response
    resp = gain * (p + (1 - p) * np.matmul(direction, x))
    if magnitude:
        return np.abs(resp)
    else:
        return resp


def cardioid_energy(p, gain=1.0):
    r"""
    This function gives the exact value of the surface integral of the cardioid
    (family) function on the unit sphere

    .. math::

        E(p, G) = \iint_{\mathbb{S}^2} G^2 \left( p + (1 - p) \boldsymbol{d}^\top \boldsymbol{r} \right)^2 d\boldsymbol{r}
        = \frac{4 \pi}{3} G^2 \left( 4 p^2 - 2 p + 1 \right).

    This can be used to normalize the energy sent/received.

    Parameters
    ---------
    p: float
        The parameter of the cardioid function (between 0 and 1)
    gain: float
        The gain of the cardioid function
    """
    return gain**2 * (4.0 * np.pi / 3.0) * (4 * p**2 - 2 * p + 1)
