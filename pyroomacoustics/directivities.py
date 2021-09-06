import abc
import numpy as np
from enum import Enum
from pyroomacoustics.doa import spher2cart
from pyroomacoustics.utilities import requires_matplotlib, all_combinations


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


class DirectionVector(object):
    """
    Object for representing direction vectors in 3D, parameterized by an azimuth and colatitude
    angle.

    Parameters
    ----------
    azimuth : float
    colatitude : float, optional
        Default to PI / 2, only XY plane.
    degrees : bool
        Whether provided values are in degrees (True) or radians (False).
    """

    def __init__(self, azimuth, colatitude=None, degrees=True):
        if degrees is True:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)
        self._azimuth = azimuth
        if colatitude is None:
            colatitude = np.pi / 2
        assert colatitude <= np.pi and colatitude >= 0
        self._colatitude = colatitude

        self._unit_v = np.array(
            [
                np.cos(self._azimuth) * np.sin(self._colatitude),
                np.sin(self._azimuth) * np.sin(self._colatitude),
                np.cos(self._colatitude),
            ]
        )

    def get_azimuth(self, degrees=False):
        if degrees:
            return np.degrees(self._azimuth)
        else:
            return self._azimuth

    def get_colatitude(self, degrees=False):
        if degrees:
            return np.degrees(self._colatitude)
        else:
            return self._colatitude

    @property
    def unit_vector(self):
        """Direction vector in cartesian coordinates."""
        return self._unit_v


class Directivity(abc.ABC):
    """
    Abstract class for directivity patterns.

    """

    def __init__(self, orientation):
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees)

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

    @abc.abstractmethod
    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles and frequency.
        """
        return


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
    """

    def __init__(self, orientation, pattern_enum, gain=1.0):
        Directivity.__init__(self, orientation)
        self._p = pattern_enum.value
        self._gain = gain
        self._pattern_name = pattern_enum.name

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
            resp = self._gain * self._p + (1 - self._p) * np.matmul(
                self._orientation.unit_vector, coord
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
    x: array_like, shape (..., n_dim)
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


def source_angle_shoebox(image_source_loc, wall_flips, mic_loc):
    """
    Determine outgoing angle for each image source for a ShoeBox configuration.

    Implementation of the method described in the paper:
    https://www2.ak.tu-berlin.de/~akgroup/ak_pub/2018/000458.pdf

    Parameters
    -----------
    image_source_loc : array_like
        Locations of image sources.
    wall_flips: array_like
        Number of x, y, z flips for each image source.
    mic_loc: array_like
        Microphone location.

    Returns
    -------
    azimuth : :py:class:`~numpy.ndarray`
        Azimith for each image source, in radians
    colatitude : :py:class:`~numpy.ndarray`
        Colatitude for each image source, in radians.

    """

    image_source_loc = np.array(image_source_loc)
    wall_flips = np.array(wall_flips)
    mic_loc = np.array(mic_loc)

    dim, n_sources = image_source_loc.shape
    assert wall_flips.shape[0] == dim
    assert mic_loc.shape[0] == dim

    p_vector_array = image_source_loc - np.array(mic_loc)[:, np.newaxis]
    d_array = np.linalg.norm(p_vector_array, axis=0)

    # Using (12) from the paper
    power_array = np.ones_like(image_source_loc) * -1
    power_array = np.power(power_array, (wall_flips + np.ones_like(image_source_loc)))
    p_dash_array = p_vector_array * power_array

    # Using (13) from the paper
    azimuth = np.arctan2(p_dash_array[1], p_dash_array[0])
    if dim == 2:
        colatitude = np.ones(n_sources) * np.pi / 2
    else:
        colatitude = np.pi / 2 - np.arcsin(p_dash_array[2] / d_array)

    return azimuth, colatitude
