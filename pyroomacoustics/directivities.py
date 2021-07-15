import abc
import numpy as np
from enum import Enum

from pyroomacoustics.doa import spher2cart
from pyroomacoustics.utilities import requires_matplotlib, all_combinations


class DirectivityPattern(Enum):
    """ Common Cardioid patterns and their corresponding coefficient. """
    FIGURE_EIGHT = 0
    HYPERCARDIOID = 0.25
    CARDIOID = 0.5
    SUBCARDIOID = 0.75
    OMNI = 1.0


class DirectionVector(object):
    """
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

        self._unit_v = np.array([
            np.cos(self._azimuth) * np.sin(self._colatitude),
            np.sin(self._azimuth) * np.sin(self._colatitude),
            np.cos(self._colatitude),
        ])

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
        return self._unit_v


class Directivity(abc.ABC):
    def __init__(self, orientation):
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees)
    
    @abc.abstractmethod
    def get_response(self, coord, magnitude=False, frequency=None):
        return

class CardioidFamily(Directivity):
    """

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
        return self._pattern_name

    def get_response(self, coord, magnitude=False, frequency=None):
        """
        Get response.

        Parameters
        ----------
        coord : array_like, shape (3, number of points)
            Cartesian coordinates for which to compute response.
        magnitude : bool
            Whether to return magnitude of response.

        """

        if self._p == DirectivityPattern.OMNI:
            return np.ones(coord.shape[1])
        else:
            resp = self._gain * self._p + (1 - self._p) \
                   * np.matmul(self._orientation.unit_vector, coord)
            if magnitude:
                return np.abs(resp)
            else:
                return resp

    @requires_matplotlib
    def plot_response(self, azimuth, colatitude=None, degrees=True):
        """
        Parameters
        ----------
        azimuth : array_like
            Azimuth values for plotting.
        colatitude : array_like, optional
            Colatitude values for plotting. If not provided, 2D plot.
        degrees : bool
            Whether provided values are in degrees (True) or radians (False).
        """
        import matplotlib.pyplot as plt

        if degrees:
            azimuth = np.radians(azimuth)

        fig = plt.figure()
        if colatitude is not None:

            if degrees:
                colatitude = np.radians(colatitude)

            spher_coord = all_combinations(azimuth, colatitude)
            azi_flat = spher_coord[:, 0]
            col_flat = spher_coord[:, 1]

            # compute response
            cart = spher2cart(azimuth=azi_flat, colatitude=col_flat)
            resp = self.get_response(coord=cart, magnitude=True)
            RESP = resp.reshape(len(azimuth), len(colatitude))

            # create surface plot, need cartesian coordinates
            AZI, COL = np.meshgrid(azimuth, colatitude)
            X = RESP.T * np.sin(COL) * np.cos(AZI)
            Y = RESP.T * np.sin(COL) * np.sin(AZI)
            Z = RESP.T * np.cos(COL)

            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_surface(X, Y, Z)
            ax.set_title("{}, azimuth={}, colatitude={}".format(
                self.directivity_pattern,
                self.get_azimuth(),
                self.get_colatitude()
            ))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

        else:

            # compute response
            cart = spher2cart(azimuth=azimuth)
            resp = self.get_response(coord=cart, magnitude=True)

            # plot
            ax = plt.subplot(111, projection="polar")
            ax.plot(azimuth, resp)


def cardioid_func(
    x,
    direction,
    coef,
    gain=1.0,
    normalize=True,
    magnitude=False
):
    """
    One-shot function for computing Cardioid response.

    Parameters
    -----------
    x: array_like, shape (..., n_dim)
         Cartesian coordinates
    direction: array_like, shape (n_dim)
         Direction vector, should be normalized.
    coef: float
         Parameter of the cardioid
    gain: float
         The gain.
    normalize : bool
        Whether to normalize coordinates and direction vector.
    magnitude : bool
        Whether to return magnitude, default is False.
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
    
    def circular_microphone_array_helper_xyplane(center, M, phi0, radius, fs, directivity_pattern): 
        """
        Creates an microphone array with directivities pointing outwards

        Parameters
        ----------
        center: array_like
            The center of the microphone array
        M: int
            The number of microphones
        phi0: float
            The counterclockwise rotation (in degrees) of the first element in the microphone array (from
            the x-axis)
        radius: float
            The radius of the microphone array
        fs: int
            The sampling frequency
        Returns
        -------
        MicrophoneArray object
        """
        
        azimuth_list = np.arange(M)*(360/M)
        phi_array = np.ones(M)*phi0
        azimuth_list = np.add(azimuth_list, phi_array)

        PATTERN = DirectivityPattern.directivity_pattern

        directivity_list = np.array([])

        for i in range(M):
            
            ORIENTATION = DirectionVector (azimuth=azimuth_list[i], colatitude=np.pi/2, degrees=False)
            dir_obj = CardioidFamily (orientation=ORIENTATION, pattern_enum=PATTERN)
            directivity_list.add(dir_obj)


        R = pra.circular_2D_array(center=center, M=M, phi0=phi0, radius=radius)
        return pra.MicrophoneArray(R, fs, directivity_list)
