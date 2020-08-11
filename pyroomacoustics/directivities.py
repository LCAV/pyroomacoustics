import abc
import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree
from .doa import GridSphere
from . import random
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

    @property
    def azimuth(self):
        return self.get_azimuth()

    @property
    def colatitude(self):
        return self.get_colatitude()

    @property
    def azimuth_deg(self):
        return self.get_azimuth(degrees=True)

    @property
    def colatitude_deg(self):
        return self.get_colatitude(degrees=True)



class Directivity(abc.ABC):
    def __init__(self, orientation):
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees)

    @abc.abstractmethod
    def get_response(self, directions, magnitude=False):
        """
        This method returns the attenuation associated to a direction for
        this directive object.

        Parameters
        ----------
        directions: array_like, shape (n_dim, n_directions)
            The direction unit vectors
        magnitude: bool, optional
            A flag indicating if the response should be given for the
             magnitude, instead of energy (the default)

        Returns
        -------
        response: numpy.ndarray, shape (n_directions)
            The attenuation corresponding to the given directions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_rays(self, n_rays):
        """
        This method samples unit vectors from the sphere according to
        the distribution of the source

        Parameters
        ----------
        n_rays: int
            The number of rays to sample

        Returns
        -------
        ray_directions: numpy.ndarray, shape (n_dim, n_rays)
            An array containing the unit vectors in its columns
        """
        raise NotImplementedError


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

        # this is the object that will allow to sample rays according to the
        # distribution corresponding to the source directivity
        self._ray_sampler = random.sampler.CardioidFamilySampler(
            loc=self._orientation.unit_vector, coeff=self._p
        )

    @property
    def directivity_pattern(self):
        return self._pattern_name

    def get_response(self, coord, magnitude=False):
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

    def sample_rays(self, n_rays):
        return self._ray_sampler(n_rays).T

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
            if degrees:
                angles = np.radians(azimuth)
            else:
                angles = azimuth
            ax.plot(angles, gains)


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


class SphericalHistogram:

    def __init__(self, n_bins, dim=3, enable_peak_finding=False):

        self._n_dim = 3
        self._n_bins = n_bins

        if self.n_dim == 3:
            self._grid = GridSphere(n_points=self.n_bins, enable_peak_finding=enable_peak_finding)
        else:
            raise NotImplementedError("Only 3D histogram has been implemented")

        # we need to know the area of each bin
        self._voronoi = SphericalVoronoi(self._grid.cartesian.T)
        self._areas = self._voronoi.calculate_areas()

        # now we also need a KD-tree to do nearest neighbor search
        self._kd_tree = cKDTree(self._grid.cartesian.T)

        # the counter variables for every bin
        self._bins = np.zeros(self.n_bins, dtype=np.int)

        # the total number of points in the histogram
        self._total_count = 0

        # we cache the histogram bins
        self._cache_dirty = False
        self._cache_histogram = np.zeros(self.n_bins)

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def histogram(self):
        if self._cache_dirty:
            # if the cache is dirty, we need to recompute
            Z = np.sum(self._areas * self._bins)  # partitioning constant
            self._cache_histogram[:] = self._bins / Z
            self._cache_dirty = False

        return self._cache_histogram

    @property
    def raw_counts(self):
        return self._bins

    @property
    def total_count(self):
        return self._total_count

    def find_peak(self, *args, **kwargs):
        return self._grid.find_peaks(self, *args, **kwargs)

    def plot(self):
        self._grid.set_values(self.histogram)
        self._grid.plot_old()

    def push(self, points):
        """
        Add new data into the histogram

        Parameters
        ----------
        points: array_like, shape (n_dim, n_points)
            The points to add to the histogram
        """
        self._total_count += points.shape[1]
        self._cache_dirty = True

        _, matches = self._kd_tree.query(points.T)
        bin_indices, counts = np.unique(matches, return_counts=True)
        self._bins[bin_indices] += counts
