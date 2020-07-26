import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree
from .doa import GridSphere
from pyroomacoustics.utilities import Options


class DirectivityPattern(Options):
    FIGURE_EIGHT = "figure_eight"
    HYPERCARDIOID = "hypercardioid"
    CARDIOID = "cardioid"
    SUBCARDIOID = "subcardioid"
    OMNI = "omni"


PATTERN_TO_CONSTANT = {
    DirectivityPattern.FIGURE_EIGHT: 0,
    DirectivityPattern.HYPERCARDIOID: 0.25,
    DirectivityPattern.CARDIOID: 0.5,
    DirectivityPattern.SUBCARDIOID: 0.75,
    DirectivityPattern.OMNI: 1.0
}


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


class Directivity(object):
    """

    Parameters
    ----------
    orientation : DirectionVector
        Indicates direction of the pattern.
    pattern : str
        One of directivities in `DirectivityPattern`.
        TODO : support arbitrary directivities.

    """
    def __init__(self, orientation, pattern):
        assert pattern in DirectivityPattern.values()
        assert isinstance(orientation, DirectionVector)
        self._orientation = orientation
        self._pattern = pattern
        self._p = PATTERN_TO_CONSTANT[pattern]

    def get_directivity_pattern(self):
        return self._pattern

    def get_azimuth(self, degrees=True):
        return self._orientation.get_azimuth(degrees)

    def get_colatitude(self, degrees=True):
        return self._orientation.get_colatitude(degrees)

    def get_response(self, direction):
        """
        Get response for a single direction.

        TODO : vectorize

        Parameters
        ----------
        direction : DirectionVector
            Direction for which to compute gain

        """
        assert isinstance(direction, DirectionVector)
        if self._pattern == DirectivityPattern.OMNI:
            return 1
        else:
            # inspiration from Habets: https://github.com/ehabets/RIR-Generator/blob/5eb70f066b74ff18c2be61c97e8e666f8492c149/rir_generator.cpp#L111
            # we use colatitude instead of elevation
            gain = np.sin(self._orientation.get_colatitude()) * \
                   np.sin(direction.get_colatitude()) * \
                   np.cos(
                       self._orientation.get_azimuth() -
                       direction.get_azimuth()
                   ) + \
                   np.cos(self._orientation.get_colatitude()) * \
                   np.cos(direction.get_colatitude())
            return np.abs(self._p + (1 - self._p) * gain)

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

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

        fig = plt.figure()
        if colatitude is not None:

            # compute response
            gains = np.zeros(shape=(len(azimuth), len(colatitude)))
            for i, a in enumerate(azimuth):
                for j, c in enumerate(colatitude):
                    direction = DirectionVector(
                        azimuth=a,
                        colatitude=c,
                        degrees=degrees
                    )
                    gains[i, j] = self.get_response(direction)

            # create surface plot, need cartesian coordinates
            if degrees:
                azimuth = np.radians(azimuth)
                colatitude = np.radians(colatitude)
            AZI, COL = np.meshgrid(azimuth, colatitude)

            X = gains.T * np.sin(COL) * np.cos(AZI)
            Y = gains.T * np.sin(COL) * np.sin(AZI)
            Z = gains.T * np.cos(COL)

            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_surface(X, Y, Z)
            ax.set_title("{}, azimuth={}, colatitude={}".format(
                self.get_directivity_pattern(),
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
            gains = np.zeros(len(azimuth))
            for i, a in enumerate(azimuth):
                direction = DirectionVector(azimuth=a, degrees=degrees)
                gains[i] = self.get_response(direction)

            # plot
            ax = plt.subplot(111, projection="polar")
            if degrees:
                angles = np.radians(azimuth)
            else:
                angles = azimuth
            ax.plot(angles, gains)


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
