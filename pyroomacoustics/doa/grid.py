'''
Routines to perform grid search on the sphere
'''
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.spatial as sp #import ConvexHull, SphericalVoronoi

from abc import ABCMeta, abstractmethod

from .detect_peaks import detect_peaks
from .utils import great_circ_dist

class Grid:
    '''
    This is an abstract class with attributes and methods for grids

    Parameters
    ----------
    n_points: int
        the number of points on the grid
    '''

    __metaclass__ = ABCMeta

    def __init__(self, n_points):

        self.n_points = n_points
        self.values = None
        self.cartesian = np.zeros((3,n_points))
        self.spherical = np.zeros((2,n_points))
        self.x = self.cartesian[0,:]
        self.y = self.cartesian[1,:]
        self.z = self.cartesian[2,:]
        self.azimuth = self.spherical[0,:]
        self.colatitude = self.spherical[1,:]
        self.dim = 0
        self.values = None

    @abstractmethod
    def apply(self, func, spherical=False):
        return NotImplemented

    @abstractmethod
    def find_peaks(self, k=1):
        return NotImplemented

    def set_values(self, vals):

        vals = np.array(vals)

        if vals.ndim == 0:
            self.values = np.ones(self.n_points) * vals

        else:
            if vals.shape != (self.n_points,):
                raise ValueError('Values should be a scalar or a 1D ndarray of the grid size.')

            self.values = vals


class GridCircle(Grid):
    '''
    Creates a grid on the circle.

    Parameters
    ----------
    n_points: int, optional
        The number of uniformly spaced points in the grid.
    azimuth: ndarray, optional
        An array of azimuth (in radians) to use for grid locations. Overrides n_points.
    '''

    def __init__(self, n_points=360, azimuth=None):
        if azimuth is not None:

            if azimuth.ndim != 1:
                raise ValueError("Azimuth must be a 1D ndarray.")

            azimuth = np.sort(azimuth)

            n_points = azimuth.shape[0]

        else:
            azimuth = np.linspace(0, 2*np.pi, n_points, endpoint=False)

        Grid.__init__(self, n_points)

        self.dim = 2

        # spherical coordinates
        self.azimuth[:] = azimuth
        self.colatitude[:] = np.pi / 2  # Fix colatitude to ecuador

        # cartesian coordinates
        self.x[:] = np.cos(azimuth)
        self.y[:] = np.sin(azimuth)

    def apply(self, func, spherical=False):

        if spherical:
            self.values = func(self.azimuth)
        else:
            self.values = func(self.x, self.y)

    def find_peaks(self, k=1):

        # make circular
        val_ext = np.append(self.values ,self.values[:10])

        # run peak finding
        indexes = detect_peaks(val_ext, show=False) % self.n_points
        candidates = np.unique(indexes)  # get rid of duplicates, if any

        # Select k largest
        peaks = self.values[candidates]
        max_idx = np.argsort(peaks)[-k:]

        # return the indices of peaks found
        return candidates[max_idx]

    def plot(self, mark_peaks=0):

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

        pts = np.append(self.azimuth, self.azimuth[0])
        vals = np.append(self.values, self.values[0])

        ax.plot(pts, vals, '-')

        if mark_peaks > 0:

            idx = self.find_peaks(k=mark_peaks)

            ax.plot(pts[idx], vals[idx], 'ro')


class GridSphere(Grid):
    '''
    This function computes nearly equidistant points on the sphere
    using the fibonacci method

    Parameters
    ----------
    n_points: int
        The number of points to sample
    spherical_points: ndarray, optional
        A 2 x n_points array of spherical coordinates with azimuth in
        the top row and colatitude in the second row. Overrides n_points.

    References
    ----------
    http://lgdv.cs.fau.de/uploads/publications/spherical_fibonacci_mapping.pdf
    http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    '''

    def __init__(self, n_points=1000, spherical_points=None):
        if spherical_points is not None:

            if spherical_points.ndim != 2 or spherical_points.shape[0] != 2:
                raise ValueError('spherical_points must be a 2D array with two rows.')

            n_points = spherical_points.shape[1]

        # Parent constructor
        Grid.__init__(self, n_points)

        self.dim = 3

        if spherical_points is not None:
            # If a list of points was provided, use it

            self.spherical[:,:] = spherical_points

            # transform to cartesian coordinates
            self.x[:] = np.cos(self.azimuth) * np.sin(self.colatitude)
            self.y[:] = np.sin(self.azimuth) * np.sin(self.colatitude)
            self.z[:] = np.cos(self.colatitude)

        else:
            # If no list was provided, samples points on the sphere
            # as uniformly as possible

            # Fibonnaci sampling
            offset = 2. / n_points
            increment = np.pi * (3. - np.sqrt(5.))

            self.z[:] = (np.arange(n_points) * offset - 1) + offset / 2
            rho = np.sqrt(1. - self.z**2)

            phi = np.arange(n_points) * increment

            self.x[:] = np.cos(phi) * rho
            self.y[:] = np.sin(phi) * rho

            # Create convenient arrays
            # to access both in cartesian and spherical coordinates
            self.azimuth[:] = np.arctan2(self.y, self.x)
            self.colatitude[:] = np.arctan2(np.sqrt(self.x**2 + self.y**2), self.z)

        
        # To perform the peak detection in 2D on a non-squared grid it is
        # necessary to know the neighboring points of each grid point.  The
        # Convex Hull of points on the sphere is equivalent to the Delauney
        # triangulation of the point set, which is what we are looking for.

        # Now we also want to compute the convex hull
        self.hull = sp.ConvexHull(self.cartesian.T)

        # and create an adjacency list
        adjacency = [ set() for pt in range(self.n_points) ]

        # Simplices contains all the triangles that form
        # the faces of the convex hull. We use this to find which
        # points are connected on the hull.
        for tri in self.hull.simplices:
            adjacency[tri[0]].add(tri[1])
            adjacency[tri[0]].add(tri[2])
            adjacency[tri[1]].add(tri[0])
            adjacency[tri[1]].add(tri[2])
            adjacency[tri[2]].add(tri[0])
            adjacency[tri[2]].add(tri[1])

        # convert to list of lists
        self.neighbors = [ list(x) for x in adjacency ]


    def apply(self, func, spherical=False):
        '''
        Apply a function to every grid point
        '''

        if spherical:
            self.values = func(self.azimuth, self.colatitude)
        else:
            self.values = func(self.x, self.y, self.z)


    def min_max_distance(self):
        ''' Compute some statistics on the distribution of the points '''

        min_dist = np.inf
        max_dist = 0

        dist = []

        for u in range(self.n_points):

            phi1, theta1 = self.spherical[:,u]

            for v in self.neighbors[u]:

                phi2, theta2 = self.spherical[:,v]

                d = great_circ_dist(1, theta1, phi1, theta2, phi2)

                dist.append(d)

                if d < min_dist:
                    min_dist = d
                
                if d > max_dist:
                    max_dist = d

        dist = np.array(dist)
        mean = dist.mean()
        median = np.median(dist)

        return min_dist, max_dist, mean, median


    def find_peaks(self, k=1):
        '''
        Find the largest peaks on the grid
        '''
        
        candidates = []

        # We start by looking at points whose neighbors all have lower values
        # than themselves
        for v in range(self.n_points):

            is_local_max = True

            for u in self.neighbors[v]:
                if self.values[u] > self.values[v]:
                    is_local_max = False
                    break

            if is_local_max:
                candidates.append(v)

        # Now sort and return k largest
        I = np.argsort(self.values[candidates])

        return [candidates[x] for x in I[-k:]]


    def regrid(self):
        ''' Regrid the non-uniform data on a regular mesh '''

        if self.values is None:
            warnings.warn('Cannont regrid: data missing.')
            return

        # First we need to interpolate the non-uniformly sampled data
        # onto a uniform grid
        from scipy.interpolate import griddata
        
        x = int(np.sqrt(self.n_points / 2))

        G = np.mgrid[-np.pi:np.pi:2j*x, 0:np.pi:1j*x]
        spherical_grid = np.squeeze(G.reshape((2,1,-1)))

        gridded = griddata(self.spherical.T, self.values, spherical_grid.T, method='nearest', fill_value=0.)
        dirty_img = gridded.reshape((2*x, x))

        return G[0], G[1], gridded.reshape((2*x, x))


    def plot(self, 
            colatitude_ref=None, azimuth_ref=None,
            colatitude_recon=None, azimuth_recon=None,
            plotly=True, projection=True, points_only=False):

        if points_only:
            dirty_img = None
            dirty_grid_x = None
            dirty_grid_y = None
        else:
            dirty_grid_x, dirty_grid_y, dirty_img = self.regrid()

        ## Then, we just need to call Hanjie's routines
        from .plotters import sph_plot_diracs, sph_plot_diracs_plotly

        if projection:
            sph_plot_diracs(
                    colatitude_ref, azimuth_ref, colatitude_recon, azimuth_recon, 
                    dirty_img=dirty_img,
                    colatitude_grid=dirty_grid_y, azimuth_grid=dirty_grid_x,
                    )

        if plotly:
            sph_plot_diracs_plotly(
                    colatitude_ref, azimuth_ref, colatitude_recon, azimuth_recon, 
                    dirty_img=dirty_img,
                    azimuth_grid=dirty_grid_x, colatitude_grid=dirty_grid_y
                    )

    
    def plot_old(self, plot_points=False, mark_peaks=0):
        ''' Plot the points on the sphere with their values '''

        from scipy import rand
        import matplotlib.colors as colors
        #from mpl_toolkits.mplot3d import Axes3D
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if plot_points:
            ax.scatter(self.x, self.y, self.z, c='b', marker='o')

        if mark_peaks > 0:

            id = self.find_peaks(k=mark_peaks)
            s = 1.05
            ax.scatter(s*self.x[id], s*self.y[id], s*self.z[id], c='k', marker='o')

        voronoi = sp.SphericalVoronoi(self.cartesian.T)
        voronoi.sort_vertices_of_regions()

        if self.values is not None:

            col_max = self.values.max()
            col_min = self.values.min()

            if col_min != col_max:
                col_map = (self.values - col_min) / (col_max - col_min)
            else:
                col_map = self.values / col_max

        else:

            col_map = np.zeros(self.n_points)

        cmap = plt.get_cmap('coolwarm')

        # plot the walls
        for v_ind, col in zip(voronoi.regions, col_map):
            triangle = a3.art3d.Poly3DCollection([voronoi.vertices[v_ind]], alpha=1., linewidth=0.0)
            triangle.set_color(cmap(col))
            triangle.set_edgecolor('k')
            ax.add_collection3d(triangle)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])

