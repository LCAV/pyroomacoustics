"""
This module contains useful functions to compute distances and errors on on
circles and spheres.
"""

from __future__ import division

import collections

import numpy as np

RegularGrid = collections.namedtuple("RegularGrid", ["azimuth", "colatitude"])


def circ_dist(azimuth1, azimuth2, r=1.0):
    """
    Returns the shortest distance between two points on a circle

    Parameters
    ----------
    azimuth1:
        azimuth of point 1
    azimuth2:
        azimuth of point 2
    r: optional
        radius of the circle (Default 1)
    """
    return np.arccos(np.cos(azimuth1 - azimuth2))


def great_circ_dist(r, colatitude1, azimuth1, colatitude2, azimuth2):
    """
    calculate great circle distance for points located on a sphere

    Parameters
    ----------
    r: radius of the sphere
    colatitude1: colatitude of point 1
    azimuth1: azimuth of point 1
    colatitude2: colatitude of point 2
    azimuth2: azimuth of point 2

    Returns
    -------
    float or ndarray
        great-circle distance
    """
    d_azimuth = np.abs(azimuth1 - azimuth2)
    dist = r * np.arctan2(
        np.sqrt(
            (np.sin(colatitude2) * np.sin(d_azimuth)) ** 2
            + (
                np.sin(colatitude1) * np.cos(colatitude2)
                - np.cos(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth)
            )
            ** 2
        ),
        np.cos(colatitude1) * np.cos(colatitude2)
        + np.sin(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth),
    )
    return dist


def cart2spher(vectors):
    """
    Parameters
    ----------
    vectors: array_like, shape (3, n_vectors)
        The vectors to transform

    Returns
    -------
    azimuth: numpy.ndarray, shape (n_vectors,)
        The azimuth of the vectors
    colatitude: numpy.ndarray, shape (n_vectors,)
        The colatitude of the vectors
    r: numpy.ndarray, shape (n_vectors,)
        The length of the vectors
    """

    r = np.linalg.norm(vectors, axis=0)

    azimuth = np.arctan2(vectors[1], vectors[0])
    colatitude = np.arctan2(np.linalg.norm(vectors[:2], axis=0), vectors[2])

    return azimuth, colatitude, r


def spher2cart(azimuth, colatitude=None, r=1, degrees=False):
    """
    Convert a spherical point to cartesian coordinates.

    Parameters
    ----------
    azimuth:
        azimuth
    colatitude:
        colatitude
    r:
        radius
    degrees:
        If True, indicates that the input angles are in degree (instead of radian)

    Returns
    -------
    ndarray
        An ndarray containing the Cartesian coordinates of the points as its columns.
    """

    if degrees:
        azimuth = np.radians(azimuth)
        if colatitude is not None:
            colatitude = np.radians(colatitude)

    if colatitude is None:
        # default to XY plane
        colatitude = np.pi / 2
        if hasattr(azimuth, "__len__"):
            colatitude = np.ones(len(azimuth)) * colatitude

    # convert to cartesian
    x = r * np.cos(azimuth) * np.sin(colatitude)
    y = r * np.sin(azimuth) * np.sin(colatitude)
    z = r * np.cos(colatitude)
    return np.array([x, y, z])


def polar_distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of
    the absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).

    Parameters
    ----------
    x1:
        vector 1
    x2:
        vector 2

    Returns
    -------
    d:
        minimum distance between d
    index:
        the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order="F")
    x2 = np.reshape(x2, (1, -1), order="F")
    N1 = x1.size
    N2 = x2.size
    diffmat = np.arccos(np.cos(x1 - np.reshape(x2, (-1, 1), order="F")))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float("inf")
            diffmat[:, index1] = float("inf")
        d = np.mean(np.arccos(np.cos(x1[:, index[:, 0]] - x2[:, index[:, 1]])))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def fibonacci_spherical_sampling(n_points):
    """
    This function computes nearly equidistant points on the sphere
    using the fibonacci method

    Parameters
    ----------
    n_points: int
        The number of points to sample

    Returns
    -------
    points: numpy.ndarray, (3, n_points)
        The cartesian coordinates of the points

    References
    ----------
    http://lgdv.cs.fau.de/uploads/publications/spherical_fibonacci_mapping.pdf
    http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """

    points = np.zeros((3, n_points))

    # Fibonnaci sampling
    offset = 2.0 / n_points
    increment = np.pi * (3.0 - np.sqrt(5.0))

    points[2, :] = (np.arange(n_points) * offset - 1) + offset / 2
    rho = np.sqrt(1.0 - points[2] ** 2)

    phi = np.arange(n_points) * increment

    points[0, :] = np.cos(phi) * rho
    points[1, :] = np.sin(phi) * rho

    return points


def detect_regular_grid(azimuth, colatitude):
    """
    This function checks that the linearized azimuth/colatitude where sampled
    from a regular grid.

    It also checks that the azimuth are uniformly spread in [0, 2 * np.pi).
    The colatitudes can have arbitrary positions.

    Parameters
    ----------
    azimuth: numpy.ndarray (npoints,)
        The azimuth values in radian
    colatitude: numpy.ndarray (npoints,)
        The colatitude values in radian

    Returns
    -------
    regular_grid: dict["azimuth", "colatitude"] or None
        A dictionary with entries for the sorted distinct azimuth an colatitude values
        of the grid, if the points form a grid.
        Returns `None` if the points do not form a grid.
    """
    if len(azimuth) != len(colatitude):
        return None

    azimuth_unique = np.unique(azimuth)
    colatitude_unique = np.unique(colatitude)
    regular_grid = None
    if len(azimuth_unique) * len(colatitude_unique) == len(azimuth):
        # check that the azimuth are uniformly spread
        az_loop = np.insert(
            azimuth_unique, len(azimuth_unique), azimuth_unique[0] + 2 * np.pi
        )
        delta_az = np.diff(az_loop)
        if np.allclose(delta_az, 2 * np.pi / len(azimuth_unique)):
            # remake the grid from the unique points and check
            # that it matches the original
            A, C = np.meshgrid(azimuth_unique, colatitude_unique)
            regrid = np.column_stack([A.flatten(), C.flatten()])
            regrid = regrid[np.lexsort(regrid.T), :]
            ogrid = np.column_stack([azimuth, colatitude])
            ogrid = ogrid[np.lexsort(ogrid.T), :]
            if np.allclose(regrid, ogrid):
                regular_grid = RegularGrid(
                    azimuth=azimuth_unique, colatitude=colatitude_unique
                )

    return regular_grid
