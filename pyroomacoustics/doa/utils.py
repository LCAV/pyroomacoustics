'''
This module contains useful functions to compute distances and errors on on
circles and spheres.
'''
from __future__ import division
import numpy as np


def circ_dist(azimuth1, azimuth2, r=1.):
    ''' 
    Returns the shortest distance between two points on a circle

    Parameters
    ----------
    azimuth1:
        azimuth of point 1
    azimuth2:
        azimuth of point 2
    r: optional
        radius of the circle (Default 1)
    '''
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
    dist = r * np.arctan2(np.sqrt((np.sin(colatitude2) * np.sin(d_azimuth)) ** 2 +
                                  (np.sin(colatitude1) * np.cos(colatitude2) -
                                   np.cos(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth)) ** 2),
                          np.cos(colatitude1) * np.cos(colatitude2) +
                          np.sin(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth))
    return dist

def spher2cart(r, azimuth, colatitude):
    """
    Convert a spherical point to cartesian coordinates.

    Parameters
    ----------
    r:
        radius
    azimuth:
        azimuth
    colatitude:
        colatitude

    Returns
    -------
    ndarray
        An ndarray containing the Cartesian coordinates of the points its columns
    """
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
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.arccos(np.cos(x1 - np.reshape(x2, (-1, 1), order='F')))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.arccos(np.cos(x1[:, index[:, 0]] - x2[:, index[:, 1]])))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index
