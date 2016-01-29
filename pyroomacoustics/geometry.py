# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np

from parameters import constants, eps


def area(corners):
    """
    Computes the signed area of a 2D surface represented by its corners.
    
    :arg corners: (np.array 2xN, N>2) list of coordinates of the corners forming the surface
    
    :returns: (float) area of the surface
        positive area means anti-clockwise ordered corners.
        negative area means clockwise ordered corners.
    """

    corners = np.array(corners)
    x = corners[0, :] - corners[0, xrange(-1, corners.shape[1]-1)]
    y = corners[1, :] + corners[1, xrange(-1, corners.shape[1]-1)]
    return -0.5 * (x * y).sum()


def side(p, p0, vector):
    """
    Compute on which side of a given point an other given point is according to a vector.
    
    :arg p: (ndarray size 2 or 3) point to be tested
    :arg p0: (ndarray size 2 or 3) origin point
    :arg vector: (ndarray size 2 or 3) directional vector
    
    :returns: (int) direction of the point
        1 : p is at the side pointed by the vector
        0 : p is in the middle (on the same line as p0)
        -1 : p is at the opposite side of the one pointed by the vector
    """
    p = np.array(p)
    p0 = np.array(p0)
    vector = np.array(vector)
    
    projection = np.dot(vector, p - p0)
    if (projection > 0):
        return 1
    elif (projection < 0):
        return -1
    else:
        return 0


def ccw3p(p1, p2, p3):
    """
    Computes the orientation of three 2D points.
    
    :arg p1: (ndarray size 2) coordinates of a 2D point
    :arg p2: (ndarray size 2) coordinates of a 2D point
    :arg p3: (ndarray size 2) coordinates of a 2D point
    
    :returns: (int) orientation of the given triangle
        1 if triangle vertices are counter-clockwise
        -1 if triangle vertices are clockwise
        0 if vertices are collinear

    :ref: https://en.wikipedia.org/wiki/Curve_orientation
    """
    
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    if (p1.shape[0] != 2 or p2.shape[0] != 2 or p3.shape[0] != 2):
        raise ValueError('geometry.ccw3p is for three 2D points')
    d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    if (np.abs(d) < eps):
        return 0
    elif (d > 0):
        return 1
    else:
        return -1


def intersection2DSegments(a1, a2, b1, b2) :
    """
    Computes the intersection between two 2D line segments.
    
    This function computes the intersection between two 2D segments
    (defined by the coordinates of their endpoints) and returns the
    coordinates of the intersection point.
    If there is no intersection, None is returned.
    If segments are collinear, None is returned.
    Two booleans are also returned to indicate if the intersection
    happened at extremities of the segments, which can be useful for limit cases
    computations.
    
    :arg a1: (ndarray size 2) coordinates of the first endpoint of segment a
    :arg a2: (ndarray size 2) coordinates of the second endpoint of segment a
    :arg b1: (ndarray size 2) coordinates of the first endpoint of segment b
    :arg b2: (ndarray size 2) coordinates of the second endpoint of segment b
    
    :returns: (tuple of 3 elements) results of the computation
        (ndarray size 2 or None) coordinates of the intersection point
        (bool) True if the intersection is at boundaries of segment a
        (bool) True if the intersection is at boundaries of segment b
    """
    
    a1=np.array(a1)
    a2=np.array(a2)
    b1=np.array(b1)
    b2=np.array(b2)
    if (a1.shape[0] != 2):
        raise NameError('utilities.intersection2DSegments input error : a1 is a ndarray of size 2')
    if (a2.shape[0] != 2):
        raise NameError('utilities.intersection2DSegments input error : a2 is a ndarray of size 2')
    if (b1.shape[0] != 2):
        raise NameError('utilities.intersection2DSegments input error : b1 is a ndarray of size 2')
    if (b2.shape[0] != 2):
        raise NameError('utilities.intersection2DSegments input error : b2 is a ndarray of size 2')
    
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = np.empty_like(da)
    dap[0] = -da[1]
    dap[1] = da[0]
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    if(denom == 0):
        p = None
        endpointA = False
        endpointB = False
    else:
        p = np.array((num / denom.astype(float))*db + b1)
        if (ccw3p(a1,a2,b1) != ccw3p(a1,a2,b2) and ccw3p(b1,b2,a1) != ccw3p(b1,b2,a2)):
            if (np.allclose(p, a1) or np.allclose(p, a2)):
                endpointA = True
            else:
                endpointA = False
            if (np.allclose(p, b1) or np.allclose(p, b2)):
                endpointB = True
            else:
                endpointB = False
        else:
            p = None
            endpointA = False
            endpointB = False
    return p, endpointA, endpointB

    
def intersectionSegmentPlane(a1, a2, p, normal):
    """
    Computes the intersection between a line segment and a plane in 3D.
    
    This function computes the intersection between a line segment (defined
    by the coordinates of two points) and a plane (defined by a point belonging
    to it and a normal vector). If there is no intersection, None is returned.
    If the segment belongs to the surface, None is returned.
    A boolean is also returned to indicate if the intersection
    happened at extremities of the segment, which can be useful for limit cases
    computations.
    
    :arg a1: (ndarray size 3) coordinates of the first endpoint of the segment
    :arg a2: (ndarray size 3) coordinates of the second endpoint of the segment
    :arg p: (ndarray size 3) coordinates of a point belonging to the plane
    :arg normal: (ndarray size 3) normal vector of the plane
    
    :returns: (tuple of 2 elements) results of the computation
        (ndarray size 3 or None) coordinates of the intersection point
        (bool) True if the intersection is at boundaries of the segment
    """

    a1=np.array(a1).astype(float)
    a2=np.array(a2).astype(float)
    p=np.array(p).astype(float)
    normal=np.array(normal).astype(float)
    if (a1.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPlane input error : a1 is a ndarray of size 3')
    if (a2.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPlane input error : a2 is a ndarray of size 3')
    if (p.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPlane input error : p is a ndarray of size 3')
    if (normal.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPlane input error : normal is a ndarray of size 3')
    
    u = a2-a1
    w = a1-p
    num = -np.dot(normal, w)
    denom = np.dot(normal, u)
    if(abs(denom) < constants.eps):
        return None, False
    else:
        s = num/denom
        if(s<0 or s>1):
            return None, False
        else:
            if (s==0 or s==1):
                limitCase = True
            else:
                limitCase = False
            return a1+s*u, limitCase


def intersectionSegmentPolygonSurface(a1, a2, corners, normal):
    """
    Computes the intersection between a line segment and a polygon surface in 3D.
    
    This function computes the intersection between a line segment (defined
    by the coordinates of two points) and a surface (defined by an array of
    coordinates of corners of the polygon and a normal vector)
    If there is no intersection, None is returned.
    If the segment belongs to the surface, None is returned.
    Two booleans are also returned to indicate if the intersection
    happened at extremities of the segment or at a border of the polygon,
    which can be useful for limit cases computations.
    
    :arg a1: (ndarray size 3) coordinates of the first endpoint of the segment
    :arg a2: (ndarray size 3) coordinates of the second endpoint of the segment
    :arg corners: (ndarray size 3xN, N>2) coordinates of the corners of the polygon
    :arg normal: (ndarray size 3) normal vector of the surface
    
    :returns: (tuple of 3 elements) results of the computation
        (ndarray size 3 or None) coordinates of the intersection point
        (bool) True if the intersection is at boundaries of the segment
        (bool) True if the intersection is at boundaries of the polygon
    """

    a1 = np.array(a1)
    a2 = np.array(a2)
    corners = np.array(corners)
    normal = np.array(normal)
    if (a1.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : a1 is a ndarray of size 3')
    if (a2.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : a2 is a ndarray of size 3')
    if (corners.shape[0] != 3 or corners.shape[1] < 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : corners is a 3xN ndarray, N>2')
    if (normal.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : normal is a ndarray of size 3')
    
    # Check if the segment intersects the plane formed by the surface
    # Get the intersection point if it is the case
    p, segmentLimit = intersectionSegmentPlane(a1, a2, corners[:,0], normal)
    if p is None:
        return None, False, False

    # Convert 3D coordinates to local 2D coordinates in the plane
    local0 = np.array(corners[:,0])
    localx = np.array(corners[:,1]-local0)
    localx = localx/np.linalg.norm(localx)
    localy = np.array(np.cross(normal, localx))
    localy = localy/np.linalg.norm(localy)
    localCorners = np.concatenate(([np.dot(corners.T-local0, localx)], [np.dot(corners.T-local0, localy)]))
    localp = np.array([np.dot(p-local0, localx), np.dot(p-local0, localy)])
    
    # Check if the intersection point is in the polygon on the plane
    inside, borderLimit = isInside2DPolygon(localp, localCorners)
    if inside:
        p = local0 + localp[0]*localx + localp[1]*localy
    else:
        p = None
    
    return p, segmentLimit, borderLimit


def isInside2DPolygon(p, corners):
    """
    Checks if a given point is inside a given polygon in 2D.
    
    This function checks if a point (defined by its coordinates) is inside
    a polygon (defined by an array of coordinates of its corners) by counting
    the number of intersections between the borders and a segment linking
    the given point with a computed point outside the polygon.
    A boolean is also returned to indicate if a point is on a border of the
    polygon (the point is still considered inside), which can be useful for
    limit cases computations.
    
    :arg p: (ndarray size 2) coordinates of the point
    :arg corners: (ndarray size 2xN, N>2) coordinates of the corners of the polygon
    
    :returns: (tuple of 2 elements) results of the computation
        (bool) True if the point is inside
        (bool) True if the intersection is at boundaries of the polygon
    """
    
    p = np.array(p)
    corners = np.array(corners)
    if (p.shape[0] != 2):
        raise NameError('utilities.isInside2DPolygon input error : p is a ndarray of size 2')
    if (corners.shape[0] != 2 or corners.shape[1] < 3):
        raise NameError('utilities.isInside2DPolygon input error : corners is a 2xN ndarray, N>2')
        
    # Compute p0, which is a point outside the room at x coordinate xMin-1
    # (where xMin is the minimum x coordinate among the corners)
    p0 = np.array([np.amin(corners[0])-1, p[1]])

    lastIntersection = 0
    count = 0
    limitCase = False
    for i in range(corners.shape[1]):
        intersection, limitA, limitB = intersection2DSegments(p0, p, corners[:, i], corners[:, (i+1)%corners.shape[1]])
        if (limitA):
            limitCase = True
        if (intersection is not None):
            count += 1
    if ((count % 2 == 1) or limitCase):
        return True, limitCase
    else:
        return False, False
