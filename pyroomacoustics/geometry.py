# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np

from .parameters import constants, eps


def area(corners):
    '''
    Computes the signed area of a 2D surface represented by its corners.
    
    :arg corners: (np.array 2xN, N>2) list of coordinates of the corners forming the surface
    
    :returns: (float) area of the surface
        positive area means anti-clockwise ordered corners.
        negative area means clockwise ordered corners.
    '''

    corners = np.array(corners)
    x = corners[0, :] - corners[0, range(-1, corners.shape[1]-1)]
    y = corners[1, :] + corners[1, range(-1, corners.shape[1]-1)]
    return -0.5 * (x * y).sum()


def side(p, p0, vector):
    '''
    Compute on which side of a given point an other given point is according to a vector.
    
    :arg p: (ndarray size 2 or 3) point to be tested
    :arg p0: (ndarray size 2 or 3) origin point
    :arg vector: (ndarray size 2 or 3) directional vector
    
    :returns: (int) direction of the point
        1 : p is at the side pointed by the vector
        0 : p is in the middle (on the same line as p0)
        -1 : p is at the opposite side of the one pointed by the vector
    '''
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
    '''
    Computes the orientation of three 2D points.
    
    :arg p1: (ndarray size 2) coordinates of a 2D point
    :arg p2: (ndarray size 2) coordinates of a 2D point
    :arg p3: (ndarray size 2) coordinates of a 2D point
    
    :returns: (int) orientation of the given triangle
        1 if triangle vertices are counter-clockwise
        -1 if triangle vertices are clockwise
        0 if vertices are collinear

    :ref: https://en.wikipedia.org/wiki/Curve_orientation
    '''
    
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


def intersection_2D_segments(a1, a2, b1, b2) :
    '''
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
    '''
    
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

    # First we weed out the simple cases where no intersection happens
    # case 1
    a1a2b1 = ccw3p(a1, a2, b1)
    a1a2b2 = ccw3p(a1, a2, b2)
    if a1a2b1 == a1a2b2:
        return None, False, False

    # case 2
    b1b2a1 = ccw3p(b1, b2, a1)
    b1b2a2 = ccw3p(b1, b2, a2)
    if b1b2a1 == b1b2a2:
        return None, False, False

    da = a2-a1
    db = b2-b1
    dap = np.empty_like(da)
    dap[0] = -da[1]
    dap[1] = da[0]
    denom = np.dot(dap, db)

    # case 3
    if denom == 0:
        return None, False, False

    # At this point, we know there is intersection
    dp = a1-b1
    num = np.dot(dap, dp)
    # This is the intersection point
    p = np.array((num / denom.astype(float))*db + b1)

    # Test if intersection is actually at one of a1 or a2
    if b1b2a1 == 0 or b1b2a2 == 0:
        endpointA = True
    else:
        endpointA = False

    # Test if intersection is actually at one of b1 or b2
    if a1a2b1 == 0 or a1a2b2 == 0:
        endpointB = True
    else:
        endpointB = False

    return p, endpointA, endpointB


    
def intersection_segment_plane(a1, a2, p, normal):
    '''
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
    '''

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
    denom = np.dot(normal, u)

    if(abs(denom) < eps):
        return None, False

    else:
        num = -np.dot(normal, w)
        s = num/denom
        if(s<0 or s>1):
            return None, False
        else:
            if (s==0 or s==1):
                limitCase = True
            else:
                limitCase = False
            return a1+s*u, limitCase


def intersection_segment_polygon_surface(a1, a2, corners_2d, normal, plane_point, plane_basis):
    '''
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
    '''

    a1 = np.array(a1)
    a2 = np.array(a2)
    corners_2d = np.array(corners_2d)
    normal = np.array(normal)
    if (a1.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : a1 is a ndarray of size 3')
    if (a2.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : a2 is a ndarray of size 3')
    if (corners_2d.shape[0] != 2 or corners_2d.shape[1] < 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : corners is a 3xN ndarray, N>2')
    if (normal.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : normal is a ndarray of size 3')
    if (plane_point.shape[0] != 3):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : point_plane is a ndarray of size 3')
    if (plane_basis.shape != (3,2)):
        raise NameError('utilities.intersectionSegmentPolygonSurface input error : point_basis is a ndarray of shape (3,2)')
    
    # Check if the segment intersects the plane formed by the surface
    # Get the intersection point if it is the case
    p, segmentLimit = intersection_segment_plane(a1, a2, plane_point, normal)
    if p is None:
        return None, False, False

    # Project intersection in the plane basis
    localp = np.dot(plane_basis.T, p - plane_point)

    # Check if the intersection point is in the polygon on the plane
    inside, borderLimit = is_inside_2D_polygon(localp, corners_2d)
    if not inside:
        p = None
    
    return p, segmentLimit, borderLimit


def is_inside_2D_polygon(p, corners):
    '''
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
    '''
    
    p = np.array(p)
    corners = np.array(corners)
    if (p.shape[0] != 2):
        raise NameError('utilities.isInside2DPolygon input error : p is a ndarray of size 2')
    if (corners.shape[0] != 2 or corners.shape[1] < 3):
        raise NameError('utilities.isInside2DPolygon input error : corners is a 2xN ndarray, N>2')
        
    # Compute p0, which is a point outside the room at x coordinate xMin-1
    # (where xMin is the minimum x coordinate among the corners)
    p0 = np.array([np.amin(corners[0])-1, p[1]])

    is_inside = False

    j = corners.shape[1] - 1
    for i in range(corners.shape[1]):

        # Check first if the point is on the segment
        # We count the border as inside the polygon
        c1c2p = ccw3p(corners[:,i], corners[:,j], p)
        if c1c2p == 0:
            # Here we know that p is co-linear with the two corners
            x_down = min(corners[0,i], corners[0,j])
            x_up = max(corners[0,i], corners[0,j])
            y_down = min(corners[1,i], corners[1,j])
            y_up = max(corners[1,i], corners[1,j])
            if x_down <= p[0] and p[0] <= x_up and y_down <= p[1] and p[1] <= y_up:
                return True, True


        # Now check the intersection using standard algorithm
        c1c2p0 = ccw3p(corners[:,i], corners[:,j], p0)
        if c1c2p == c1c2p0:
            # we know there is no intersection
            j = i
            continue

        pp0c1 = ccw3p(p, p0, corners[:,i])
        pp0c2 = ccw3p(p, p0, corners[:,j])
        if pp0c1 == pp0c2:
            # we know there is no intersection
            j = i
            continue

        # At this point, we know there is an intersection

        # the second condition takes care of horizontal edges and intersection on vertex
        if p[1] < max(corners[1,i], corners[1,j]):
            is_inside = not is_inside

        # circular move around polygon
        j = i

    if is_inside:
        return True, False
    else:
        return False, False

