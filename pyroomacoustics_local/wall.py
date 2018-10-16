# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np

from . import geometry as geom



class Wall(object):
    '''
    This class represents a wall instance. A room instance is formed by these.
    
    :attribute corners: (np.array dim 2x2 or 3xN, N>2) endpoints forming the wall
    :attribute absorption: (float) attenuation reflection factor
    :attribute name: (string) name given to the wall, which can be reused to reference it in the Room object
    :attribute normal: (np.array dim 2 or 3) normal vector pointing outward the room
    :attribute dim: (int) dimension of the wall (2 or 3, meaning 2D or 3D)
    '''
    
    
    def __init__(
        self,
        corners,
        absorption = 1.,
        name = None):
        
        self.corners = np.array(corners, order='F', dtype=np.float32)
        self.absorption = absorption

        # set first corner as origin of plane
        self.plane_point = np.array(self.corners[:,0])

        if (self.corners.shape == (2, 2)):
            self.normal = np.array([(self.corners[:, 1] - self.corners[:, 0])[1], (-1)*(self.corners[:, 1] - self.corners[:, 0])[0]])
            self.dim = 2
        elif (self.corners.shape[0] == 3 and self.corners[0].shape[0] > 2):
            # compute the normal assuming the vertices are aranged counter
            # clock wise when the normal defines "up"
            i_min = np.argmin(self.corners[0,:])
            i_prev = i_min - 1 if i_min > 0 else self.corners.shape[1] - 1
            i_next = i_min + 1 if i_min < self.corners.shape[1] - 1 else 0
            self.normal = np.cross(self.corners[:, i_next] - self.corners[:, i_min], 
                                   self.corners[:, i_prev] - self.corners[:, i_min])
            self.dim = 3

            # Compute a basis for the plane and project the corners into that basis
            self.plane_basis = np.zeros((3,2), order='F', dtype=np.float32)
            localx = np.array(self.corners[:,1]-self.plane_point)
            self.plane_basis[:,0] = localx/np.linalg.norm(localx)
            localy = np.array(np.cross(self.normal, localx))
            self.plane_basis[:,1] = localy/np.linalg.norm(localy)
            self.corners_2d = np.concatenate((
                [ np.dot(self.corners.T - self.plane_point, self.plane_basis[:,0]) ], 
                [ np.dot(self.corners.T - self.plane_point, self.plane_basis[:,1]) ]
                ))
            self.corners_2d = np.array(self.corners_2d, order='F', dtype=np.float32)
        else:
            raise NameError('Wall.__init__ input error : corners must be an np.array dim 2x2 or 3xN, N>2')
        self.normal = self.normal/np.linalg.norm(self.normal)
        if (name is not None):
            self.name = name

    def intersection(self, p1, p2):
        '''
        Returns the intersection point between the wall and a line segment.
        
        :arg p1: (np.array dim 2 or 3) first end point of the line segment
        :arg p2: (np.array dim 2 or 3) second end point of the line segment
        
        :returns: (np.array dim 2 or 3 or None) intersection point between the wall and the line segment
        '''
        
        p1 = np.array(p1)
        p2 = np.array(p2)
    
        if (self.dim == 2):
            return geom.intersection_2D_segments(p1, p2, self.corners[:,0], self.corners[:,1])
            
        if (self.dim == 3):
            return geom.intersection_segment_polygon_surface(p1, p2, self.corners_2d, self.normal, self.plane_point, self.plane_basis)
        
    def intersects(self, p1, p2):
        '''
        Tests if the given line segment intersects the wall.
        
        :arg p1: (ndarray size 2 or 3) first endpoint of the line segment
        :arg p2: (ndarray size 2 or 3) second endpoint of the line segment
        
        :returns: (tuple size 3)
            (bool) True if the line segment intersects the wall
            (bool) True if the intersection happens at a border of the wall
            (bool) True if the intersection happens at the extremity of the segment
        '''
        
        if (self.dim == 2):
            intersection, borderOfSegment, borderOfWall = geom.intersection_2D_segments(p1, p2, self.corners[:,0], self.corners[:,1])

        if (self.dim == 3):
            intersection, borderOfSegment, borderOfWall = geom.intersection_segment_polygon_surface(p1, p2, self.corners_2d, self.normal,
                                                                                                 self.plane_point, self.plane_basis)

        if intersection is None:
            intersects = False
        else:
            intersects = True
            
        return intersects, borderOfWall, borderOfSegment
            
    def side(self, p):
        '''
        Computes on which side of the wall the point p is.
        
        :arg p: (np.array dim 2 or 3) coordinates of the point
        
        :returns: (int) integer representing on which side the point is
            -1 : opposite to the normal vector (going inside the room)
            0 : on the wall
            1 : in the direction of the normal vector (going outside of the room)
        '''
        
        p = np.array(p)
        if (self.dim != p.shape[0]):
            raise NameError('Wall.side input error : dimension of p and the wall must match.')

        return geom.side(p, self.corners[:,0], self.normal)
