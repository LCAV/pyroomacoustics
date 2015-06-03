# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import numpy as np
from utilities import ccw3p



class Wall(object):
    """
    This class represents a wall instance. A room instance is formed by theses.
    
    :attribute corners: (np.array dim 2x2 or 3xN, N>2) endpoints forming the wall
    :attribute absorption: (float) attenuation reflection factor
    :attribute name: (string) name given to the wall, which can be reused to reference it in the Room object
    :attribute normal: (np.array dim 2 or 3) normal vector pointing outward the room
    :attribute dim: (int) dimension of the wall (2 or 3, meaning 2D or 3D)
    """
    
    
    def __init__(
        self,
        corners,
        absorption = 1.,
        name = None):
        
        self.corners = np.array(corners)
        self.absorption = absorption
        if (self.corners.shape == (2, 2)):
            self.normal = np.array([(self.corners[:, 1] - self.corners[:, 0])[1], (-1)*(self.corners[:, 1] - self.corners[:, 0])[0]])
            self.dim = 2
        elif (self.corners.shape[0] == 3 and self.corners[0].shape[0] > 2):
            normal = np.cross(self.corners[:, 1] - self.corners[:, 0], self.corners[:, 2] - self.corners[:, 0])
            self.dim = 3
        else:
            raise NameError('Wall.__init__ input error : corners must be an np.array dim 2x2 or 3xN, N>2')
        self.normal = self.normal/np.linalg.norm(self.normal)
        if (name is not None):
            self.name = name

    def intersection(self, p1, p2):
        """
        Returns the intersection point between the wall and a line segment.
        
        :arg p1: (np.array dim 2 or 3) first end point of the line segment
        :arg p2: (np.array dim 2 or 3) second end point of the line segment
        
        :returns: (np.array dim 2 or 3) intersection point between the wall and the line segment
            can return NaN values and crash if there is no or infinite number of intersection points
        """
        
        p1 = np.array(p1)
        p2 = np.array(p2)
    
        if (self.dim == 2):
            if (p1.shape[0] != 2 or p2.shape[0] != 2):
                raise NameError('Wall.intersection input error : requires two 2D points.')
            da = self.corners[:, 1]-self.corners[:, 0]
            db = p2-p1
            dp = self.corners[:, 0]-p1
            dap = np.empty_like(da)
            dap[0] = -da[1]
            dap[1] = da[0]
            denom = np.dot(dap, db)
            num = np.dot(dap, dp )
            return (num/denom.astype(float))*db + p1
            
        if (self.dim == 3):
            raise NameError('Wall.intersection error : not implemented for 3D walls!')
        
    def intersects(self, p1, p2):
        """
        Tests if the given line segment intersects the wall.
        
        :arg p1: (np.array dim 2 or 3) first endpoint of the line segment
        :arg p2: (np.array dim 2 or 3) second endpoint of the line segment
        
        :returns: (int)
            0 (False) = not intersecting
            1 (True) = intersecting (standard case)
            2 (True) = intersecting at the extremity of the line segment (might be at the extremity of the wall as well)
            3 (True) = intersecting at the extremity of the wall
            4 (True) = collinear
        """
        
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        if (self.dim == 2):
            if (p1.shape[0] != 2 or p2.shape[0] != 2):
                raise NameError('Wall.intersects input error : requires two 2D points.')
            if (p1[0] == p2[0] and p1[1] == p2[1]):
                raise NameError('Wall.intersects input error : points must be different.')
     
            # Build all possible triangles from the points forming the line segments
            t1 = np.array([[self.corners[0, 0], self.corners[0, 1], p1[0]], [self.corners[1, 0], self.corners[1, 1], p1[1]]])
            t2 = np.array([[self.corners[0, 0], self.corners[0, 1], p2[0]], [self.corners[1, 0], self.corners[1, 1], p2[1]]])
            t3 = np.array([[p1[0], p2[0], self.corners[0, 0]], [p1[1], p2[1], self.corners[1, 0]]])
            t4 = np.array([[p1[0], p2[0], self.corners[0, 1]], [p1[1], p2[1], self.corners[1, 1]]])
        
            # Compute the orientation of the triangles
            d1 = ccw3p(t1)
            d2 = ccw3p(t2)
            d3 = ccw3p(t3)
            d4 = ccw3p(t4)
        
            # Standard intersection case
            if (((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0))):
                return 1

            # Special case (collinear)
            elif (d1==0 and d2==0 and d3==0 and d4==0):
                return 4
            
            # Special case (extremity of a segment)
            elif (d1==0 or d2==0 or d3==0 or d4==0):
                if (d1==0):
                    endpoint1 = np.array([self.corners[0,0], self.corners[1,0]])
                    endpoint2 = np.array([self.corners[0,1], self.corners[1,1]])
                    midpoint = np.array([p1[0], p1[1]])
                    ret = 2
                elif (d2==0):
                    endpoint1 = np.array([self.corners[0,0], self.corners[1,0]])
                    endpoint2 = np.array([self.corners[0,1], self.corners[1,1]])
                    midpoint = np.array([p2[0], p2[1]])
                    ret = 2
                elif (d3==0):
                    endpoint1 = np.array([p1[0], p1[1]])
                    endpoint2 = np.array([p2[0], p2[1]])
                    midpoint = np.array([self.corners[0,0], self.corners[1,0]])
                    ret = 3
                elif (d4==0):
                    endpoint1 = np.array([p1[0], p1[1]])
                    endpoint2 = np.array([p2[0], p2[1]])
                    midpoint = np.array([self.corners[0,1], self.corners[1,1]])
                    ret = 3
                if ((min(endpoint1[0], endpoint2[0]) <= midpoint[0] <= max(endpoint1[0], endpoint2[0])) and
                    (min(endpoint1[1], endpoint2[1]) <= midpoint[1] <= max(endpoint1[1], endpoint2[1]))):
                    return ret
            return 0
            
        if (self.dim == 3):
            raise NameError('Wall.intersects error : not implemented for 3D walls!')
            
    def side(self, p):
        """
        Computes on which side of the wall the point p is.
        
        :arg p: (np.array dim 2 or 3) coordinates of the point
        
        :returns: (int) integer representing on which side the point is
            -1 : opposite to the normal vector (going inside the room)
            0 : on the wall
            1 : in the direction of the normal vector (going outside of the room)
        """
        
        p = np.array(p)
        if (self.dim != self.corners[0].shape[0]):
            raise NameError('Wall.side input error : dimension of p and the wall must match.')
        
        projection = np.dot(self.normal, (p - self.corners[0]))
        if (projection > 0):
            return 1
        elif (projection < 0):
            return -1
        else:
            return 0