"""
In this example, we compute the path followed by a ray emitted in an L-shape 2D room
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra
import math
import scipy
from scipy.io import wavfile
from scipy import signal
import os
from joblib import Parallel, delayed
import sys
from pyroomacoustics.utilities import fractional_delay


PI = 3.141592653589793

# ==================== VECTOR SPACE FUNCTIONS ====================


def norm(v):
    """
    Computes the norm of a vector
    :param v: an array of length 2 or 3 representing a vector
    :return: a positive scalar : the norm of v
    """
    if len(v) == 2:
        return math.sqrt(v[0]*v[0] + v[1]*v[1])
    if len(v) == 3:
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    raise ValueError("The norm(v) function only supports 2 or 3 long vectors")


def normalize(v):
    """
    Returns the unit vector of the vector.
    :param v: an array of length 2 or 3 representing a vector
    :return: the same vector but with a magnitude of 1
    """

    vnorm = norm(v)

    if len(v) == 2:
        return v[0]/vnorm, v[1]/vnorm
    if len(v) == 3:
        return v[0]/vnorm, v[1]/vnorm, v[2]/vnorm

    raise ValueError("The normalize(v) function only supports 2 or 3 long vectors.")


def dist(p1, p2):
    """
    Returns the euclidean distance between p1 and p2
    :param p1: an array of length 2 or 3 representing the first point
    :param p2: an array of length 2 or 3 representing the second point
    :return: a double, the euclidean distance between p1 and p2
    """
    if len(p1) == len(p2) and len(p1) == 2:
        return math.sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) )

    if len(p1) == len(p2) and len(p1) == 3:
        return math.sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]))

    raise ValueError("Function dist(p1,p2) only supports vectors of same length (2 or 3).")


def dot(v1, v2):
    """
    Computes the dot product of 2 2-dim array
    :param v1: an array of length 2 or 3 representing a vector
    :param v2: an array of length 2 or 3 representing a vector
    :return: a scalar representing the dot product v1.v2
    """
    if len(v1) == len(v2) and len(v1) == 2:
        return v1[0]*v2[0] + v1[1]*v2[1]

    if len(v1) == len(v2) and len(v1) == 3:
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    raise ValueError("Function dot(v1,v2) only supports vectors of same length (2 or 3).")


def scale(v, k):
    """
    Scales the vector v by k
    :param v: an array of length 2 or 3 defining the  vector
    :param k: the scalar factor
    :return: an array of length 2 or 3 defining k*v
    """

    if len(v) == 2:
        return k*v[0], k*v[1]

    if len(v) == 3:
        return k*v[0], k*v[1], k*v[2]

    raise ValueError("Function scalar_prod(v,k) only supports vector of length 2 or 3.")


def substract(v1, v2):
    """
    Computes the substraction of 2 vectors
    :param v1: an array of length 2 or 3 defining the first vector
    :param v2: an array of length 2 or 3 defining the second vector
    :return: an array of length 2 or 3 defining v1 - v2
    """

    if len(v1) == len(v2) and len(v1) == 2:
        return v1[0]-v2[0], v1[1]-v2[1]

    if len(v1) == len(v2) and len(v1) == 3:
        return v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]

    raise ValueError("Function substract(v1,v2) only supports vectors of same length (2 or 3).")


def add(v1, v2):
    """
    Computes the addition of 2 vectors
    :param v1: an array of length 2 or 3 defining the first vector
    :param v2: an array of length 2 or 3 defining the second vector
    :return: an array of length 2 or 3 defining v1 + v2
    """

    if len(v1) == len(v2) and len(v1) == 2:
        return v1[0]+v2[0], v1[1]+v2[1]

    if len(v1) == len(v2) and len(v1) == 3:
        return v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]

    raise ValueError("Function add(v1,v2) only supports vectors of same length (2 or 3).")


def make_vector(start_point, end_point):
    """
    Computes the vector going from the starting point to the end point
    :param start_point: an array of length 2 or 3 representing the start point
    :param end_point: an array of length 2 or 3 representing the destination point
    :return: an array of the same length than the parameters representing a vector going from start to end
    """
    if len(start_point) == len(end_point) and len(end_point) == 2:
        return end_point[0] - start_point[0], end_point[1] - start_point[1]

    if len(start_point) == len(end_point) and len(end_point) == 3:
        return end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2]

    raise ValueError("Function make_vector(p1,p2) only supports vectors of same length (2 or 3).")


def reverse_vector(v):
    """
    Computes the vector going in the opposite direction than v
    :param v: an array of length 2 or 3 representing a vector to be reversed
    :return: an array of length 2 or 3 representing a vector going in the opposite direction than v
    """

    if len(v) == 2:
        return -v[0], -v[1]

    if len(v) == 3:
        return -v[0], -v[1], -v[2]

    raise ValueError("Function reverse_vector(v) only supports a vector of length 2 or 3.")


def clip(value, down, up):
    """
    Clips the value in parameter to the value down or up
    :param value: a scalar, the value to clip
    :param down: a scalar, the minimum value to be accepted
    :param up: a scalar, the maximum value to be accepted
    :return: a scalar :     - value if value is in [down, up]
                            - down if value < down
                            - up if value > up
    """
    if value < down : return down
    if value > up : return up
    return value


def equation(p1, p2):
    """
    Computes 'a' and 'b' coefficients in the expression y = a*x +b for the line defined by the two points in argument
    :param p1: a 2 dim array representing a first point on the line
    :param p2: a 2 dim array representing an other point on the line
    :return: the coefficients 'a' and 'b' that fully describe the line algebraically
    """

    a = (p2[1]-p1[1])/(p2[0]-p1[0])

    return a, p1[1] - a*p1[0]


def get_quadrant(vec):
    """
    Outputs the quadrant that the vector in parameter belongs to
    :param vec: a 2D vector
    :return: an integer:
                - 1 if the vector (starting from (0,0)) belongs to the first quadrant ([0, pi/2])
                - 2 if the vector (starting from (0,0)) belongs to the second quadrant ([pi/2, pi])
                - 3 if the vector (starting from (0,0)) belongs to the third quadrant ([pi, 3pi/2])
                - 4 if the vector (starting from (0,0)) belongs to the last quadrant ([3pi/2, 2pi])
    """

    if len(vec) != 2:
        raise ValueError("The function get_quadrant(vec) only supports a vector of length 2")

    if vec[0] >= 0:
        if vec[1] >= 0:
            return 1
        return 4

    if vec[1] >= 0:
        return 2
    return 3


def angle_between(v1, v2):
    """
    Returns the angle in radians between two 2D vectors 'v1' and 'v2'

    :param v1: an N dim array representing the first vector
    :param v2: an N dim array representing the first vector
    :return: the angle formed by the two vectors. WARNING : the angle is not signed, hence it belongs to [0,pi]
    """

    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return math.acos(clip(dot(v1_u, v2_u), -1.0, 1.0))


def cross_product(u, v):
    """
    Computes the cross product of 2 vectors of length 3
    :param u: an array of length 3 representing the first vector
    :param v: an array of length 3 representing the second vector
    :return: an array of length 3 representing the cross product of the two input vectors
    """

    if len(u) != 3 or len(v) != 3 :
        raise ValueError("The function cross_product(v1,v2) only supports vectors of length 3 ")

    return u[1]*v[2]-v[1]*u[2], v[0]*u[2] - u[0]*v[2], u[0]*v[1]-v[0]*u[1]


def get_phi_theta(v):

    """
    2D : returns the phi angle that describes the vector v in polar coordinates.
    3D : returns the phi and theta angles that describe the vector v in sperical coordinates.
    :param v: an array of length 2 or 3 defining the vector to be considered.
    :return: an array  [phi, theta]

                - phi is in [0, 2*pi] and describes the direction of the vector on the (x,y) plane
                - theta is in [0, pi] and describes the elevation angle with respect to the z axis
                   (when in 2D, theta is pi/2 by default).
                   WARNING : when theta is 0 or PI, then the phi angle is not well defined and should not be used"""
                   # There should be no problem since those angles are used only in the 'compute_segment_end' function
                   # where everything is multiplied by sin(theta)=0 which 'kills' the term in function of phi.

    # It is essential to normalize the vector
    nv = normalize(v)

    if len(nv) == 2:

        phi = math.acos(nv[0]) if nv[1] > 0 else (-1)*math.acos(nv[0])
        return [phi, PI/2]

    if len(nv) == 3:

        theta = math.acos(nv[2])

        if theta == PI or theta == 0.:
            return [0., theta ] # WARNING : here nv is perpendicular to (x,y) plane
                                # The phi angle is not defined !

        val = clip(nv[0]/math.sin(theta), -1., 1.)
        phi = math.acos(val) if nv[1] > 0 else (-1)*math.acos(val)

        return [phi, theta]



# ==================== ALGO FUNCTIONS ====================


def get_max_distance(room):
    """
    Computes the maximum distance that a ray could possibly cover inside the room without hitting a wall.
    Every ray will be seen as a segment of this length+1, so that it is sure that it hits at least one wall.
    This allows us to use the wall.intersection function to compute the hitting point

    :arg room: the room that is studied, can be in 2D or 3D
    :returns: a double corresponding to the max distance
    """

    # NOTE : Storage of corners coordinates for each wall :
    # [[x_corner0, x_corner1, x_corner2, x_corner3]
    # [y_corner0, y_corner1, y_corner2, y_corner3]
    # [z_corner0, z_corner1, z_corner2, z_corner3]...]


    # Every 2D wall has 2 corners (2D) or 4 corners (3D)
    # So for every wall in 2 we take the largest x and y among the 2 corners
    # So for every wall in 3D we take the largest x, y and z among the 4 corners
    largest_xy = np.array([np.ndarray.max(w.corners, 1) for w in room.walls])

    # Return a new point made of the maxX, maxY, maxZ (in case of 3D room)
    largest_point =  np.ndarray.max(largest_xy, 0)


    # Do the same as above for the smallest values of x, y and z
    smallest_xy = np.array([np.ndarray.min(w.corners, 1) for w in room.walls])
    smallest_point =  np.ndarray.min(smallest_xy, 0)

    return dist(largest_point, smallest_point) + 1


def compute_segment_end(start, length, phi, theta=PI/2):
    """
    Computes the end point of a segment, given its starting point, its angle and its length
    :param start: an array of length 2 or 3 defining the starting position
    :param length: the length of the segment
    :param phi: the angle (rad) defining the direction of the segment
                in the (x,y) plane with respect to the reference vector [x=1, y=0]
    :param theta: the angle (rad) defining the elevation of the segment in the 3rd dimension
                when theta belongs to [0 ; pi/2] the segment goes upward
                when theta belongs to [pi/2 ; pi] the segment goes downward
    :return: an array of length 2 or 3 containing the end point of the segment
    """

    if len(start) == 2:
        return [start[0] + length*np.cos(phi), start[1] + length*np.sin(phi)]

    if len(start) == 3:
        # Spherical coordinates
        return [start[0] + length*np.sin(theta)*np.cos(phi), start[1] + length*np.sin(theta)*np.sin(phi), start[2] + length*np.cos(theta)]

    raise ValueError("Function compute_segment_end() only supports points of dimension 2 or 3.")


def same_wall(w1, w2):
    """
    Returns True if both walls are the same
    :param w1: the first wall
    :param w2: the second wall
    :return: True if they are the same, False otherwise
    """

    if w1 is None:
        return False

    c1 = w1.corners
    c2 = w2.corners

    if len(c1) != len(c2) or len(c1[0]) != len(c2[0]):
        return False


    for k in range(len(c1)):
        for l in range(len(c1[0])):

            if c1[k][l] != c2[k][l]:
                return False


    return True


def get_intersected_walls(start, end, room, previous_wall):
    """
    This methode returns a list with all the walls the the [start, end] segment will intersect.
    Except from the case where start is the source, start is the last hit point on a wall. We make sure that this previous
    wall is not included in the output list
    :param start: An array of length 2 or 3 representing the start of the segment
    :param end: An array of length 2 or 3 representing the end of the segment
    :param room: The room that is studied
    :param previous_wall: The wall where the start is located. If start is the source (first iteration) then
                            previous_wall is None
    :return: a list containing all wall objects that are intersected by the segment
    """

    intersected_walls = []


    # We collect the walls that are intersected and that are not the previous wall
    for w in room.walls:

        # Boolean conditions
        different_than_previous = previous_wall is not None   and   not same_wall(previous_wall, w)
        w_intersects_segment = w.intersects(start, end)[0]

        # Candidate walls for first hit
        if w_intersects_segment and (previous_wall is None or different_than_previous):
            intersected_walls = intersected_walls + [w]

    return intersected_walls


def next_wall_hit(start, end, room, previous_wall):
    """
    Finds the next wall that will be hit by the ray (represented as a segment here) and outputs the hitting point.
    For non-shoebox rooms, there may be several walls intersected by the ray.
    In this case we compute the intersection points for all those walls and only keep the closest point to the start.
    :param start: an array of length 2 or 3 representing the starting point of the ray
    :param end: an array of length 2 or 3 representing the end point of the ray. Recall that thanks to get_max_distance, we are sure
                that there is at least one wall between start and end.
    :param room: the room in which the ray propagates
    :param previous_wall : a wall object representing the last wall that the ray has hit.
                            It is None before the first hit.
    :return: an array with three elements
                - a 2 dim array representing the place where the ray hits the next wall
                - the distance between start and the hit point
                - the wall that is going to be hit
                NOTE : Those 3 elements will be None in case no wall intersection has been found
                (in case of rounding errors when hit points are close to a 3D corner)
    """

    intersected_w = get_intersected_walls(start, end, room, previous_wall)

    # If no wall has been intersected, there might be a rounding error
    # WARNING : in this case just stop tracing the ray
    if len(intersected_w) == 0:
        print("Note : One ray did not intersect any wall (rounding error).")
        sys.stdout.flush()
        return [None, None, None]


    # If only 1 wall is intersected
    if len(intersected_w) == 1:
        intersection = intersected_w[0].intersection(start, end)[0]
        return intersection, dist(intersection, start), intersected_w[0]

    # If we are here it means that several walls have been intersected (non shoebox room)
    intersections = [w.intersection(start, end)[0] for w in intersected_w]
    dist_from_start = [dist(start, p) for p in intersections]

    # Returns the closest point to 'start', ie. the one corresponding to the correct wall
    correct_wall = np.argmin(dist_from_start)
    return intersections[correct_wall], dist_from_start[correct_wall], intersected_w[correct_wall]


def compute_new_end(start, hit_point, wall_normal, segment_length):
    """
    This function computes the reflection of the ray on the wall and outputs the end point of this segment,
    located at a distance 'segment_length' of the hit_point
    :param start: an array of length 2 or 3 representing the point that originated the ray before the hit.
                This point is either the previous hit point, or the source of the sound (for the first iteration).
    :param hit_point: an array of length 2 or 3 representing the intersection between the ray and the wall
    :param wall_normal: an array of length 2 or 3 representing the normal vector of the wall
    :param segment_length: a double representing a distance that cannot be travelled inside the room without hurting a wall
    :return: an array of length 2 or 3 representing the end point of the segment representing the reflection of the ray on a wall
    """


    d = normalize(make_vector(start, hit_point))
    reversed_incident = normalize(reverse_vector(d))

    if angle_between(reversed_incident, wall_normal) > PI / 2:
        wall_normal = reverse_vector(wall_normal)

    n = normalize(wall_normal)

    # ref : https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    reflected = normalize(substract(d, scale(n,2*dot(d,n))))
    reflected = scale(reflected, segment_length)

    return add(hit_point, reflected)

# ==================== MICROPHONE FUNCTIONS ====================


def dist_line_point(start, end, point):
    """
    Computes the distance between a segment and a point
    :param start: an array of length 2 or 3 defining the starting point of the segment
    :param end: an array of length 2 or 3 defining the end point of the segment
    :param point: an array of length 2 or 3 the point that we want to know the distance to the segment
    :return: the distance between the point and the segment
    """

    # 2D case
    if len(start) == len(end) and len(start) == len(point) and len(start) == 2 :

        if start[0] == end[0]:  # Here we cannot use algebra since the segment is vertical
            return abs(point[0] - start[0])

        a, b = equation(start, end)

        return abs(point[1] - a*point[0] - b) / math.sqrt(a*a + 1)

    # 3D case
    if len(start) == len(end) and len(start) == len(point) and len(start) == 3:
        return norm(cross_product(substract(point,start),substract(point,end))) / norm(substract(end,start))

    raise ValueError("The function dist_line_point only supports arrays of the same size (2 or 3).")


def intersects_mic(start, end, center, radius):
    """
    Returns True iff the segment between the points 'start' and 'end' intersect the circle defined by center and radius
    :param start: an array of length 2 or 3 defining the starting point of the segment
    :param end: an array of length 2 or 3 defining the end point of the segment
    :param center: an array of length 2 or 3 defining the center point of the circle
    :param radius: the radius of the microphone
    :return: True if the segment intersects the circle, False otherwise
    """

    start_end_vect = make_vector(start, end)
    start_center_vect = make_vector(start, center)

    end_start_vect = reverse_vector(start_end_vect)
    end_center_vect = make_vector(end, center)

    # Boolean
    intersection = dist_line_point(start, end, center) <= radius

    # Boolean to check that the microphone is "between" start and end
    # Handle situations of non-convex rooms where the mic can be very close to
    # the infinite line (start, end) without being reachable by the ray
    between_start_and_end = angle_between(start_end_vect, start_center_vect) < PI/2 \
                  and angle_between(end_start_vect, end_center_vect) < PI/2

    return intersection and between_start_and_end


def mic_intersection(start, end, center, radius):
    """
    Computes the intersection point between a segment and a circle/sphere (depending on the dimensionality of the room.
    As there are often 2 such points, this function returns only the closest point to the start of the segment,
    ie the point where the ray is detected by the circular receiver.
    The function also outputs the distance between this point and the start of the segment in order
    to facilitate the computation of air absoption and travelling time.
    :param start: an array of length 2 or 3 defining the starting point of the segment
    :param end: an array of length 2 or 3 defining the end point of the segment
    :param center: an array of length 2 or 3 array defining the center of the circle (detector)
    :param radius: the radius of the circle or sphere
    :return: an array of length 2 : [intersection, distance]
                - intersection is an array of length 2 or 3 defining the intersection between the segment and the microphone
                - distance is the euclidean distance between 'intersection' and 'start'
    """


    def solve_quad(A, B, C):
        """
        Computes the real roots of a quadratic polynomial defined as Ax² + Bx + C = 0
        :param A: a real number
        :param B: a real number
        :param C: a real number
        :return: a 2 dim array [x1,x2] containing the 2 roots of the polynomial.
                - If the polynomial has only 1 real solution x1, then the function returns [x1,x1]
        """

        delta = B * B - 4 * A * C
        if delta >= 0:
            return (-B + math.sqrt(delta)) / (2 * A), (-B - math.sqrt(delta)) / (2 * A)

        return -B/(2*A), -B/(2*A)

        # Note : Due to rounding errors, delta is sometimes negative (close to zero)
        # In those cases I approximate it to be 0

    # 2D case
    if len(start) == len(end) and len(start) == len(center) and len(start) == 2:
        p, q = center

        if start[0] == end[0]:  # When the segment is vertical, we already know the x coordinate of the intersection
            A = 1
            B = -2 * q
            C = q * q + (start[0] - p) * (start[0] - p) - radius * radius
            x1 = start[0]
            x2 = start[0]
            y1, y2 = solve_quad(A, B, C)

        else:   # See the formula on the first answer :
                # https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle

            m, c = equation(start, end)

            A = m * m + 1
            B = 2 * (m * c - m * q - p)
            C = (q * q - radius * radius + p * p - 2 * c * q + c * c)

            x1, x2 = solve_quad(A, B, C)

            y1 = m * x1 + c
            y2 = m * x2 + c

        d1 = dist([x1, y1], start)
        d2 = dist([x2, y2], start)

        return [[x1, y1], d1] if d1 <= d2 else [[x2, y2], d2]

    # 3D case
    if len(start) == len(end) and len(start) == len(center) and len(start) == 3:

        # Following https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        # Checked with : http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
        l = normalize(make_vector(start, end))
        o = start
        c = center
        r = radius

        # We are searching d such that a point x on the sphere is also on the line
        # ie : x = o + k*l  (recall l has unit norm)

        o_c = substract(o,c)

        delta = dot(l, o_c)**2 - norm(o_c)**2 + r*r

        if delta > 0:
            k1 = (-1) * dot(l, o_c) + math.sqrt(delta)
            k2 = (-1) * dot(l, o_c) - math.sqrt(delta)

            inter1 = add(o, scale(l, k1))
            d1 = dist(inter1, start)

            inter2 = add(o, scale(l, k2))
            d2 = dist(inter2, start)

            return [inter1, d1] if d1 <= d2 else [inter2, d2]

        # The delta should never be negative since we call this function once
        # we know that there is indeed an intersection.
        # (rounding error can lead to negative delta close to zero)
        k = (-1) * dot(l, o_c)
        inter = add(o, scale(l, k))
        return [inter, dist(inter, start)]



    raise ValueError("The function mic_intersection() only supports points of same dimension (2 or 3)")


# ==================== TIME ENERGY FUNCTIONS ====================


def update_travel_time(previous_travel_time, current_hop_length, speed):
    """
    Computes the total travel time of the ray knowing the travel time to reach the previous hit_point, the new distance
    to travel, and the speed of the ray
    :param previous_travel_time: the total travel time from the source to the previous hit_point (seconds)
    :param current_hop_length: the distance between previous hit_point and next hit_point (meters)
    :param speed: the sound speed (may depend on several factors) (meters/seconds)
    :return: the total travel time from the source to next hit_point (seconds)
    """
    return previous_travel_time + current_hop_length / speed


def distance_attenuation(previous_energy, new_dist, total_dist):
    """
    Computes the distance attenuation of the sound
    :param previous_energy: The energy the ray had at its last hit point
    :param new_dist: the distance that the ray has to travel between the last hit point and its next hit point X
    :param total_dist: the total distance travelled by the ray from the beginning to its next hit_point X
    :return: the new energy of the ray


    """

    # Arbitrary choice : no attenuation before 1 meter
    if total_dist < 1.:
        return previous_energy

    # For first wall hit
    if total_dist==new_dist:
        return previous_energy / total_dist

    # For next wall hits
    return previous_energy * (total_dist-new_dist) / float(total_dist)


def wall_absorption(previous_energy, wall):
    return previous_energy * math.sqrt(1 - wall.absorption)


def stop_ray(actual_travel_time, time_thresh, actual_energy, energy_thresh=0.1, sound_speed=340.):
    """
    Returns True if the ray must be stopped according to the 'which' condition
    :param actual_travel_time: total travel time of the ray from the source to the point of evaluation
    :param time_thresh: the maximum travel time for the ray
    :return:
    """
    return actual_travel_time > time_thresh or actual_energy/(sound_speed*actual_travel_time) < energy_thresh


def compute_scat_energy(energy, scatter_coef, wall, start, hit_point, mic_pos):
    """
    Computes the scattering energy
    :param energy: The energy of the ray after its rebound at hit_point
    :param scatter_coef: The scattering coefficient
    :param wall: the wall where hit_point is located
    :param start: an array of length 2 or 3 defining the previous hit point of the ray
    :param hit_point: an array of length 2 or 3 defining the actual hit_point of the ray
    :param mic_pos: an array of length 2 or 3 defining the center of the microphone
    :return: the amount of energy of the scattering ray, which depends on the angle between
            the [hit_point, mic_pos] vector and the wall normal
    """

    # First we need to check that the wall normal is pointing inside the room
    # Recall the angle between the reversed incident vector and the normal is less than pi/2
    # in the case where the normal points inside the room
    n = wall.normal
    if angle_between(n, make_vector(hit_point, start)) > PI/2:
        n = reverse_vector(n)

    # Now we must compute the angle between the normal and the vector hit_point -> mic_pos
    hit_to_mic = make_vector(hit_point, mic_pos)

    return energy * scatter_coef * np.cos(angle_between(n,hit_to_mic))


def scattering_ray(room,
                   last_wall,
                   last_hit,
                   mic_pos,
                   scat_energy,
                   actual_travel_time,
                   time_thresh,
                   total_dist,
                   sound_speed,
                   plot=False):
    """
    Trace a one-hop scattering ray from the last wall hit to the microphone
    :param room: The room in which the ray propagates
    :param last_wall: The wall object where last_hit is located
    :param last_hit: An array of length 2 or 3 defining the last wall hit position
    :param mic_pos: An array of length 2 or 3 defining the position of the microphone
    :param scat_energy: The energy of the scattering ray
    :param actual_travel_time: The cumulated travel time of the ray
    :param time_thresh: The time threshold of the ray
    :param sound_speed: The speed of sound
    :param plot: a boolean defining if we must plot the scattered ray or not
    :return: a  tuple (travel_time, remaining_energy)
                - travel_time is the time it took the ray to go from the source to the microphone.
                - remaining_energy is the amount of energy that the ray has left when it hits the microphone

    """
    intersects_no_wall = len(get_intersected_walls(last_hit, mic_pos, room, last_wall)) == 0

    # In case the scattering way can reach the mic with a straight line
    if intersects_no_wall:

        hit_point, distance = mic_intersection(last_hit, mic_pos, mic_pos, mic_radius)
        total_dist += distance
        #scat_energy = distance_attenuation(scat_energy, distance, total_dist)
        travel_time = update_travel_time(actual_travel_time, distance, sound_speed)

        if not stop_ray(travel_time, time_thresh, scat_energy):
            if plot:
                draw_segment(last_hit, hit_point, red=False)

            return [[travel_time, scat_energy]]

    return []


# ==================== DRAWING FUNCTIONS ====================


def draw_point(pos,  marker='o', markersize=10, color="red" ):
    """
    Draw a point on a plot at position pos
    :param pos: an array of length 2 or 3 representing the position of the point
    :param marker: a char representing the shape of the marker to use
    :param markersize: an int representing the size of the marker
    :param color: a string representing the color of the marker
    :return: Nothing
    """
    if len(pos) == 2:
        plt.plot([pos[0]], [pos[1]], marker=marker, markersize=markersize, color=color)

    elif len(pos) == 3:
        plt.plot([pos[0]], [pos[1]], [pos[2]], marker=marker, markersize=markersize, color=color)

    else :
        raise ValueError("The function draw_point can only draw a 2D or 3D point")


def draw_segment(source, hit, red=True):

    option = 'ro-' if red else 'bo-'

    if len(source) == len(hit) and len(source) == 2:
        plt.plot([source[0], hit[0]], [source[1], hit[1]], option)

    elif len(source) == len(hit) and len(source) == 3:
        plt.plot([source[0], hit[0]], [source[1], hit[1]], [source[2], hit[2]], option)

    else:
        raise ValueError("The function draw_segment only supports points of same dimension (2 or 3)")


def highpass(audio, fs, cutoff=200, butter_order=5):
    nyq = 0.5 * fs
    fc_norm = cutoff / nyq
    b, a = signal.butter(butter_order, fc_norm, btype="high", analog=False)
    return signal.lfilter(b, a, audio)

# ==================== SIMULATION ====================


def simul_ray(room,
              ray_segment_length,
              init_phi,
              init_energy,
              mic_pos,
              mic_radius,
              scatter_coef,
              init_theta = PI/2,
              time_thres = 0.3,  # seconds
              sound_speed=340.,
              plot = False):
    """
    Simulate the evolution a ray emitted by the source with an initial angle and an inital energy in a room.
    :param room: a room object in which the ray propagates
    :param ray_segment_length: the length of the segment that ensures that a wall will be intersected by the ray
    :param init_phi: the angle (rad) with respect to the vector [1, 0] which gives its first 2D direction to the ray
    :param init_energy: the inital amount of energy of the ray
    :param mic_pos: a 2 dim array representing the position of the circular microphone
    :param mic_radius: the radius of the circular microphone (meters)
    :param scatter_coef: the scattering coefficient of the walls
    :param init_theta: the angle (rad) which gives its first "elevation" to the ray
                        - if init_theta = 0 rad : the ray goes vertically up
                        - if init_theta = pi/2 : the ray is parallele to the (x,y) plane
                        - if init_theta = pi : the ray goes vertically down
    :param time_thres: the time threshold. If the travelling time of the ray exceeds this value, then the ray disappears
    :param sound_speed: the speed of sound (meters/sec)
    :param plot : a boolean that controls if the ray is going to be plotted or not
                - IMPORTANT : if plot=True then the function room.plot() must be called before simul_ray, and the function plt.plot() must be called after simul_ray
    :return: a tuple (travel_time, remaining_energy)
                - travel_time is the time it took the ray to go from the source to the microphone.
                - remaining_energy is the amount of energy that the ray has left when it hits the microphone
    """



    phi = init_phi
    theta = init_theta

    # First segment
    start = room.sources[0].position
    end = compute_segment_end(start, ray_segment_length, phi, theta=theta)


    energy = init_energy

    wall = None
    travel_time = 0.
    total_dist = 0.

    # To be filled with tuples (time, energy) for the main ray and all scattered rays that will hit the receiver.
    output = []

    if plot:
        draw_point(start, color="green")


    while True:

        hit_point, distance, wall = next_wall_hit(start, end, room, wall)

        # If no hit point is found (in rare cases of rounding errors), then stop tracing the ray
        if hit_point is None :
            break


        # If the ray hits the microphone
        if intersects_mic(start, hit_point, mic_pos, mic_radius):

            hit_point, distance = mic_intersection(start, hit_point, mic_pos, mic_radius)
            total_dist += distance
            #energy = distance_attenuation(energy, distance, total_dist)
            travel_time = update_travel_time(travel_time, distance, sound_speed)

            if not stop_ray(travel_time, time_thres, energy) :
                if plot:
                    draw_segment(start, hit_point)
                    draw_point(hit_point, markersize=3, color='purple')

                # Ajouter des brackets si on parallelise pas
                output = output + [[travel_time, energy]]

            break


        # Update the ray's variable
        total_dist += distance
        #energy = distance_attenuation(energy, distance, total_dist)

        travel_time = update_travel_time(travel_time, distance, sound_speed)

        # If the ray cannot reach the next hit_point on 'wall'
        if stop_ray(travel_time, time_thres, energy):
            if plot:
                draw_point(start, marker='o', color='blue')
                draw_point(hit_point, marker = 'x', color='blue')
            break

        # Energy after wall hit
        energy = wall_absorption(energy, wall)

        # Let's apply the scattering coefficient
        if scatter_coef > 0:
            energy_scat = compute_scat_energy(energy, scatter_coef, wall, start, hit_point, mic_pos)
            energy -= energy_scat

            # Add the scattered ray to the output (if there is no wall between hit_point and mic_pos)
            output = output + scattering_ray(room, wall, hit_point, mic_pos, energy_scat, travel_time, time_thres, total_dist, sound_speed)


        if plot:
            draw_segment(start, hit_point)


        # ==== Update for next rebound ====

        end = compute_new_end(start, hit_point, wall.normal, ray_segment_length)
        start = hit_point.copy()


    return output


def get_rir_rt(room,
               nb_phis,
               time_thres,
               mic_pos,
               mic_radius,
               scatter_coef,
               init_energy = 1000,
               nb_thetas=1,
               sound_speed = 340.,
               plot_RIR=False,
               plot_rays=False):

    """
    :param room: a room object in which the ray propagates
    :param nb_phis: the number of different 2D directions used to compute the RIR
    :param time_thres: the time threshold. If the travelling time of the ray exceeds this value, then the ray disappears
    :param init_energy: the inital amount of energy of the ray
    :param mic_pos: a 2 dim array representing the position of the circular microphone
    :param mic_radius: the radius of the circular microphone (meters)
    :param scatter_coef: the scattering coefficient of the walls
    :param nb_thetas: the number of different elevation per flat angle phi
    :param sound_speed: the speed of sound (meters/sec)
    :param plot_RIR : the RIR will be plotted only if this boolean is True
    :param plot_rays : a boolean that controls if the ray is going to be plotted or not
                - IMPORTANT : if plot_rays=True then the function room.plot() must be called before simul_ray, and the function plt.plot() must be called after simul_ray
    :return: The Room Impulse Response computed for the microphone
    """

    # ========== TRACE ALL THE RAYS ==========

    if dist(mic_pos, room.sources[0].position) <= mic_radius:
        raise ValueError("The source is in the microphone !")

    phis = np.linspace(1, 2 * PI, nb_phis)
    thetas = np.linspace(0, PI, nb_thetas)  # For 3D rooms
    parallelize = not plot_rays

    if room.dim == 2:
        thetas = [PI/2]

    nb_rays = nb_thetas*nb_phis
    max_dist = get_max_distance(room)


    if plot_rays:
        room.plot(img_order=1)
        draw_point(mic_pos,marker='x')

    str_scat = "(no scattering)"
    if scatter_coef > 0:
        str_scat = "(with scattering)"

    # Compute all the possible pairs of theta and phi
    # => Parallelize on all those pairs
    angles = [(p,t) for p in phis for t in thetas]

    print("\nSet up done. Starting Ray Tracing", str_scat)
    print("Nb phi =", nb_phis, " | Nb theta =", nb_thetas)
    start_time = time.time()

    # To store info about rays that reach the mic
    log = []

    if parallelize:
        log = log + Parallel(n_jobs=2)(delayed(simul_ray)(room,
                                                          max_dist,
                                                          phi,
                                                          init_energy,
                                                          mic_pos,
                                                          mic_radius,
                                                          scatter_coef,
                                                          init_theta=theta,
                                                          time_thres=time_thres,
                                                          sound_speed=sound_speed,
                                                          plot=plot_rays) for (phi, theta) in angles)

        log = [item for sublist in log for item in sublist]

    else:
        for index, phi in enumerate(phis):

            if index % ((nb_phis // 100) + 1) == 0:
                print("\r", 100 * index // nb_phis, "%", end='', flush=True)

            for theta in thetas:
                log = log + simul_ray(room,
                                      max_dist,
                                      phi,
                                      init_energy,
                                      mic_pos,
                                      mic_radius,
                                      scatter_coef,
                                      init_theta=theta,
                                      time_thres=time_thres,
                                      sound_speed=sound_speed,
                                      plot=plot_rays)


    print("\rDone.")
    print("Running time for", nb_rays, "rays:", time.time() - start_time)

    print("log length =", len(log))
    if plot_rays:
        plt.show()


    # ===== PUT EVERYTHING TOGETHER TO COMPUTE RIR ======

    TIME = 0
    ENERGY = 1

    use_frac = True

    if use_frac:

        #======= PART WITH FRACTIONAL DELAY ========
        fdl = pra.constants.get('frac_delay_length')
        fdl2 = (fdl - 1) // 2  # Integer division

        ir = np.zeros(int(time_thres*room.fs) + fdl)


        for entry in log:
            time_ip = int(np.floor(entry[TIME]*room.fs))

            if time_ip > len(ir)-fdl2 or time_ip < fdl2:
                continue

            time_fp = (entry[TIME]*room.fs) - time_ip

            # We apply the distance attenuation here
            ir[time_ip - fdl2:time_ip + fdl2 + 1] += (entry[ENERGY] * fractional_delay(time_fp)) / (sound_speed*entry[TIME])


    else:
        # ======= PART WITHOUT FRACTIONAL DELAY ========

        ir = np.zeros(int(time_thres * room.fs) + 1)
        for entry in log:
            time_ip = int(np.floor(entry[TIME] * room.fs))

            if time_ip > len(ir):
                continue

            # We apply the distance attenuation
            ir[time_ip] += entry[ENERGY] / (sound_speed*entry[TIME])


    if plot_RIR:
        x = np.arange(len(ir)) / room.fs
        plt.figure()
        plt.plot(x, ir)
        plt.title("RIR")
        plt.show()

    return ir


def apply_rir(rir, wav_data, cutoff, fs=16000, result_name="result.wav"):
    """
    This function applies a RIR to sound data coming from a .wav file
    The result is written in the current directory
    :param rir: the room impulse response
    :param wav_data: an array of data with wav format
    :param fs: the sampling frequency used to write the result in local directory
    :param result_name: the name of the resulting .wav file
    :return: Nothing
    """

    # Compute the convolution and set all coefficients between -1 and 1 (range for float32 .wav files)
    result = scipy.signal.fftconvolve(rir, wav_data)

    if cutoff > 0:
        result = highpass(result, fs, cutoff)

    result /= np.abs(result).max()
    result -= np.mean(result)
    wavfile.write(result_name, rate=fs, data=result.astype('float32'))


def wall_area(wall):

    """Computes the area of a 3D planar wall.
    :param wall: the wall object that is defined in the 3D space"""

    # Algo : http://geomalgorithms.com/a01-_area.

    # Recall that the wall corners have the following shape :
    # [  [x1, x2, ...], [y1, y2, ...], [z1, z2, ...]  ]

    c = wall.corners
    n = normalize(wall.normal)


    if len(c) != 3 :
        raise ValueError("The function wall_area3D only supports ")


    sum_vect = [0.,0.,0.]
    num_vertices = len(c[0])

    for i in range(num_vertices):
        sum_vect = add(sum_vect, cross_product(c[:,(i-1)%num_vertices], c[:,i]))

    return abs(dot(n,sum_vect))/2.


def get_volume(room):

    """
    Computes the volume of a room
    :param room: the room object
    :return: the volume in cubic unit
    """

    wall_sum = 0.

    for w in room.walls :

        n = normalize(w.normal)
        one_point = w.corners[:,0]

        wall_sum += dot(n, one_point) * wall_area(w)

    return wall_sum / 3.


def get_total_abs(room):
    """
    Computes the total absorption of a room in Sabins.
    This value is defined as the sum{absorption*area} over all the walls
    Here the units for the walls dimensions are meters
    :param room: the room that is considered
    :return: the total absorption of the wall in sabins
    """

    return sum([w.absorption*wall_area(w) for w in room.walls])

def get_RT60(room):
    """
    Computes the RT60 reverberation time for a room with Sabine's formula
    Warning : the unit for the room dimensions must be meters
    :param room: The considered room
    :return: the RT60 reverberation time approximation derived with Sabine's formula
    """

    return 0.161*get_volume(room)/get_total_abs(room)

# ==================== ROOM SETUP ====================

_3D = True
max_order = 8

nb_phis = 10
nb_thetas = 10 if _3D else 1

scatter_coef = 0.1
absor = 0.1
ray_simul_time = 0.8

mic_radius = 0.05  # meters

fs0, audio_anechoic = wavfile.read(os.path.join(os.path.dirname(__file__),"input_samples", 'arctic_a0010.wav'))


## Decomment in case the audio file has several channels
#audio_anechoic = audio_anechoic[:,0]
audio_anechoic = audio_anechoic-np.mean(audio_anechoic)

# Lshape room
pol = 8*np.array([[0., 0.], [0., 3.], [5., 3.], [5., 1.], [3.,1.], [3.,0.]]).T

# Very long room
#pol = np.array([[0., 0.], [0., 20.], [10., 20.], [10., 0.]]).T




d= "3D" if _3D else "2D"

if _3D:

    # Add the circular microphone
    mic_pos = np.array([3.5, 2., 0.5])
    source = [1., 1., 0.5]

    # Create the room from its corners
    room = pra.Room.from_corners(pol,fs=16000, max_order=max_order, absorption=absor)
    room.extrude(2., absorption=absor)

    # Add a source somewhere in the room
    room.add_source(source, signal=audio_anechoic)

    ## To perform ISM
    # R = np.array([[3.5], [2.], [0.5]])  # [[x], [y], [z]]
    # room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

else:

    # Add the circular microphone
    mic_pos = np.array([2.,2.])

    source = [1,1]

    # Create the room from its corners
    room = pra.Room.from_corners(pol,fs=16000, max_order=max_order, absorption=absor)

    # Add a source somewhere in the room
    room.add_source(source, signal=audio_anechoic)

    R = np.array([[2.5], [2.5]])  # [[x], [y]]
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


# ==================== MAIN ====================


rir_rt = get_rir_rt(room, nb_phis, ray_simul_time, mic_pos, mic_radius, scatter_coef, nb_thetas=nb_thetas, plot_rays=False, plot_RIR=True)

apply_rir(rir_rt, audio_anechoic, cutoff=0., fs = fs0, result_name='aaa.wav')


for w in room.walls :
    print("Area :", wall_area(w))

print ("Volume :", get_volume(room))

print("Total abs :", get_total_abs(room))
print("RT60 : %.3f" % get_RT60(room))

## === COMPUTING THE RUNNING TIME
# ray_number = np.array([10,20,30,40,50,60,70,80,90,100])
# abcisse = (ray_number/2)**2
#
# nb_total_rays = [4306, 18149, 40092, 71580, 111105, 159857, 217429, 283640, 358910, 443121]
#
# plt.figure()
# plt.plot(abcisse, nb_total_rays, 'ro-')
# plt.xlabel("Number of rays")
# plt.ylabel("Number of entries in the log")
# plt.title("Number of rays reaching the microphone (including scattered rays)")
# plt.show()



# T = [0.] * len(ray_number)
#
# for k, elem in enumerate(ray_number):
#
#     start = time.time()
#     get_rir_rt(room, elem/2., ray_simul_time, init_energy, mic_pos, mic_radius, scatter_coef, nb_thetas=elem/2.,
#                plot_rays=False, plot_RIR=False)
#     T[k] = time.time()-start
#
# plt.figure()
# plt.plot(abcisse, T, 'ro-')
# plt.xlabel("Number of rays")
# plt.ylabel("Time [s]")
# plt.title("Evolution of running time in function of the number of rays")
# plt.show()