/* 
 * Geometric routines used in libroom core of pyroomacoustics
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */

/*
 * This file contains geometric routines 
 * - find the orientation of three points in 2D
 * - to find intersection between line segments
 * - to find intersection between line segment and planes
 * - cross product of two 3D vectors
 * - area of 2D polygon
 * - to find whether a point is within or without a polygon
 */
 
#include <iostream>
#include <cmath>
#include "common.hpp"
#include "geometry.hpp"


double clamp(double value, double min, double max)
{
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}


int ccw3p(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2, const Eigen::Vector2f &p3)
{
  /*
     Computes the orientation of three 2D points.

     p1: (array size 2) coordinates of a 2D point
     p2: (array size 2) coordinates of a 2D point
     p3: (array size 2) coordinates of a 2D point

     :returns: (int) orientation of the given triangle
         1 if triangle vertices are counter-clockwise
         -1 if triangle vertices are clockwise
         0 if vertices are collinear

     :ref: https://en.wikipedia.org/wiki/Curve_orientation
     */

  float d = (p2.coeff(0) - p1.coeff(0)) * (p3.coeff(1) - p1.coeff(1)) -
    (p3.coeff(0) - p1.coeff(0)) * (p2.coeff(1) - p1.coeff(1));

  if (d < libroom_eps && d > -libroom_eps)
    return 0;
  else if (d > 0.)
    return 1;
  else
    return -1;
}

int check_intersection_2d_segments(
    const Eigen::Vector2f &a1, const Eigen::Vector2f &a2,
    const Eigen::Vector2f &b1, const Eigen::Vector2f &b2
    )
{
  /*
   * Returns:
   * -1: no intersection
   *  0: proper intersection
   *  1: intersection at endpoint of segment a
   *  2: intersection at endpoint of segment b
   *  3: both of the above at the same time
   */
  int ret = 0;
  int a1a2b1, a1a2b2, b1b2a1, b1b2a2;
  a1a2b1 = ccw3p(a1, a2, b1);
  a1a2b2 = ccw3p(a1, a2, b2);

  if (a1a2b1 == a1a2b2) return -1;

  b1b2a1 = ccw3p(b1, b2, a1);
  b1b2a2 = ccw3p(b1, b2, a2);

  if (b1b2a1 == b1b2a2) return -1;

  // At this point, there is intersection, but we need to check limit cases
  ret = 0;
  if (b1b2a1 == 0 || b1b2a2 == 0) ret |= 1;  // a1 or a2 between (b1,b2)
  if (a1a2b1 == 0 || a1a2b2 == 0) ret |= 2;  // b1 or b2 between (a1, a2)

  return ret;

}

int intersection_2d_segments(
    const Eigen::Vector2f &a1, const Eigen::Vector2f &a2,
    const Eigen::Vector2f &b1, const Eigen::Vector2f &b2,
    Eigen::Ref<Eigen::Vector2f> intersection
    )
{
  /*
    Computes the intersection between two 2D line segments.

    This function computes the intersection between two 2D segments
    (defined by the coordinates of their endpoints) and returns the
    coordinates of the intersection point.
    If there is no intersection, None is returned.
    If segments are collinear, None is returned.
    Two booleans are also returned to indicate if the intersection
    happened at extremities of the segments, which can be useful for limit cases
    computations.

    a1: (array size 2) coordinates of the first endpoint of segment a
    a2: (array size 2) coordinates of the second endpoint of segment a
    b1: (array size 2) coordinates of the first endpoint of segment b
    b2: (array size 2) coordinates of the second endpoint of segment b
    p: (array size 2) coordinates of intersection

    :returns:
    -1: no intersection
  0: proper intersection
    1: intersection at boundaries of segment a
    2: intersection at boundaries of segment b
    3: intersection at boundaries of segment a and b
  */

  int ret = 0;
  float denom, num;

  ret = check_intersection_2d_segments(a1, a2, b1, b2);

  if (ret < 0)  // no intersection
    return ret;

  // normal to a1 <-> a2 segment
  Eigen::Vector2f normal;
  normal.coeffRef(0) = a1.coeff(1) - a2.coeff(1);
  normal.coeffRef(1) = a2.coeff(0) - a1.coeff(0);

  // denominator of the slope
  Eigen::Vector2f db = b2 - b1;
  denom = normal.adjoint() * db;

  if (fabsf(denom) < libroom_eps)
    return -1;

  // Compute intersection point
  num = normal.adjoint() * (a1 - b1);
  intersection = (num / denom) * db + b1;

  return ret;
}

int intersection_3d_segment_plane(
    const Eigen::Vector3f &a1, const Eigen::Vector3f &a2,
    const Eigen::Vector3f &p, const Eigen::Vector3f &normal,
    Eigen::Ref<Eigen::Vector3f> intersection)
{
  /*
     Computes the intersection between a line segment and a plane in 3D.

     This function computes the intersection between a line segment (defined
     by the coordinates of two points) and a plane (defined by a point belonging
     to it and a normal vector). If there is no intersection, -1 is returned.
     If the segment belongs to the surface, -1 is returned.
     If the intersection happened at extremities of the segment 1 is returned, which
     can be useful for limit cases computations. Otherwise 0 is returned.

     a1: (array size 3) coordinates of the first endpoint of the segment
     a2: (array size 3) coordinates of the second endpoint of the segment
     p: (array size 3) coordinates of a point belonging to the plane
     normal: (array size 3) normal vector of the plane
     intersection: (array size 3) array to store the intersection if necessary

     :returns: -1: no intersection
                0: intersection
                1: intersection and one of the end points of the segment is in the plane
    */

  float num=0, denom=0;

  Eigen::Vector3f u = a2 - a1;
  denom = normal.adjoint() * u;

  if (fabsf(denom) > libroom_eps)
  {

    Eigen::Vector3f w = a1 - p;
    num = -normal.adjoint() * w;

    float s = num / denom;

    if (0 - libroom_eps <= s && s <= 1 + libroom_eps)
    {
      // compute intersection point
      intersection = s * u + a1;

      // check limit case
      if (fabsf(s) < libroom_eps || fabsf(s - 1) < libroom_eps)
        return 1;  // a1 or a2 belongs to plane
      else
        return 0;  // plane is between a1 and a2
    }
  }

  return -1;  // no intersection
}


Eigen::Vector3f cross(Eigen::Vector3f v1, Eigen::Vector3f v2)
{
  /* Convenience function to take cross product of two vectors */
  return v1.cross(v2);
}

bool on_segment(const Eigen::Vector2f &c1, const Eigen::Vector2f &c2, const Eigen::Vector2f &p) {
  // Given three collinear points c1, c2, p, the function checks if 
  // point p lies on line segment c1 <-> c2

  float x_down, x_up, y_down, y_up;
  x_down = fminf(c1.coeff(0), c2.coeff(0));
  x_up = fmaxf(c1.coeff(0), c2.coeff(0));
  y_down = fminf(c1.coeff(1), c2.coeff(1));
  y_up = fmaxf(c1.coeff(1), c2.coeff(1));
  return (x_down <= p.coeff(0) && p.coeff(0) <= x_up && y_down <= p.coeff(1) && p.coeff(1) <= y_up);
}

inline int is_left(const Eigen::Vector2f &P0, const Eigen::Vector2f &P1, const Eigen::Vector2f &P2)
{
  // isLeft(): tests if a point is Left|On|Right of an infinite line.
  // on the line is defined as within epsilon
  // Input: three points P0, P1, and P2
  // Return: >0 for P2 left of the line through P0 and P1
  // =0 for P2 on the line
  // <0 for P2 right of the line
  // See: Algorithm 1 "Area of Triangles and Polygons"
  float test_value = (
      (P1.coeff(0) - P0.coeff(0)) * (P2.coeff(1) - P0.coeff(1))
      - (P2.coeff(0) - P0.coeff(0)) * (P1.coeff(1) - P0.coeff(1))
  );

  if (fabsf(test_value) < libroom_eps)
    return 0;
  else if (test_value > 0)
    return 1;
  else
    return -1;
}

int is_inside_2d_polygon(const Eigen::Vector2f &p_test,
    const Eigen::Matrix<float,2,Eigen::Dynamic> &corners)
{
  /*
    Checks if a given point is inside a given polygon in 2D using the winding
    number method.

    This function checks if a point (defined by its coordinates) is inside
    a polygon (defined by an array of coordinates of its corners) by counting
    the number of left intersections between the edges and a half-line starting from
    the test point.

    This is Dave Sunday's winding number algorithm described here:
    http://profs.ic.uff.br/~anselmo/cursos/CGI/slidesNovos/Inclusion%20of%20a%20Point%20in%20a%20Polygon.pdf
    It was modified to explicitely check for points on the edges of the polygon.

    p: (array size 2) coordinates of the point
    corners: (array size 2xN, N>2) coordinates of the corners of the polygon
    n_corners: the number of corners

    returns: 
    -1 : if the point is outside
    0 : the point is inside
    1 : the point is on the boundary
    */
  int n_corners = corners.cols();
  int wn = 0; // the winding number counter
              // loop through all edges of the polygon
  for (int i = 0, j = n_corners-1 ; i < n_corners ; j=i++)
  {
    if (ccw3p(corners.col(j), corners.col(i), p_test) == 0 && on_segment(corners.col(j), corners.col(i), p_test))
    {
      // point is on the edge, we consider it inside
      return 1;
    }
    if (corners.coeff(1,j) <= p_test.coeff(1)) { // start y <= P.y
      if (corners.coeff(1,i) > p_test.coeff(1)) { // an upward crossing
        if (is_left(corners.col(j), corners.col(i), p_test) > 0) // P left of edge
          ++wn; // have a valid up intersect
      }
    } else { // start y > P.y (no test needed)
      if (corners.coeff(1, i) <= p_test.coeff(1)) { // a downward crossing
        if (is_left(corners.col(j), corners.col(i), p_test) < 0) // P right of edge
          --wn; // have a valid down intersect
      }
    }
  }

  if (wn == 0)
    return -1;
  else
    return 0;
}

float area_2d_polygon(const Eigen::Matrix<float, 2, Eigen::Dynamic> &corners)
{
  /*
    Computes the signed area of a 2D surface represented by its corners.
    
    :arg corners: (Eigen::Matrix 2xN, N>2) list of coordinates of the corners forming the surface
    
    :returns: (float) area of the surface
        positive area means anti-clockwise ordered corners.
        negative area means clockwise ordered corners.
   */
  float a = 0;
  for (int c1 = 0 ; c1 < corners.cols() ; c1++)
  {
    int c2 = (c1 == corners.cols() - 1) ? 0 : c1 + 1;
    float base = 0.5 * (corners.coeff(1, c2) + corners.coeff(1, c1));
    float height = corners.coeff(0, c2) - corners.coeff(0, c1);
    a -= height * base;
  }
  return a;
}


float cos_angle_between(
    const Eigen::VectorXf &v1,
    const Eigen::VectorXf &v2)
{
	  
  /* This function computes the cosinus of the angle between two vectors.
   
   v1: array of length N defining the first vector
   v2: array of length N defining the second vector
    
   :returns: a value in [-1;1] representing the cosinus of the angle
     between the two vectors*/

  return clamp(v1.normalized().dot(v2.normalized()), -1., 1.);
}


float dist_line_point(
    const Eigen::VectorXf &start,
    const Eigen::VectorXf &end,
    const Eigen::VectorXf &point)
{
	  
  /* This function computes the smallest distance between a point and an 
   endless line.
    
   start: (array size 2 or 3) defines one point on the line.
   end: (array size 2 or 3) defines a second point on the line.
   point: (array size 2 or 3) defines the point not on the line
    
   :returns: the smallest distance between 'point' and the line defined
     by 'start' and 'end'*/

  Eigen::VectorXf unit_vec = (end - start).normalized(); // vector
  Eigen::VectorXf v = point - start; // vector

  float proj = v.adjoint() * unit_vec; // scalar

  return (v - proj * unit_vec).norm(); // scalar
}						  

