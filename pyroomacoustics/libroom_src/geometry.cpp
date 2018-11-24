/*
 * This file contains geometric routines 
 * - find the orientation of three points in 2D
 * - to find intersection between line segments
 * - to find intersection between line segment and planes
 * - cross product of two 3D vectors
 * - area of 2D polygon
 * - to find whether a point is within or without a polygon
 *
 * (c) 2018 Robin Scheibler
 * MIT License
 */
 
#include <iostream>
#include <cmath>
#include "utility.hpp"
#include "geometry.hpp"


int ccw3p(const Eigen::VectorXf &p1, const Eigen::VectorXf &p2, const Eigen::VectorXf &p3)
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

  float d = (p2.coeff(0) - p1.coeff(0)) * (p3.coeff(1) - p1.coeff(1))
    - (p3.coeff(0) - p1.coeff(0)) * (p2.coeff(1) - p1.coeff(1));

  if (d < libroom_eps && d > -libroom_eps)
    return 0;
  else if (d > 0.)
    return 1;
  else
    return -1;
}

int check_intersection_2d_segments(
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &b1, const Eigen::VectorXf &b2
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
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &b1, const Eigen::VectorXf &b2,
    Eigen::Ref<Eigen::VectorXf> intersection
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
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &p, const Eigen::VectorXf &normal,
    Eigen::Ref<Eigen::VectorXf> intersection)
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
    num = - normal.adjoint() * w;

    float s = num / denom;

    if (0-libroom_eps <= s && s <= 1+libroom_eps)
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

int is_inside_2d_polygon(const Eigen::Vector2f &p,
    const Eigen::MatrixXf &corners)
{
  /*
    Checks if a given point is inside a given polygon in 2D.

    This function checks if a point (defined by its coordinates) is inside
    a polygon (defined by an array of coordinates of its corners) by counting
    the number of intersections between the borders and a segment linking
    the given point with a computed point outside the polygon.
    A boolean is also returned to indicate if a point is on a border of the
    polygon (the point is still considered inside), which can be useful for
    limit cases computations.

    p: (array size 2) coordinates of the point
    corners: (array size 2xN, N>2) coordinates of the corners of the polygon
    n_corners: the number of corners

    returns: 
    -1 : if the point is outside
    0 : the point is inside
    1 : the point is on the boundary
    */

  if (corners.rows() != 2)
  {
    std::cerr << "Only 2D polygons are supported" << std::endl;
    throw std::exception();
  }

  if (corners.cols() < 3)
  {
    std::cerr << "The polygon should have more than 2 points" << std::endl;
    throw std::exception();
  }

  bool is_inside = false;  // initialize point not in the polygon
  int c1c2p, c1c2p0, pp0c1, pp0c2;
  int n_corners = corners.cols();

  // find a point outside the polygon
  int i_min;
  corners.row(0).minCoeff(&i_min);
  Eigen::Vector2f p_out;
  p_out.resize(2);
  p_out.coeffRef(0) = corners.coeff(0,i_min) - 1;
  p_out.coeffRef(1) = p.coeff(1);

  /*
  std::cout << "The corners: " << corners << std::endl;
  std::cout << "The point: " << p << std::endl;
  std::cout << "Aux point: " << p_out << std::endl;
  */


  // Now count intersections
  for (int i = 0, j = n_corners-1 ; i < n_corners ; j=i++)
  {

    // Check first if the point is on the segment
    // We count the border as inside the polygon
    c1c2p = ccw3p(corners.col(i), corners.col(j), p);
    if (c1c2p == 0)
    {
      // Here we know that p is co-linear with the two corners
      float x_down, x_up, y_down, y_up;
      x_down = fminf(corners.coeff(0,i), corners.coeff(0,j));
      x_up = fmaxf(corners.coeff(0,i), corners.coeff(0,j));
      y_down = fminf(corners.coeff(1,i), corners.coeff(1,j));
      y_up = fmaxf(corners.coeff(1,i), corners.coeff(1,j));
      if (x_down <= p.coeff(0) && p.coeff(0) <= x_up && y_down <= p.coeff(1) && p.coeff(1) <= y_up)
        return 1;
    }

    // Now check intersection with standard method
    c1c2p0 = ccw3p(corners.col(i), corners.col(j), p_out);
    if (c1c2p == c1c2p0)  // no intersection
      continue;

    pp0c1 = ccw3p(p, p_out, corners.col(i));
    pp0c2 = ccw3p(p, p_out, corners.col(j));
    if (pp0c1 == pp0c2)  // no intersection
      continue;

    // at this point we are sure there is an intersection

    // the second condition takes care of horizontal edges and intersection on vertex
    float c_max = fmaxf(corners.coeff(1,i), corners.coeff(1,j));
    if (p.coeff(1) + libroom_eps < c_max)
    {
      /*
      std::cout << "wall " << j << " " << i 
        << " p[1]=" << p.coeff(1) << " c1[1]=" << corners.coeff(1,i)
        << " c2[1]=" << corners.coeff(1,j) << " c_max=" << c_max
        << " p[1] < c_max:" << (p.coeff(1) < c_max)
        << " diff=" << (c_max - p.coeff(1)) << std::endl;
      */
      is_inside = !is_inside;
    }

  }

  // for a odd number of intersections, the point is in the polygon
  if (is_inside)
    return 0;  // point strictly inside
  else
    return -1; // point is outside
}


float area_2d_polygon(const Eigen::MatrixXf &corners)
{
  /*
    Computes the signed area of a 2D surface represented by its corners.
    
    :arg corners: (Eigen::Matrix 2xN, N>2) list of coordinates of the corners forming the surface
    
    :returns: (float) area of the surface
        positive area means anti-clockwise ordered corners.
        negative area means clockwise ordered corners.
   */
  if (corners.rows() != 2)
  {
    std::cerr << "Only 2D polygons are supported" << std::endl;
    throw std::exception();
  }

  if (corners.cols() < 3)
  {
    std::cerr << "The corners should have more than 2 points" << std::endl;
    throw std::exception();
  }

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


float angle_between_3D(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2)
{
	/* Function that takes 2 vectors with 3 components and computes the angle between them.
	 * The result is in radians and belongs to the unsigned [0;pi] interval*/

	Eigen::Vector3f v1n = v1.normalized();
	Eigen::Vector3f v2n = v2.normalized();
		
	return 	acos(clamp(v1n.dot(v2n), -1., 1.));
}

float angle_between_2D(const Eigen::Vector2f &v1, const Eigen::Vector2f &v2)
{
	/* Function that takes 2 vectors with 2 components and computes the angle between them.
	 * The result is in radians and belongs to the unsigned [0;pi] interval*/
	 
	Eigen::Vector2f v1n = v1.normalized();
	Eigen::Vector2f v2n = v2.normalized();
		
	return 	acos(clamp(v1n.dot(v2n), -1., 1.));
}





















