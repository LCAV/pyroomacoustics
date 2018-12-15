#include "wall.hpp"
#include "geometry.hpp"
#include <iostream>
#include <cmath>

Wall::Wall(const Eigen::MatrixXf & _corners, float _absorption,
  const std::string & _name): absorption(_absorption), name(_name), corners(_corners) {
	  
  // corners should be D x N for N corners in D dimensions
  dim = corners.rows(); // assign the attribute
  int n_corners = corners.cols();

  // Sanity check
  if (dim == 2 && n_corners != 2) {
    fprintf(stderr, "2D walls have only two corners.\n");
    throw std::exception();
    
  } else if (dim < 2 || dim > 3) {
    fprintf(stderr, "Only 2D and 3D walls are supported.\n");
    throw std::exception();
  }

  // Construction is different for 2D and 3D
  if (dim == 2) {
	  
    // Pick one of the corners as the origin of the wall
    origin.resize(2);
    origin = corners.col(0);

    // compute normal (difference of 2 corners, swap x-y, change 1 sign)
    normal.resize(2);
    normal.coeffRef(0) = corners.coeff(1, 1) - corners.coeff(1, 0);
    normal.coeffRef(1) = corners.coeff(0, 0) - corners.coeff(0, 1);
    normal = normal.normalized();
    
  } else if (dim == 3) {
	  
    // In 3D things are a little more complicated
    // We need to compute a 2D basis for the plane and find the normal

    // Pick the origin as the first corner
    origin = corners.col(0);

    // The basis and normal are found by SVD
    Eigen::JacobiSVD < Eigen::MatrixXf > svd(corners.colwise() - origin, Eigen::ComputeThinU);

    // The corners matrix should be rank defficient, check the smallest eigen value
    if (svd.singularValues().coeff(2) > libroom_eps) {
      fprintf(stderr, "The corners of the wall should lie in a plane");
      throw std::exception();
    }

    // The basis is the leading two left singular vectors
    basis.resize(dim, 2);
    basis.col(0) = svd.matrixU().col(0);
    basis.col(1) = svd.matrixU().col(1);

    // The normal correspons to the smallest singular value
    normal = svd.matrixU().col(2);

    // Project the 3d corners into 2d plane
    flat_corners = basis.adjoint() * (corners.colwise() - origin);

    // Our convention is that the vertices are arranged counter-clockwise
    // around the normal. In that case, the area computation should be positive.
    // If it is positive, we need to swap the basis.
    float a = area();
    if (a < 0) {
      // exchange the other two basis vectors
      basis.rowwise().reverseInPlace();
      flat_corners.colwise().reverseInPlace();
    }

    // Now the normal is computed as the cross product of the two basis vectors
    normal = cross(basis.col(0), basis.col(1));
  }
}

float Wall::area() {
  if (dim == 2) {
    return (corners.col(1) - corners.col(0)).norm();
  } else // dim == 3
  {
    return area_2d_polygon(flat_corners);
  }
}

int Wall::intersection(
  const Eigen::VectorXf & p1,
  const Eigen::VectorXf & p2,
  Eigen::Ref < Eigen::VectorXf > intersection) {
	  
  if (p1.size() != dim || p2.size() != dim) {
    std::cerr << "The wall and points dimensionalities differ" << std::endl;
    std::cerr << "  - p1: " << p1 << std::endl;
    std::cerr << "  - p2: " << p2 << std::endl;
    throw std::exception();
  }

  if (dim == 2) {
    return intersection_2d_segments(p1, p2, corners.col(0), corners.col(1), intersection);
  } else // dim == 3
  {
    return _intersection_segment_3d(p1, p2, intersection);
  }

  return -1;
}

int Wall::intersects(const Eigen::VectorXf & p1,
  const Eigen::VectorXf & p2) {
  Eigen::VectorXf v;
  v.resize(dim);
  return intersection(p1, p2, v);
}

int Wall::reflect(const Eigen::VectorXf & p, Eigen::Ref < Eigen::VectorXf > p_reflected) {
  /*
   * Reflects point p across the wall 
   *
   * wall: a wall object (2d or 3d)
   * p: a point in space
   * p_reflected: a pointer to a buffer large enough to receive
   *              the location of the reflected point
   *
   * Returns: 1 if reflection is in the same direction as the normal
   *          0 if the point is within tolerance of the wall
   *         -1 if the reflection is in the opposite direction of the normal
   */

  // projection onto normal axis
  float distance_wall2p = normal.adjoint() * (origin - p);

  // compute reflected point
  p_reflected = p + 2 * distance_wall2p * normal;

  if (distance_wall2p > libroom_eps)
    return 1;
  else if (distance_wall2p < -libroom_eps)
    return -1;
  else
    return 0;
}

/* checks on which side of a wall a point is */
int Wall::side(const Eigen::VectorXf & p) {
  // Essentially, returns the sign of the inner product with the normal vector
  float ip = (p - origin).adjoint() * normal;

  if (ip > libroom_eps)
    return 1;
  else if (ip < -libroom_eps)
    return -1;
  else
    return 0;
}

int Wall::_intersection_segment_3d( // intersection routine specialized for 3D
  const Eigen::VectorXf & a1,
  const Eigen::VectorXf & a2,
  Eigen::Ref < Eigen::VectorXf > intersection) {
  /*
    Computes the intersection between a line segment and a polygon surface in 3D.

    This function computes the intersection between a line segment (defined
    by the coordinates of two points) and a surface (defined by an array of
    coordinates of corners of the polygon and a normal vector)
    If there is no intersection, None is returned.
    If the segment belongs to the surface, None is returned.
    Two booleans are also returned to indicate if the intersection
    happened at extremities of the segment or at a border of the polygon,
    which can be useful for limit cases computations.

    a1: (array size 3) coordinates of the first endpoint of the segment
    a2: (array size 3) coordinates of the second endpoint of the segment
    corners: (array size 3xN, N>2) coordinates of the corners of the polygon
    normal: (array size 3) normal vector of the surface
    intersection: (array size 3) store the intersection point

    :returns: 
           -1 if there is no intersection
            0 if the intersection striclty between the segment endpoints and in the polygon interior
            1 if the intersection is at endpoint of segment
            2 if the intersection is at boundary of polygon
            3 if both the above are true
    */

  int ret1, ret2, ret = 0;

  ret1 = intersection_3d_segment_plane(a1, a2, origin, normal, intersection);

  if (ret1 == -1)
    return -1; // there is no intersection

  if (ret1 == 1) // intersection at endpoint of segment
    ret = 1;

  /* project intersection into plane basis */
  Eigen::Vector2f flat_intersection = basis.adjoint() * (intersection - origin);

  /* check in flatland if intersection is in the polygon */
  ret2 = is_inside_2d_polygon(flat_intersection, flat_corners);

  if (ret2 < 0) // intersection is outside of the wall
    return -1;

  if (ret2 == 1) // intersection is on the boundary of the wall
    ret |= 2;

  return ret; // no intersection
}

bool Wall::same_as(const Wall & that) {

  if (dim != that.dim) {
    std::cerr << "The two walls are not of the same dimensions !" << std::endl;
    throw std::exception();
  }

  // Not the same number of corners
  if (corners.cols() != that.corners.cols()) {
    return false;
  }

  return (corners - that.corners).cwiseAbs().sum() == 0.;
}






















