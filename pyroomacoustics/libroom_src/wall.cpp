
#include "wall.hpp"
#include "geometry.hpp"
#include <iostream>
#include <cmath>

template<>
float Wall<2>::area()
{
  return (corners.col(1) - corners.col(0)).norm();
}

template<>
float Wall<3>::area()
{
  return area_2d_polygon(flat_corners);
}

template<size_t D>
float Wall<D>::area()
{
  return 0.;
}

template<>
Wall<2>::Wall(
    const Eigen::Matrix<float,2,Eigen::Dynamic> &_corners,
    float _absorption,
    const std::string &_name
    )
  : absorption(_absorption), name(_name), corners(_corners)
{
  // corners should be D x N for N corners in D dimensions
  dim = 2;  // assign the attribute

  // Pick one of the corners as the origin of the wall
  origin = corners.col(0);

  // compute normal (difference of 2 corners, swap x-y, change 1 sign)
  normal.coeffRef(0) = corners.coeff(1,1) - corners.coeff(1,0);
  normal.coeffRef(1) = corners.coeff(0,0) - corners.coeff(0,1);
  normal = normal.normalized();
}

template<>
Wall<3>::Wall(
    const Eigen::Matrix<float,3,Eigen::Dynamic> &_corners,
    float _absorption,
    const std::string &_name
    )
  : absorption(_absorption), name(_name), corners(_corners)
{
  // corners should be D x N for N corners in D dimensions
  dim = 3;  // assign the attribute

  // In 3D things are a little more complicated
  // We need to compute a 2D basis for the plane and find the normal

  // Pick the origin as the first corner
  origin = corners.col(0);

  // The basis and normal are found by SVD
  Eigen::JacobiSVD<Eigen::Matrix<float,3,Eigen::Dynamic>> svd(corners.colwise() - origin, Eigen::ComputeThinU);

  // The corners matrix should be rank defficient, check the smallest eigen value
  // The rank deficiency is because all the corners are in a 2D subspace of 3D space
  if (svd.singularValues().coeff(2) > libroom_eps)
  {
    fprintf(stderr, "The corners of the wall should lie in a plane");
    throw std::exception();
  }

  // The basis is the leading two left singular vectors
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
  if (a < 0)
  {
    // exchange the other two basis vectors
    basis.rowwise().reverseInPlace();
    flat_corners.colwise().reverseInPlace();
  }

  // Now the normal is computed as the cross product of the two basis vectors
  normal = cross(basis.col(0), basis.col(1));
}

template<>
int Wall<2>::intersection(
    const Eigen::Matrix<float,2,1> &p1,
    const Eigen::Matrix<float,2,1> &p2,
    Eigen::Ref<Eigen::Matrix<float,2,1>> intersection
    )
{
  return intersection_2d_segments(p1, p2, corners.col(0), corners.col(1), intersection);
}

template<>
int Wall<3>::intersection(
    const Eigen::Matrix<float,3,1> &p1,
    const Eigen::Matrix<float,3,1> &p2,
    Eigen::Ref<Eigen::Matrix<float,3,1>> intersection
    )
{
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

  ret1 = intersection_3d_segment_plane(p1, p2, origin, normal, intersection);

  if (ret1 == -1)
    return -1;  // there is no intersection

  if (ret1 == 1)  // intersection at endpoint of segment
    ret = 1;

  /* project intersection into plane basis */
  Eigen::Vector2f flat_intersection = basis.adjoint() * (intersection - origin);

  /* check in flatland if intersection is in the polygon */
  ret2 = is_inside_2d_polygon(flat_intersection, flat_corners);

  if (ret2 < 0)  // intersection is outside of the wall
    return -1;

  if (ret2 == 1) // intersection is on the boundary of the wall
    ret |= 2;

  return ret;  // no intersection
}

template<size_t D>
int Wall<D>::intersects(const Eigen::Matrix<float,D,1> &p1, const Eigen::Matrix<float,D,1> &p2)
{
  Eigen::VectorXf v;
  v.resize(dim);
  return intersection(p1, p2, v);
}

template<size_t D>
int Wall<D>::reflect(const Eigen::Matrix<float,D,1> &p, Eigen::Ref<Eigen::Matrix<float,D,1>> p_reflected)
{
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
template<size_t D>
int Wall<D>::side(const Eigen::Matrix<float,D,1> &p)
{
  // Essentially, returns the sign of the inner product with the normal vector
  float ip = (p - origin).adjoint() * normal;

  if (ip > libroom_eps)
    return 1;
  else if (ip < -libroom_eps)
    return -1;
  else
    return 0;
}

template<size_t D>
bool Wall<D>::same_as(const Wall & that)
{
  /*
  Checks if two walls are the same, based on the corners of the walls.
  Be careful : it will return true for two identical walls that belongs
  to two different rooms !
  */

  if (dim != that.dim)
  {
    std::cerr << "The two walls are not of the same dimensions !" << std::endl;
    throw std::exception();
    return false;
  }

  // Not the same number of corners
  if (corners.cols() != that.corners.cols())
  {
    return false;
  }

  return (corners - that.corners).cwiseAbs().sum() == 0.;
}

