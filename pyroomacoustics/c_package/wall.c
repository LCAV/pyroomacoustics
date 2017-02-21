
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "room.h"

float eps = 1e-5;

wall_t *new_wall(int dim, int n_corners, float *corners, float absorption)
{
  int i;

  // Sanity check
  if (dim == 2 && n_corners != 2)
  {
    fprintf(stderr, "2D walls have only two corners.\n");
    return NULL;
  }
  else if (dim < 2 || dim > 3)
  {
    fprintf(stderr, "Only 2D and 3D walls are supported.\n");
    return NULL;
  }

  // allocate new wall
  wall_t *wall = (wall_t *)malloc(sizeof(wall_t));
  wall->dim = dim;
  wall->absorption = absorption;
  wall->n_corners = n_corners;

  // copy over the corners
  wall->corners = (float *)malloc(dim * n_corners * sizeof(float));
  for (i = 0 ; i < dim * n_corners ; i++)
    wall->corners[i] = corners[i];


  if (dim == 2)
  {
    // compute normal (difference of 2 corners, swap x-y, change 1 sign)
    wall->normal[0] = wall->corners[3] - wall->corners[1];
    wall->normal[1] = wall->corners[0] - wall->corners[2];
    normalize(wall->normal, wall->dim);
    wall->flat_corners = NULL;

    // Pick one of the corners as the origin of the wall
    for (i = 0 ; i < dim ; i++)
      wall->origin[i] = wall->corners[i];
  }
  else if (dim == 3)
  {
    // In 3D things are a little more complicated
    // We need to compute a 2D basis for the plane and find the normal
    // For that we first need to find a point of the convex hull,
    // we take the corner with smallest x
    int i_min = 0, i_prev, i_next;
    for (i = dim ; i < dim * n_corners ; i += dim)
      if (wall->corners[i] < wall->corners[i_min])
        i_min = i;
    // get previous and next, taking care of wraparound
    i_prev = (i_min == 0) ? (n_corners - 1) * dim : i_min - dim;
    i_next = (i_min == (n_corners - 1) * dim) ? 0 : i_min + dim;

    // Save the (non-orthogonal at this point) basis for the wall plane
    for (i = 0 ; i < dim ; i++)
    {
      wall->origin[i] = wall->corners[i_min+i];
      wall->basis[i] = wall->corners[i_next+i] - wall->origin[i];
      wall->basis[i+dim] = wall->corners[i_prev+i] - wall->origin[i];
    }

    // orthogonalize
    gram_schmidt(wall->basis, 2, dim);

    // compute the normal with cross product
    cross(wall->basis, wall->basis + dim, wall->normal);
    
    // Project the 3d corners into 2d plane
    wall->flat_corners = (float *)malloc(2 * n_corners * sizeof(float));
    float tmp[3];
    for (i = 0 ; i < n_corners ; i++)
    {
      int d;
      // difference with origin
      for (d = 0 ; d < 3 ; d++)
        tmp[d] = wall->corners[i * 3 + d] - wall->origin[d];

      // project to plane basis
      for (d = 0 ; d < 2 ; d++)
        wall->flat_corners[i * 2 + d] = inner(tmp, wall->basis + d * 3, 3);
    }
  }

  return wall;
}

void free_wall(wall_t *wall)
{
  // free all allocated memory
  free(wall->corners);

  if (wall->dim == 3)
    free(wall->flat_corners);

  free(wall);
}

int ccw3p(float *p1, float *p2, float *p3)
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

  float d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);

  if (d < eps && d > -eps)
    return 0;
  else if (d > 0.)
    return 1;
  else
    return -1;
}

/* checks if line segment (p1, p2) intersects the wall */
int wall_intersection(wall_t *wall, float *p1, float *p2, float *intersection)
{
  /*
   * Compute intersection between line segment (p1, p2) and wall (2D or 3D)
   *
   * Returns
   * -1 : no intersection
   *  0 : proper intersection
   *  1 : intersection point is an endpoint of the segment
   *  2 : intersection point is on the border of the wall
   *  3 : both of the above at the same time
   */
  if (wall->dim == 2)
    return intersection_2d_segments(p1, p2, wall->corners, wall->corners+2, intersection);
  else if (wall->dim == 3)
    return intersection_segment_wall_3d(p1, p2, wall, intersection);
  else
    fprintf(stderr, "Walls can only be 2D or 3D.\n");

  return -1;
}

int wall_reflect(wall_t *wall, float *p, float *p_reflected)
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

  int i;
  float distance_wall2p;

  // vector from wall to point
  for (i = 0 ; i < wall->dim ; i++)
    p_reflected[i] = wall->origin[i] - p[i];

  // projection onto normal axis
  distance_wall2p = inner(wall->normal, p_reflected, wall->dim);

  // compute reflected point
  for (i = 0 ; i < wall->dim ; i++)
    p_reflected[i] = p[i] + 2 * distance_wall2p * wall->normal[i];

  if (distance_wall2p > eps)
    return 1;
  else if (distance_wall2p < -eps)
    return -1;
  else
    return 0;
}

/* checks on which side of a wall a point is */
int wall_side(wall_t *wall, float *p)
{
  // Essentially, returns the sign of the inner product with the normal vector
  int i;
  float vec[3];

  for (i = 0 ; i < wall->dim ; i++)
    vec[i] = p[i] - wall->origin[i];

  float ip = inner(vec, wall->normal, wall->dim);

  if (ip > eps)
    return 1;
  else if (ip < -eps)
    return -1;
  else
    return 0;
}

int check_intersection_2d_segments(float *a1, float *a2, float *b1, float *b2)
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

int intersection_2d_segments(float *a1, float *a2, float *b1, float *b2, float *p)
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
    1: intersection at boundaries of segment a
    2: intersection at boundaries of segment b
    3: intersection at boundaries of segment a and b
  */

  int ret = 0;
  float normal[2], db[2], dp[2];
  float denom, num, slope;

  ret = check_intersection_2d_segments(a1, a2, b1, b2);

  if (ret < 0)  // no intersection
    return ret;

  // normal
  normal[0] = a1[1] - a2[1];
  normal[1] = a2[0] - a1[0];

  db[0] = b2[0] - b1[0];
  db[1] = b2[1] - b1[1];

  denom = inner(normal, db, 2);

  if (fabsf(denom) < eps)
    return -1;

  dp[0] = a1[0] - b1[0];
  dp[1] = a1[1] - b1[1];

  // Compute intersection point
  num = inner(normal, dp, 2);
  slope = num / denom;
  p[0] = slope * db[0] + b1[0];
  p[1] = slope * db[1] + b1[1];

  return ret;
}

int intersection_segment_plane(float *a1, float *a2, float *p, float *normal, float *intersection)
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
  float u[3], w[3];
  int i;

  for (i = 0 ; i < 3 ; i++)
    u[i] = a2[i] - a1[i];
  denom = inner(normal, u, 3);

  if (fabsf(denom) > eps)
  {

    for (i = 0 ; i < 3 ; i++)
      w[i] = a1[i] - p[i];
    num = -inner(normal, w, 3);

    float s = num / denom;

    if (0-eps <= s && s <= 1+eps)
    {
      // compute intersection point
      for (i = 0 ; i < 3 ; i++)
        intersection[i] = s*u[i] + a1[i];

      // check limit case
      if (fabsf(s) < eps || fabsf(s - 1) < eps)
        return 1;  // a1 or a2 belongs to plane
      else
        return 0;  // plane is between a1 and a2
    }
  }

  return -1;  // no intersection
  
}

int intersection_segment_wall_3d(float *a1, float *a2, wall_t *wall, float *intersection)
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
  int i;
  float delta[3];
  float flat_intersection[2];

  ret1 = intersection_segment_plane(a1, a2, wall->origin, wall->normal, intersection);

  if (ret1 == -1)
    return -1;  // there is no intersection

  if (ret1 == 1)  // intersection at endpoint of segment
    ret = 1;

  /* substract origin of plane */
  for (i = 0 ; i < 3 ; i++)
    delta[i] = intersection[i] - wall->corners[i];

  /* project intersection into plane basis */
  for (i = 0 ; i < 2 ; i++)
    flat_intersection[i] = inner(delta, wall->basis + 3*i, 3);

  /* check in flatland if intersection is in the polygon */
  ret2 = is_inside_2d_polygon(flat_intersection, wall->flat_corners, wall->n_corners);

  if (ret2 < 0)  // intersection is outside of the wall
    return -1;

  if (ret2 == 1) // intersection is on the boundary of the wall
    ret |= 2;

  return ret;  // no intersection
}

int is_inside_2d_polygon(float *p, float *corners, int n_corners)
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

  int is_inside = 0;  // initialize point not in the polygon
  int c1c2p, c1c2p0, pp0c1, pp0c2;
  int i, j;
  float p_out[2];

  // find a point outside the polygon
  p_out[0] = corners[0];
  for (i = 1 ; i < n_corners ; i++)
    if (corners[2*i] < p_out[0])
      p_out[0] = corners[2*i];
  p_out[0]--;
  p_out[1] = p[1];

  // Now counter intersections
  for (i = 0, j = n_corners-1 ; i < n_corners ; j=i++)
  {

    // Check first if the point is on the segment
    // We count the border as inside the polygon
    c1c2p = ccw3p(corners + 2*i, corners + 2*j, p);
    if (c1c2p == 0)
    {
      // Here we know that p is co-linear with the two corners
      float x_down, x_up, y_down, y_up;
      x_down = fminf(corners[2*i], corners[2*j]);
      x_up = fmaxf(corners[2*i], corners[2*j]);
      y_down = fminf(corners[2*i+1], corners[2*j+1]);
      y_up = fmaxf(corners[2*i+1], corners[2*j+1]);
      if (x_down <= p[0] && p[0] <= x_up && y_down <= p[1] && p[1] <= y_up)
        return 1;
    }

    // Now check intersection with standard method
    c1c2p0 = ccw3p(corners + 2*i, corners + 2*j, p_out);
    if (c1c2p == c1c2p0)  // no intersection
      continue;

    pp0c1 = ccw3p(p, p_out, corners + 2*i);
    pp0c2 = ccw3p(p, p_out, corners + 2*j);
    if (pp0c1 == pp0c2)  // no intersection
      continue;

    // at this point we are sure there is an intersection

    // the second condition takes care of horizontal edges and intersection on vertex
    if (p[1] < fmaxf(corners[2*i+1], corners[2*j+1]))
      is_inside = ~is_inside;

  }

  // for a odd number of intersections, the point is in the polygon
  if (is_inside)
    return 0;  // point strictly inside
  else
    return -1; // point is outside

}

