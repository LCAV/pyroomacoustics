/* 
 * Definition of the geometry routines
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

#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <Eigen/Dense>

#include "common.hpp"

int ccw3p(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2, const Eigen::Vector2f &p3);

int check_intersection_2d_segments(
    const Eigen::Vector2f &a1, const Eigen::Vector2f &a2,
    const Eigen::Vector2f &b1, const Eigen::Vector2f &b2
    );

int intersection_2d_segments(
    const Eigen::Vector2f &a1, const Eigen::Vector2f &a2,
    const Eigen::Vector2f &b1, const Eigen::Vector2f &b2,
    Eigen::Ref<Eigen::Vector2f> intersection
    );

int intersection_3d_segment_plane(
    const Eigen::Vector3f &a1, const Eigen::Vector3f &a2,
    const Eigen::Vector3f &p, const Eigen::Vector3f &normal,
    Eigen::Ref<Eigen::Vector3f> intersection);

Eigen::Vector3f cross(Eigen::Vector3f v1, Eigen::Vector3f v2);

int is_inside_2d_polygon(const Eigen::Vector2f &p,
    const Eigen::Matrix<float,2,Eigen::Dynamic> &corners);
    
float area_2d_polygon(const Eigen::Matrix<float, 2, Eigen::Dynamic> &corners);

float cos_angle_between(const Eigen::VectorXf & v1,
  const Eigen::VectorXf & v2);

float dist_line_point(const Eigen::VectorXf & start,
  const Eigen::VectorXf & end,
  const Eigen::VectorXf & point);

template<size_t D>
class Line
{
  Vectorf<D> unit_vec;  // direction of the Line
  Vectorf<D> origin;  // point in the Line

  public:
  Line(const Vectorf<D> &_unit, const Vectorf<D> &_p) : unit_vec(_unit), origin(_p) {}
  ~Line() {}

  // Create a line from two points
  static Line from_points(const Vectorf<D> &_origin, const Vectorf<D> &_other_point)
  {
    return Line((_other_point - _origin).normalize(), _origin);
  }

  // returns the distance between the line and a point p
  float distance(const Vectorf<D> &p)
  {
    return (p - project(p)).norm();
  }

  // signed distance from line origin to the projection of point p onto the line
  float projected_distance(const Vectorf<D> &p)
  {
    return (p - origin).adjoint() * unit_vec; 
  }

  // returns orthogonal projection of p onto the line
  Vectorf<D> project(const Vectorf<D> &p)
  {
    return origin + projected_distance(p) * unit_vec;
  }

  // returns the symmetric point with respect to line
  Vectorf<D> reflect(const Vectorf<D> &p)
  {
    return 2.f * project(p) - p;
  }
};

#include "geometry.cpp"

#endif // __GEOMETRY_H__
