#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <Eigen/Dense>


extern float libroom_eps;

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
    
float area_2d_polygon(const Eigen::MatrixXf &);

float cos_angle_between(const Eigen::VectorXf & v1,
  const Eigen::VectorXf & v2);

float dist_line_point(const Eigen::VectorXf & start,
  const Eigen::VectorXf & end,
  const Eigen::VectorXf & point);


#include "geometry.cpp"

#endif // __GEOMETRY_H__
