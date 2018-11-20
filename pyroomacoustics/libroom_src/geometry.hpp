#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <Eigen/Dense>

extern float libroom_eps;

int ccw3p(const Eigen::VectorXf &p1, const Eigen::VectorXf &p2, const Eigen::VectorXf &p3);

int check_intersection_2d_segments(
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &b1, const Eigen::VectorXf &b2
    );

int intersection_2d_segments(
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &b1, const Eigen::VectorXf &b2,
    Eigen::Ref<Eigen::VectorXf> intersection
    );

int intersection_3d_segment_plane(
    const Eigen::VectorXf &a1, const Eigen::VectorXf &a2,
    const Eigen::VectorXf &p, const Eigen::VectorXf &normal,
    Eigen::Ref<Eigen::VectorXf> intersection);

Eigen::Vector3f cross(Eigen::Vector3f v1, Eigen::Vector3f v2);

int is_inside_2d_polygon(const Eigen::Vector2f &p,
    const Eigen::MatrixXf &corners);

float area_2d_polygon(const Eigen::MatrixXf &corners);

#endif // __GEOMETRY_H__
