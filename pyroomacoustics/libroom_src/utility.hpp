#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>

using namespace Eigen;

double clamp(double value, double min, double max);

Vector2f equation(const Vector2f &p1, const Vector2f &p2);

VectorXf compute_segment_end(const VectorXf start_point, float_t length, float phi, float theta);

VectorXf compute_reflected_end(const VectorXf &start,
							   const VectorXf &hit_point,
							   const VectorXf &wall_normal,
							   float_t length);



#endif // __UTILITY_H__
