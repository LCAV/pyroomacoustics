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
							   
							   
bool intersects_mic(const VectorXf &start,
					const VectorXf &end,
					const VectorXf &center,
					float_t radius);							   

Vector2f solve_quad(float A, float B, float C);

VectorXf mic_intersection(const VectorXf &start,
					const VectorXf &end,
					const VectorXf &center,
					float_t radius);

#endif // __UTILITY_H__
