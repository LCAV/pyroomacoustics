#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>
#include "wall.hpp"

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
					
void update_travel_time(float_t &travel_time,
						const float_t hop_length,
						const float_t sound_speed);
						
float compute_scat_energy(float_t energy, float_t scat_coef,
						  const Wall &wall, 
						  const VectorXf &start,
						  const VectorXf &hit_point,
						  const VectorXf &mic_pos);						

#endif // __UTILITY_H__
