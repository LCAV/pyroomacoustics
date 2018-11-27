#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>
#include "wall.hpp"
#include <vector>
#include <array>


/* The 'entry' type is simply defined as an array of 2 floats.
 * It represents an entry that is logged by the microphone
 * during the ray_tracing execution.
 * The first one of those float will be the energy of a ray reaching
 * the microphone. The second one will be the travel time of this ray.*/
typedef std::array<float,2> entry;


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
						  
void append(float energy, float travel_time, std::vector<entry> &output);				

#endif // __UTILITY_H__
