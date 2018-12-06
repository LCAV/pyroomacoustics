#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>
#include "wall.hpp"
#include "geometry.hpp"
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

VectorXf compute_segment_end(const VectorXf start_point, float length, float phi, float theta);

VectorXf compute_reflected_end(const VectorXf &start,
							   const VectorXf &hit_point,
							   const VectorXf &wall_normal,
							   float length);
							   
							   
bool intersects_mic(const VectorXf &start,
					const VectorXf &end,
					const VectorXf &center,
					float radius);							   

Vector2f solve_quad(float A, float B, float C);

VectorXf mic_intersection(const VectorXf &start,
					const VectorXf &end,
					const VectorXf &center,
					float radius);
					
void update_travel_time(float &travel_time,
						const float hop_length,
						const float sound_speed);
						
void update_energy_wall(float &energy,
						const Wall &wall);						
						
float compute_scat_energy(float energy, float scat_coef,
						  const Wall &wall, 
						  const VectorXf &start,
						  const VectorXf &hit_point,
						  const VectorXf &mic_pos);
						  
void append(float travel_time, float energy,  std::vector<entry> &output);				

#endif // __UTILITY_H__
