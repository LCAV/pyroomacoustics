#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>
#include "wall.hpp"
#include "geometry.hpp"
#include <vector>
#include <array>
#include <list>
#include <vector>
#include <cmath>

using namespace Eigen;
using namespace std;

double clamp(double value, double min, double max);

Vector2f equation(const Vector2f &p1, const Vector2f &p2);

VectorXf compute_segment_end(const VectorXf start_point, float length, float phi, float theta);

VectorXf compute_reflected_end(const VectorXf & start,
  const VectorXf & hit_point,
  const VectorXf & wall_normal,
  float length);

bool intersects_mic(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius);

Vector2f solve_quad(float A, float B, float C);

VectorXf mic_intersection(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius);

void update_travel_time(float & travel_time,
  const float hop_length,
  const float sound_speed);

void update_energy_wall(float & energy, const Wall & wall);

float compute_scat_energy(float energy, float scat_coef,
  const Wall & wall,
  const VectorXf & start,
  const VectorXf & hit_point,
  const VectorXf & mic_pos,
  float radius,
  float total_dist);

int test();











#endif // __UTILITY_H__
