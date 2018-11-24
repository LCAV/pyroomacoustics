#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>


double clamp(double value, double min, double max);

Eigen::Vector2f equation(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2);



#endif // __UTILITY_H__
