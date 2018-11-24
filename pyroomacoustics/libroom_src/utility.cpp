/* This file contains some utility functions that are convenient for 
 * the project
 */

#include <iostream>
#include "utility.hpp"

double clamp(double value, double min, double max){
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

Eigen::Vector2f equation(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2)
{
	/* Function that computes the a and be coefficients of the line defined
	 * by the equation y = a*x + b.
	 * The function is given two points p1 and p2, each one defined by its
	 * x and y coordinates */
	 
	 
	 if (p1[0] == p2[0]) {
		std::cerr << "The line passing by those two points cannot be described by a linear function of x." << std::endl;
		throw std::exception();
	 }
	
	float a = (p2[1]-p1[1])/(p2[0]-p1[0]); 
	float b = p1[1] - a*p1[0];
	
	Eigen::Vector2f result(a,b);
	return result;
}
