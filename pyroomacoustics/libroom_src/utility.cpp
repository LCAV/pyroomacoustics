/* This file contains some utility functions that are convenient for 
 * the project
 */

#include <iostream>
#include <cmath>
#include "utility.hpp"

using namespace Eigen;

double clamp(double value, double min, double max){
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

Vector2f equation(const Vector2f &p1, const Vector2f &p2)
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
	
	Vector2f result(a,b);
	return result;
}


VectorXf compute_segment_end(const VectorXf start, float_t length, float phi, float theta){
						
	/* Given a starting point, the length of the segment
	 * and a 2D or 3D orientation, this function
	 * computes the end point of the segment.
	 * It returns 0 if everything went fine, 1 otherwise*/
	
	if (start.size() == 2){
		return start + length*Vector2f(cos(phi), sin(phi));
		//return 0;
	}
	
	if (start.size() == 3){
		return start + length*Vector3f(sin(theta)*cos(phi),
									  sin(theta)*sin(phi),
									  cos(theta));
		
		//return 0;
	}
	
	std::cerr << "The starting point has dimension higher than 3 !" << std::endl;
	std::cerr << "Actual dim = " << start.size() << std::endl;
	throw std::exception();
}


