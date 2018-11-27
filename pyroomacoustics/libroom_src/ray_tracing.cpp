#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <cmath>

#include "wall.hpp"
#include "room.hpp"
#include "geometry.hpp"
#include "utility.hpp"
#include "ray_tracing.hpp"

using namespace Eigen;
using namespace std;

/* Important general note :
 * The ray_tracing function outputs a SINGLE std::vector<entry>
 * 
 * The 'entry' type is simply defined as an array of 2 floats.
 * The first one of those float will be the energy of a ray reaching
 * the microphone. The second one will be the travel time of this ray.*/

std::vector<entry> bilbo(){
	
	vector<entry> hobbit;
	hobbit.reserve(5);
	
	for (size_t i(0); i<14; ++i){
		append(i+0.4, i+0.5, hobbit);
	}
	
	return hobbit;
}






int print(std::string message){
	std::cout << message << std::endl;
	return 0;
}

int print(double val){
	std::cout << val << std::endl;
	return 0;
}


