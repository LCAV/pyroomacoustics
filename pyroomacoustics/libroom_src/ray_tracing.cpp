#include <vector>
#include <stack>
#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <cmath>

#include "wall.hpp"
#include "room.hpp"
#include "geometry.hpp"
#include "utility.hpp"

using namespace Eigen;
using namespace std;

int print(std::string message);
int print(double value);


int main(){
	
	Vector2f vec(1,1);
	Vector2f vec2(-1,-1);
		
	print(angle_between_2D(vec, vec2));

	return 0;
}

int print(std::string message){
	std::cout << message << std::endl;
	return 0;
}

int print(double val){
	std::cout << val << std::endl;
	return 0;
}


