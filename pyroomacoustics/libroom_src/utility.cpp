/* This file contains some utility functions that are convenient for 
 * the project
 */

#include <iostream>
#include "utility.hpp"

using namespace Eigen;
extern float libroom_eps;

double clamp(double value, double min, double max)
{
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

Vector2f equation(const Vector2f & p1,
  const Vector2f & p2)
  {
  /* Function that computes the 'a' and 'b' coefficients in the expression
  y = a*x + b that defines the line passing by the points p1 and p2.
  */

  if (p1[0] == p2[0])
  {
    std::cerr << "The line passing by those two points cannot be described by a linear function of x." << std::endl;
    throw std::exception();
  }

  float a = (p2[1] - p1[1]) / (p2[0] - p1[0]);
  float b = p1[1] - a * p1[0];

  Vector2f result(a, b);
  return result;
}


VectorXf compute_segment_end(const VectorXf start, float length, float phi, float theta)
{

  /* Given a starting point, the length of the segment and a 2D or 3D
   orientation, this function computes the end point of the segment.
    
   start: (array size 2 or 3) defines the start point of the segment
   length: defines the desired length of the segment
   phi: angle that defines the planar orientation of the segment
   theta: elevation angle that defines the z-axis orientation of the segment
    
   :returns:
   an array of size 2 or 3 corresponding to the end point of this segment*/

  if (start.size() == 2)
  {
    return start + length * Vector2f(cos(phi), sin(phi));
    //return 0;
  }

  if (start.size() == 3)
  {
    return start + length * Vector3f(sin(theta) * cos(phi),
      sin(theta) * sin(phi),
      cos(theta));
  }

  std::cerr << "The starting point has dimension higher than 3 !" << std::endl;
  std::cerr << "Actual dim = " << start.size() << std::endl;
  throw std::exception();
}

VectorXf compute_reflected_end(const VectorXf & start,
  const VectorXf & hit_point,
  const VectorXf & wall_normal,
  float length)
  {
	  
  /* This method computes the reflection of one point with respect to
   a precise hit_point on a wall. Also, the distance between the
   wall hit point and the reflected point is defined by the 'length'
   parameter.
    
   start: (array size 2 or 3) defines the point to be reflected
   hit_point: (array size 2 or 3) defines a point on a wall that will
     serve as the reference point for the reflection
   wall_normal: (array size 2 or 3) defines the normal of the reflecting
     wall. It will be used as if it was anchored at hit_point
   length : the desired distance between hit_point and the reflected point
   
   :returns: an array of size 2 or 3 representing the reflected point
   */

  VectorXf incident = (hit_point - start).normalized();

  VectorXf n = wall_normal.normalized();

  // Reverse the normal if the angle between the incoming ray and 
  // the normal is more than PI/2 rad

  if (cos_angle_between((-1) * incident, n) < 0.)
  {
    n = (-1) * wall_normal;
  }

  return hit_point + length * ((incident - n * 2 * incident.dot(n)).normalized());

}


bool intersects_mic(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius)
  {

  /* This function checks if a segment crosses the microphone. 
   We must pay attention : the function dist_line_point considers
   an endless line while we are working with a segment here !
   => Hence the additional conditions.
    
   start: (array size 2 or 3) defines the start point of the segment
   end: (array size 2 or 3) defines the end point of the segment
   center: (array size 2 or 3) defines the center of the microphone
   radius: (array size 2 or 3) defines the radius of the microphone
    
   :returns: true iff the segment intersects the microphone
   * */

  size_t s_start = start.size();
  size_t s_end = end.size();
  size_t s_center = center.size();

  if (s_start < 2 or s_start > 3 or s_end < 2 or s_end > 3 or s_center < 2 or s_center > 3)
  {
    std::cerr << "intersects_mic : Only 2D and 3D vectors are supported" << std::endl;
    std::cerr << "Dim start = " << s_start << std::endl;
    std::cerr << "Dim end = " << s_end << std::endl;
    std::cerr << "Dim center = " << s_center << std::endl;
    throw std::exception();
  }

  if (s_start != s_center or s_start != s_end or s_end != s_center)
  {
    std::cerr << "The 3 vectors objects must have the same dimension !" << std::endl;
    std::cerr << "Dim start = " << s_start << std::endl;
    std::cerr << "Dim end = " << s_end << std::endl;
    std::cerr << "Dim center = " << s_center << std::endl;
    throw std::exception();
  }

  // Here we make sure that the ray hits the microphone 
  // We take a small margin to avoid bugs due to rounding errors
  // in the mic_intersection function
  if ((start - center).norm() < radius - libroom_eps or(end - center).norm() < radius - libroom_eps)
  {
    std::cerr << "One of the end point is inside the mic !" << std::endl;
  }

  VectorXf start_end = end - start; // vector from start to end
  VectorXf start_center = center - start;
  VectorXf end_center = center - end;
  VectorXf end_start = (-1) * start_end;

  bool intersects = dist_line_point(start, end, center) <= radius - libroom_eps;

  // This boolean checks that the projection of the center of the mic
  // on the segment is between start and end points
  bool on_segment = (cos_angle_between(start_end, start_center) >= 0.
    and cos_angle_between(end_start, end_center) >= 0.);

  return intersects and on_segment;
}


Vector2f solve_quad(float A, float B, float C)
{
  /* This function outputs the two solutions of the equation
   Ax^2 + Bx + C = 0
    
   We consider that no imaginary roots can be found since we are
   dealing with real distances and coefficients
    
   Delta < 0 can only be caused by rounding errors so we
   will consider it to be 0 in those cases
    
   A, B, C: the coefficients of Ax^2 + Bx + C = 0
    
   :returns: A vector containing 2 floats (the two roots of the polynome)*/

  float delta = B * B - 4 * A * C;

  float sqrt_delta(0);

  if (delta > 0) {
    sqrt_delta = sqrt(delta);
  }

  return Vector2f(-B - sqrt_delta, -B + sqrt_delta) / (2 * A);

}


VectorXf mic_intersection(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius)
  {

  /* This function computes the precise intersection point between the mic and the ray.
   This function should ONLY be called only if there is an intersection.
    
   start: (array size 2 or 3) defines the start point of the segment
   end: (array size 2 or 3) defines the end point of the segment
   center: (array size 2 or 3) defines the center of the microphones
   radius: defines the radius of the microphone
   
   :returns: (array size 2 or 3) defining the precise intersection point
     between the segment and the microphone*/

  size_t s_start = start.size();
  size_t s_end = end.size();
  size_t s_center = center.size();

  if (s_start < 2 or s_start > 3 or s_end < 2 or s_end > 3 or s_center < 2 or s_center > 3)
  {
    std::cerr << "mic_intersection : Only 2D and 3D vectors are supported" << std::endl;
    std::cerr << "Dim start = " << s_start << std::endl;
    std::cerr << "Dim end = " << s_end << std::endl;
    std::cerr << "Dim center = " << s_center << std::endl;
    throw std::exception();
  }

  if (s_start != s_center or s_start != s_end or s_end != s_center)
  {
    std::cerr << "The 3 vectors objects must have the same dimension !" << std::endl;
    std::cerr << "Dim start = " << s_start << std::endl;
    std::cerr << "Dim end = " << s_end << std::endl;
    std::cerr << "Dim center = " << s_center << std::endl;
    throw std::exception();
  }

  // Here we make sure that the ray hits the microphone 
  // We take a small margin to avoid bugs due to rounding errors
  // in the mic_intersection function
  if ((start - center).norm() < radius - libroom_eps)
  {
    std::cerr << "The start point is inside the mic !" << std::endl;
  }

  if (start.size() == 2)
  {

    Vector2f result1;
    Vector2f result2;

    float p = center[0];
    float q = center[1];

    // When the segment is vertical, we already know the x axis of the intersection point
    if (start[0] == end[0]) {
      float A = 1;
      float B = (-2) * q;
      float C = q * q + (start[0] - p) * (start[0] - p) - radius * radius;

      Vector2f ys = solve_quad(A, B, C);
      result1 = Vector2f(start[0], ys[0]);
      result2 = Vector2f(start[0], ys[1]);

      // See the formulat in the first answer of this post :
      // https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle

    } else
    {

      Vector2f eq = equation(start, end);
      float m = eq[0];
      float c = eq[1];

      float A = m * m + 1;
      float B = 2 * (m * c - m * q - p);
      float C = q * q - radius * radius + p * p - 2 * c * q + c * c;

      Vector2f xs = solve_quad(A, B, C);
      Vector2f ys = Vector2f(m * xs[0] + c, m * xs[1] + c);

      result1 = Vector2f(xs[0], ys[0]);
      result2 = Vector2f(xs[1], ys[1]);
    }

    // Now we must return only the closest intersection point 
    // to the start of the segment

    if ((start - result2).norm() < (start - result1).norm())
    {
      return result2;
    }
    return result1;
  }

  if (start.size() == 3)
  {

    Vector3f result1;
    Vector3f result2;

    // Here we are going to follow this formula :
    // https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    // With respect to the formula given on this page :
    // o = start
    // c = center
    // r = radius

    // We can check the results with this app :
    // http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm

    Vector3f l = (end - start).normalized();

    // We are searching d such that a point x on the sphere is also on the line
    // ie : x = o + k*l  (recall l has unit norm)		

    Vector3f o_c = start - center;
    float ocl_dot = o_c.dot(l);
    float delta = ocl_dot * ocl_dot - o_c.norm() * o_c.norm() + radius * radius;

    if (delta > 0)
    {

      result1 = start - l * (ocl_dot + sqrt(delta));
      result2 = start - l * (ocl_dot - sqrt(delta));

      if ((start - result2).norm() < (start - result1).norm())
      {
        return result2;
      }
      return result1;
    }

    // As this function is only called when there is an intersection,
    // delta should not be negative, unless because of rounding errors.
    // If this happens, we consider the delta to be 0

    return start - l * ocl_dot;

  }

  std::cerr << "Error : the vectors should be 2D or 3D" << s_center << std::endl;
  throw std::exception();
}


void update_travel_time(float & travel_time, float hop_length, float sound_speed)
{

  /* This function updates the travel time according to the newly 
   travelled distance
    
   travel_time: the cumulated travel time of the ray until before this newest hop
   hop_length: the distance travelled by the ray during the newest hop
   sound_speed: the constant speed of sound*/

  travel_time = travel_time + hop_length / sound_speed;
}

template<size_t D>
void update_energy_wall(float & energy,
  const Wall<D> & wall)
  {

  /* This function updates the ray's energy with respect to the wall's
   absorption coefficient.
    
   energy: the ray's energy just before the wall hit
   wall: the wall that is it by the ray*/

  energy = energy * sqrt(1 - wall.absorption);
}


int test()
{
  return 1;
}
























