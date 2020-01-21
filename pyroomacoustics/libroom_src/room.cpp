/* 
 * Implementation of the Room class
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include "room.hpp"

const double pi = 3.14159265358979323846;
const double pi_2 = 1.57079632679489661923;

template<size_t D>
Room<D>::Room(
    const std::vector<Wall<D>> &_walls,
    const std::vector<int> &_obstructing_walls,
    const std::vector<Microphone<D>> &_microphones,
    float _sound_speed,
    // parameters for the image source model
    int _ism_order,
    // parameters for the ray tracing
    float _energy_thres,
    float _time_thres,
    float _mic_radius,
    float _mic_hist_res,
    bool _is_hybrid_sim
    )
  : walls(_walls), obstructing_walls(_obstructing_walls), microphones(_microphones),
  sound_speed(_sound_speed), ism_order(_ism_order),
  energy_thres(_energy_thres), time_thres(_time_thres), mic_radius(_mic_radius),
  mic_hist_res(_mic_hist_res), is_hybrid_sim(_is_hybrid_sim), is_shoebox(false)
{
  init();
}

template<size_t D>
Room<D>::Room(
    const Vectorf<D> &_room_size,
    const Eigen::Array<float,Eigen::Dynamic,2*D> &_absorption,
    const Eigen::Array<float,Eigen::Dynamic,2*D> &_scattering,
    const std::vector<Microphone<D>> &_microphones,
    float _sound_speed,
    // parameters for the image source model
    int _ism_order,
    // parameters for the ray tracing
    float _energy_thres,
    float _time_thres,
    float _mic_radius,
    float _mic_hist_res,
    bool _is_hybrid_sim
    )
  : microphones(_microphones),
  sound_speed(_sound_speed), ism_order(_ism_order), energy_thres(_energy_thres), time_thres(_time_thres),
  mic_radius(_mic_radius), mic_hist_res(_mic_hist_res), is_hybrid_sim(_is_hybrid_sim),
  is_shoebox(true), shoebox_size(_room_size), shoebox_absorption(_absorption),
  shoebox_scattering(_scattering)
{
  if (shoebox_absorption.rows() != _scattering.rows())
  {
    throw std::runtime_error("Error: The same number of absorption and scattering coefficients are required");
  }

  make_shoebox_walls(shoebox_size, _absorption, _scattering);
  init();
}


template<>
void Room<2>::make_shoebox_walls(
    const Vectorf<2> &rs,  // room_size
    const Eigen::Array<float,Eigen::Dynamic,4> &abs,
    const Eigen::Array<float,Eigen::Dynamic,4> &scat
    )
{
  Eigen::Matrix<float,2,Eigen::Dynamic> corners;
  corners.resize(2, 2);

  corners << 0.f, rs[0], 0.f, 0.f;
  walls.push_back(Wall<2>(corners, abs.col(2), scat.col(2), "south"));

  corners << rs[0], rs[0], 0.f, rs[1];
  walls.push_back(Wall<2>(corners, abs.col(1), scat.col(1), "east"));

  corners << rs[0], 0.f, rs[1], rs[1];
  walls.push_back(Wall<2>(corners, abs.col(3), scat.col(3), "north"));

  corners << 0.f, 0.f, rs[1], 0.f;
  walls.push_back(Wall<2>(corners, abs.col(0), scat.col(0), "west"));
}


template<>
void Room<3>::make_shoebox_walls(
    const Vectorf<3> &rs,  // room_size
    const Eigen::Array<float,Eigen::Dynamic,6> &abs,
    const Eigen::Array<float,Eigen::Dynamic,6> &scat
    )
{
  Eigen::Matrix<float,3,Eigen::Dynamic> corners;
  corners.resize(3,4);

  corners << 0.f, 0.f, 0.f, 0.f,
             rs[1], 0.f, 0.f, rs[1],
             0.f, 0.f, rs[2], rs[2];
  walls.push_back(Wall<3>(corners, abs.col(0), scat.col(0), "west"));

  corners << rs[0], rs[0], rs[0], rs[0], 
             0.f, rs[1], rs[1], 0.f, 
             0.f, 0.f, rs[2], rs[2];
  walls.push_back(Wall<3>(corners, abs.col(1), scat.col(1), "east"));

  corners << 0.f, rs[0], rs[0], 0.f,
             0.f, 0.f, 0.f, 0.f, 
             0.f, 0.f, rs[2], rs[2];
  walls.push_back(Wall<3>(corners, abs.col(2), scat.col(2), "south"));

  corners << rs[0], 0.f, 0.f, rs[0],
             rs[1], rs[1], rs[1], rs[1],
             0.f, 0.f, rs[2], rs[2];
  walls.push_back(Wall<3>(corners, abs.col(3), scat.col(3), "north"));

  corners << rs[0], 0.f, 0.f, rs[0],
             0.f, 0.f, rs[1], rs[1],
             0.f, 0.f, 0.f, 0.f;
  walls.push_back(Wall<3>(corners, abs.col(4), scat.col(4), "floor"));

  corners << rs[0], rs[0], 0.f, 0.f,
             0.f, rs[1], rs[1], 0.f,
             rs[2], rs[2], rs[2], rs[2];
  walls.push_back(Wall<3>(corners, abs.col(5), scat.col(5), "ceiling"));
}


template<size_t D>
void Room<D>::init()
{
  /*
   * Constructor for non-shoebox rooms
   */
  if (walls.size() > D)
  {
    n_bands = walls[0].get_n_bands();
    for (auto &wall : walls)
      if (n_bands != wall.get_n_bands())
      {
        throw std::runtime_error("Error: All walls should have the same number of frequency bands");
      }
  }
  else
  {
    if (D == 2)
      throw std::runtime_error("Error: The minimum number of walls is 3");
    else if (D == 3)
      throw std::runtime_error("Error: The minimum number of walls is 4");
    else
      throw std::runtime_error("Rooms of dimension other than 2 and 3 not supported");
  }

  // Useful for ray tracing
  max_dist = get_max_distance();
}


template<size_t D>
int Room<D>::image_source_model(const Vectorf<D> &source_location)
{
  /*
   * This is the top-level method to run the image source model
   */

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();

  if (is_shoebox)
  {
    return image_source_shoebox(source_location);
  }
  else
  {
    // add the original (real) source
    ImageSource<D> real_source(source_location, n_bands);

    // Run the image source model algorithm
    image_sources_dfs(real_source, ism_order);

    // fill the sources array in room and return
    return fill_sources();
  }
}


template<size_t D>
int Room<D>::fill_sources()
{
  int n_sources = visible_sources.size();

  // Create linear arrays to store the image sources
  if (n_sources > 0)
  {
    // resize all the arrays
    sources.resize(D, n_sources);
    orders.resize(n_sources);
    gen_walls.resize(n_sources);
    attenuations.resize(n_bands, n_sources);
    visible_mics.resize(microphones.size(), n_sources);

    for (int i = n_sources - 1 ; i >= 0 ; i--)
    {
      ImageSource<D> &top = visible_sources.top();  // sample top of stack

      // fill the arrays
      sources.col(i) = top.loc;
      gen_walls.coeffRef(i) = top.gen_wall;
      orders.coeffRef(i) = top.order;
      attenuations.col(i) = top.attenuation;
      visible_mics.col(i) = top.visible_mics;

      visible_sources.pop();  // unstack
    }
  }

  return n_sources;
}


template<size_t D>
void Room<D>::image_sources_dfs(ImageSource<D> &is, int max_order)
{
  /*
   * This function runs a depth first search (DFS) on the tree of image sources
   */

  ImageSource<D> new_is(n_bands);

  // Check the visibility of the source from the different microphones
  bool any_visible = false;
  int m = 0;
  for (auto mic = microphones.begin() ; mic != microphones.end() ; ++mic, ++m)
  {
    bool is_visible = is_visible_dfs(mic->get_loc(), is);
    if (is_visible && !any_visible)
    {
      any_visible = is_visible;
      is.visible_mics.resize(microphones.size());
      is.visible_mics.setZero();
    }
    if (any_visible)
      is.visible_mics.coeffRef(m) = is_visible;
  }

  if (any_visible)
    visible_sources.push(is);  // this should push a copy onto the stack
  
  // If we reached maximal depth, stop
  if (max_order == 0)
    return;

  // Then, check all the reflections across the walls
  for (size_t wi=0 ;  wi < walls.size() ; wi++)
  {
    int dir = walls[wi].reflect(is.loc, new_is.loc);  // the reflected location

    // We only check valid reflections (normals should point outward from the room
    if (dir <= 0)
      continue;

    // The reflection is valid, fill in the image source attributes
    new_is.attenuation *= walls[wi].get_transmission();
    if (walls[wi].scatter.maxCoeff() > 0.f && is_hybrid_sim)
    {
      new_is.attenuation *= (1 - walls[wi].scatter).sqrt();
    }
    new_is.order = is.order + 1;
    new_is.gen_wall = wi;
    new_is.parent = &is;

    // Run the DFS recursion (on the last element in the array, the one we just added)
    image_sources_dfs(new_is, max_order - 1);
  }
}


template<size_t D>
bool Room<D>::is_visible_dfs(const Vectorf<D> &p, ImageSource<D> &is)
{
  /*
     Returns true if the given sound source (with image source id) is visible from point p.

     room - the structure that contains all the sources and stuff
     p (np.array size 2 or 3) coordinates of the point where we check visibility
     imageId (int) id of the image within the SoundSource object
     
     Returns
     False (0) : not visible
     True (1) :  visible
     */

  if (is_obstructed_dfs(p, is))
    return false;

  if (is.parent != NULL)
  {
    Vectorf<D> intersection;

    // get generating wall id
    int wall_id = is.gen_wall;

    // check if the generating wall is intersected
    int ret = walls[wall_id].intersection(p, is.loc, intersection);

    // The source is not visible if the ray does not intersect
    // the generating wall
    if (ret >= 0)
      // Check visibility of intersection point from parent source
      return is_visible_dfs(intersection, *(is.parent));
    else
      return false;
  }

  // If we get here this is the original, unobstructed, source
  return true;
}


template<size_t D>
bool Room<D>::is_obstructed_dfs(const Vectorf<D> &p, ImageSource<D> &is)
{
  /*
     Checks if there is a wall obstructing the line of sight going from a source to a point.

     room - the structure that contains all the sources and stuff
     p (np.array size 2 or 3) coordinates of the point where we check obstruction
     imageId (int) id of the image within the SoundSource object

     Returns (bool)
     False (0) : not obstructed
     True (1) :  obstructed
     */
  int gen_wall_id = is.gen_wall;

  // Check candidate walls for obstructions
  for (size_t ow = 0 ; ow < obstructing_walls.size() ; ow++)
  {
    int wall_id = obstructing_walls[ow];

    // generating wall can't be obstructive
    if (wall_id != gen_wall_id)
    {
      Vectorf<D> intersection;
      int ret = walls[wall_id].intersection(is.loc, p, intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == Wall<D>::Isect::VALID || ret == Wall<D>::Isect::BNDRY)
      {
        if (is.parent != NULL)
        {
          // Test if the intersection point and the image are on
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = walls[is.gen_wall].side(is.loc);
          int intersection_side = walls[is.gen_wall].side(intersection);

          if (img_side != intersection_side && intersection_side != 0)
            return true;
        }
        else
          return true;
      }
    }
  }

  return false;
}


template<size_t D>
int Room<D>::image_source_shoebox(const Vectorf<D> &source)
{
  // precompute powers of the transmission coefficients
  std::vector<Eigen::ArrayXXf> transmission_pwr;
  for (int i(0) ; i <= ism_order ; ++i)
    transmission_pwr.push_back(Eigen::ArrayXXf(n_bands, 2*D));

  transmission_pwr[0].setOnes();
  if (ism_order > 0)
  {
    transmission_pwr[1] = (1.f - shoebox_absorption).sqrt();
    if (shoebox_scattering.maxCoeff() > 0.f && is_hybrid_sim)
    {
      transmission_pwr[1] *= (1 - shoebox_scattering).sqrt();
    }
  }
  for (int i = 2 ; i <= ism_order ; ++i)
    transmission_pwr[i] = transmission_pwr[i-1] * transmission_pwr[1];

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();
  
  // L1 ball of room images
  int point[3] = {0, 0, 0};

  // Take 2D case into account
  int z_max = ism_order;
  if (D == 2)
    z_max = 0;

  // Walk on all the points of the discrete L! ball of radius ism_order
  for (point[2] = -z_max ; point[2] <= z_max ; point[2]++)
  {
    int y_max = ism_order - abs(point[2]);
    for (point[1] = -y_max ; point[1] <= y_max ; point[1]++)
    {
      int x_max = y_max - abs(point[1]);
      if (x_max < 0) x_max = 0;

      for (point[0] = -x_max ; point[0] <= x_max ; point[0]++)
      {
        visible_sources.push(ImageSource<D>(n_bands));
        ImageSource<D> &is = visible_sources.top();
        is.visible_mics.resize(microphones.size());
        is.visible_mics.setOnes();  // everything is visible

        // Now compute the reflection, the order, and the multiplicative constant
        for (size_t d = 0 ; d < D ; d++)
        {
          // Compute the reflected source
          float step = abs(point[d]) % 2 == 1 ? shoebox_size.coeff(d) - source.coeff(d) : source.coeff(d);
          is.loc[d] = point[d] * shoebox_size.coeff(d) + step;

          // source order is just the sum of absolute values of reflection indices
          is.order += abs(point[d]);

          // attenuation can also be computed this way
          int p1 = 0, p2 = 0;
          if (point[d] > 0)
          {
            p1 = point[d]/2; 
            p2 = (point[d]+1)/2;
          }
          else if (point[d] < 0)
          {
            p1 = abs((point[d]-1)/2);
            p2 = abs(point[d]/2);
          }
          is.attenuation *= transmission_pwr[p1].col(2*d);  // 'west' absorption factor
          is.attenuation *= transmission_pwr[p2].col(2*d+1);  // 'east' absorption factor
        }
      }
    }
  }

  // fill linear arrays and return status
  return fill_sources();
}


template<size_t D>
float Room<D>::get_max_distance()
{

  /* This function outputs a value L that is strictly larger than any distance
   that a ray can travel in straight line in the room without hitting anything.
   In other words, this function outputs the diagonal of the bounding box 
   of this room.
   
   As we are sure that a ray reflecting from a hit_point H will hit
   another wall in less than L meters, we can use Wall::intersection()
   with the segment starting at H and of length L
  */

  float maxX(0), maxY(0), maxZ(0);
  float minX(0), minY(0), minZ(0);
  size_t n_walls = walls.size();
  
  // The first step is to go over the corners of all the walls to extract
  // the min x, y and z and the max x,y and z.

  for (size_t i(0); i < n_walls; ++i)
  {
    Wall<D> wi = this -> get_wall(i);

    Eigen::Vector3f max_coord(0, 0, 0);
    Eigen::Vector3f min_coord(0, 0, 0);

    if (D == 2)
    {
      max_coord.head(2) = wi.corners.topRows(D).rowwise().maxCoeff();
      min_coord.head(2) = wi.corners.topRows(D).rowwise().minCoeff();
    } else
    {
      max_coord = wi.corners.topRows(D).rowwise().maxCoeff();
      min_coord = wi.corners.topRows(D).rowwise().minCoeff();
    }

	// For the first wall, we have nothing to compare to
    if (i == 0)
    {
      maxX = max_coord[0];
      minX = min_coord[0];
      maxY = max_coord[1];
      minY = min_coord[1];
      maxZ = max_coord[2];
      minZ = min_coord[2];

    }
    // For the next walls, we compare to the stored min and max x,y and z
    else
    {

      // Update max point
      if (max_coord[0] > maxX)
        maxX = max_coord[0];
      if (max_coord[1] > maxY)
        maxY = max_coord[1];
      if (max_coord[2] > maxZ)
        maxZ = max_coord[2];

      // Update min point
      if (min_coord[0] < minX)
        minX = min_coord[0];
      if (min_coord[1] < minY)
        minY = min_coord[1];
      if (min_coord[2] < minZ)
        minZ = min_coord[2];
    }
  }

  // If we are in 2D, maxZ = minZ = 0
  // so there won't be any difference
  Eigen::Vector3f max_point(maxX, maxY, maxZ);
  Eigen::Vector3f min_point(minX, minY, minZ);

  // Return the length of the diagonal of the bounding box,  + 1
  return (min_point - max_point).norm() + 1;

}


template<size_t D>
std::tuple < Vectorf<D>, int > Room<D>::next_wall_hit(
  const Vectorf<D> &start,
  const Vectorf<D> &end,
  bool scattered_ray)
  {

  /* This function is called in 2 different contexts :
   * 
   * 1) When we trace the main ray : it computes the next next wall_hit
   * position given a segment defined by its endpoints.It also returns the
   * index of the wall that contains this next hit point.
   * 
   * start: (array size 2 or 3) the start point of the segment. Except for
   * 		the very first ray where this point is the source, this point 
   * 		is located on a wall.
   * end: (array size 2 or 3) the end point of the segment. Recall that 
   * 		this point must be set so that a wall is intersected between
   * 		start and end (dist(start,end) = Room->max_dist).
   * scattered_ray: false
   * 
   * :return: tuple <next_wall_hit_position, next_wall_index>
   * 
   * ==================================================================
   * 
   * 2) When we trace a scattered ray from the previous wall_hit point
   * back to the microphone : it checks if an (obstructing) wall stands
   * between that previous wall_hit position and the center of the microphone.
   * In that case, the scattered ray cannot reach the microphone.
   * 
   * start: (array size 2 or 3) the last wall_hit position
   * end: (array size 2 or 3) the end point of the segment, ie the center 
   * 		of the microphone in this case.
   * scattered_ray: true
   * 
   * :return: tuple <potentially_obstructing_wall_hit_position, potentially_obstructing_wall_index>
   * If no wall is intersected, then potentially_obstructing_wall_index will be -1
   * 
   * In fact here we are only interested in the second element of the tuple.
   * */

  Vectorf<D> result;

  int next_wall_index = -1;
  
  
  // For case 1) in non-convex rooms, the segment might intersect several
  // walls. In this case, we are only interested on the closest wall to
  // 'start'. That's why we need a min_dist variable
  // Upperbound on the min distance that we could find
  float min_dist(max_dist);

  for (size_t i(0); i < walls.size(); ++i)
  {

    // Scattered rays can only intersects 'obstructing' walls
    // So a wall is not obstructing, we skip it
    if (scattered_ray && (std::find(obstructing_walls.begin(), obstructing_walls.end(), i) == obstructing_walls.end())) {
      continue;
    }

    
    Wall<D> & w = walls[i];

    // To store the result of this iteration
    Vectorf<D> temp_hit;

    // As a side effect, temp_hit gets a value (VectorXf) here
    bool intersects = (w.intersection(start, end, temp_hit) > -1);

    if (intersects)
    {
      float temp_dist = (start - temp_hit).norm();

      // Compare to min dist to see if this wall is the closest to 'start'
      // Compare to libroom_eps to be sure that this wall w is not the wall
      //   where 'start' is located ('intersects' could be true because of
      //   rounding errors)
      if (temp_dist > libroom_eps && temp_dist < min_dist)
      {
        min_dist = temp_dist;
        result = temp_hit;
        next_wall_index = i;
      }
    }
  }
  return std::make_tuple(result, next_wall_index);
}


template<size_t D>
bool Room<D>::scat_ray(
    const Eigen::ArrayXf &transmitted,
    const Wall<D> &wall,
    const Vectorf<D> &prev_last_hit,
    const Vectorf<D> &hit_point,
    float travel_dist
    )
{

  /*
    Traces a one-hop scattered ray from the last wall hit to each microphone.
    In case the scattering ray can indeed reach the microphone (no wall in
    between), we log the hit in a histogram

    float energy: The energy of the ray right after last_wall has absorbed
      a part of it
    wall: The wall object where last_hit is located
    prev_last_hit: (array size 2 or 3) the previous last wall hit_point position (needed to check that 
      the wall normal is correctly oriented)
    hit_point: (array size 2 or 3) defines the last wall hit position
    travel_dist: The total distance travelled by the ray from source to hit_point

  :return : true if the scattered ray reached ALL the microphones, false otw
  */

  // Convert the energy threshold to transmission threshold (make this more efficient at some point)
  float distance_thres = time_thres * sound_speed;

  bool ret = true;  
  for(size_t k(0); k < microphones.size(); ++k)
  {

    Vectorf<D> mic_pos = microphones[k].get_loc();

    /* 
     * We also need to check that both the microphone and the
     * previous hit point are on the same side of the wall
     */
    if (wall.side(mic_pos) != wall.side(prev_last_hit))
    {
      ret = false;
      continue;
    }

    // Prepare the output tupple of next_wall_hit()
    Vectorf<D> dont_care;
    int next_wall_index(-1);
    std::tie(dont_care, next_wall_index) = next_wall_hit(hit_point, mic_pos, true);

    // If no wall obstructs the scattered ray
    if (next_wall_index == -1)
    {
      // As the ray is shot towards the microphone center,
      // the hop dist can be easily computed
      Vectorf<D> hit_point_to_mic = mic_pos - hit_point;
      float hop_dist = hit_point_to_mic.norm();
      float travel_dist_at_mic = travel_dist + hop_dist;

      // compute the scattered energy reaching the microphone
      float m_sq = mic_radius * mic_radius;
      float h_sq = hop_dist * hop_dist;
      float p_hit_equal = 1.f - sqrt(1.f - m_sq / h_sq);
      // cosine angle should be positive, but could be negative if normal is
      // facing out of room so we take abs
      float p_lambert = 2 * std::abs(wall.cosine_angle(hit_point_to_mic));
      Eigen::VectorXf scat_trans = wall.scatter * transmitted * p_hit_equal * p_lambert;

      // We add an entry to output and we increment the right element
      // of scat_per_slot
      if (travel_dist_at_mic < distance_thres && scat_trans.maxCoeff() > energy_thres)
      {
        //output[k].push_back(Hit(travel_dist_at_mic, scat_trans));        
        //microphones[k].log_histogram(output[k].back(), hit_point);
        float r_sq = travel_dist_at_mic * travel_dist_at_mic;
        auto p_hit = (1 - sqrt(1 - m_sq / std::max(m_sq, r_sq)));
        Eigen::ArrayXf energy = scat_trans / (r_sq * p_hit) ;
        microphones[k].log_histogram(travel_dist_at_mic, energy, hit_point);
      }
      else
        ret = false;
    }
    else
    {
      ret = false;  // if a wall intersects the scattered ray, we return false
    }
  }

  return ret;
}

template<size_t D>
void Room<D>::simul_ray(
    float phi,
    float theta,
    const Vectorf<D> source_pos,
    float energy_0
    )
{

  /*This function simulates one ray and fills the output vectors of 
   every microphone with all the entries produced by this ray
   (any specular or scattered ray reaching a microphone)
    
   phi and theta : give the orientation of the ray (2D or 3D)
   source_pos: (array size 2 or 3) is the location of the sound source (NOT AN IMAGE SOURCE)
  energy_0: (float) the initial energy of one ray
   output: is the std::vector that contains the entries for all the simulated rays */

  // ------------------ INIT --------------------
  // What we need to trace the ray
  Vectorf<D> start = source_pos;

  Vectorf<D> end;
  if(D == 2)
    end.head(2) = start.head(2)
      + max_dist * Eigen::Vector2f(cos(phi), sin(phi));
  else if (D == 3)
    end.head(3) = start.head(3)
      + max_dist * Eigen::Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

  // The following initializations are arbitrary and does not count since we set
  // the boolean to false
  int next_wall_index(0);

  // The ray's characteristics
  Eigen::ArrayXf transmitted = Eigen::ArrayXf::Ones(n_bands) * energy_0;
  Eigen::ArrayXf energy = Eigen::ArrayXf::Ones(n_bands);
  float travel_dist = 0;
  
  // To count the number of times the ray bounces on the walls
  // For hybrid generation we add a ray to output only if specular_counter
  // is higher than the ism order.
  int specular_counter(0);

  // Convert the energy threshold to transmission threshold
  float e_thres = energy_0 * energy_thres;
  float distance_thres = time_thres * sound_speed;

  double m_sq = mic_radius * mic_radius;

  //---------------------------------------------


  //------------------ RAY TRACING --------------------

  while (true)
  {
    Vectorf<D> hit_point;
    std::tie(hit_point, next_wall_index) = next_wall_hit(start, end, false);

    // If no wall is hit (rounding errors), stop the ray
    if (next_wall_index == -1)
    {
      break;
    }

    // Intersected wall
    Wall<D> &wall = walls[next_wall_index];

    // Initialization needed for the next if
    // Defines the length of the actual hop
    float distance(0);

    bool already_in_ism = is_hybrid_sim && specular_counter < ism_order;

    // Check if the specular ray hits any of the microphone
    if (!already_in_ism)
    {
      for(size_t k(0) ; k < microphones.size() ; k++)
      {
        // Compute the distance between the line defined by (start, hit_point)
        // and the center of the microphone (mic_pos)
        Vectorf<D> seg = hit_point - start;
        float seg_length = seg.norm();
        seg /= seg_length;
        Vectorf<D> to_mic = microphones[k].get_loc() - start;
        float impact_distance = to_mic.dot(seg);

        bool impacts = -libroom_eps < impact_distance && impact_distance < seg_length + libroom_eps;

        // If yes, we compute the ray's transmitted amplitude at the mic
        // and we continue the ray
        if (impacts &&
            (to_mic - seg * impact_distance).norm() < mic_radius + libroom_eps)
        {
          // The length of this last hop
          distance = fabsf(impact_distance);

          // Updating travel_time and transmitted amplitude for this ray
          // We DON'T want to modify the variables transmitted amplitude and travel_dist
          //   because the ray will continue its way          
          float travel_dist_at_mic = travel_dist + distance;

          double r_sq = double(travel_dist_at_mic) * travel_dist_at_mic;
          auto p_hit = (1 - sqrt(1 - m_sq / std::max(m_sq, r_sq)));
          energy = transmitted / (r_sq * p_hit);
          microphones[k].log_histogram(travel_dist_at_mic, energy, start);
        }
      }
    }

    // Update the characteristics
    distance = (start - hit_point).norm();
    travel_dist += distance;
    transmitted *= wall.get_energy_reflection();

    // Let's shoot the scattered ray induced by the rebound on the wall
    if (wall.scatter.maxCoeff() > 0.f)
    {
      // Shoot the scattered ray
      scat_ray(
          transmitted,
          wall,
          start,
          hit_point,
          travel_dist
          );

      // The overall ray's energy gets decreased by the total
      // amount of scattered energy
      transmitted *= (1.f - wall.scatter);
    }

    // Check if we reach the thresholds for this ray
    if (travel_dist > distance_thres || transmitted.maxCoeff() < e_thres)
      break;

    // set up for next iteration
    specular_counter += 1;
    end = wall.normal_reflect(start, hit_point, max_dist);
    start = hit_point;
  }
}


template<size_t D>
void Room<D>::ray_tracing(
  const Eigen::Matrix<float,D-1,Eigen::Dynamic> &angles,
  const Vectorf<D> source_pos
  )
{
  // float energy_0 = 2.f / (mic_radius * mic_radius * angles.cols());
  float energy_0 = 2.f / angles.cols();

  for (int k(0) ; k < angles.cols() ; k++)
  {
    float phi = angles.coeff(0,k);
    float theta = pi_2;

    if (D == 3)
      theta = angles.coeff(1,k);

    simul_ray(phi, theta, source_pos, energy_0);
  }
}


template<size_t D>
void Room<D>::ray_tracing(
    size_t nb_phis,
    size_t nb_thetas,
    const Vectorf<D> source_pos
    )
{


  /*This method produced all the time/energy entries needed to compute
   the RIR using ray-tracing with the following parameters

   nb_phis: the number of different planar directions that will be used
   nb_thetas: the number of different elevation angles that will be used
     (NOTE: nb_phis*nb_thetas is the number of simulated rays
   source_pos: (array size 2 or 3) represents the position of the sound source

   :returns: 
   a std::vector where each entry is a tuple (time, energy)
   reprensenting a ray (scattered or not) reaching the microphone
   */


  // ------------------ INIT --------------------
  // initial energy of one ray
  float energy_0 = 2.f / (nb_phis * nb_thetas);

  // ------------------ RAY TRACING --------------------

  for (size_t i(0); i < nb_phis; ++i)
  {
    float phi = 2 * pi * (float) i / nb_phis;

    for (size_t j(0); j < nb_thetas; ++j)
    {
      // Having a 3D uniform sampling of the sphere surrounding the room
      float theta = std::acos(2 * ((float) j / nb_thetas) - 1);

      // For 2D, this parameter means nothing, but we set it to
      // PI/2 to be consistent
      if (D == 2) {
        theta = pi_2;
      }

      // Trace this ray
      simul_ray(phi, theta, source_pos, energy_0);

      // if we work in 2D rooms, only 1 elevation angle is needed
      // => get out of the theta loop
      if (D == 2)
      {
        break;
      }
    }
  }
}


template<size_t D>
void Room<D>::ray_tracing(
    size_t n_rays,
    const Vectorf<D> source_pos
    )
{
  /*This method produced all the time/energy entries needed to compute
   the RIR using ray-tracing with the following parameters

   n_rays: the number of rays to use, rays are sampled pseudo-uniformly from
      the sphere using the Fibonacci algorithm
   source_pos: (array size 2 or 3) represents the position of the sound source
   */


  // ------------------ INIT --------------------
  // initial energy of one ray
  float energy_0 = 2.f / n_rays;

  // ------------------ RAY TRACING --------------------
  if (D == 3)
  {
    auto offset = 2.f / n_rays;
    auto increment = pi * (3.f - sqrt(5.f));  // phi increment

    for (size_t i(0); i < n_rays ; ++i)
    {
      auto z = (i * offset - 1) + offset / 2.f;
      auto rho = sqrt(1.f - z * z);

      float phi = i * increment;

      auto x = cos(phi) * rho;
      auto y = sin(phi) * rho;

      auto azimuth = atan2(y, x);
      auto colatitude = atan2(sqrt(x * x + y * y), z);

      simul_ray(azimuth, colatitude, source_pos, energy_0);
    }
  }
  else if (D == 2)
  {
    float offset = 2. * pi / n_rays;
    for (size_t i(0) ; i < n_rays ; ++i)
      simul_ray(i * offset, 0.f, source_pos, energy_0);
  }
}


template<size_t D>
bool Room<D>::contains(const Vectorf<D> point)
{

  /*This methods checks if a point is contained in the room
   
   point: (array size 2 or 3) representing a point in the room
    
   :returs: true if the point is inside the room, false otherwise*/
   
   
  // ------- USING RAY CASTING ALGO -------
  // First we need to build a point outside the room
  // For this we take the min (x,y,z) point and subtract 1 (arbitrary) to each coordinate

  size_t n_walls = walls.size();

  Eigen::Matrix<float,D,2> min_coord;
  min_coord.setZero();

  for (size_t i(0); i < n_walls; ++i)
  {
    Wall<D> &wi = this->get_wall(i);

    // First iteration		
    if (i == 0)
    {
      min_coord.col(0) = wi.corners.rowwise().minCoeff();
    }
    else
    {
      min_coord.col(1) = wi.corners.rowwise().minCoeff();
      min_coord.col(0) = min_coord.rowwise().minCoeff();
    }
  }

  Vectorf<D> outside_point = min_coord.col(0);

  // ------------------------------------------

  // Now we build a segment between 'outside_point' and 'point' 
  // We must look at the number of walls that intersect this segment

  size_t n_intersections(0);
  bool ambiguous_intersection = false;

  do
  {
	// We restart the computation with a modified output_point as long as we find
	// an ambiguous intersection (on the edge or on the endpoint of a segment)
	// Note : ambiguous intersection means that Wall::intersects() method
	// gives a result strictly above 0
	
    n_intersections = 0;
    ambiguous_intersection = false;

    outside_point[0] -= (float)(rand() % 27) / 50;
    outside_point[1] -= (float)(rand() % 22) / 26;

    if (D == 3)
    {
      outside_point[2] -= (float)(rand() % 24 / 47);
    }

    for (size_t i(0); i < n_walls; ++i)
    {

      Wall<D> & w = walls[i];
      int result = w.intersects(outside_point, point);
      ambiguous_intersection = ambiguous_intersection || (result > 0);

      if (result > -1)
      {
        n_intersections++;
      }
    }
  } while (ambiguous_intersection);

  // If an odd number of walls have been intersected,
  // then the point is in the room  => return true
  return ((n_intersections % 2) == 1);
}

