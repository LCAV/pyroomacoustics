#include <iostream>
#include "room.hpp"


template<size_t D>
int Room<D>::image_source_model(const Eigen::Matrix<float,D,1> &source_location, int max_order)
{
  /*
   * This is the top-level method to run the image source model
   */

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();

  // add the original (real) source
  ImageSource<D> real_source;

  real_source.loc = source_location;
  real_source.attenuation = 1.;
  real_source.order = 0;
  real_source.gen_wall = -1;
  real_source.parent = NULL;

  // Run the image source model algorithm
  image_sources_dfs(real_source, max_order);

  // fill the sources array in room and return
  return fill_sources();
}


template<size_t D>
int Room<D>::fill_sources()
{
  int n_sources = visible_sources.size();

  // Create linear arrays to store the image sources
  if (n_sources > 0) {
    // resize all the arrays
    sources.resize(D, n_sources);
    orders.resize(n_sources);
    gen_walls.resize(n_sources);
    attenuations.resize(n_sources);
    visible_mics.resize(microphones.cols(), n_sources);

    for (int i = n_sources - 1; i >= 0; i--)
    {

      ImageSource<D> &top = visible_sources.top();  // sample top of stack

      // fill the arrays
      sources.col(i) = top.loc;
      gen_walls.coeffRef(i) = top.gen_wall;
      orders.coeffRef(i) = top.order;
      attenuations.coeffRef(i) = top.attenuation;
      visible_mics.col(i) = top.visible_mics;

      visible_sources.pop(); // unstack
    }
  }

  return 0;
}


template<size_t D>
void Room<D>::image_sources_dfs(ImageSource<D> &is, int max_order)
{
  /*
   * This function runs a depth first search (DFS) on the tree of image sources
   */

  ImageSource<D> new_is;

  // Check the visibility of the source from the different microphones
  bool any_visible = false;

  for (int mic = 0 ; mic < microphones.cols() ; mic++)
  {
    bool is_visible = is_visible_dfs(microphones.col(mic), is);
    if (is_visible && !any_visible)
    {
      any_visible = is_visible;
      is.visible_mics.resize(microphones.cols());
      is.visible_mics.setZero();
    }
    if (any_visible)
      is.visible_mics.coeffRef(mic) = is_visible;
  }

  if (any_visible)
    visible_sources.push(is); // this should push a copy onto the stack

  // If we reached maximal depth, stop
  if (max_order == 0)
    return;

  // Then, check all the reflections across the walls
  for (size_t wi = 0; wi < walls.size(); wi++)
  {
    int dir = walls[wi].reflect(is.loc, new_is.loc); // the reflected location

    // We only check valid reflections (normals should point outward from the room
    if (dir <= 0)
      continue;

    // The reflection is valid, fill in the image source attributes
    new_is.attenuation = is.attenuation * (1. - walls[wi].absorption);
    new_is.order = is.order + 1;
    new_is.gen_wall = wi;
    new_is.parent = & is;

    // Run the DFS recursion (on the last element in the array, the one we just added)
    image_sources_dfs(new_is, max_order - 1);
  }
}


template<size_t D>
bool Room<D>::is_visible_dfs(const Eigen::Matrix<float,D,1> &p, ImageSource<D> &is)
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
    Eigen::Matrix<float,D,1> intersection;

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
bool Room<D>::is_obstructed_dfs(const Eigen::Matrix<float,D,1> &p, ImageSource<D> &is)
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
  for (size_t ow = 0; ow < obstructing_walls.size(); ow++) {
    int wall_id = obstructing_walls[ow];

    // generating wall can't be obstructive

    if (wall_id != gen_wall_id)
    {
      Eigen::Matrix<float,D,1> intersection;
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
        } else
          return true;
      }
    }
  }

  return false;
}


template<size_t D>
int Room<D>::image_source_shoebox(
    const Eigen::Matrix<float,D,1> &source,
    const Eigen::Matrix<float,D,1> &room_size,
    const Eigen::Matrix<float,2*D,1> &absorption,
    int max_order
    )
{
  // precompute powers of the absorption coefficients
  std::vector<float> transmission_pwr((max_order + 1) * 2 * D);
  for (size_t d = 0 ; d < 2 * D ; d++)
    transmission_pwr[d] = 1.;
  for (int i = 1 ; i <= max_order ; i++)
    for (size_t d = 0 ; d < 2 * D ; d++)
      transmission_pwr[i * 2 * D + d] = (1. - absorption[d]) * transmission_pwr[(i-1)*2*D + d];

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();

  // L1 ball of room images
  int point[3] = {0,0,0};

  // Take 2D case into account
  int z_max = max_order;
  if (D == 2)
    z_max = 0;

  // Walk on all the points of the discrete L! ball of radius max_order
  for (point[2] = -z_max; point[2] <= z_max; point[2]++)
  {
    int y_max = max_order - abs(point[2]);
    for (point[1] = -y_max; point[1] <= y_max; point[1]++)
    {
      int x_max = y_max - abs(point[1]);
      if (x_max < 0) x_max = 0;

      for (point[0] = -x_max; point[0] <= x_max; point[0]++)
      {

        visible_sources.push(ImageSource<D>());
        ImageSource<D> &is = visible_sources.top();
        is.visible_mics = VectorXb::Ones(microphones.cols());  // everything is visible

        is.order = 0;
        is.attenuation = 1.;
        is.gen_wall = -1;

        // Now compute the reflection, the order, and the multiplicative constant

        for (size_t d = 0 ; d < D ; d++)
        {
          // Compute the reflected source
          float step = abs(point[d]) % 2 == 1 ? room_size.coeff(d) - source.coeff(d) : source.coeff(d);
          is.loc[d] = point[d] * room_size.coeff(d) + step;

          // source order is just the sum of absolute values of reflection indices
          is.order += abs(point[d]);

          // attenuation can also be computed this way
          int p1 = 0, p2 = 0;
          if (point[d] > 0)
          {
            p1 = point[d] / 2;
            p2 = (point[d] + 1) / 2;
          } else if (point[d] < 0)
          {
            p1 = abs((point[d] - 1) / 2);
            p2 = abs(point[d] / 2);
          }

          is.attenuation *= transmission_pwr[2 * D * p1 + 2*d];  // 'west' absorption factor
          is.attenuation *= transmission_pwr[2 * D * p2 + 2*d+1];  // 'east' absorption factor
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
std::tuple < Eigen::VectorXf, int > Room<D>::next_wall_hit(
  const Eigen::VectorXf & start,
  const Eigen::VectorXf & end,
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

  Eigen::VectorXf result;
  result.resize(D);

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
    if (scattered_ray and(std::find(obstructing_walls.begin(), obstructing_walls.end(), i) == obstructing_walls.end())) {
      continue;
    }
    
    Wall<D> & w = walls[i];

    // To store the result of this iteration
    Eigen::VectorXf temp_hit;
    temp_hit.resize(D);

    // As a side effect, temp_hit gets a value (VectorXf) here
    bool intersects = (w.intersection(start, end, temp_hit) > -1);

    if (intersects)
    {
      float temp_dist = (start - temp_hit).norm();

      // Compare to min dist to see if this wall is the closest to 'start'
      // Compare to libroom_eps to be sure that this wall w is not the wall
      //   where 'start' is located ('intersects' could be true because of
      //   rounding errors)
      if (temp_dist > libroom_eps and temp_dist < min_dist)
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
float Room<D>::compute_scat_energy(
  float energy,
  float scat_coef,
  const Wall<D> & wall,
  const VectorXf & start,
  const VectorXf & hit_point,
  const VectorXf & mic_pos,
  float radius,
  float total_dist,
  bool for_hybrid_rir)
  {
	  
  /* This function computes the energy of a scattered ray, including the 
   distance attenuation. This energy depends on several factors as explained below
    
   energy: the ray's energy (wall absorption already taken into account)
   scat_coef: the scattering coefficients of the wall
   wall: the wall being intersected (we need its normal vector)
   start: (array size 2 or 3) the previous wall hit_point position (needed to check that 
     the wall normal is correctly oriented)
   hit_point: (array size 2 or 3) the actual wall hit_point position
   mic_pos: (array size 2 or 3) the position of the microphone's center
   radius: the radius of the microphone
   float total_dist : the distance travelled by the ray between the source and hit_point.
   for_hybrid_rir : boolean to know if Ray Tracing is used in a hybrid way with ISM
     In this case, we might define another formula for the scattered energy.
   *  */
   
  // Make sure the normal points inside the room
  VectorXf n = wall.normal;
  if (cos_angle_between(start - hit_point, n) < 0.)
  {
    n = (-1) * n;
  }
  
  // The formula takes into account the scattered coef but also the
  // distance attenuation. 
  // The latter is the same as in the Image Source Method
  
  VectorXf r = mic_pos - hit_point;
  float B = r.norm();
  float cos_theta = cos_angle_between(n, r);
  float gamma_term =  1- sqrt(B*B-radius*radius)/B;
  
  
  //return energy*scat_coef*cos_theta /(4*M_PI*(total_dist+r.norm())); 
  
  return energy * sqrt( scat_coef * 2 * cos_theta * gamma_term ) / (4*M_PI*(total_dist+B-radius));
}

template<size_t D>
bool Room<D>::scat_ray(
  float energy,
  float scatter_coef,
  const Wall<D> & wall,
  const Eigen::VectorXf & prev_last_hit,
  const Eigen::VectorXf & hit_point,
  float radius,
  float total_dist,
  float travel_time,
  float time_thres,
  float energy_thres,
  float sound_speed,
  bool for_hybrid_rir,
  room_log & output)
  {

  /*
	Traces a one-hop scattered ray from the last wall hit to each microphone.
	In case the scattering ray can indeed reach the microphone (no wall in
	between), then we add an entry to the output passed by reference.
	
	float energy: The energy of the ray right after last_wall has absorbed
	  a part of it
	scatter_coef: The scattering coefficient
    wall: The wall object where last_hit is located
    prev_last_hit: (array size 2 or 3) the previous last wall hit_point position (needed to check that 
      the wall normal is correctly oriented)
    hit_point: (array size 2 or 3) defines the last wall hit position
    radius : The radius of the circle/sphere microphone
    total_dist: The total distance travelled by the ray from source to hit_point
    travel_time: The cumulated travel time of the ray from source to hit_point
    time_thresh: The time threshold for the ray
    energy_thresh: The energy threshold for the ray
    sound_speed: The speed of sound
    for_hybrid_rir : a boolean that indicate if we are going to use this
      room_log to compute a hybrid RIR (true) or a pure ray tracing rir (false).
      This is important because for hybrid RIR, we must scale the energy of each
      particle to make it coherent with ISM
    fs : the sampling frequency used for the rir
    scat_per_slot : a 2D vector counting for each microphone and each time
      slot, the number of scattered rays reaching the microphone
    output: the std::vector containing the time/energy entries for each microphone to build the rir
     
    :return : true if the scattered ray reached ALL the microphones, false otw
	*/

  bool result = true;  
  for(int k(0); k < n_mics; ++k){
	  
	Eigen::VectorXf mic_pos = microphones.col(k);
	mic_pos.resize(D);
	
    float scat_energy = compute_scat_energy(
      energy,
	  scatter_coef,
      wall,
      prev_last_hit,
      hit_point,
      mic_pos,
      radius,
      total_dist,
      for_hybrid_rir);
        
    // If we have multiple microphones, we must keep travel_time untouched !
    float travel_time_at_mic = travel_time;
       
    // Prepare the output tupple of next_wall_hit()
    Eigen::VectorXf dont_care;
    int next_wall_index(-1);
    std::tie(dont_care, next_wall_index) = next_wall_hit(hit_point, mic_pos, true);

	
    // If no wall obstructs the scattered ray
    if (next_wall_index == -1)
    {

      // As the ray is shot towards the microphone center,
      // the hop dist can be easily computed
      float hop_dist = (hit_point - mic_pos).norm() - radius;
      travel_time_at_mic += hop_dist/sound_speed;

      // We add an entry to output and we increment the right element
      // of scat_per_slot
      if (travel_time_at_mic < time_thres and scat_energy > energy_thres)
      {
        output[k].push_back(entry{{travel_time_at_mic,scat_energy}});        
        result = result and true;
      }
      else
      {
		// if a threshold is met
	    result = false;
	  }
    }
    else
    {
	  // if a wall intersects the scattered ray, we return false
	  result = false;
	}
  }

  return result;
}


template<size_t D>
void Room<D>::simul_ray(float phi,
  float theta,
  const Eigen::VectorXf source_pos,
  float mic_radius,
  float scatter_coef,
  float time_thres,
  float energy_thres,
  float sound_speed,
  bool for_hybrid_rir,
  int ism_order,
  room_log & output)
  {

  /*This function simulates one ray and fills the output vectors of 
   every microphone with all the entries produced by this ray
   (any specular or scattered ray reaching a microphone)
    
   phi and theta : give the orientation of the ray (2D or 3D)
   source_pos: (array size 2 or 3) is the location of the sound source (NOT AN IMAGE SOURCE)
   mic_radius: (array size 2 or 3) is the radius of the microphone that is represented like a circle or a sphere
   scatter coef: determines the amount of the energy that gets scattered every time the ray hits a wall
   time_thres: is the upperbound on the travel time of the ray
   energy_thresh: The energy threshold for the ray
   sound_speed: is the speed of sound (may be dependent on humidity etc...)
   for_hybrid_rir : a boolean that indicate if we are going to use this
     room_log to compute a hybrid RIR (true) or a pure ray tracing rir (false).
     This is important because for hybrid RIR, we must scale the energy of each
     particle to make it coherent with ISM
   ism_order : the maximum order of the Image Sources when Ray Tracing is called
     for a hybrid RIR computation. In ray tracing, we don't take into account the specular rays
     of order less than ism_order.
   fs : the sampling frequency used for the rir
   scat_per_slot : a 2D vector counting for each microphone and each time
     slot the number of scattered rays reaching the microphone
   output: is the std::vector that contains the entries for all the simulated rays */

  // ------------------ INIT --------------------
  // What we need to trace the ray
  float energy = 1;
  VectorXf start = source_pos;
  VectorXf end = compute_segment_end(start, max_dist, phi, theta);

  // The following initializations are arbitrary and does not count since we set
  // the boolean to false
  Wall<D> & wall = walls[0];
  int next_wall_index(0);

  // The ray's characteristics
  float travel_time = 0;
  float total_dist = 0;
  
  // To count the number of times the ray bounces on the walls
  // For hybrid generation we add a ray to output only if specular_counter
  // is higher than the ism order.
  int specular_counter(0);
  //---------------------------------------------


  //------------------ RAY TRACING --------------------

  while (true)
  {
    VectorXf hit_point;
    std::tie(hit_point, next_wall_index) = next_wall_hit(start, end, false);

    // If no wall is hit (rounding errors), stop the ray
    if (next_wall_index == -1)
    {
      break;
    }

    // Intersected wall
    wall = walls[next_wall_index];

    // Initialization needed for the next if
    // Defines the length of the actual hop
    float distance(0);

	bool already_in_ism = for_hybrid_rir and specular_counter < ism_order;
	
    // Check if the specular ray hits any of the microphone
    if (not already_in_ism)
	{
      for(int k(0); k<n_mics; k++)
      {
		
	    Eigen::VectorXf mic_pos = microphones.col(k);
	   	
        // If yes, we compute the ray's energy at the mic
        // and we continue the ray
        if (intersects_mic(start, hit_point, mic_pos, mic_radius))
        {
	      // The length of this last hop
          distance = (start - mic_pos).norm();
      
          // Updating travel_time and energy for this ray
          // We DON'T want to modify the variables energy and travel_time
          //   because the ray will continue its way          
          float travel_time_at_mic = travel_time + distance/sound_speed;
          float energy_at_mic = energy / (4*M_PI* (total_dist + distance - mic_radius));
	   	  
          if (travel_time_at_mic < time_thres and energy_at_mic > energy_thres)
          {		  
            output[k].push_back(entry{{travel_time_at_mic, energy_at_mic}});
          }
        }
      }
    }
    
    // Update the characteristics

    distance = (start - hit_point).norm();
    total_dist += distance;
    
    travel_time = travel_time + distance/sound_speed;
    energy = energy * sqrt(1 - wall.absorption);
    

    // Check if we reach the thresholds for this ray
    if (travel_time > time_thres or energy < energy_thres)
    {
      break;
    }

    // Let's shoot the scattered ray induced by the rebound on the wall
	if (scatter_coef > 0 and not already_in_ism)
	{
	// Shoot the scattered ray
	scat_ray(
	  energy,
	  scatter_coef,
	  wall,
	  start,
	  hit_point,
	  mic_radius,
	  total_dist,
	  travel_time,
	  time_thres,
	  energy_thres,
	  sound_speed,
	  for_hybrid_rir,
	  output);

	  // The overall ray's energy gets decreased by the total
	  // amount of scattered energy
	  energy = energy * sqrt(1 - scatter_coef);
    }

	// set up for next iteration
    specular_counter += 1;
    end = compute_reflected_end(start, hit_point, wall.normal, max_dist);
    start = hit_point;
  }
}

template<size_t D>
room_log Room<D>::get_rir_entries(size_t nb_phis,
  size_t nb_thetas,
  const Eigen::VectorXf source_pos,
  float mic_radius,
  float scatter_coef,
  float time_thres,
  float energy_thres,
  float sound_speed,
  bool for_hybrid_rir,
  int ism_order)
  {


  /*This method produced all the time/energy entries needed to compute
   the RIR using ray-tracing with the following parameters
    
   nb_phis: the number of different planar directions that will be used
   nb_thetas: the number of different elevation angles that will be used
     (NOTE: nb_phis*nb_thetas is the number of simulated rays
   source_pos: (array size 2 or 3) represents the position of the sound source
   mic_radius: the radius of the circular(2D) or spherical(3D) microphone
   scatter_coef: the scattering coefficients used for the walls of the room
   time_thres: the simulation time threshold for each ray
   energy_thresh: The energy threshold for the ray
   sound_speed: the constant speed of sound
   for_hybrid_rir : a boolean that indicate if we are going to use this
     room_log to compute a hybrid RIR (true) or a pure ray tracing rir (false).
     This is important because for hybrid RIR, we must scale the energy of each
     particle to make it coherent with ISM
   ism_order : the maximum order of the Image Sources when Ray Tracing is called
     for a hybrid RIR computation. In ray tracing, we don't take into account the specular rays
     of order less than ism_order.
   fs : the sampling frequency used for the rir
   
   :returns: 
   a std::vector where each entry is a tuple (time, energy)
   reprensenting a ray (scattered or not) reaching the microphone
   */


  // ------------------ INIT --------------------
  room_log output;
  
  for (int k(0); k<n_mics; k++)
  {
	
	// push a vector for each mic (log every entries)
	output.push_back(mic_log());
	
	VectorXf mic_pos = microphones.col(k);
    
    if (not contains(mic_pos))
    {
      std::cerr << "One microphone is not inside the room ! " << std::endl;
      throw std::exception();
    }
    
    if ((source_pos - mic_pos).norm() < mic_radius)
    {
      std::cerr << "The source is inside the microphone ! " << std::endl;
      throw std::exception();
    }
  }
  
  if (not contains(source_pos))
  {
    std::cerr << "The source is not inside the room ! " << std::endl;
    throw std::exception();
  }


  // ------------------ RAY TRACING --------------------

  for (size_t i(0); i < nb_phis; ++i)
  {
    float phi = 2 * M_PI * (float) i / nb_phis;

    for (size_t j(0); j < nb_thetas; ++j)
    {
	  // Having a 3D uniform sampling of the sphere surrounding the room
      float theta = std::acos(2 * ((float) j / nb_thetas) - 1);

      // For 2D, this parameter means nothing, but we set it to
      // PI/2 to be consistent
      if (D == 2) {
        theta = M_PI_2;
      }
      
      // Trace this ray
      simul_ray(phi, theta, source_pos, mic_radius, scatter_coef,
        time_thres, energy_thres, sound_speed, for_hybrid_rir, ism_order, output);

      // if we work in 2D rooms, only 1 elevation angle is needed
      // => get out of the theta loop
      if (D == 2)
      {
        break;
      }
    }
  }
  
  return output;

}


template<size_t D>
bool Room<D>::contains(const Eigen::VectorXf point)
{

  /*This methods checks if a point is contained in the room
   
   point: (array size 2 or 3) representing a point in the room
    
   :returs: true if the point is inside the room, false otherwise*/
   
   
  // Sanity check
  if (D != point.size())
  {
    std::cerr << "Error in Room::contains()\nThe room and the point have different dimensions !" << std::endl;
    throw std::exception();
  }

  // ------- USING RAY CASTING ALGO -------
  // First we need to build a point outside the room
  // For this we take the min (x,y,z) point and subtract 1 (arbitrary) to each coordinate

  float minX(0), minY(0), minZ(0);

  size_t n_walls = walls.size();

  for (size_t i(0); i < n_walls; ++i)
  {
    Wall<D> &wi = this -> get_wall(i);

    Eigen::Vector3f min_coord(0, 0, 0);

    if (D == 2)
    {
      min_coord.head(2) = wi.corners.topRows(D).rowwise().minCoeff();
    } 
    else
    {
      min_coord = wi.corners.topRows(D).rowwise().minCoeff();
    }

    // First iteration		
    if (i == 0)
    {
      minX = min_coord[0];
      minY = min_coord[1];
      minZ = min_coord[2];

      // Other iterations
    }
    else
    {

      if (min_coord[0] < minX)
        minX = min_coord[0];
      if (min_coord[1] < minY)
        minY = min_coord[1];
      if (min_coord[2] < minZ)
        minZ = min_coord[2];
    }
  }

  Eigen::VectorXf outside_point;
  outside_point.resize(D);

  outside_point[0] = minX - 1.;
  outside_point[1] = minY - 1.;

  if (D == 3) {
    outside_point[2] = minZ - 1.;
  }

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
      ambiguous_intersection = ambiguous_intersection or result > 0;

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















