#include <iostream>
#include "room.hpp"

int Room::image_source_model(const Eigen::VectorXf &source_location, int max_order)
{
  /*
   * This is the top-level method to run the image source model
   */

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();

  // add the original (real) source
  ImageSource real_source;
  real_source.loc.resize(dim);

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

int Room::fill_sources()
{
  int n_sources = visible_sources.size();

  // Create linear arrays to store the image sources
  if (n_sources > 0)
  {
    // resize all the arrays
    sources.resize(dim, n_sources);
    orders.resize(n_sources);
    gen_walls.resize(n_sources);
    attenuations.resize(n_sources);
    visible_mics.resize(microphones.cols(), n_sources);

    for (int i = n_sources - 1 ; i >= 0 ; i--)
    {
      ImageSource &top = visible_sources.top();  // sample top of stack

      // fill the arrays
      sources.col(i) = top.loc;
      gen_walls.coeffRef(i) = top.gen_wall;
      orders.coeffRef(i) = top.order;
      attenuations.coeffRef(i) = top.attenuation;
      visible_mics.col(i) = top.visible_mics;

      visible_sources.pop();  // unstack
    }
  }

  return 0;
}

void Room::image_sources_dfs(ImageSource &is, int max_order)
{
  /*
   * This function runs a depth first search (DFS) on the tree of image sources
   */

  ImageSource new_is;
  new_is.loc.resize(dim);

  // Check the visibility of the source from the different microphones
  bool any_visible = false;
  is.visible_mics.resize(microphones.cols());
  for (int mic = 0 ; mic < microphones.cols() ; mic++)
  {
    is.visible_mics.coeffRef(mic) = is_visible_dfs(microphones.col(mic), is);
    any_visible = any_visible || is.visible_mics.coeff(mic);
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
    new_is.attenuation = is.attenuation * (1. - walls[wi].absorption);
    new_is.order = is.order + 1;
    new_is.gen_wall = wi;
    new_is.parent = &is;

    // Run the DFS recursion (on the last element in the array, the one we just added)
    image_sources_dfs(new_is, max_order - 1);
  }
}

bool Room::is_visible_dfs(const Eigen::VectorXf &p, ImageSource &is)
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
    Eigen::VectorXf intersection;
    intersection.resize(dim);

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

bool Room::is_obstructed_dfs(const Eigen::VectorXf &p, ImageSource &is)
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
      Eigen::VectorXf intersection;
      intersection.resize(dim);
      int ret = walls[wall_id].intersection(is.loc, p, intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == Wall::Isect::VALID || ret == Wall::Isect::BNDRY)
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

int Room::image_source_shoebox(
    const Eigen::VectorXf &source,
    const Eigen::VectorXf &room_size,
    const Eigen::VectorXf &absorption,
    int max_order
    )
{
  // precompute powers of the absorption coefficients
  std::vector<float> transmission_pwr((max_order + 1) * 2 * dim);
  for (int d = 0 ; d < 2 * dim ; d++)
    transmission_pwr[d] = 1.;
  for (int i = 1 ; i <= max_order ; i++)
    for (int d = 0 ; d < 2 * dim ; d++)
      transmission_pwr[i * 2 * dim + d] = (1. - absorption[d]) * transmission_pwr[(i-1)*2*dim + d];

  // make sure the list is empty
  while (visible_sources.size() > 0)
    visible_sources.pop();
  
  // L1 ball of room images
  int point[3] = {0, 0, 0};

  // Take 2D case into account
  int z_max = max_order;
  if (dim == 2)
    z_max = 0;

  // Walk on all the points of the discrete L! ball of radius max_order
  for (point[2] = -z_max ; point[2] <= z_max ; point[2]++)
  {
    int y_max = max_order - abs(point[2]);
    for (point[1] = -y_max ; point[1] <= y_max ; point[1]++)
    {
      int x_max = y_max - abs(point[1]);
      if (x_max < 0) x_max = 0;

      for (point[0] = -x_max ; point[0] <= x_max ; point[0]++)
      {
        visible_sources.push(ImageSource());
        ImageSource &is = visible_sources.top();
        is.loc.resize(dim);
        is.visible_mics = VectorXb::Ones(microphones.cols());  // everything is visible

        is.order = 0;
        is.attenuation = 1.;
        is.gen_wall = -1;

        // Now compute the reflection, the order, and the multiplicative constant
        for (int d = 0 ; d < dim ; d++)
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
            p1 = point[d]/2; 
            p2 = (point[d]+1)/2;
          }
          else if (point[d] < 0)
          {
            p1 = abs((point[d]-1)/2);
            p2 = abs(point[d]/2);
          }
          is.attenuation *= transmission_pwr[2 * dim * p1 + 2*d];  // 'west' absorption factor
          is.attenuation *= transmission_pwr[2 * dim * p2 + 2*d+1];  // 'east' absorption factor
        }
      }
    }
  }

  // fill linear arrays and return status
  return fill_sources();
}


float Room::get_max_distance(){
	
	/* This function outputs a value L that is strictly larger than any distance
	 * that a ray could travel in straight line in the 2D or 3D room without
	 * hitting anything.
	 * As we are sure that a ray reflecting from a hit_point H will hit
	 * another wall in less than L meters, we can use the wall intersection
	 * functions with the segment starting at H and of length V*/
	
	// Useless initialization to avoid compilation warnings
	float maxX(0), maxY(0), maxZ(0);
	float minX(0), minY(0), minZ(0);
	
	size_t n_walls = walls.size();
	
	for (size_t i(0); i<n_walls; ++i){
		Wall wi = this -> get_wall(i);
		
		Eigen::Vector3f max_coord(0,0,0);
		Eigen::Vector3f min_coord(0,0,0);
		
		if (this -> dim == 2){
			max_coord.head(2) = wi.corners.rowwise().maxCoeff();
			min_coord.head(2) = wi.corners.rowwise().minCoeff();
		}else{
			max_coord = wi.corners.rowwise().maxCoeff();
			min_coord = wi.corners.rowwise().minCoeff();
		}
		
		
		
		if (i == 0){
			maxX = max_coord[0]; minX = min_coord[0];
			maxY = max_coord[1]; minY = min_coord[1];
			maxZ = max_coord[2]; minZ = min_coord[2];
			
		}else{
			
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
	
	return (min_point - max_point).norm() + 1;

}



Eigen::VectorXf Room::next_wall_hit(
					const Eigen::VectorXf &start,
					const Eigen::VectorXf &end,
					bool there_is_prev_wall,
					const Wall &previous_wall,
					Eigen::Ref<Vector1i> next_wall_index){
						
	/* Computes the next next wall_hit position given a segment defined
	 * by its endpoints. This method also stores the index of the wall
	 * that contains this next hit point (we needed to pass it inside a 
	 * Vector1i to be able to call this function from python).
	 * 
	 * If no wall is intersected, then the next_wall_index will be -1*/
						
	Eigen::VectorXf result;
	result.resize(dim);
	
	next_wall_index[0] = -1;
	
	// Upperbound on the min distance that we could find
	float min_dist(max_dist);
	
	for (size_t i(0); i < walls.size(); ++i){
		
		Wall w = walls[i];
		
		// To store the results of this iteration
		Eigen::VectorXf temp_hit;
		temp_hit.resize(dim);
		
		// The 2 conditions for wall hit (temp_hit computed as a side effect)
		bool different_than_prev = there_is_prev_wall and not w.same_as(previous_wall);
		bool intersects = (w.intersection(start, end, temp_hit) > -1);
		
		
		
		if (intersects and
			(different_than_prev or not there_is_prev_wall)){
				
			float temp_dist = (start-temp_hit).norm();
			
			if (temp_dist < min_dist){
				
				min_dist = temp_dist;
				result = temp_hit;
				
				// update the index of the wall that is hit
				next_wall_index[0] = i;
			}
		}
		
	}
	return result;
}					
					
					
bool Room::scat_ray(const Wall &last_wall,
					const Eigen::VectorXf &hit_point,
					float radius,
					float scat_energy,
					float travel_time,
					float time_thres,
					float sound_speed,
					std::vector<entry> &output){
						
	/*
	 * Trace a one-hop scattering ray from the last wall hit to the microphone.
	 * In case the scattering ray can reach the microphone (no wall in
	 * between), then we add an entry to the output passed by reference.
	 * 
    - last_wall: The wall object where last_hit is located
    - radius : The radius of the circle/sphere microphone
    - last_hit: An array of length 2 or 3 defining the last wall hit position
    - scat_energy: The energy of the scattering ray
    - actual_travel_time: The cumulated travel time of the ray
    - time_thresh: The time threshold of the ray
    - sound_speed: The speed of sound
    * 
    * Return : true if the scattered ray reached the mic, false otw
	*/
	
	bool result = false;
	
	Vector1i next_wall_idx(-1);
	
	// Just to see the evolution of next_wall_idx
	// parameter there_is_prev_wall is set to true since we consider
	// scattered rays only when we hit a wall
	next_wall_hit(hit_point, mic_pos, true, last_wall, next_wall_idx);
	
	
	// No wall intersecting the direct scattered ray
	if (next_wall_idx[0] == -1){
		
		VectorXf mic_hit = mic_intersection(hit_point, mic_pos, mic_pos, radius);
		float hop_dist = (hit_point - mic_hit).norm();
		
		update_travel_time(travel_time, hop_dist, sound_speed);
		
		// Remember that the energy distance attenuation will be 
		// performed once all the ray arrived.
		
		if (travel_time < time_thres){
			append(travel_time, scat_energy, output);
			result = true;
						
		}
		
	}
	
	return result;
	
}

void Room::simul_ray(float phi,
				   float theta,
				   Eigen::VectorXf source_pos,
				   float mic_radius,
				   float scatter_coef,
				   float time_thres,
				   float sound_speed,
				   std::vector<entry> &output){
					   
					   
	/*This function simulates one ray and fills the output vector with
	 * all the entries that can be produced by this ray (either when the
	 * ray reaches the microphone or all the scattered rays)
	 * 
	 * INPUT :
	 * - phi and theta give the orientation of the ray (2D or 3D)
	 * - source_pos is the location of the sound source (NOT AN IMAGE SOURCE)
	 * - mic_radius is the radius of the microphone that is represented like a circle or a sphere
	 * - scatter coef determines the amount of the energy that gets scattered every time the ray hits a wall
	 * - time_thres is the upperbound on the travel time of the ray
	 * - sound_speed is the speed of sound (may be dependent on humidity etc...)
	 * - output is the vector that contains the entries for all the simulated rays */


	// ------------------ INIT --------------------
	// What we need to trace the ray
	float energy = 1000;
	VectorXf start = source_pos;
	VectorXf end = compute_segment_end(start, max_dist, phi, theta);
	
	// The following initializations are arbitrary and does not count since we set
	// the boolean to false
	Wall wall = walls[0];
	Vector1i next_wall_index(0);
	bool there_is_prev_wall = false;
	
	// The ray's characteristics
	float travel_time = 0;
	float total_dist = 0;
	//---------------------------------------------
	
	//------------------ RAY TRACING --------------------
	
	while(true){
		
		
		/* //Debugging
		std::cout << "---\n" << "start : " << start[0] << " " << start[1] << " " << start[2] << std::endl;
		std::cout << "end : " << end[0] << " " << end[1] << " " << end[2] << std::endl;
		std::cout << "there_is_prev_wall : " <<  there_is_prev_wall <<std::endl;
		std::cout << "next_wall_index : " <<  next_wall_index[0] <<std::endl;
		* 
		* */
		
		
		VectorXf hit_point = next_wall_hit(start,
										   end,
										   there_is_prev_wall,
										   wall,
										   next_wall_index);


		// Debugging
		//std::cout << "hit_point : " << hit_point[0]<< " " << hit_point[1]<< " " << hit_point[2] << std::endl;
		
		// The wall that has just been hit
		wall = walls[next_wall_index[0]];
		there_is_prev_wall = true;
		
		// Initialization needed for the next if
		float distance(0);
		
		
		
		
		// Before updatign the ray's characteristic, we must see
		// if the mic is intersected on the [start, hit_point] segment
		
		// - If yes, we compute the energy at the mic_intersection point
		// and we continue the ray
		if (intersects_mic(start, hit_point, mic_pos, mic_radius)){
			
			Eigen::VectorXf hit_point_mic = mic_intersection(start, hit_point, mic_pos, mic_radius);
			
			//std::cout << "MICROPHONE hit_point : " << hit_point_mic[0] << " "<< hit_point_mic[1] << " "<< hit_point_mic[2] << std::endl;
			distance = (start - hit_point_mic).norm();
			
			float travel_time_to_mic = travel_time;

			update_travel_time(travel_time_to_mic, distance, sound_speed);
			
			if (travel_time_to_mic < time_thres){
				append(travel_time_to_mic, energy, output);
			}
			
		}
		
		
		// Update the characteristics
		
		//std::cout << "\n\n-------\nprev travel time : " <<  travel_time <<std::endl;
		//std::cout << "prev energy : " <<  energy <<std::endl;
		distance = (start - hit_point).norm();
		total_dist += distance;
		update_travel_time(travel_time, distance, sound_speed);
		update_energy_wall(energy, wall);
		
		//std::cout << "post travel time : " <<  travel_time <<std::endl;
		//std::cout << "post energy : " <<  energy <<std::endl;
		
		// Check if we reach the time threshold for this ray
		if (travel_time > time_thres){
			break;
		}
		
		// Let's simulate the scattered ray induced by the rebound on
		// the wall
		
		if (scatter_coef > 0){
			
			float ray_scat_energy = compute_scat_energy(energy,
														scatter_coef,
														wall,
														start,
														hit_point,
														mic_pos);
														
			// Shoot the scattered ray
			scat_ray(wall,
					 hit_point,
					 mic_radius,
					 ray_scat_energy,
					 travel_time,
					 time_thres,
					 sound_speed,
					 output);
			
			// The overall ray's energy gets decreased by the total
			// amount of scattered energy
			energy = energy*(1-scatter_coef);
			
		}
				
		// Now we just need to update the start and end positions for
		// next hop
		end = compute_reflected_end(start, hit_point, wall.normal, max_dist);
		start = hit_point;		
	}
	
	
}

std::vector<entry> Room::get_rir_entries(size_t nb_phis,
								   size_t nb_thetas,
								   Eigen::VectorXf source_pos,
								   float mic_radius,
								   float scatter_coef,
								   float time_thres,
								   float sound_speed){
									   
	/*This method simulates a certain number of rays and output the
	 * vector of entries produced by the latter.
	 * 
	 * INPUT
	 * - nb_phis is the number of different planar directions that will be used
	 * - nb_thetas is the number of different elevation angles that will be used
	 *   (NOTE : nb_phis*nb_thetas is the number of simulated raysUT
	 * 
	 * OUTPUT 
	 * - a 2D vector where each entry is a tuple (time, energy)
	 * reprensenting a ray (scattered or not) reaching the microphone*/
	 
	if ( (source_pos-mic_pos).norm() < mic_radius ){
		std::cerr << "The source is inside the microphone ! " <<std::endl;
		throw std::exception();
	}
	
	std::vector<entry> output;
	output.reserve(10000);
	
	for (size_t i(0); i<nb_phis; ++i)  {
		//std::cout << "\n===============\ni="<< i << std::endl;
		float phi = i*2*M_PI/nb_phis;
		
		for (size_t j(0); j<nb_thetas; ++j){
			
			//std::cout << "j=" << j << std::endl;
						
			float theta = j*M_PI/nb_thetas;
			
			// For 2D, this parameter means nothing, but we set it to
			// PI/2 to be consistent
			if(dim==2){
				theta = M_PI_2;
			}
			
			
			simul_ray(phi, theta, source_pos, mic_radius, scatter_coef,
					  time_thres, sound_speed, output);
			
			
			// if we work in 2D rooms, only 1 elevation angle is needed
			if(dim==2){
				break;
			}
		}
		
	}
	
	// Now we must apply the distance attenuation for every ray
	for (size_t k(0); k < output.size(); k++){
		output[k][1] /= (output[k][0] * sound_speed);
	}
	
	return output;

	
}


// In get_rir method, set init_theta to M_PI_2 for 2D rooms















