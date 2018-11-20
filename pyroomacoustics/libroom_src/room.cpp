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
          int img_side = walls[wall_id].side(is.loc);
          int intersection_side = walls[wall_id].side(intersection);

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
