#include <iostream>
#include "room.hpp"

Room::Room(py::list _walls, py::list _obstructing_walls, const Eigen::MatrixXf _microphones)
  : microphones(_microphones)
{
  for (auto wall : _walls)
    walls.push_back(wall.cast<Wall>());

  for (auto owall : _obstructing_walls)
    obstructing_walls.push_back(owall.cast<int>());

  dim = walls[0].dim;
}

void Room::check_visibility_all()
{
  /*
   * Just a big loop checking all sources and mics
   * at some point, parallelize with pthreads
   */
  visible_mics.resize(microphones.cols(), sources.cols());

  for (int m = 0 ; m < microphones.cols() ; m++)
    for (int s = 0 ; s < sources.cols() ; s++)
      visible_mics.coeffRef(m, s) = is_visible(microphones.col(m), s);
}

int Room::image_source_model(const Eigen::VectorXf source_location, int max_order)
{
  /*
   * This is the top-level method to run the image source model
   */
  visible_sources.clear();  // make sure the list is empty

  // add the original (real) source
  visible_sources.push_back(ImageSource());
  ImageSource &real_source = visible_sources[0];
  real_source.loc.resize(dim);

  real_source.loc = source_location;
  real_source.attenuation = 1.;
  real_source.order = 0;
  real_source.gen_wall = -1;
  real_source.parent = -1;

  // Run the image source model algorithm
  image_sources_dfs(0, max_order);

  // fill the sources array in room and return
  return fill_sources();
}

int Room::fill_sources()
{
  int n_sources = visible_sources.size();

  // Create linear arrays to store the image sources
  if (n_sources > 0)
  {
    sources.resize(dim, n_sources);
    parents.resize(n_sources);
    orders.resize(n_sources);
    gen_walls.resize(n_sources);
    attenuations.resize(n_sources);
    visible_mics.resize(microphones.cols(), n_sources);

    for (int i = 0 ; i < n_sources ; i++)
    {
      sources.col(i) = visible_sources[i].loc;
      parents.coeffRef(i) = visible_sources[i].parent;
      gen_walls.coeffRef(i) = visible_sources[i].gen_wall;
      orders.coeffRef(i) = visible_sources[i].order;
      attenuations.coeffRef(i) = visible_sources[i].attenuation;
      visible_mics.col(i) = visible_sources[i].visible_mics;
    }
  }

  return 0;
}

void Room::image_sources_dfs(int image_id, int max_order)
{
  /*
   * This function runs a depth first search (DFS) on the tree of image sources
   */

  // Check the initial size of the IS stack
  size_t size_before = visible_sources.size();

  // Check the visibility of the source from the different microphones
  int any_visible = 0;
  visible_sources[image_id].visible_mics.resize(microphones.cols());
  for (int mic = 0 ; mic < microphones.cols() ; mic++)
  {
    visible_sources[image_id].visible_mics.coeffRef(mic) = is_visible_dfs(microphones.col(mic), image_id);
    any_visible = any_visible || visible_sources[image_id].visible_mics.coeff(mic);
  }
  
  // If we reached maximal depth, stop
  if (max_order == 0)
    return;

  // Then, check all the reflections across the walls
  for (size_t wi=0 ;  wi < walls.size() ; wi++)
  {
    visible_sources.push_back(ImageSource());  // add a new image source
    ImageSource &new_is = visible_sources.back();  // ref to newly added image source
    new_is.loc.resize(dim);

    int dir = walls[wi].reflect(visible_sources[image_id].loc, new_is.loc);  // the reflected location
    // We only check valid reflections (normals should point outward from the room
    if (dir <= 0)
    {
      visible_sources.pop_back();  // remove newly created source
      continue;
    }

    // The reflection is valid, fill in the image source attributes
    new_is.attenuation = visible_sources[image_id].attenuation * (1. - walls[wi].absorption);
    new_is.order = visible_sources[image_id].order + 1;
    new_is.gen_wall = wi;
    new_is.parent = image_id;

    // Run the DFS recursion (on the last element in the array, the one we just added)
    image_sources_dfs(visible_sources.size() - 1, max_order - 1);
  }

  if (visible_sources.size() == size_before)
    visible_sources.pop_back();

}

bool Room::is_visible_dfs(const Eigen::Ref<Eigen::VectorXf> p, int image_id)
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

  if (is_obstructed_dfs(p, image_id))
    return false;

  if (image_id > 0)
  {
    Eigen::VectorXf intersection;
    intersection.resize(dim);

    // get generating wall id
    int wall_id = visible_sources[image_id].gen_wall;

    // check if the generating wall is intersected
    int ret = walls[wall_id].intersection(p, visible_sources[image_id].loc, intersection);

    // The source is not visible if the ray does not intersect
    // the generating wall
    if (ret >= 0)
      // Check visibility of intersection point from parent source
      return is_visible_dfs(intersection, visible_sources[image_id].parent);
    else
      return false;
  }

  // If we get here this is the original, unobstructed, source
  return true;
}

bool Room::is_obstructed_dfs(const Eigen::Ref<Eigen::VectorXf> p, int image_id)
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
  int gen_wall_id = visible_sources[image_id].gen_wall;

  // Check candidate walls for obstructions
  for (size_t ow = 0 ; ow < obstructing_walls.size() ; ow++)
  {
    int wall_id = obstructing_walls[ow];

    // generating wall can't be obstructive
    if (wall_id != gen_wall_id)
    {
      Eigen::VectorXf intersection;
      intersection.resize(dim);
      int ret = walls[wall_id].intersection(visible_sources[image_id].loc, p, intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == WALL_ISECT_VALID || ret == WALL_ISECT_VALID_BNDRY)
      {
        if (visible_sources[image_id].parent != -1)
        {
          // Test if the intersection point and the image are on
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = walls[wall_id].side(visible_sources[image_id].loc);
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

bool Room::is_visible(const Eigen::Ref<Eigen::VectorXf> p, int image_id)
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

  if (is_obstructed(p, image_id))
    return false;

  if (orders[image_id] > 0)
  {
    Eigen::VectorXf intersection;
    intersection.resize(p.size());

    // get generating wall id
    int wall_id = gen_walls[image_id];

    // check if the generating wall is intersected
    int ret = walls[wall_id].intersection(p, sources.col(image_id), intersection);

    // The source is not visible if the ray does not intersect
    // the generating wall
    if (ret >= 0)
      // Check visibility of intersection point from parent source
      return is_visible(intersection, parents[image_id]);
    else
      return false;
  }

  // If we get here this is the original, unobstructed, source
  return true;
}

bool Room::is_obstructed(const Eigen::Ref<Eigen::VectorXf> p, int image_id)
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
  int gen_wall_id = gen_walls[image_id];

  // Check candidate walls for obstructions
  for (size_t ow = 0 ; ow < obstructing_walls.size() ; ow++)
  {
    // reference to the current possibly obstructing wall
    int wall_id = obstructing_walls[ow];

    // generating wall can't be obstructive
    if (wall_id != gen_wall_id)
    {
      Eigen::VectorXf intersection;
      intersection.resize(p.size());
      int ret = walls[wall_id].intersection(sources.col(image_id), p, intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == WALL_ISECT_VALID || ret == WALL_ISECT_VALID_BNDRY)
      {
        if (orders[image_id] > 0)
        {
          // Test if the intersection point and the image are on
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = walls[wall_id].side(sources.col(image_id));
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
