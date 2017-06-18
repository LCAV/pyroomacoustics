
#include "room.h"

#include <stdlib.h>
#include <stdio.h>

// The number of threads to use for multithreading
int num_threads = 1;

// The head of the linked list of visible image sources
is_ll_t *visible_sources = NULL;

void set_num_threads(int n)
{
  num_threads = n;
}

void check_visibility_all(room_t *room)
{
  /*
   * Just a big loop checking all sources and mics
   * at some point, parallelize with pthreads
   */
  int m, s;
  float *mic_loc;

  for (m = 0 ; m < room->n_microphones ; m++)
  {
    mic_loc = room->microphones + m * room->dim;
    for (s = 0 ; s < room->n_sources ; s++)
      room->is_visible[m * room->n_sources + s] = is_visible(room, mic_loc, s);
  }
}

int is_visible(room_t *room, float *p, int image_id)
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

  if (is_obstructed(room, p, image_id))
    return 0;

  if (room->orders[image_id] > 0)
  {
    float intersection[3];

    // get generating wall id
    int wall_id = room->gen_walls[image_id];

    // check if the generating wall is intersected
    int ret = wall_intersection(room->walls + wall_id, 
                                p, 
                                room->sources + image_id * room->dim, 
                                intersection);

    // The source is not visible if the ray does not intersect
    // the generating wall
    if (ret >= 0)
      // Check visibility of intersection point from parent source
      return is_visible(room, intersection, room->parents[image_id]);
    else
      return 0;
  }

  // If we get here this is the original, unobstructed, source
  return 1;
}

int is_obstructed(room_t *room, float *p, int image_id)
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
  int ow;
  int gen_wall_id = room->gen_walls[image_id];

  // Check candidate walls for obstructions
  for (ow = 0 ; ow < room->n_obstructing_walls ; ow++)
  {
    int wall_id = room->obstructing_walls[ow];

    // generating wall can't be obstructive
    if (wall_id != gen_wall_id)
    {
      float intersection[3];
      int ret = wall_intersection(room->walls + wall_id,
                                  room->sources + image_id * room->dim,
                                  p,
                                  intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == 0 || ret == 2)
      {
        if (room->orders[image_id] > 0)
        {
          // Test if the intersection point and the image are on
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = wall_side(room->walls + gen_wall_id, room->sources + image_id * room->dim);
          int intersection_side = wall_side(room->walls + gen_wall_id, intersection);

          if (img_side != intersection_side && intersection_side != 0)
            return 1;
        }
        else
          return 1;
      }

    }

  }

  return 0;

}

int image_source_model(room_t *room, float *source_location, int max_order)
{
  int d;
  image_source_t source;

  // That's the original source
  for (d = 0 ; d < room->dim ; d++)
    source.loc[d] = source_location[d];
  source.order = 0;
  source.parent = NULL;
  source.attenuation = 1.;
  source.gen_wall = -1;

  image_sources_dfs(room, &source, max_order);

  // fill the sources array in room and return
  return fill_sources(room, &visible_sources);
}

int fill_sources(room_t *room, is_ll_t **stack)
{
  int i, d;
  room->n_sources = is_list_count(*stack);

  // Create linear arrays to store the image sources
  if (room->n_sources > 0)
  {
    room->sources = (float *)malloc(room->n_sources * room->dim * sizeof(float));
    room->orders = (int *)malloc(room->n_sources * sizeof(int));
    room->gen_walls = (int *)malloc(room->n_sources * sizeof(int));
    room->attenuations = (float *)malloc(room->n_sources * sizeof(float));
    room->is_visible = (int *)malloc(room->n_sources * room->n_microphones * sizeof(int));
    room->parents = NULL;
    if (room->sources == NULL || room->orders == NULL || room->gen_walls == NULL || room->attenuations == NULL || room->is_visible == NULL)
    {
      fprintf(stderr, "Error: Couldn't allocate memory.\n");
      return -1;
    }

    // Copy from linked list to linear array (in reverse order)
    i = room->n_sources - 1;
    while (*stack != NULL)
    {
      for (d = 0 ; d < room->dim ; d++)
        room->sources[i * room->dim + d] = (*stack)->is.loc[d];
      room->orders[i] = (*stack)->is.order;
      room->gen_walls[i] = (*stack)->is.gen_wall;
      room->attenuations[i] = (*stack)->is.attenuation;
      for (d = 0 ; d < room->n_microphones ; d++)
        if ((*stack)->visible_mics != NULL)
          room->is_visible[d * room->n_sources + i] = (*stack)->visible_mics[d];
        else
          room->is_visible[d * room->n_sources + i] = 1;  // for shoebox room, all mics are visible

      // Remove top of linked list
      is_list_pop(stack);

      i--;
    }
  }
  else
  {
    // No source is visible
    room->sources = NULL;
    room->orders = NULL;
    room->gen_walls = NULL;
    room->attenuations = NULL;
    room->is_visible = NULL;
    room->parents = NULL;
  }

  return 0;
}

void free_sources(room_t *room)
{
  /* free all the malloc'ed stuff here */
  if (room->sources != NULL)
    free(room->sources);
  if (room->orders != NULL)
    free(room->orders);
  if (room->gen_walls != NULL)
    free(room->gen_walls);
  if (room->attenuations != NULL)
    free(room->attenuations);
  if (room->is_visible != NULL)
    free(room->is_visible);

  room->sources = room->attenuations = NULL;
  room->orders = room->gen_walls = room->is_visible = NULL;
}

// Performs a depth first search of visible image sources
void image_sources_dfs(room_t *room,  image_source_t *is, int max_order)
{
  wall_t *wall;
  int wi, mic;
  int visible_mics[room->n_microphones];
  int dir;
  image_source_t new_is;

  // Check the visibility of the source from the different microphones
  int any_visible = 0;
  for (mic = 0 ; mic < room->n_microphones ; mic++)
  {
    visible_mics[mic] = is_visible_dfs(room, room->microphones + mic * room->dim, is);
    any_visible = any_visible || visible_mics[mic];
  }
  
  // If any of the microphone is visible, add to the linked list
  if (any_visible)
  {
    // add the image source to the list
    is_ll_t *new_node = (is_ll_t *)malloc(sizeof(is_ll_t));

    new_node->is = *is;

    new_node->visible_mics = (int *)malloc(room->n_microphones * sizeof(int));
    for (mic = 0 ; mic < room->n_microphones ; mic++)
      new_node->visible_mics[mic] = visible_mics[mic];

    //insert_visible_source(new_node);
    is_list_insert(&visible_sources, new_node);
  }

  // If we reached maximal depth, stop
  if (max_order == 0)
    return;

  // Then, check all the reflections across the walls
  for (wi=0 ;  wi < room->n_walls ; wi++)
  {
    wall = room->walls + wi;
    dir = wall_reflect(wall, is->loc, new_is.loc);

    // We only check valid reflections (normals should point outward from the room
    if (dir <= 0)
      continue;

    new_is.attenuation = is->attenuation * (1. - wall->absorption);
    new_is.order = is->order + 1;
    new_is.gen_wall = wi;
    new_is.parent = is;


    // Run the DFS recursion
    image_sources_dfs(room, &new_is, max_order - 1);
  }

}

int is_visible_dfs(room_t *room, float *p, image_source_t *image)
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

  if (is_obstructed_dfs(room, p, image))
    return 0;

  if (image->parent != NULL)
  {
    float intersection[3];

    // get generating wall id
    int wall_id = image->gen_wall;

    // check if the generating wall is intersected
    int ret = wall_intersection(room->walls + wall_id, 
                                p, 
                                image->loc, 
                                intersection);

    // The source is not visible if the ray does not intersect
    // the generating wall
    if (ret >= 0)
      // Check visibility of intersection point from parent source
      return is_visible_dfs(room, intersection, image->parent);
    else
      return 0;
  }

  // If we get here this is the original, unobstructed, source
  return 1;
}

int is_obstructed_dfs(room_t *room, float *p, image_source_t *image)
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
  int ow;
  int gen_wall_id = image->gen_wall;

  // Check candidate walls for obstructions
  for (ow = 0 ; ow < room->n_obstructing_walls ; ow++)
  {
    int wall_id = room->obstructing_walls[ow];

    // generating wall can't be obstructive
    if (wall_id != gen_wall_id)
    {
      float intersection[3];
      int ret = wall_intersection(room->walls + wall_id,
                                  image->loc,
                                  p,
                                  intersection);

      // There is an intersection and it is distinct from segment endpoints
      if (ret == 0 || ret == 2)
      {
        if (image->parent != NULL)
        {
          // Test if the intersection point and the image are on
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = wall_side(room->walls + gen_wall_id, image->loc);
          int intersection_side = wall_side(room->walls + gen_wall_id, intersection);

          if (img_side != intersection_side && intersection_side != 0)
            return 1;
        }
        else
          return 1;
      }

    }

  }

  return 0;

}
