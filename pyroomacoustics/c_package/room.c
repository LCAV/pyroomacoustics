
#include "room.h"

#include <stdlib.h>
#include <stdio.h>

int num_threads = 1;

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
  int wall_id;
  int gen_wall_id = room->gen_walls[image_id];

  // Check candidate walls for obstructions
  for (wall_id = 0 ; wall_id < room->n_obstructing_walls ; wall_id++)
  {
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
        if (room->orders[image_id > 0])
        {
          // Test if the intersection point and the image are at
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
