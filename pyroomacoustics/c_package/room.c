
#include "room.h"

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
    int wall_id = room->gen_walls[image_id];
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
  int wall_id = room->gen_walls[image_id];

  // Check candidate walls for obstructions
  for (ow = 0 ; ow < room->n_obstructing_walls ; ow++)
  {
    // generating wall can't be obstructive
    if (ow != wall_id)
    {
      float intersection[3];
      int ret = wall_intersection(room->walls + ow,
                                  room->sources + image_id * room->dim,
                                  p,
                                  intersection);
      // There is an intersection and it is distinct from segment endpoints
      if (ret >= 0 && ~(ret & 1))
      {
        if (room->orders[image_id > 0])
        {
          // Test if the intersection point and the image are at
          // opposite sides of the generating wall 
          // We ignore the obstruction if it is inside the
          // generating wall (it is what happens in a corner)
          int img_side = wall_side(room->walls + wall_id, room->sources + image_id * room->dim);
          int intersection_side = wall_side(room->walls + wall_id, intersection);

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
