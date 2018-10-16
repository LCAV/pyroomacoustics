
#include "room.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

int image_source_shoebox(room_t *room, float *source, float *room_size, float *absorption, int max_order)
{
  int i, d;

  // precompute powers of the absorption coefficients
  float *transmission_pwr = (float *)malloc((max_order + 1) * 2 * room->dim * sizeof(int));
  for (d = 0 ; d < 2 * room->dim ; d++)
    transmission_pwr[d] = 1.;
  for (i = 1 ; i <= max_order ; i++)
    for (d = 0 ; d < 2 * room->dim ; d++)
      transmission_pwr[i * 2 * room->dim + d] = (1. - absorption[d]) * transmission_pwr[(i-1)*2*room->dim + d];

  /*
  for (i = 0 ; i <= max_order ; i++)
    for (d = 0 ; d < 2 * room->dim ; d++)
      printf("abs wall=%d pwr=%d abs=%f\n", d, i, transmission_pwr[i * 2 * room->dim + d]);
  */

  // The linked list of image sources
  is_ll_t *source_list = NULL;

  // L1 ball of room images
  int point[3] = {0, 0, 0};

  // Take 2D case into account
  int z_max = max_order;
  if (room->dim == 2)
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
        is_ll_t *node = (is_ll_t *)malloc(sizeof(is_ll_t));
        node->is.order = 0;
        node->is.attenuation = 1.;
        node->is.gen_wall = -1;
        node->visible_mics = NULL;

        // Now compute the reflection, the order, and the multiplicative constant
        for (d = 0 ; d < room->dim ; d++)
        {
          // Compute the reflected source
          float step = abs(point[d]) % 2 == 1 ? room_size[d] - source[d] : source[d];
          node->is.loc[d] = point[d] * room_size[d] + step;

          // source order is just the sum of absolute values of reflection indices
          node->is.order += abs(point[d]);

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
          float a1 = transmission_pwr[2 * room->dim * p1 + 2*d];
          float a2 = transmission_pwr[2 * room->dim * p2 + 2*d+1];
          /*
          printf("(%d,%d) dim=%d p1=%d a[%d]=%f p2=%d a[%d]=%f\n", point[0], point[1], d, 
              p1, 2 * room->dim * p1 + 2*d, a1, 
              p2, 2 * room->dim * p2 + 2*d+1, a2);
          */
          node->is.attenuation *= a1;       // 'west' absorption factor
          node->is.attenuation *= a2; // 'east' absorption factor
        }

        // add to list and increment counter
        is_list_insert(&source_list, node);
      }
    }
  }

  // Clean up allocated memory
  free(transmission_pwr);

  // fill linear arrays and return status
  return fill_sources(room, &source_list);
}
