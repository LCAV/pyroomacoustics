
#include <stdio.h>
#include <math.h>

#include "room.h"

// All the test routines.
int test_ccw3p();
int test_intersection_2d_segments();

// Run them all in the main
int main()
{
  int ret;
  int all_ret = 0;

  // 3 points orientation (clockwise / counter-clockwise / colinear)
  ret = test_ccw3p();
  if (ret != 0)
    printf("Test ccw3p fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // 2d segments intersection
  ret = test_intersection_2d_segments();
  if (ret != 0)
    printf("Test intersection_2D_segments fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // Final report
  if (!all_ret)
    printf("All tests succeeded.\n");
  else
    printf("There were some errors.\n");
}

int test_ccw3p()
{
  float p1[2], p2[2], p3[2];
  int ret;

  // case 1: counter-clockwise
  p1[0] = 0.; p1[1] = 0.;
  p2[0] = 1.; p2[1] = 0.;
  p3[0] = 0.; p3[1] = 1.;

  ret = ccw3p(p1, p2, p3);
  if (ret != 1)
  {
    printf("Test ccw3p counter-clockwise returns %d\n", ret);
    return 1;
  }
  
  // case 1: clockwise
  p1[0] = 0.; p1[1] = 0.;
  p2[0] = 0.; p2[1] = 1.;
  p3[0] = 1.; p3[1] = 0.;

  ret = ccw3p(p1, p2, p3);
  if (ret != -1)
  {
    printf("Test ccw3p clockwise returns %d\n", ret);
    return 2;
  }

  // case2: co-linear 1
  p1[0] = 0.; p1[1] = 0.;
  p2[0] = 0.; p3[1] = 1.;
  p3[0] = 1.; p2[1] = 0.;

  ret = ccw3p(p1, p2, p3);
  if (ret != 0)
  {
    printf("Test ccw3p co-linear returns %d\n", ret);
    return 3;
  }

  return 0;
}

int test_intersection_2d_segments()
{
  float a1[2], a2[2], b1[2], b2[2], p1[2], p2[2];
  int ret, ret2;

  // "normal" intersection
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = 0;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2 || distance(p1, p2, 2) > eps)
  {
    printf("Intersection fails ret=%d (expects %d), p=(%f, %f) (expects (%f, %f))\n",
        ret, ret2, p1[0], p1[1], p2[0], p2[1]);
    return 1;
  }

  // a1 on b1-b2 segment
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 0.; a2[1] = 0.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = 1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2 || distance(p1, p2, 2) > eps)
  {
    printf("Intersection fails ret=%d (expects %d), p=(%f, %f) (expects (%f, %f))\n",
        ret, ret2, p1[0], p1[1], p2[0], p2[1]);
    return 2;
  }

  // a2 on b1-b2 segment
  a1[0] = 0.; a1[1] = 0;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = 1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2 || distance(p1, p2, 2) > eps)
  {
    printf("Intersection fails ret=%d (expects %d), p=(%f, %f) (expects (%f, %f))\n",
        ret, ret2, p1[0], p1[1], p2[0], p2[1]);
    return 3;
  }

  // b1 on a1-a2
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = 0; b1[1] = 0.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = 2;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2 || distance(p1, p2, 2) > eps)
  {
    printf("Intersection fails ret=%d (expects %d), p=(%f, %f) (expects (%f, %f))\n",
        ret, ret2, p1[0], p1[1], p2[0], p2[1]);
    return 4;
  }

  // b2 on a1-a2
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 0.; b2[1] = 0.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = 2;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2 || distance(p1, p2, 2) > eps)
  {
    printf("Intersection fails ret=%d (expects %d), p=(%f, %f) (expects (%f, %f))\n",
        ret, ret2, p1[0], p1[1], p2[0], p2[1]);
    return 5;
  }

  // a1 and a2 on b1-b2 segment (no intersection)
  a1[0] = -0.5; a1[1] = 0.5;
  a2[0] = 0.5; a2[1] = -0.5;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = -1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2)
  {
    printf("Intersection fails ret=%d (expects %d)\n",
        ret, ret2);
    return 6;
  }

  // no intersection
  a1[0] = 0.5; a1[1] = 0.5;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = -1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2)
  {
    printf("Intersection fails ret=%d (expects %d)\n",
        ret, ret2);
    return 7;
  }

  // no intersection
  a1[0] = -1.; a1[1] = -1;
  a2[0] = -0.5; a2[1] = -0.5;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = -1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2)
  {
    printf("Intersection fails ret=%d (expects %d)\n",
        ret, ret2);
    return 8;
  }

  // no intersection
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = 0.5; b1[1] = -0.5;
  b2[0] = 1.; b2[1] = -1.;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = -1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2)
  {
    printf("Intersection fails ret=%d (expects %d)\n",
        ret, ret2);
    return 9;
  }

  // no intersection
  a1[0] = -1.; a1[1] = -1;
  a2[0] = 1.; a2[1] = 1.;
  b1[0] = -1; b1[1] = 1.;
  b2[0] = -0.5; b2[1] = 0.5;
  p2[0] = 0.; p2[1] = 0.;
  ret2 = -1;

  ret = intersection_2d_segments(a1, a2, b1, b2, p1);

  if (ret != ret2)
  {
    printf("Intersection fails ret=%d (expects %d)\n",
        ret, ret2);
    return 10;
  }

  return 0;

}

int test_interesection_segment_plane()
{
  int ret, ret_e;
  float a1[3], a2[3], p[3], normal[3], intersection[3], inter_e[3];
  int i;


  // plane normal to [1,1,1]/sqrt(3) containing point [2,2,2]
  p[0] = 2.; p[1] = 2.; p[2] = 2.;
  normal[0] = 1./sqrtf(3); normal[1] = 1./sqrtf(3); normal[2] = 1./sqrtf(3);

  // plant the intersection in the plane
  inter_e[0] = p[0] + 0.5; inter_e[1] = p[1] + 0.5; inter_e[2] = p[2] - 1.;
  
  // intersection happens, points outside plane
  ret_e = 0;

  a1[0] = -1; a1[1] = -1.; a1[1] = -1;
  for (i = 0 ; i < 3 ; i++)
    a2[i] = 2 * inter_e[i] - a1[i];

  ret = intersection_segment_plane(a1, a2, p, normal, intersection);
  if (ret != ret_e || distance(intersection, inter_e, 3) > eps)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, intersection[0], intersection[1], intersection[2], 
        inter_e[0], inter_e[1], inter_e[2]);
    return 1;
  }

  // intersection happens, points on plane
  ret_e = 1;

  a1[0] = -1; a1[1] = -1.; a1[1] = -1;
  for (i = 0 ; i < 3 ; i++)
    a2[i] = inter_e[i];

  ret = intersection_segment_plane(a1, a2, p, normal, intersection);
  if (ret != ret_e || distance(intersection, inter_e, 3) > eps)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, intersection[0], intersection[1], intersection[2], 
        inter_e[0], inter_e[1], inter_e[2]);
    return 2;
  }

  ret = intersection_segment_plane(a2, a1, p, normal, intersection);
  if (ret != ret_e || distance(intersection, inter_e, 3) > eps)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, intersection[0], intersection[1], intersection[2], 
        inter_e[0], inter_e[1], inter_e[2]);
    return 3;
  }
  
  // no intersection
  inter_e[0] = p[0] + 0.5; inter_e[1] = p[1] + 0.5; inter_e[2] = p[2] - 1.;
  ret_e = -1;

  a1[0] = -1; a1[1] = -1.; a1[1] = -1;
  for (i = 0 ; i < 3 ; i++)
    a2[i] = 0.9 * inter_e[i] + a1[i];

  ret = intersection_segment_plane(a1, a2, p, normal, intersection);
  if (ret != ret_e)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d)\n", ret, ret_e);
    return 4;
  }

  // no intersection but on other side
  a1[0] = -1; a1[1] = -1.; a1[1] = -1;
  for (i = 0 ; i < 3 ; i++)
  {
    a2[i] = 1.1 * inter_e[i] + a1[i];
    a1[i] = 2 * inter_e[i] + a1[i];
  }

  ret = intersection_segment_plane(a2, a1, p, normal, intersection);
  if (ret != ret_e || distance(intersection, inter_e, 3) > eps)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, intersection[0], intersection[1], intersection[2], 
        inter_e[0], inter_e[1], inter_e[2]);
    return 5;
  }

  return 0;

}


