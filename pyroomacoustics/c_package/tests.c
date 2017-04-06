
#include <stdio.h>
#include <math.h>

#include "room.h"

// All the test routines.
int test_side();
int test_ccw3p();
int test_intersection_2d_segments();
int test_intersection_segment_plane();
int test_intersection_segment_wall_3d();
int test_is_inside_2d_polygon();
int test_wall_reflection();

// Run them all in the main
int main()
{
  int ret;
  int all_ret = 0;

  // Test side detection for point / line or point / plane
  ret = test_side();
  if (ret != 0)
    printf("Test side fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // 3 points orientation (clockwise / counter-clockwise / colinear)
  ret = test_ccw3p();
  if (ret != 0)
    printf("Test ccw3p fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // 2d segments intersection
  ret = test_intersection_2d_segments();
  if (ret != 0)
    printf("Test intersection_2d_segments fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // 3d plane segment intersection
  ret = test_intersection_segment_plane();
  if (ret != 0)
    printf("Test intersection_segment_plane fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  ret = test_intersection_segment_wall_3d();
  if (ret != 0)
    printf("Test intersection_segment_wall_3d fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  ret = test_is_inside_2d_polygon();
  if (ret != 0)
    printf("Test is_inside_2d_polygon fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  ret = test_wall_reflection();
  if (ret != 0)
    printf("Test wall_reflect fails with ret=%d\n", ret);
  all_ret = all_ret || ret;

  // Final report
  if (!all_ret)
    printf("All tests succeeded.\n");
  else
    printf("There were some errors.\n");
}

int test_side()
{
  int ret, ret_e;
  float p[2] = {-1, 0};
  wall_t wall;

  wall.dim = 2;
  wall.origin[0] = 0; wall.origin[1] = 0;
  wall.normal[0] = 1; wall.normal[1] = 0;
  

  /* 2d, left */
  ret_e = -1;

  ret = wall_side(&wall, p);
  if (ret != ret_e)
  {
    printf("Test wall_side left returns %d (expected %d)\n", ret, ret_e);
    return 1;
  }

  /* 2d, right */
  p[0] = 1; p[1] = 0;
  ret_e = 1;

  ret = wall_side(&wall, p);
  if (ret != ret_e)
  {
    printf("Test wall_side right returns %d (expected %d)\n", ret, ret_e);
    return 2;
  }

  /* 2d, middle */
  p[0] = 0; p[1] = 1;
  ret_e = 0;

  ret = wall_side(&wall, p);
  if (ret != ret_e)
  {
    printf("Test wall_side middle returns %d (expected %d)\n", ret, ret_e);
    return 3;
  }

  return 0;

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

int test_intersection_segment_plane()
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

  a1[0] = -1; a1[1] = -1.; a1[2] = -1;
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

  a1[0] = -1; a1[1] = -1.; a1[2] = -1;
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

  a1[0] = -1; a1[1] = -1.; a1[2] = -1;
  for (i = 0 ; i < 3 ; i++)
    a2[i] = 0.9 * inter_e[i] + a1[i];

  ret = intersection_segment_plane(a1, a2, p, normal, intersection);
  if (ret != ret_e)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d)\n", ret, ret_e);
    return 4;
  }

  // no intersection but on other side
  a1[0] = -1; a1[1] = -1.; a1[2] = -1;
  for (i = 0 ; i < 3 ; i++)
  {
    a2[i] = 1.1 * inter_e[i] + a1[i];
    a1[i] = 2 * inter_e[i] + a1[i];
  }
  a1[0] = a1[1] = a1[2] = -3;
  a2[0] = a2[1] = a2[2] = -4;

  ret = intersection_segment_plane(a2, a1, p, normal, intersection);
  if (ret != ret_e || distance(intersection, inter_e, 3) > eps)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, intersection[0], intersection[1], intersection[2], 
        inter_e[0], inter_e[1], inter_e[2]);
    return 5;
  }

  normal[0] = 1; normal[1] = normal[2] = 0;
  p[0] = 3; p[1] = p[2] = 0;
  a1[0] = -1.5; a1[1] = 1.2; a1[2] = 0.5;
  a2[0] = 1.5; a2[1] = 1.2; a2[2] = 0.5;
  ret_e = -1;
  ret = intersection_segment_plane(a1, a2, p, normal, intersection);
  if (ret != ret_e)
  {
    printf("Intersection seg/plane fails ret=%d (expects %d)\n", ret, ret_e);
    return 6;
  }


  return 0;

}

int test_intersection_segment_wall_3d()
{
  int ret, ret_e;
  int test_result = 0;
  float inters[3], inters_e[3], p1[3], p2[3];
  float corners[12] = { 0, 0, 0,  4, 0, 0,  4, 4, 0,  0, 4, 0 };
  wall_t *wall = new_wall(3, 4, corners, 1.);

  /* through */
  p1[0] = p1[1] = p1[2] = 2;
  p2[0] = p2[1] = 2; p2[2] = -2;
  ret_e = 0;
  inters_e[0] = 2; inters_e[1] = 2; inters_e[2] = 0;

  ret = intersection_segment_wall_3d(p1, p2, wall, inters);
  
  if (ret != ret_e || distance(inters_e, inters, 3) > eps)
  {
    printf("Intersection seg/wall 3d fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, inters[0], inters[1], inters[2], 
        inters_e[0], inters_e[1], inters_e[2]);

    test_result = 1;
    goto test_end;
  }

  /* touching */
  ret_e = 1;

  ret = intersection_segment_wall_3d(p1, inters_e, wall, inters);
  
  if (ret != ret_e || distance(inters_e, inters, 3) > eps)
  {
    printf("Intersection seg/wall 3d fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, inters[0], inters[1], inters[2], 
        inters_e[0], inters_e[1], inters_e[2]);

    test_result = 2;
    goto test_end;
  }

  /* border */
  p1[0] = p1[1] = p2[0] = p2[1] = 0;
  p1[2] = 2;
  p2[2] = -2;
  ret_e = 2;
  inters_e[0] = inters_e[1] = inters_e[2] = 0;

  ret = intersection_segment_wall_3d(p1, p2, wall, inters);

  if (ret != ret_e || distance(inters_e, inters, 3) > eps)
  {
    printf("Intersection seg/wall 3d fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, inters[0], inters[1], inters[2], 
        inters_e[0], inters_e[1], inters_e[2]);

    test_result = 3;
    goto test_end;
  }

  /* touching border */
  p1[0] = p1[1] = p2[0] = p2[1] = p2[2] = 0;
  p1[2] = 2;
  ret_e = 3;
  inters_e[0] = inters_e[1] = inters_e[2] = 0;

  ret = intersection_segment_wall_3d(p1, p2, wall, inters);

  if (ret != ret_e || distance(inters_e, inters, 3) > eps)
  {
    printf("Intersection seg/wall 3d fails ret=%d (expects %d), [%f,%f,%f] but expects [%f,%f,%f]\n",
        ret, ret_e, inters[0], inters[1], inters[2], 
        inters_e[0], inters_e[1], inters_e[2]);

    test_result = 4;
    goto test_end;
  }

  /* miss */
  p1[0] = p1[1] = p2[0] = p2[1] = -1;
  p1[2] = 2;
  p2[2] = -2;
  ret_e = -1;
  inters_e[0] = inters_e[1] = inters_e[2] = 0;

  ret = intersection_segment_wall_3d(p1, p2, wall, inters);

  if (ret != ret_e)
  {
    printf("Intersection seg/wall 3d fails ret=%d (expects %d)\n", ret, ret_e);

    test_result = 5;
    goto test_end;
  }


test_end:
  free_wall(wall);
  return test_result;

}

int test_is_inside_2d_polygon()
{
  int ret, ret_e;
  float polygon1[8] = { 0, 0, 4, 0, 4, 4, 0, 4 };
  float polygon2[12] = { 0, 0, 0, 1, 1, 1, 1, 2, 3, 2, 3, 0 };
  float polygon3[10] = { 0, 0, 1, 1, 1, 2, 3, 2, 3, 0 };
  float p[2];

  /* inside */
  p[0] = p[1] = 2;
  ret_e = 0;

  ret = is_inside_2d_polygon(p, polygon1, 4);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 1;
  }

  /* on border */
  p[0] = 0; p[1] = 2;
  ret_e = 1;

  ret = is_inside_2d_polygon(p, polygon1, 4);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 2;
  }

  /* on corner */
  p[0] = p[1] = 4;
  ret_e = 1;

  ret = is_inside_2d_polygon(p, polygon1, 4);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 3;
  }

  /* outside */
  p[0] = p[1] = 5;
  ret_e = -1;

  ret = is_inside_2d_polygon(p, polygon1, 4);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 4;
  }

  /* horizontal wall aligned with point */
  p[0] = 2; p[1] = 1;
  ret_e = 0;

  ret = is_inside_2d_polygon(p, polygon2, 6);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 5;
  }

  /* ray is going through vertex */
  p[0] = 2; p[1] = 1;
  ret_e = 0;

  ret = is_inside_2d_polygon(p, polygon3, 5);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 6;
  }

  /* point is at the same height as top of polygon, but outside */
  p[0] = 4; p[1] = 2;
  ret_e = -1;

  ret = is_inside_2d_polygon(p, polygon3, 5);

  if (ret != ret_e)
  {
    printf("Point in polygon failed ret=%d (expects %d)\n", ret, ret_e);

    return 7;
  }


  return 0;
}

int test_wall_reflection()
{
  float corners_3d[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  float corners_2d[4] = { 1, 0, 0, 1 };
  float p2d[2] = { -1, -1 };
  float p3d[3] = { -1, -1, -1 };
  float reflection_exp_2d[2] = { 2, 2 };
  float reflection_exp_3d[3] = { 5./3., 5./3., 5./3. };
  float reflection[3];
  int ret, ret_e;

  wall_t *wall2d = new_wall(2, 2, corners_2d, 1.);
  wall_t *wall3d = new_wall(3, 3, corners_3d, 1.);

  ret = wall_reflect(wall2d, p2d, reflection);
  ret_e = 1;
  if (ret != ret_e || distance(reflection, reflection_exp_2d, 2) > eps)
  {
    printf("Test wall_reflect fails ret=%d (expected %d) point=[%f, %f] (expected [%f, %f])\n",
        ret, ret_e, reflection[0], reflection[1], reflection_exp_2d[0], reflection_exp_2d[1]);
    return 1;
  }

  ret = wall_reflect(wall3d, p3d, reflection);
  ret_e = 1;
  if (ret != ret_e || distance(reflection, reflection_exp_3d, 3) > eps)
  {
    printf("Test wall_reflect fails ret=%d (expected %d) point=[%f, %f, %f] (expected [%f, %f, %f])\n",
        ret, ret_e, reflection[0], reflection[1], reflection[2], reflection_exp_3d[0], reflection_exp_3d[1], reflection_exp_3d[2]);
    return 2;
  }

  free_wall(wall2d);
  free_wall(wall3d);

  return 0;
}

