#ifndef __ROOM_H__
#define __ROOM_H__

/*
 * Structure that can store a 2D or 3D wall
 * corners is a flat list of coordinates of size (dim * n_corners)
 */
typedef struct wall_struct
{
  int dim;
  float absorption;
  float normal[3];
  int n_corners;
  float *corners;

  /* for 3D wall, provide local basis for plane of wall */
  float origin[3];
  float basis[6];
  float *flat_corners;  /* corners projected in plane of wall */
} 
wall_t;

/*
 * Structure for a room as a list of walls
 * with a few sources and microphones around
 */
typedef struct room_struct
{
  int dim;
  int n_walls;
  wall_t *walls;

  // This is a list of image sources
  int n_sources;
  float *sources;
  int *parents;
  int *gen_walls;
  int *orders;

  // List of obstructing walls
  int n_obstructing_walls;
  int *obstructing_walls;

  // The microphones are in the room
  int n_microphones;
  float *microphones;
  
  // This array will get filled by visibility status
  // its size is n_microphones * n_sources
  int *is_visible;
}
room_t;

/* Linear algebra routines */
float distance(float *p1, float *p2, int dim);
float inner(float *p1, float *p2, int dim);
void cross(float *p1, float *p2, float *xprod);
float norm(float *p, int dim);
void normalize(float *p, int dim);
void gram_schmidt(float *vec, int n_vec, int dim);
void print_vec(float *p, int dim);

/* segment line/plane intersection routines */
wall_t *new_wall(int dim, int n_corners, float *corners, float absorption);
void free_wall(wall_t *wall);
int wall_side(wall_t *wall, float *p);
int ccw3p(float *p1, float *p2, float *p3);
int wall_intersection(wall_t *wall, float *p1, float *p2, float *intersection);
int check_intersection_2d_segments(float *a1, float *a2, float *b1, float *b2);
int intersection_2d_segments(float *a1, float *a2, float *b1, float *b2, float *p);
int intersection_segment_plane(float *a1, float *a2, float *p, float *normal, float *intersection);
int intersection_segment_wall_3d(float *a1, float *a2, wall_t *wall, float *intersection);
int is_inside_2d_polygon(float *p, float *corners, int n_corners);

/* visibility and obstruction routines */
void check_visibility_all(room_t *room);
int is_visible(room_t *room, float *p, int image_id);
int is_obstructed(room_t *room, float *p, int image_id);

void set_num_threads(int n);


extern float eps;

#endif /* __ROOM_H__ */
