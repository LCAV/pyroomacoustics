#ifndef __CWALL_H__
#define __CWALL_H__

#include <Eigen/Dense>

extern float libroom_eps;

#define WALL_ISECT_NONE        -1  // if there is no intersection
#define WALL_ISECT_VALID        0  // if the intersection striclty between the segment endpoints and in the polygon interior
#define WALL_ISECT_VALID_ENDPT  1  // if the intersection is at endpoint of segment
#define WALL_ISECT_VALID_BNDRY  2  // if the intersection is at boundary of polygon
#define ENDPOINT_BOUNDARY       3  // if both the above are true

class Wall
{
  public:

    int dim;
    float absorption;
    
    Eigen::VectorXf  normal;
    Eigen::MatrixXf corners;

    /* for 3D wall, provide local basis for plane of wall */
    Eigen::VectorXf origin;
    Eigen::MatrixXf basis;
    Eigen::MatrixXf flat_corners;

    // Constructor
    Wall(const Eigen::MatrixXf &_corners, float _absorption);

    // methods
    float area();  // compute the area of the wall
    int intersection(  // compute the intersection of line segment (p1 <-> p2) with wall
        const Eigen::Ref<Eigen::VectorXf> p1, const Eigen::Ref<Eigen::VectorXf> p2,
        Eigen::Ref<Eigen::VectorXf> intersection);

    int reflect(const Eigen::Ref<Eigen::VectorXf> p, Eigen::Ref<Eigen::VectorXf> p_reflected);
    int side(const Eigen::Ref<Eigen::VectorXf> p);

  private:
    int _intersection_segment_3d(  // intersection routine specialized for 3D
        const Eigen::Ref<Eigen::VectorXf> a1, const Eigen::Ref<Eigen::VectorXf> a2,
        Eigen::Ref<Eigen::VectorXf> intersection);

}; 

#endif // __CWALL_H__
