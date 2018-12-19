#ifndef __ROOM_H__
#define __ROOM_H__

#include <vector>
#include <stack>
#include <Eigen/Dense>
#include "wall.hpp"

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

template<size_t D>
struct ImageSource
{
  /*
   * A class to hold the information relating to an Image source when running the ISM
   */

  Eigen::Matrix<float,D,1> loc;
  float attenuation;
  int order;
  int gen_wall;
  ImageSource *parent;
  VectorXb visible_mics;
};

/*
 * Structure for a room as a list of walls
 * with a few sources and microphones around
 */
template<size_t D>
class Room
{
  public:
    int dim;
    std::vector<Wall<D>> walls;

    // List of obstructing walls
    std::vector<int> obstructing_walls;

    // The microphones are in the room
    Eigen::Matrix<float,D,Eigen::Dynamic> microphones;

    // This is a list of image sources
    Eigen::Matrix<float,D,Eigen::Dynamic> sources;
    Eigen::VectorXi gen_walls;
    Eigen::VectorXi orders;
    Eigen::VectorXf attenuations;

    // This array will get filled by visibility status
    // its size is n_microphones * n_sources
    MatrixXb visible_mics;

    // Constructor
    Room() {}  // default

    Wall<D> &get_wall(int w) { return walls[w]; }

    // Image source model methods
    int image_source_model(const Eigen::Matrix<float,D,1> &source_location, int max_order);

    // A specialized method for the shoebox room case
    int image_source_shoebox(
        const Eigen::Matrix<float,D,1> &source,
        const Eigen::Matrix<float,D,1> &room_size,
        const Eigen::Matrix<float,2*D,1> &absorption,
        int max_order
        );

  private:
    // We need a stack to store the image sources during the algorithm
    std::stack<ImageSource<D>> visible_sources;
    
    // Image source model internal methods
    void image_sources_dfs(ImageSource<D> &is, int max_order);
    bool is_visible_dfs(const Eigen::Matrix<float,D,1> &p, ImageSource<D> &is);
    bool is_obstructed_dfs(const Eigen::Matrix<float,D,1> &p, ImageSource<D> &is);
    int fill_sources();

};

#include "room.cpp"

#endif // __ROOM_H__
