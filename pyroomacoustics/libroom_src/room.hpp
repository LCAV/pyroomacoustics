#ifndef __ROOM_H__
#define __ROOM_H__

#include <vector>
#include <pybind11/pybind11.h>  // needed for python list type
#include <Eigen/Dense>
#include "wall.hpp"

namespace py = pybind11;

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

struct ImageSource
{
  /*
   * A class to hold the information relating to an Image source when running the ISM
   */

  Eigen::VectorXf loc;
  float attenuation;
  int order;
  int gen_wall;
  int parent;
  ImageSource *parent;
  VectorXb visible_mics;
};

/*
 * Structure for a room as a list of walls
 * with a few sources and microphones around
 */
class Room
{
  public:
    int dim;
    std::vector<Wall> walls;

    // List of obstructing walls
    std::vector<int> obstructing_walls;

    // The microphones are in the room
    Eigen::MatrixXf microphones;

    // This is a list of image sources
    Eigen::MatrixXf sources;
    VectorXi parents;
    VectorXi gen_walls;
    VectorXi orders;
    Eigen::VectorXf attenuations;

    // This array will get filled by visibility status
    // its size is n_microphones * n_sources
    MatrixXb visible_mics;

    // Constructor
    Room(py::list _walls, py::list _obstructing_walls, const Eigen::MatrixXf _microphones);

    Wall &get_wall(int w) { return walls[w]; }

    // Image source model methods
    int image_source_model(const Eigen::VectorXf source_location, int max_order);

  private:
    // We need a stack to store the image sources during the algorithm
    std::vector<ImageSource> visible_sources;
    
    // Image source model internal methods
    void image_sources_dfs(int image_id, int max_order);
    bool is_visible_dfs(const Eigen::Ref<Eigen::VectorXf> p, int image_id);
    bool is_obstructed_dfs(const Eigen::Ref<Eigen::VectorXf> p, int image_id);
    int fill_sources();

    /* visibility and obstruction routines */
    void check_visibility_all();
    bool is_visible(const Eigen::Ref<Eigen::VectorXf> p, int image_id);
    bool is_obstructed(const Eigen::Ref<Eigen::VectorXf> p, int image_id);

};


#endif // __ROOM_H__
