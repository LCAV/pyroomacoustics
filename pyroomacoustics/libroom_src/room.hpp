#ifndef __ROOM_H__
#define __ROOM_H__

#include <vector>
#include <stack>
#include <Eigen/Dense>
#include "wall.hpp"
#include "utility.hpp"

extern float libroom_eps;

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;
typedef Eigen::Matrix<int, 1, 1> Vector1i;
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
    
    // Very useful for raytracing
    // 1. Position format that fits the need of ray_tracing
    Eigen::VectorXf mic_pos;
    // 2. A distance after which a ray must have hit at least 1 wall
    float max_dist;

    // This is a list of image sources
    Eigen::MatrixXf sources;
    VectorXi gen_walls;
    VectorXi orders;
    Eigen::VectorXf attenuations;

    // This array will get filled by visibility status
    // its size is n_microphones * n_sources
    MatrixXb visible_mics;

    // Constructor
    Room() {}  // default

    Wall &get_wall(int w) { return walls[w]; }

    // Image source model methods
    int image_source_model(const Eigen::VectorXf &source_location, int max_order);

    // A specialized method for the shoebox room case
    int image_source_shoebox(
        const Eigen::VectorXf &source,
        const Eigen::VectorXf &room_size,
        const Eigen::VectorXf &absorption,
        int max_order);
        
    float get_max_distance();
    
	Eigen::VectorXf next_wall_hit(
					const Eigen::VectorXf &start,
					const Eigen::VectorXf &end,
					Eigen::Ref<Vector1i> next_wall_index);
					
	bool scat_ray(const Wall &last_wall,
					const Eigen::VectorXf &hit_point,
					float radius,
					float scat_energy,
					float travel_time,
					float time_thres,
					float sound_speed,
					std::vector<entry> &output);
					
	void simul_ray(float init_phi,
				   float init_theta,
				   const Eigen::VectorXf source_pos,
				   float mic_radius,
				   float scatter_coef,
				   float time_thres,
				   float sound_speed,
				   std::vector<entry> &output);
				   
	std::vector<entry> get_rir_entries(size_t nb_phis,
									   size_t nb_thetas,
									   const Eigen::VectorXf source_pos,
								       float mic_radius,
								       float scatter_coef,
								       float time_thres,
								       float sound_speed);
								       
	bool contains(const Eigen::VectorXf point);
					

  private:
    // We need a stack to store the image sources during the algorithm
    std::stack<ImageSource> visible_sources;
    
    // Image source model internal methods
    void image_sources_dfs(ImageSource &is, int max_order);
    bool is_visible_dfs(const Eigen::VectorXf &p, ImageSource &is);
    bool is_obstructed_dfs(const Eigen::VectorXf &p, ImageSource &is);
    int fill_sources();

};


#endif // __ROOM_H__
