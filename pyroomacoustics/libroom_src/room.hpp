/* 
 * Definition of the Room class
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */
#ifndef __ROOM_H__
#define __ROOM_H__

#include <vector>
#include <stack>
#include <tuple>
#include <Eigen/Dense>
#include <algorithm>
#include <ctime>

#include "common.hpp"
#include "wall.hpp"

template<size_t D>
struct ImageSource
{
  /*
   * A class to hold the information relating to an Image source when running the ISM
   */

  Vectorf<D> loc;
  Eigen::ArrayXf attenuation;
  int order;
  int gen_wall;
  ImageSource *parent;
  VectorXb visible_mics;

  ImageSource(int n_bands)
    : order(0), gen_wall(-1), parent(NULL)
  {
    loc.setZero();
    attenuation.resize(n_bands);
    attenuation.setOnes();
  }

  ImageSource(const Vectorf<D> &_loc, int n_bands)
    : loc(_loc), order(0), gen_wall(-1), parent(NULL)
  {
    attenuation.resize(n_bands);
    attenuation.setOnes();
  }
};

/*
 * Structure for a room as a list of walls
 * with a few sources and microphones around
 */
template<size_t D>
class Room
{
  public:
    static const int dim = D;
    int n_bands;
    std::vector<Wall<D>> walls;

    // List of obstructing walls
    std::vector<int> obstructing_walls;

    // The microphones are in the room
    std::vector<Microphone<D>> microphones;

    // Very useful for raytracing
    int n_mics;
    // 2. A distance after which a ray must have hit at least 1 wall
    float max_dist;

    // This is a list of image sources
    Eigen::Matrix<float,D,Eigen::Dynamic> sources;
    Eigen::VectorXi gen_walls;
    Eigen::VectorXi orders;
    Eigen::MatrixXf attenuations;

    // This array will get filled by visibility status
    // its size is n_microphones * n_sources
    MatrixXb visible_mics;

    // Simulation parameters
    float sound_speed = 343.;
    float energy_thres = 1e-7;
    float time_thres = 1.;
    float mic_radius = 0.15;
    bool is_hybrid_sim = true;
    int ism_order = 0.;

    Eigen::ArrayXf air_absorption;

    // Constructor
    Room() {}  // default

    void set_params(
        float _sound_speed,
        float _energy_thres,
        float _time_thres,
        float _mic_radius,
        bool _is_hybrid_sim,
        int _ism_order
        )
    {
      sound_speed = _sound_speed;
      energy_thres = _energy_thres;
      time_thres = _time_thres;
      mic_radius = _mic_radius;
      is_hybrid_sim = _is_hybrid_sim;
      ism_order = _ism_order;
    }


    void add_mic(const Vectorf<D> &loc, int n_bands, const std::vector<float> &dist_bins)
    {
      microphones.push_back(Microphone<D>(loc, n_bands, dist_bins));
    }

    Wall<D> &get_wall(int w) { return walls[w]; }

    // Image source model methods
    int image_source_model(const Vectorf<D> &source_location, int max_order);

    // A specialized method for the shoebox room case
    int image_source_shoebox(
        const Vectorf<D> &source,
        const Vectorf<D> &room_size,
        const Eigen::Array<float,Eigen::Dynamic,2*D> &absorption,
        int max_order
        );

    float get_max_distance();

    std::tuple < Vectorf<D>, int > next_wall_hit(
        const Vectorf<D> &start,
        const Vectorf<D> &end,
        bool scattered_ray
        );

    Eigen::ArrayXf compute_scat_energy(
        const Eigen::ArrayXf &transmitted,
        float scat_coef,
        const Wall<D> & wall,
        const Vectorf<D> & start,
        const Vectorf<D> & hit_point,
        const Vectorf<D> & mic_pos,
        float total_dist
        );

    bool scat_ray(
        const Eigen::ArrayXf &transmitted,
        float scatter_coef,
        const Wall<D> &wall,
        const Vectorf<D> &prev_last_hit,
        const Vectorf<D> &hit_point,
        float travel_dist,
        HitLog & output
        );

    void simul_ray(
        float phi,
        float theta,
        const Vectorf<D> source_pos,
        float scatter_coef,
        HitLog & output
        );

    HitLog get_rir_entries(
        const Eigen::Matrix<float,D-1,Eigen::Dynamic> &angles,
        const Vectorf<D> source_pos,
        float scatter_coef
        );

    HitLog get_rir_entries(
        size_t nb_phis,
        size_t nb_thetas,
        const Vectorf<D> source_pos,
        float scatter_coef
        );

    bool contains(const Vectorf<D> point);

  private:
    // We need a stack to store the image sources during the algorithm
    std::stack<ImageSource<D>> visible_sources;

    // Image source model internal methods
    void image_sources_dfs(ImageSource<D> &is, int max_order);
    bool is_visible_dfs(const Vectorf<D> &p, ImageSource<D> &is);
    bool is_obstructed_dfs(const Vectorf<D> &p, ImageSource<D> &is);
    int fill_sources();

};

#include "room.cpp"

#endif // __ROOM_H__
