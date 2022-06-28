/*
 * Microphone and receiver classes
 * Copyright (C) 2019  Robin Scheibler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program.
 * If not, see <https://opensource.org/licenses/MIT>.
 */
#ifndef __MICROPHONE_HPP__
#define __MICROPHONE_HPP__

#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <list>
#include <nanoflann.hpp>
#include <vector>

#include "common.hpp"

template <size_t D>
class Microphone {
  /*
   * This is the basic microphone class. It works as an omnidirectional
   * microphone.
   */
 private:
  using my_kd_tree_t =
      nanoflann::KDTreeEigenMatrixAdaptor<MatrixXf, -1 /*dyn size*/,
                                          nanoflann::metric_L2_Simple>;
  // the kd-tree
  my_kd_tree_t *kdtree = nullptr;

  MatrixXf directions;
  size_t n_dist_bins_init = 1;

 public:
  Vectorf<D> loc;

  int n_dirs = 1;
  int n_bands = 1;        // the number of frequency bands in the histogram
  float hist_resolution;  // the size of one bin in meters
  std::vector<float> distance_bins = {
      0.f};  // a list of distances forming the boundaries of the bins in the
             // time histogram

  // and an Energy histogram for the tail
  std::vector<Histogram2D> histograms;

  // Constructor for omni microphones (kd-tree not necessary)
  Microphone(const Vectorf<D> &_loc, int _n_bands, float _hist_res,
             float max_dist_init)
      : loc(_loc), n_dirs(1), n_bands(_n_bands), hist_resolution(_hist_res) {
    n_dist_bins_init = size_t(max_dist_init / hist_resolution) + 1;
    // Initialize the histograms
    histograms.resize(n_dirs);
    for (auto &hist : histograms) hist.init(n_bands, n_dist_bins_init);
  }

  Microphone(const Vectorf<D> &_loc, int _n_bands, float _hist_res,
             float max_dist_init, const MatrixXf &directions)
      : Microphone(_loc, _n_bands, _hist_res, max_dist_init) {
    set_directions(directions);
  }

  ~Microphone() { delete_kdtree(); };

  void set_directions(const MatrixXf &_directions) {
    delete_kdtree();
    directions = _directions;
    kdtree = new my_kd_tree_t(D, std::cref(directions), 10 /* max leaf */);
    n_dirs = directions.rows();

    // Initialize the histograms
    histograms.resize(n_dirs);
    for (auto &hist : histograms) hist.init(n_bands, n_dist_bins_init);
  }

  void make_omni() {
    delete_kdtree();
    n_dirs = 1;
    directions.setZero(1, D);
  }

  void delete_kdtree() {
    if (kdtree) {
      delete kdtree;
      kdtree = nullptr;
    }
  }

  void reset() {
    for (auto h = histograms.begin(); h != histograms.end(); ++h) h->reset();
  }

  const Vectorf<D> &get_loc() const { return loc; };

  int get_dir_bin(const Vectorf<D> &origin) const {
    if (n_dirs == 1) {
      return 0;  // only one direction is logged (omni)
    } else {
      // direction of incoming ray
      auto dir = (origin - loc).normalized();

      // find in kd-tree
      MatrixXf::Index ret_index = 0;
      float out_dist_sqr = 0.0f;
      float *ptr = dir.data();
      kdtree->query(ptr, 1, &ret_index, &out_dist_sqr);

      return int(ret_index);
    }
  }

  void log_histogram(float distance, const Eigen::ArrayXf &energy,
                     const Vectorf<D> &origin) {
    // first find the bin index
    auto dist_bin_index = size_t(distance / hist_resolution);
    auto dir_index = get_dir_bin(origin);
    histograms[dir_index].log_col(dist_bin_index, energy);
  }

  void log_histogram(const Hit &the_hit, const Vectorf<D> &origin) {
    log_histogram(the_hit.distance, the_hit.transmitted, origin);
  }
};

#endif  // __MICROPHONE_HPP__
