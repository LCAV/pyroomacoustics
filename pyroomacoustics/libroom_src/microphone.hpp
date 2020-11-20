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
#ifndef __MICROPHONE_HPP__
#define __MICROPHONE_HPP__

#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <Eigen/Dense>

#include "common.hpp"

template<size_t D>
class Microphone
{
  /*
   * This is the basic microphone class. It works as an omnidirectional microphone.
   */
  public:
    Vectorf<D> loc;

    int n_dirs = 1;
    int n_bands = 1;  // the number of frequency bands in the histogram
    float hist_resolution;  // the size of one bin in meters
    std::vector<float> distance_bins = { 0.f };  // a list of distances forming the boundaries of the bins in the time histogram

    // We keep a log of discrete hits
    std::list<Hit> hits;

    // and an Energy histogram for the tail
    std::vector<Histogram2D> histograms;

    Microphone(const Vectorf<D> &_loc, int _n_bands, float _hist_res, float max_dist_init)
      : loc(_loc), n_dirs(1), n_bands(_n_bands), hist_resolution(_hist_res)
    {
      size_t n_dist_bins_init = size_t(max_dist_init / hist_resolution) + 1;
      // Initialize the histograms
      histograms.resize(n_dirs);
      for (auto &hist : histograms)
        hist.init(n_bands, n_dist_bins_init);
    }
    ~Microphone() {};

    void reset()
    {
      for (auto h = histograms.begin() ; h != histograms.end() ; ++h)
        h->reset();
    }

    const Vectorf<D> &get_loc() const
    {
      return loc;
    };

    float get_dir_gain(const Vectorf<D> &origin, int band_index) const
    {
      return 1.;  // omnidirectional
    }
    
    float get_dir_bin(const Vectorf<D> &origin) const
    {
      return 0;  // only one direction is logged (omni)
    }

    void log_hit(const Hit &the_hit, const Vectorf<D> &origin)
    {
      Hit copy_hit(the_hit);

      // Correct transmitted amplitude with directivity
      for (int f(0) ; f < n_bands ; f++)
        copy_hit.transmitted[f] *= get_dir_gain(origin, f);
        
      hits.push_back(copy_hit);
    }

    void log_histogram(float distance, const Eigen::ArrayXf &energy, const Vectorf<D> &origin)
    {
      // first find the bin index
      auto dist_bin_index = size_t(distance / hist_resolution);
      auto dir_index = get_dir_bin(origin);
      histograms[dir_index].log_col(dist_bin_index, energy);
    }

    void log_histogram(const Hit &the_hit, const Vectorf<D> &origin)
    {
      log_histogram(the_hit.distance, the_hit.transmitted, origin);
    }
};

#endif // __MICROPHONE_HPP__
