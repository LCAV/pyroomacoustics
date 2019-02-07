/* 
 * Type and constant definitions
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

/* This file contains type and constants definitions */
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <Eigen/Dense>

extern float libroom_eps;  // epsilon is the precision for floating point computations. It is defined in libroom.cpp

template<size_t D>
using Vectorf = Eigen::Matrix<float, D, 1>;

using MatrixXf = Eigen::MatrixXf;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

/* The 'entry' type is simply defined as an array of 2 floats.
 * It represents an entry that is logged by the microphone
 * during the ray_tracing execution.
 * The first one of those float will be the travel time of a ray reaching
 * the microphone. The second one will be the energy of this ray.*/
typedef std::array<float,2> entry;
typedef std::list<entry> mic_log;
typedef std::vector<mic_log> HitLog;

struct Hit
{
  float distance = 0.f;
  std::vector<float> transmitted;  // vector of transmitted amplitude over frequency bands

  Hit(int _nfreq)
  {
    transmitted.resize(_nfreq);
  };
  Hit(const float _d, const std::vector<float> &_t)
    : distance(_d), transmitted(_t) {}
};

class Histogram2D
{
  size_t rows, cols;
  Eigen::ArrayXXf array;
  Eigen::ArrayXXi counts;

  public:
    Histogram2D() {}  // empty constructor
    Histogram2D(int _r, int _c) : rows(_r), cols(_c)
    {
      init(rows, cols);
    }

    void init(int rows, int cols)
    {
      array.resize(rows, cols);
      array.setZero();
      counts.resize(rows, cols);
      counts.setZero();
    }

    void log(int row, int col, float val)
    {
      array.coeffRef(row, col) += val;
      counts.coeffRef(row, col)++;
    }

    float bin(int row, int col) const
    {
      if (counts.coeff(row, col) != 0)
        return array.coeff(row, col) / counts.coeff(row, col);
      else
        return 0.f;
    }

    Eigen::ArrayXXf get_hist() const
    {
      return array / counts.max(1).cast<float>();
    }
};

#endif // __COMMON_HPP__
