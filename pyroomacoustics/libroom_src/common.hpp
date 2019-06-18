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

#include <iostream>
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

struct Hit
{
  float distance = 0.f;
  Eigen::ArrayXf transmitted;  // vector of transmitted energy over frequency bands

  Hit(int _nfreq)
  {
    transmitted.resize(_nfreq);
    transmitted.setOnes();
  };
  Hit(const float _d, const Eigen::ArrayXf &_t)
    : distance(_d), transmitted(_t) {}
};

typedef std::vector<std::list<Hit>> HitLog;

size_t get_new_size(size_t val, size_t cur_size)
{
  size_t new_size = cur_size;
  while (val >= new_size)
    new_size *= 2;
  return new_size;
}

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

    void reset()
    {
      array.setZero();
      counts.setZero();
    }

    void resize_rows(int new_rows)
    {
      auto old_rows = array.rows();
      // this will resize the array while preserving the content
      array.conservativeResize(new_rows, Eigen::NoChange);
      counts.conservativeResize(new_rows, Eigen::NoChange);
      // We need to initialize the new elements
      if (new_rows > old_rows)
      {
        array.bottomRows(new_rows - old_rows).setZero();
        counts.bottomRows(new_rows - old_rows).setZero();
      }
    }

    void resize_cols(int new_cols)
    {
      auto old_cols = array.cols();
      // this will resize the array while preserving the content
      array.conservativeResize(Eigen::NoChange, new_cols);
      counts.conservativeResize(Eigen::NoChange, new_cols);
      // We need to initialize the new elements
      if (new_cols > old_cols)
      {
        array.rightCols(new_cols - old_cols).setZero();
        counts.rightCols(new_cols - old_cols).setZero();
      }
    }

    void log(Eigen::Index row, Eigen::Index col, float val)
    {
      if (row >= array.rows())
        resize_rows(get_new_size(row, array.rows()));

      if (col >= array.cols())
        resize_cols(get_new_size(col, array.cols()));

      array.coeffRef(row, col) += val;
      counts.coeffRef(row, col)++;
    }

    void log_col(Eigen::Index col, const Eigen::ArrayXf &val)
    {
      if (col >= array.cols())
        resize_cols(get_new_size(col, array.cols()));

      array.col(col) += val;
      counts.col(col) += 1;
    }

    void log_row(Eigen::Index row, const Eigen::ArrayXf &val)
    {
      if (row >= array.rows())
        resize_rows(get_new_size(row, array.rows()));

      array.row(row) += val;
      counts.row(row) += 1;
    }

    float bin(Eigen::Index row, Eigen::Index col) const
    {
      if (counts.coeff(row, col) != 0)
        return array.coeff(row, col) / counts.coeff(row, col);
      else
        return 0.f;
    }

    Eigen::ArrayXXf get_hist() const
    {
      return array;
    }
};

#endif // __COMMON_HPP__
