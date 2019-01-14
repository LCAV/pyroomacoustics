/* 
 * A few auxilliary routines used in libroom
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
#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <Eigen/Dense>
#include "wall.hpp"
#include "geometry.hpp"
#include <vector>
#include <array>
#include <list>
#include <vector>
#include <cmath>

using namespace Eigen;
using namespace std;

double clamp(double value, double min, double max);

Vector2f equation(const Vector2f &p1, const Vector2f &p2);

VectorXf compute_segment_end(const VectorXf start_point, float length, float phi, float theta);

VectorXf compute_reflected_end(const VectorXf & start,
  const VectorXf & hit_point,
  const VectorXf & wall_normal,
  float length);

bool intersects_mic(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius);

Vector2f solve_quad(float A, float B, float C);

VectorXf mic_intersection(const VectorXf & start,
  const VectorXf & end,
  const VectorXf & center,
  float radius);



#include "utility.cpp"

#endif // __UTILITY_H__
