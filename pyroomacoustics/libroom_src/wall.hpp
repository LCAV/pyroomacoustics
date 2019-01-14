/* 
 * Definition of the Wall class used in libroom core of pyroomacoustics
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
#ifndef __CWALL_H__
#define __CWALL_H__

#include <string>
#include <Eigen/Dense>

extern float libroom_eps;

#define WALL_ISECT_NONE        -1  // if there is no intersection
#define WALL_ISECT_VALID        0  // if the intersection striclty between the segment endpoints and in the polygon interior
#define WALL_ISECT_VALID_ENDPT  1  // if the intersection is at endpoint of segment
#define WALL_ISECT_VALID_BNDRY  2  // if the intersection is at boundary of polygon
#define ENDPOINT_BOUNDARY       3  // if both the above are true

template<size_t D>
class Wall
{
  public:
    enum Isect {  // The different cases for intersections
      NONE = -1,  // - There is no intersection
      VALID = 0,  // - There is a valid intersection
      ENDPT = 1,  // - The intersection is on the endpoint of the segment
      BNDRY = 2   // - The intersection is on the boundary of the wall
    };

    int dim;
    float absorption;
    std::string name;
    
    Eigen::Matrix<float, D, 1>  normal;
    Eigen::Matrix<float, D, Eigen::Dynamic> corners;

    /* for 3D wall, provide local basis for plane of wall */
    Eigen::Matrix<float, D, 1> origin;
    Eigen::Matrix<float, D, 2> basis;
    Eigen::Matrix<float, 2, Eigen::Dynamic> flat_corners;

    // Constructor
    Wall(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        float _absorption,
        const std::string &_name
        );
    Wall(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        float _absorption
        ) : Wall(_corners, _absorption, "") {}

    // methods
    float area();  // compute the area of the wall
    int intersection(  // compute the intersection of line segment (p1 <-> p2) with wall
        const Eigen::Matrix<float,D,1> &p1,
        const Eigen::Matrix<float,D,1> &p2,
        Eigen::Ref<Eigen::Matrix<float,D,1>> intersection
        );

    int intersects(
        const Eigen::Matrix<float,D,1> &p1,
        const Eigen::Matrix<float,D,1> &p2
        );

    int reflect(
        const Eigen::Matrix<float,D,1> &p,
        Eigen::Ref<Eigen::Matrix<float,D,1>> p_reflected
        );
    int side(const Eigen::Matrix<float,D,1> &p);
    bool same_as(const Wall & that);
}; 

#include "wall.cpp"

#endif // __CWALL_H__
