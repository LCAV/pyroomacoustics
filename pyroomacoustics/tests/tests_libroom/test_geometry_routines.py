# Test of the geometry routines
# Copyright (C) 2019  Robin Scheibler, Cyril Cadoux, Sidney Barthe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.
from __future__ import division

import unittest
import numpy as np

import pyroomacoustics as pra

class TestGeometryRoutines(unittest.TestCase):

    def test_area_square(self):
        self.assertEqual(pra.libroom.area_2d_polygon([[0, 4, 4, 0], [0, 0, 4, 4]]), 4*4)

    def test_area_triangle(self):
        self.assertEqual(pra.libroom.area_2d_polygon([[0, 4, 2], [0, 0, 2]]), (4*2)/2)

    def test_ccw3p_counterclockwise(self):
        self.assertEqual(pra.libroom.ccw3p([0, 0], [2, 0], [1, 1]), 1)

    def test_ccw3p_clockwise(self):
        self.assertEqual(pra.libroom.ccw3p([1, 1], [2, 0], [0, 0]), -1)

    def test_ccw3p_collinear(self):
        self.assertEqual(pra.libroom.ccw3p([0, 0], [1, 0], [2, 0]), 0)

    def test_intersection2DSegments_cross(self):
        loc = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([-2, 0], [2, 0], [0, -2], [0, 2], loc)
        i = np.allclose(loc, np.zeros(2))
        self.assertTrue(all([i, ret == 0]))  # intersection is valid

    def test_intersection2DSegments_T(self):
        loc = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([0, 0], [2, 0], [2, -2], [2, 2], loc)
        i = np.allclose(loc, np.r_[2,0])
        self.assertTrue(all([i, ret == 1]))  # intersection on endpoint of 1st seg

    def test_intersection2DSegments_T2(self):
        loc = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([-2, 0], [2, 0], [0, -2], [0, 0], loc)
        i = np.allclose(loc, np.r_[0,0])
        self.assertTrue(all([i, ret == 2]))

    def test_intersection2DSegments_L(self):
        loc = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([0, 0], [2, 0], [2, 2], [2, 0], loc)
        i = np.allclose(loc, np.r_[2,0])
        self.assertTrue(all([i, ret == 3]))

    def test_intersection2DSegments_parallel(self):
        loc = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([0, 0], [2, 0], [0, 1], [2, 1], loc)
        self.assertTrue(ret == -1)

    def test_intersection2DSegments_notTouching(self):
        p = np.zeros(2, dtype=np.float32)
        ret = pra.libroom.intersection_2d_segments([0, 0], [2, 0], [1, 4], [1, 1], p)
        self.assertTrue(ret == -1)

    def test_intersectionSegmentPlane_through(self):
        p = np.zeros(3, dtype=np.float32)
        ret = pra.libroom.intersection_3d_segment_plane([2, 2, 2], [2, 2, -2], [2, 2, 0], [0, 0, -1], p)
        i = np.allclose(p, np.r_[2,2,0])
        self.assertTrue(all([i, ret == 0]))

    def test_intersectionSegmentPlane_touching(self):
        p = np.zeros(3, dtype=np.float32)
        ret = pra.libroom.intersection_3d_segment_plane([2, 2, 2], [2, 2, 0], [2, 2, 0], [0, 0, -1], p)
        i = np.allclose(p, np.r_[2,2,0])
        self.assertTrue(all([i, ret == 1]))

    def test_intersectionSegmentPlane_notTouching(self):
        p = np.zeros(3, dtype=np.float32)
        ret = pra.libroom.intersection_3d_segment_plane([2, 2, 2], [2, 2, 1], [2, 2, 0], [0, 0, -1], p)
        self.assertTrue(all([ret == -1]))

    def test_intersectionSegmentPlane_inside(self):
        p = np.zeros(3, dtype=np.float32)
        ret = pra.libroom.intersection_3d_segment_plane([0, 2, 0], [2, 2, 0], [2, 2, 0], [0, 0, -1], p)
        self.assertTrue(all([ret == -1]))

    def test_intersectionSegmentPolygonSurface_through(self):
        wall = pra.libroom.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p = np.zeros(3, dtype=np.float32)
        ret = wall.intersection([2,2,2], [2,2,-2], p)
        i = np.allclose(p, np.r_[2,2,0])
        self.assertTrue(all([i, ret == pra.libroom.Wall.Isect.VALID]))

    def test_intersectionSegmentPolygonSurface_touching(self):
        wall = pra.libroom.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p = np.zeros(3, dtype=np.float32)
        ret = wall.intersection([2,2,2], [2,2,0], p)
        i = np.allclose(p, np.r_[2,2,0])
        self.assertTrue(all([i, ret == pra.libroom.Wall.Isect.ENDPT]))

    def test_intersectionSegmentPolygonSurface_border(self):
        wall = pra.libroom.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p = np.zeros(3, dtype=np.float32)
        ret = wall.intersection([0,0,2], [0,0,-2], p)
        i = np.allclose(p, np.r_[0,0,0])
        self.assertTrue(all([i, ret == pra.libroom.Wall.Isect.BNDRY]))

    def test_intersectionSegmentPolygonSurface_miss(self):
        wall = pra.libroom.Wall([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]])
        p = np.zeros(3, dtype=np.float32)
        ret = wall.intersection([-1,-1,2], [-1,-1,-2], p)
        self.assertTrue(all([ret == pra.libroom.Wall.Isect.NONE]))

    def test_intersectionSegmentPolygonSurface_miss_2(self):
        corners = np.array([
            [0, 0, 0, 0],
            [3, 0, 0, 3],
            [0, 0, 4, 4]
            ])
        wall = pra.libroom.Wall(corners)
        p = np.array([ 2. ,  3. ,  1.7])
        p0 = np.array([-2.348,  3.27 ,  8.91 ])
        loc = np.zeros(p.shape[0], dtype=np.float32)
        ret = wall.intersection(p, p0, loc)
        self.assertTrue(ret == pra.libroom.Wall.Isect.NONE)

    def test_isInside2DPolygon_inside1(self):
        p = np.r_[0, -2]
        corners = np.array([
            [ 0,     -2,           0,  2,],
            [-0,     -2, -np.sqrt(8), -2,],
            ])
        ret = pra.libroom.is_inside_2d_polygon(p, corners)
        self.assertTrue(ret == 0)
        
    def test_isInside2DPolygon_inside(self):
        ret = pra.libroom.is_inside_2d_polygon([2, 2], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([ret == 0]))
        
    def test_isInside2DPolygon_onBorder(self):
        ret = pra.libroom.is_inside_2d_polygon([0, 2], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([ret == 1]))
        
    def test_isInside2DPolygon_onCorner(self):
        ret = pra.libroom.is_inside_2d_polygon([4, 4], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([ret == 1]))
        
    def test_isInside2DPolygon_outside(self):
        ret = pra.libroom.is_inside_2d_polygon([5, 5], [[0, 4, 4, 0], [0, 0, 4, 4]])
        self.assertTrue(all([ret == -1]))

    def test_isInside2DPolygon_parallel_wall(self):
        corners = [[0, 0, 1, 1, 3, 3], [0, 1, 1, 2, 2, 0]]
        ret = pra.libroom.is_inside_2d_polygon([2, 1], corners)
        self.assertTrue(all([ret == 0]))

    def test_isInside2DPolygon_at_corner(self):
        corners = [[0, 1, 1, 3, 3], [0, 1, 2, 2, 0]]
        ret = pra.libroom.is_inside_2d_polygon([2, 1], corners)
        self.assertTrue(all([ret == 0]))

    def test_isInside2DPolygon_just_out(self):
        corners = [[0, 1, 1, 3, 3], [0, 1, 2, 2, 0]]
        ret = pra.libroom.is_inside_2d_polygon([4, 2], corners)
        self.assertTrue(all([ret == -1]))

    def test_cosangleBetween2D(self):
        eps = 0.001
        result = pra.libroom.cos_angle_between([1.,0.],[-1.,-1.])
        self.assertTrue(np.allclose(result, -np.sqrt(2)/2., atol=eps))

    def test_cosangleBetween3D_orthog(self):
        eps = 0.001
        result = pra.libroom.cos_angle_between([1.,0.,0.],[0.,0.,1.])
        self.assertTrue(np.allclose(result, 0, atol=eps))

    def test_cosangleBetween3D_colinear(self):
        eps = 0.001
        result = pra.libroom.cos_angle_between([1.,1.,0.],[-1.,-1.,0.])
        self.assertTrue(np.allclose(result, -1, atol=eps))

    def test_cosangleBetween3D_special(self):
        eps = 0.001
        result = pra.libroom.cos_angle_between([36.,1.,-4.],[12.,-1.,2.])
        self.assertTrue(np.allclose(result,0.956345613,  atol=eps))


    def test_dist_line_point2D(self):

        start = [3,3]
        end = [6,9]
        point = [11,4]

        eps = 0.001
        res = pra.libroom.dist_line_point(start, end, point)
        self.assertTrue(abs(res - np.sqrt(6*6+3*3)) < eps)

    def test_dist_line_point2D_vert(self):

        start = [-4,12]
        end = [-4,27]
        point = [7,10]

        eps = 0.001
        res = pra.libroom.dist_line_point(start, end, point)
        self.assertTrue(abs(res - 11) < eps)

    def test_dist_line_point3D(self):

        start = [0,0,0]
        end = [1,2,3]
        point = [4,5,6]

        eps = 0.001
        res = pra.libroom.dist_line_point(start, end, point)
        self.assertTrue(abs(res - 1.963961012) < eps)

    def test_dist_line_point3D_online(self):

        start = [0,0,0]
        end = [0,0,3]
        point = [0,0,6]

        eps = 0.001
        res = pra.libroom.dist_line_point(start, end, point)
        self.assertTrue(abs(res) < eps)


if __name__ == '__main__':
    unittest.main()
