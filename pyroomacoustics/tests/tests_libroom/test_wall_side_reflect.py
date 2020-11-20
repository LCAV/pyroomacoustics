# Test of Wall reflection side test
# Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
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

eps = 1e-6

corners = {
        '3d' : np.array([
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            ]),
        '2d' : np.array([
            [ -1, 1 ],
            [ -2, 2 ],
            ]),
        }

points = {
        '3d_up' : {
            'd' : '3d',
            'p' : [-1,-1,-1],
            'expect' : -1,
            'reflect' : [5/3, 5/3, 5/3],
            },
        '3d_down' : {
            'd' : '3d',
            'p' : [1,1,1],
            'expect' : 1,
            'reflect' : [-1/3, -1/3, -1/3],
            },
        '3d_on' : {
            'd' : '3d',
            'p' : [1/3, 1/3, 1/3],
            'expect' : 0,
            'reflect' : [1/3, 1/3, 1/3],
            },

        '2d_down' : {
            'd' : '2d',
            'p' : [-2, 1],
            'expect' : -1,
            'reflect' : [2, -1],
            },
        '2d_up' : {
            'd' : '2d',
            'p' : [2, -1],
            'expect' : 1,
            'reflect' : [-2, 1],
            },
        '2d_on' : {
            'd' : '2d',
            'p' : [0.5, 1],
            'expect' : 0,
            'reflect' : [0.5, 1],
            },
        }


def run_side(label):
    p = points[label]['p']
    r_exp = points[label]['expect']
    d = points[label]['d']

    wall = pra.wall_factory(corners[d], [0.1], [0.1])
    r = wall.side(p)

    print('{}: returned={} expected={}'.format(label, r, r_exp))
    return r == r_exp


def run_reflect(label):

    p = points[label]['p']
    p_refl = np.array(points[label]['reflect'])
    r_exp = points[label]['expect']
    d = points[label]['d']

    wall = pra.wall_factory(corners[d], [0.1], [0.1])

    x = np.zeros(wall.dim, dtype=np.float32)
    wall.reflect(p, x)

    err = np.linalg.norm(x - p_refl) < eps

    print('{}: error={}'.format(label, err))
    return err


class TestUtilityRoutines(unittest.TestCase):

    def test_side_3d_up(self):
        ret = run_side('3d_up')
        self.assertTrue(ret)

    def test_side_3d_down(self):
        ret = run_side('3d_down')
        self.assertTrue(ret)

    def test_side_3d_on(self):
        ret = run_side('3d_on')
        self.assertTrue(ret)

    def test_side_2d_up(self):
        ret = run_side('2d_up')
        self.assertTrue(ret)

    def test_side_2d_down(self):
        ret = run_side('2d_down')
        self.assertTrue(ret)

    def test_side_2d_on(self):
        ret = run_side('2d_on')
        self.assertTrue(ret)

    def test_reflect_3d_up(self):
        ret = run_reflect('3d_up')
        self.assertTrue(ret)

    def test_reflect_3d_down(self):
        ret = run_reflect('3d_down')
        self.assertTrue(ret)

    def test_reflect_3d_on(self):
        ret = run_reflect('3d_on')
        self.assertTrue(ret)

    def test_reflect_2d_up(self):
        ret = run_reflect('2d_up')
        self.assertTrue(ret)

    def test_reflect_2d_down(self):
        ret = run_reflect('2d_down')
        self.assertTrue(ret)

    def test_reflect_2d_on(self):
        ret = run_reflect('2d_on')
        self.assertTrue(ret)

    def test_reflected_end2D(self):

        eps = 0.001
        start = [1,3]

        hit = [5,3]
        #normal = [-1, -1]
        corners = np.array([
            [ 6, 4 ],
            [ 4, 6 ],
            ])
        wall = pra.wall_factory(corners, [0.1], [0.1])
        length = 4

        res = wall.normal_reflect(start, hit, length)
        self.assertTrue(np.allclose(res, [5.,-1.], atol=eps))

    def test_reflected_end3D(self):

        eps = 0.001
        start = [1,1,1]
        hit = [-1, 1, 3]
        #normal = [1, 0, 0]
        corners = np.array([
            [ -1, -1, -1, -1, ],
            [  0,  2,  2,  0, ],
            [  2,  2,  4,  4, ],
            ])
        wall = pra.wall_factory(corners, [0.1], [0.1])

        length = 2*np.sqrt(2)

        res = wall.normal_reflect(start, hit, length)
        self.assertTrue(np.allclose(res, [1,1,5], atol=eps))


if __name__ == '__main__':
    unittest.main()
