# Test of Wall constructor
# Copyright (C) 2019  Robin, Scheibler, Cyril Cadoux
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

import numpy as np
import pyroomacoustics as pra

eps = 1e-6

# The vertices of the wall are assumed to turn counter-clockwise around the
# normal of the wall

# A very simple wall
walls = [
        {
            'corners' : np.array([
                [0., 1., 1., 0.],
                [0., 0., 1., 1.],
                [0., 0., 0., 0.],
                ]),
            'area' : 1,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'area' : 3.4641016151377557,  # this is an equilateral triangle with side sqrt(8)
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        ]



def test_wall_3d_construct_0():
    ''' Tests construction of a wall '''
    w_info = walls[0]
    wall = pra.wall_factory(w_info['corners'], [0.2], [0.1])

    return wall

def test_wall_3d_construct_1():
    ''' Tests construction of a wall '''
    w_info = walls[1]
    wall = pra.wall_factory(w_info['corners'], [0.2], [0.1])

    return wall

def test_wall_3d_area_0():
    ''' Tests the area computation '''
    w_info = walls[0]
    wall = pra.wall_factory(w_info['corners'], w_info['absorption'], w_info['scattering'])
    err = abs(wall.area() - w_info['area'])
    assert err < 1, 'The error is {}'.format(err)

def test_wall_3d_area_1():
    ''' Tests the area computation '''
    w_info = walls[1]
    wall = pra.wall_factory(w_info['corners'], w_info['absorption'], w_info['scattering'])
    err = abs(wall.area() - w_info['area'])
    assert err < 1, 'The error is {}'.format(err)

def test_wall_3d_normal_0():
    ''' Tests direction of normal wrt to point arrangement '''
    w_info = walls[0]
    wall1 = pra.wall_factory(w_info['corners'], [0.2], [0.1])
    # the same wall with normal pointing the other way
    wall2 = pra.wall_factory(w_info['corners'][:,::-1], [0.2], [0.1])

    err = np.linalg.norm(wall1.normal + wall2.normal)
    assert err < eps, 'The error is {}'.format(err)

def test_wall_3d_normal_1():
    ''' Tests direction of normal wrt to point arrangement '''
    w_info = walls[1]
    wall1 = pra.wall_factory(w_info['corners'], [0.2], [0.1])
    # the same wall with normal pointing the other way
    wall2 = pra.wall_factory(w_info['corners'][:,::-1], [0.2], [0.1])

    err = np.linalg.norm(wall1.normal + wall2.normal)
    assert err < eps, 'The error is {}'.format(err)

if __name__ == '__main__':

        wall0 = test_wall_3d_construct_0()
        test_wall_3d_normal_0()
        test_wall_3d_area_0()
        wall1 = test_wall_3d_construct_1()
        test_wall_3d_normal_1()
        test_wall_3d_area_1()

