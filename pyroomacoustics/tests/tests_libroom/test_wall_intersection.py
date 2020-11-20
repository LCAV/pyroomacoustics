# Test of Wall intersection routines
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

import numpy as np
import pyroomacoustics as pra

eps = 1e-6

cases = {
        '3d_valid' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-1,-1,-1], [1,1,1]]),
            'intersect' : np.array([1/3,1/3,1/3]),
            'type' : pra.libroom.WALL_ISECT_VALID,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '3d_endpt_1' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[1/3,1/3,1/3], [1,1,1]]),
            'intersect' : np.array([1/3,1/3,1/3]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_endpt_2' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[1,1,1], [1/3,1/3,1/3]]),
            'intersect' : np.array([1/3,1/3,1/3]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '3d_bndry' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[1,0,1], [-1,0,1]]),
            'intersect' : np.array([0,0,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_vertex' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-2,1,1], [0,1,1]]),
            'intersect' : np.array([-1,1,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '3d_endpt_bndry_1' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[0,0,1], [-1,0,1]]),
            'intersect' : np.array([0,0,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY | pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_endpt_bndry_2' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[1,0,1],[0,0,1]]),
            'intersect' : np.array([0,0,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY | pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_vertex_endpt_1' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-2,1,1], [-1,1,1]]),
            'intersect' : np.array([-1,1,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY | pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_vertex_endpt_2' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-1,1,1], [0,1,1]]),
            'intersect' : np.array([-1,1,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY | pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '3d_none_1' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-10,-5,-3], [-2,-1,-3]]),
            'intersect' : None,
            'type' : pra.libroom.WALL_ISECT_NONE,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '3d_none_2' : {
            'corners' : np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            'seg' : np.array([[-1,-1,2], [1,1,2]]),
            'intersect' : None,
            'type' : pra.libroom.WALL_ISECT_NONE,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_valid' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-1,1], [1,1]]),
            'intersect' : np.array([1/3,1]),
            'type' : pra.libroom.WALL_ISECT_VALID,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_endpt_1' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[1/3,1], [1,1]]),
            'intersect' : np.array([1/3,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '2d_endpt_2' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-1,1], [1/3,1]]),
            'intersect' : np.array([1/3,1]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_bndry_1' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[0,2], [2,2]]),
            'intersect' : np.array([1,2]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '2d_bndry_2' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-2,-1], [0,-1]]),
            'intersect' : np.array([-1,-1]),
            'type' : pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_bndry_endpt_1' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-1,-1], [0,1]]),
            'intersect' : np.array([-1,-1]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT | pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_bndry_endpt_2' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-2,-1], [-1,-1]]),
            'intersect' : np.array([-1,-1]),
            'type' : pra.libroom.WALL_ISECT_VALID_ENDPT | pra.libroom.WALL_ISECT_VALID_BNDRY,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        '2d_none_1' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-2,3], [3,3]]),
            'intersect' : None,
            'type' : pra.libroom.WALL_ISECT_NONE,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },
        '2d_none_2' : {
            'corners' : np.array([  # [-1,-1] -> [1/3,1] -> [1, 2]
                [-1, 1],
                [-1, 2],
                ]),
            'seg' : np.array([[-4,1], [-1,1]]),
            'intersect' : None,
            'type' : pra.libroom.WALL_ISECT_NONE,
            'absorption' : [0.2],
            'scattering' : [0.1],
            },

        }

def run_intersect(lbl):

    case = cases[lbl]
    wall = pra.wall_factory(case['corners'], case['absorption'], case['scattering'])
    p1, p2 = case['seg']

    isect_exp = case['intersect']
    r_exp = case['type']
    isect = np.zeros(wall.dim, dtype=np.float32)
    ret = wall.intersection(p1, p2, isect)

    assert ret == r_exp, 'Wall intersect: expects={} got={}'.format(r_exp, ret)

    if r_exp != pra.libroom.WALL_ISECT_NONE:
        err = np.linalg.norm(isect - case['intersect'])
        assert err < eps, 'Wall intersect: err={} (expects={} got={})'.format(err, isect_exp, isect)


def test_3d_valid():
    run_intersect('3d_valid')

def test_3d_endpt_1():
    run_intersect('3d_endpt_1')

def test_3d_endpt_2():
    run_intersect('3d_endpt_2')

def test_3d_bndry():
    run_intersect('3d_bndry')
    
def test_3d_vertex():
    run_intersect('3d_vertex')

def test_3d_endpt_bndry_1():
    run_intersect('3d_endpt_bndry_1')

def test_3d_endpt_bndry_2():
    run_intersect('3d_endpt_bndry_2')
    
def test_3d_vertex_endpt_1():
    run_intersect('3d_vertex_endpt_1')

def test_3d_vertex_endpt_2():
    run_intersect('3d_vertex_endpt_2')

def test_3d_none_1():
    run_intersect('3d_none_1')

def test_3d_none_2():
    run_intersect('3d_none_2')

def test_2d_valid():
    run_intersect('2d_valid')

def test_2d_endpt_1():
    run_intersect('2d_endpt_1')

def test_2d_endpt_2():
    run_intersect('2d_endpt_2')

def test_2d_bndry_1():
    run_intersect('2d_bndry_1')

def test_2d_bndry_2():
    run_intersect('2d_bndry_2')

def test_2d_bndry_endpt_1():
    run_intersect('2d_bndry_endpt_1')

def test_2d_bndry_endpt_2():
    run_intersect('2d_bndry_endpt_2')

def test_2d_none_1():
    run_intersect('2d_none_1')

def test_2d_none_2():
    run_intersect('2d_none_2')


if __name__ == '__main__':

    test_3d_valid()
    test_3d_endpt_1()
    test_3d_endpt_2()
    test_3d_bndry()
    test_3d_vertex()
    test_3d_endpt_bndry_1()
    test_3d_endpt_bndry_2()
    test_3d_vertex_endpt_1()
    test_3d_vertex_endpt_2()
    test_3d_none_1()
    test_3d_none_2()

    test_2d_valid()
    test_2d_endpt_1()
    test_2d_endpt_2()
    test_2d_bndry_1()
    test_2d_bndry_2()
    test_2d_bndry_endpt_1()
    test_2d_bndry_endpt_2()
    test_2d_none_1()
    test_2d_none_2()
