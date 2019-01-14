# Test of the CCW3P routine
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

import numpy as np
import pyroomacoustics as pra

cases = {
        'anti-clockwise' : {
            'points' : np.array([
                [1,-1], [2,-1], [1,0]
                ]),
            'expected' : 1,  # anti-clockwise
            'label' : 'Test: CCW3P anti-clockwise',
            },
        'clockwise' : {
            'points' : np.array([
                [1,-1], [1,0], [2,-1]
                ]),
            'expected' : -1,  # clockwise
            'label' : 'Test: CCW3P clockwise',
            },
        'co-linear' : {
            'points' : np.array([
                [0,0], [0.5,0.5], [1,1]
                ]),
            'expected' : 0,  # co-linear
            'label' : 'Test: CCW3P co-linear',
            },
        }

def ccw3p(case):

    p1, p2, p3 = case['points']

    r = pra.libroom.ccw3p(p1, p2, p3)

    assert r == case['expected'], (case['label']
            + ' returned: {}, expected {}'.format(r, case['expected']))

def test_ccw3p_anticlockwise():
    ccw3p(cases['anti-clockwise'])

def test_ccw3p_clockwise():
    ccw3p(cases['clockwise'])

def test_ccw3p_colinear():
    ccw3p(cases['co-linear'])

if __name__ == '__main__':

    for lbl, case in cases.items():

        try:
            ccw3p(case)
        except:
            print('{} failed'.format(lbl))
