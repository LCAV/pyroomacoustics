# Test of auxilliary routines
# Copyright (C) 2019  Cyril Cadoux
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

import unittest
import numpy as np

import pyroomacoustics as pra

class TestUtilityRoutines(unittest.TestCase):

    def test_segment_end2D(self):

        eps = 0.001

        start = [0, 0]
        phi = np.pi
        theta = 0.
        length = 2.

        res = pra.libroom.compute_segment_end(start, length, phi, theta)

        self.assertTrue(np.allclose(res, [-2.,0.], atol=eps))

    def test_segment_end3D(self):

        eps = 0.001

        start = [1, 1, 1]
        phi = 0.35
        theta = 2.
        length = 2.

        res = pra.libroom.compute_segment_end(start, length, phi, theta)

        ok0 = abs(res[0] - (1+ length*np.sin(theta)*np.cos(phi))) < eps
        ok1 = abs(res[1] - (1+ length*np.sin(theta)*np.sin(phi))) < eps
        ok2 = abs(res[2] - (1+ length*np.cos(theta))) < eps
        self.assertTrue(all([ok0, ok1, ok2]))

    def test_intersects_mic3D_true(self):

        start = [4,4,4]
        end = [0,0,0]
        center = [2,2,2.2]
        radius = 0.5

        self.assertTrue(pra.libroom.intersects_mic(start, end, center, radius))

    def test_intersects_mic3D_false(self):

        start = [4,4,4]
        end = [0,0,0]
        center = [2,-2,0]
        radius = 1.5

        self.assertTrue(not pra.libroom.intersects_mic(start, end, center, radius))

    def test_intersects_mic2D_true(self):

        start = [10,10]
        end = [0,0]
        center = [5,5]
        radius = 2*np.sqrt(2)

        self.assertTrue(pra.libroom.intersects_mic(start, end, center, radius))

    def test_intersects_mic2D_false(self):

        start = [0,0]
        end = [0,2]
        center = [5,5]
        radius = 2*np.sqrt(2)

        self.assertTrue(not pra.libroom.intersects_mic(start, end, center, radius))


if __name__ == '__main__':
    unittest.main()
