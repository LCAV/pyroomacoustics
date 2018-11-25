# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import unittest
import numpy as np

import pyroomacoustics as pra

class TestUtilityRoutines(unittest.TestCase):

    def test_equation_flat_line(self):
        p1 = [1.,1.]
        p2 = [4., 1.]
        a,b = pra.libroom.equation(p1, p2)
        self.assertTrue(all([a == 0, b == 1.]))

    def test_equation_increasing_line(self):
        p1 = [1., 3.]
        p2 = [2., 9.]
        a,b = pra.libroom.equation(p1, p2)
        self.assertTrue(all([a == 6, b == -3]))

    def test_equation_decreasing_line(self):
        p1 = [-4., 8.]
        p2 = [0., 6.]
        a,b = pra.libroom.equation(p1, p2)
        self.assertTrue(all([a == -0.5, b == 6.]))

    def test_segment_end2D(self):

        eps = 0.001

        start = [0, 0]
        phi = np.pi
        theta = 0.
        length = 2.

        res = pra.libroom.compute_segment_end(start, length, phi, theta)

        ok0 = abs(res[0] - (-2.0)) < eps
        ok1 = abs(res[1] - .0) < eps
        self.assertTrue(all([ok0, ok1]))

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

    def test_reflected_end2D(self):

        eps = 0.001
        start = [1,3]
        hit = [5,3]
        normal = [1, 1]
        length = 4

        res = pra.libroom.compute_reflected_end(start, hit, normal, length)
        ok1 = res[0] - 5. < eps
        ok2 = res[1] - 3. < eps
        self.assertTrue(all([ok1, ok2]))

    def test_reflected_end3D(self):

        eps = 0.001
        start = [1,1,1]
        hit = [-1, 1, 3]
        normal = [1, 0, 0]
        length = 2*np.sqrt(2)

        res = pra.libroom.compute_reflected_end(start, hit, normal, length)
        ok1 = res[0] - 1 < eps
        ok2 = res[1] == 1.
        ok3 = res[2] - 5. < eps
        self.assertTrue(all([ok1, ok2, ok3]))

if __name__ == '__main__':
    unittest.main()
