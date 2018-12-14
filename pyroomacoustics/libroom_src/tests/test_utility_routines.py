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

    def test_reflected_end2D(self):

        eps = 0.001
        start = [1,3]
        hit = [5,3]
        normal = [-1, -1]
        length = 4

        res = pra.libroom.compute_reflected_end(start, hit, normal, length)
        self.assertTrue(np.allclose(res, [5.,-1.], atol=eps))

    def test_reflected_end3D(self):

        eps = 0.001
        start = [1,1,1]
        hit = [-1, 1, 3]
        normal = [1, 0, 0]
        length = 2*np.sqrt(2)

        res = pra.libroom.compute_reflected_end(start, hit, normal, length)
        self.assertTrue(np.allclose(res, [1,1,5], atol=eps))

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

    def test_quad_solve(self):

        eps = 0.001
        res = pra.libroom.solve_quad(-1,0,1)

        self.assertTrue(abs(res[0]-1)<eps and abs(res[1]+1)<eps)

    def test_quad_solve_uniqueroot(self):

        eps = 0.001
        res = pra.libroom.solve_quad(1,0,0)

        self.assertTrue(sum(abs(res)) < eps)

    def test_mic_intersection_3D(self):

        start = [5,-5,-1]
        end = [0,10,2]
        center = [0,0,0]
        radius = 5.

        res = pra.libroom.mic_intersection(start, end, center, radius)

        # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
        self.assertTrue(np.allclose(res, [4.21, -2.64, -0.53], atol=0.01))


    def test_mic_intersection_2D(self):

        start = [5,-5]
        end = [-6,0]
        center = [0,0]
        radius = 5.

        res = pra.libroom.mic_intersection(start, end, center, radius)

        # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
        self.assertTrue(np.allclose(res, [2.924 , -4.056], atol=0.01))

    def test_mic_intersection_2D_samex(self):

        start = [0,-6]
        end = [0,6]
        center = [0,0]
        radius = 5.

        res = pra.libroom.mic_intersection(start, end, center, radius)

        # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
        self.assertTrue(np.allclose(res, [0,-5], atol=0.01))


    def test_tuple(self):

        a = pra.libroom.test_tuple(-15)
        print(a)


if __name__ == '__main__':
    unittest.main()
