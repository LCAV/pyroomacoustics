import numpy as np
from pyroomacoustics import angle_function
from unittest import TestCase


pi = np.pi

# for 3-D coordinates
a1 = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
a2 = np.array([[0], [0], [1]])
a3 = np.array([0, 0, 1]).T

b1 = np.array([[0], [0], [0]])
b2 = np.array([1, -1, 1]).T


a1_b1 = np.array([[0, 0, pi / 4], [0, 0, pi / 4]])
a1_b2 = np.array([[3 * pi / 4, 3 * pi / 4, pi / 2], [3 * pi / 4, pi / 2, pi / 2]])
a2_b1 = np.array([[0], [0]])
a2_b2 = np.array([[3 * pi / 4], [pi / 2]])
a3_b1 = np.array([[0], [0]])
a3_b2 = np.array([[3 * pi / 4], [pi / 2]])


# for 2-D coordinates
c1 = np.array([[0, 0], [0, 0]])
c2 = np.array([[1], [1]])
c3 = np.array([1, 1]).T

d1 = np.array([[0], [0]])
d2 = np.array([0, 1]).T


c1_d1 = np.array([[0, 0], [pi / 2, pi / 2]])
c1_d2 = np.array([[-pi / 2, -pi / 2], [pi / 2, pi / 2]])
c2_d1 = np.array([[pi / 4], [pi / 2]])
c2_d2 = np.array([[0], [pi / 2]])
c3_d1 = np.array([[pi / 4], [pi / 2]])
c3_d2 = np.array([[0], [pi / 2]])


class TestAngleFunction(TestCase):
    def test_set_3d(self):
        self.assertTrue(angle_function(a1, b1).all() == a1_b1.all())
        self.assertTrue(angle_function(a1, b2).all() == a1_b2.all())

    def test_point_3d_1(self):
        self.assertTrue(angle_function(a2, b1).all() == a2_b1.all())
        self.assertTrue(angle_function(a2, b2).all() == a2_b2.all())

    def test_point_3d_2(self):
        self.assertTrue(angle_function(a3, b1).all() == a3_b1.all())
        self.assertTrue(angle_function(a3, b2).all() == a3_b2.all())

    def test_set_2d(self):
        self.assertTrue(angle_function(c1, d1).all() == c1_d1.all())
        self.assertTrue(angle_function(c1, d2).all() == c1_d2.all())

    def test_point_2d_1(self):
        self.assertTrue(angle_function(c2, d1).all() == c2_d1.all())
        self.assertTrue(angle_function(c2, d2).all() == c2_d2.all())

    def test_point_2d_2(self):
        self.assertTrue(angle_function(c3, d1).all() == c3_d1.all())
        self.assertTrue(angle_function(c3, d2).all() == c3_d2.all())


def find_error(type_coordinates):
    if type_coordinates == "3-D":
        print("-" * 40)
        print("type_coordinates = 3-D")
        print("-" * 40)

        a_range = [a1, a2, a3]
        b_range = [b1, b2]
        a_b_range = [a1_b1, a1_b2, a2_b1, a2_b2, a3_b1, a3_b2]

        a_b_index = 0
        for a in a_range:
            for b in b_range:

                error_azimuth = (angle_function(a, b) - a_b_range[a_b_index])[0]
                error_colatitude = (angle_function(a, b) - a_b_range[a_b_index])[1]

                print("for points :")
                print(a)
                print(b)
                print(
                    "error in azimuth calculation: {}".format(np.average(error_azimuth))
                )
                print(
                    "error in colatitude calculation: {}".format(
                        np.average(error_colatitude)
                    )
                )
                print()
                a_b_index += 1

    elif type_coordinates == "2-D":
        print("-" * 40)
        print("type_coordinates = 2-D")
        print("-" * 40)

        c_range = [c1, c2, c3]
        d_range = [d1, d2]
        c_d_range = [c1_d1, c1_d2, c2_d1, c2_d2, c3_d1, c3_d2]

        c_d_index = 0
        for c in c_range:
            for d in d_range:

                error_azimuth = (angle_function(c, d) - c_d_range[c_d_index])[0]
                error_colatitude = (angle_function(c, d) - c_d_range[c_d_index])[1]

                print("for points :")
                print(c)
                print(d)
                print(
                    "error in azimuth calculation: {}".format(np.average(error_azimuth))
                )
                print(
                    "error in colatitude calculation: {}".format(
                        np.average(error_colatitude)
                    )
                )
                print()
                c_d_index += 1


if __name__ == "__main__":
    find_error("3-D")
    find_error("2-D")
