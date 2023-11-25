from unittest import TestCase

import numpy as np

from pyroomacoustics import angle_function

pi = np.pi

# for 3-D coordinates
a1 = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
a2 = np.array([[0], [0], [1]])
a3 = np.array([0, 0, 1]).T
a4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

b1 = np.array([[0], [0], [0]])
b2 = np.array([1, -1, 1]).T
b3 = np.array([1, 0, 0]).T


a1_b1 = np.array([[0, 0, pi / 4], [0, 0, np.arctan(np.sqrt(2.0))]])
a1_b2 = np.array(
    [
        [3 * pi / 4, 3 * pi / 4, pi / 2],
        [pi / 2 + np.arctan(1.0 / np.sqrt(2)), pi / 2, pi / 2],
    ]
)
a2_b1 = np.array([[0], [0]])
a2_b2 = np.array([[3 * pi / 4], [pi / 2]])
a3_b1 = np.array([[0], [0]])
a3_b2 = np.array([[3 * pi / 4], [pi / 2]])
a4_b1 = np.array([[0, pi / 2, 0], [0, pi / 2, pi / 2]])
a4_b3 = np.array([[pi, 3 * pi / 4, 0], [pi / 4, pi / 2, 0]])
a2_b3 = np.array([[pi], [pi / 4]])
a3_b3 = np.array([[pi], [pi / 4]])


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


def test_set_3d():
    assert np.allclose(angle_function(a1, b1), a1_b1)
    assert np.allclose(angle_function(a1, b2), a1_b2)


def test_point_3d_1():
    assert np.allclose(angle_function(a2, b1), a2_b1)
    assert np.allclose(angle_function(a2, b2), a2_b2)
    assert np.allclose(angle_function(a2, b3), a2_b3)


def test_point_3d_2():
    assert np.allclose(angle_function(a3, b1), a3_b1)
    assert np.allclose(angle_function(a3, b2), a3_b2)
    assert np.allclose(angle_function(a3, b3), a3_b3)


def test_point_3d_3():
    assert np.allclose(angle_function(a4, b1), a4_b1)
    assert np.allclose(angle_function(a4, b3), a4_b3)


def test_set_2d():
    assert np.allclose(angle_function(c1, d1), c1_d1)
    assert np.allclose(angle_function(c1, d2), c1_d2)


def test_point_2d_1():
    assert np.allclose(angle_function(c2, d1), c2_d1)
    assert np.allclose(angle_function(c2, d2), c2_d2)


def test_point_2d_2():
    assert np.allclose(angle_function(c3, d1), c3_d1)
    assert np.allclose(angle_function(c3, d2), c3_d2)
