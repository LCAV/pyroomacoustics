from unittest import TestCase

from random import randint
import numpy as np

import pyroomacoustics as pra

class TestGeometryRoutines(TestCase):

    def test_orientation_anticlockwise(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0], roomDimensions, 0)
        self.assertTrue(room.ccw3p(np.array([[6, 4, 2], [0, 2, 0]])) == 1)

    def test_orientation_clockwise(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0], roomDimensions, 0)
        self.assertTrue(room.ccw3p(np.array([[2, 4, 6], [0, 2, 0]])) == -1)

    def test_intersection_separated(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        s1 = [[4, 4], [0, 4]]
        s2 = [[0, 2], [2, 2]]
        self.assertTrue(room.intersects(np.array(s1), np.array(s2)) == 0)

    def test_intersection_cross(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        s1 = [[0, 4], [2, 2]]
        s2 = [[2, 2], [0, 4]]
        self.assertTrue(room.intersects(np.array(s1), np.array(s2)) == 1)
        
    def test_intersection_t(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        s1 = [[0, 4], [2, 2]]
        s2 = [[4, 4], [0, 4]]
        self.assertTrue(room.intersects(np.array(s1), np.array(s2)) == 3)

    def test_inside_middle(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        self.assertTrue(room.isInside(np.array([2,2]), room.corners, True))
        
    def test_inside_outsideOnTheRight(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        self.assertTrue(not room.isInside(np.array([5,2]), room.corners, True))
        
    def test_inside_onBorderInclusive(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        self.assertTrue(room.isInside(np.array([0,2]), room.corners, True))
        
    def test_inside_onBorderExclusive(self):
        roomDimensions = [4,4]
        room = pra.Room.shoeBox2D([0,0],roomDimensions,0)
        self.assertTrue(room.isInside(np.array([0,2]), room.corners, False))