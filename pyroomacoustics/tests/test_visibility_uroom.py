# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from unittest import TestCase
import numpy as np

import pyroomacoustics as pra

fs = 8000
t0 = 1./(fs*np.pi*1e-2)
absorption = 0.90
max_order_sim = 2
sigma2_n = 5e-7

corners = np.array([[0,5,5,3,3,2,2,0], [0,0,5,5,2,2,5,5]])
room1 = pra.Room.fromCorners(
    corners,
    absorption,
    fs,
    t0,
    max_order_sim,
    sigma2_n)
    
room1.addSource([1, 4], None, 0)

class TestVisibilityURoom(TestCase):

    def test_visibility_oppositeSide(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([4, 4]))
        expected = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.assertTrue(all(computed == expected))

    def test_visibility_middle(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([2.5, 1]))
        expected = [0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0]
        self.assertTrue(all(computed == expected))

    def test_visibility_outside(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([2.5, 4]))
        expected = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.assertTrue(all(computed == expected))