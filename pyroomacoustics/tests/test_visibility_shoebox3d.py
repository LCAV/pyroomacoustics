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

room_dim = [6, 6, 6]
room1 = pra.ShoeBox(
    room_dim,
    absorption,
    fs,
    t0,
    max_order_sim,
    sigma2_n)
    
room1.addSource([3, 3, 3], None, 0)

class TestVisibilityShoeBox3D(TestCase):

    def test_visibility_middle(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([3, 3, 3]))
        expected = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.assertTrue(all(computed == expected))

    def test_visibility_east(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([5, 3, 3]))
        expected = [1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,1,1]
        I = np.where(computed != np.array(expected))
        self.assertTrue(all(computed == expected))
        
    def test_visibility_southWestBottomCorner(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([0, 0, 0]))
        expected = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1]
        self.assertTrue(all(computed == expected))
        
    def test_visibility_outside(self):
        computed = room1.checkVisibilityForAllImages(room1.sources[0], np.array([7, 7, 7]))
        expected = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.assertTrue(all(computed == expected))
