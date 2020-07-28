# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import unittest
import numpy as np

import pyroomacoustics as pra

fs = 8000
t0 = 0.
absorption = 0.90
max_order_sim = 2
sigma2_n = 5e-7

corners = np.array([[0,0,4,4],[0,4,4,0]])
room = pra.Room.from_corners(
    corners,
    absorption=absorption,
    fs=fs,
    t0=t0,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n)
    
    
room.add_source([2, 2], None, 0)

# place 3 microphones in the room
mics = pra.MicrophoneArray(np.array([[2, 5, 3, 1, 2,],
                                     [2, 5, 3, 1, 1,]]), fs)
room.add_microphone_array(mics)

# run the image source model
room.image_source_model()

# we order the sources lexicographically
ordering = np.lexsort(room.sources[0].images)
images_found = room.sources[0].images[:,ordering]
visibilities = np.array(room.visibility[0][:,ordering], dtype=bool)

# This should be all visible sources, from the 3 mic locations
sources = np.array([[  2.,  -2.,  -2.,   2.,   6.,   6.,  -6.,  -2.,   2.,   6.,  10., -2.,  -2.,   2.,   6.,   6.,   2.],
                    [ -6.,  -2.,  -2.,  -2.,  -2.,  -2.,   2.,   2.,   2.,   2.,   2.,  6.,   6.,   6.,   6.,   6.,  10.]])

# These are the visibilities for each individual microphone
visible_middle = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
visible_outside = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
visible_upperRight = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
visible_lowerLeft = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
visible_lowMiddle = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], dtype=bool)


class TestVisibilityShoeBox2D(unittest.TestCase):

    def test_sources(self):
        self.assertTrue(np.allclose(images_found, sources))

    def test_visibility_middle(self):
        self.assertTrue(np.allclose(images_found[:,visibilities[0,:]], sources[:,visible_middle]))
        
    def test_visibility_outside(self):
        self.assertTrue(np.allclose(images_found[:,visibilities[1,:]], sources[:,visible_outside]))

    def test_visibility_upperRight(self):
        self.assertTrue(np.allclose(images_found[:,visibilities[2,:]], sources[:,visible_upperRight]))
        
    def test_visibility_lowerLeft(self):
        self.assertTrue(np.allclose(images_found[:,visibilities[3,:]], sources[:,visible_lowerLeft]))
        
    def test_visibility_lowMiddle(self):
        self.assertTrue(np.allclose(images_found[:,visibilities[4,:]], sources[:,visible_lowMiddle]))

if __name__ == '__main__':
    unittest.main()
