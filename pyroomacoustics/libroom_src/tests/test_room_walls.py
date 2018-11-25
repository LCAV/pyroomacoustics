# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import unittest
import numpy as np

import pyroomacoustics as pra


wall_corners_3D = [
        np.array([  # left
            [ 0, 3, 3, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 2, 2],
            ]),
        np.array([  # right
            [ 0, 0, 6, 6],
            [ 8, 8, 8, 8],
            [ 0, 4, 4, 0],
            ]),
        np.array([  # floor
            [ 0, 0, 6, 3, ],
            [ 0, 8, 8, 0, ],
            [ 0, 0, 0, 0, ],
            ]),
        np.array([  # ceiling
            [ 0, 3, 6, 0, ],
            [ 0, 0, 8, 8, ],
            [ 2, 2, 4, 4, ],
            ]),
        np.array([  # back
            [ 0, 0, 0, 0, ],
            [ 0, 0, 8, 8, ],
            [ 0, 2, 4, 0, ],
            ]),
        np.array([  # front
            [ 3, 6, 6, 3, ],
            [ 0, 8, 8, 0, ],
            [ 0, 0, 4, 2, ],
            ]),
        ]


absorptions_3D = [ 0.1, 0.25, 0.25, 0.25, 0.2, 0.15 ]


# Let's describe a pentagonal room with corners :
# (-1,0) (-1,2) (0,3) (2,2) (2,-1)
wall_corners_2D = [
        np.array([  # side1
            [ -1, -1],
            [ 0, 2],
            ]),
        np.array([  # side2
            [ -1, 0],
            [ 2, 3],
            ]),
        np.array([  # side3
            [ 0, 2],
            [ 3, 2],
            ]),
        np.array([  # side4
            [ 2, 2],
            [ 2, -1],
            ]),
        np.array([  # side5
            [ 2, -1],
            [ -1, 0],
            ]),
        ]

absorptions_2D = [ 0.1, 0.25, 0.25, 0.25, 0.2]


class TestRoomWalls(unittest.TestCase):

    def test_max_dist_3D(self):

        walls = [pra.libroom_new.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
            [1, ],
        ])

        room = pra.libroom_new.Room(walls, obstructing_walls, microphones)

        eps = 0.001
        result = room.get_max_distance()
        correct = np.sqrt(116)+1
        self.assertTrue(all([abs(result - correct) < eps]))


    def test_max_dist_2D(self):

        walls = [pra.libroom_new.Wall(c, a) for c, a in zip(wall_corners_2D, absorptions_2D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
        ])

        room = pra.libroom_new.Room(walls, obstructing_walls, microphones)

        eps = 0.001
        result = room.get_max_distance()
        self.assertEqual(result, np.sqrt(25)+1)

    def test_same_wall_true3D(self):
        w1 = pra.libroom_new.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom_new.Wall(wall_corners_3D[0], absorptions_3D[0])
        self.assertTrue(all([w1.same_as(w2)]))

    def test_same_wall_true2D(self):
        w1 = pra.libroom_new.Wall(wall_corners_2D[0], absorptions_3D[0])
        w2 = pra.libroom_new.Wall(wall_corners_2D[0], absorptions_3D[0])
        self.assertTrue(all([w1.same_as(w2)]))

    def test_same_wall_false3D(self):
        w1 = pra.libroom_new.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom_new.Wall(wall_corners_3D[1], absorptions_3D[0])
        self.assertTrue(all([not w1.same_as(w2)]))

    def test_same_wall_false3D_more_corners(self):

        # Modification of wall_corners_3D[0]: adding a corner => 5 corners wall
        c1 = np.array([
            [0, 3, 3, 1.5, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 1.5, 2]])

        w1 = pra.libroom_new.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom_new.Wall(c1, absorptions_3D[0])
        self.assertTrue(all([not w1.same_as(w2)]))


if __name__ == '__main__':
    unittest.main()
