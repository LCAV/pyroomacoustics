# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

import unittest
import numpy as np

import pyroomacoustics as pra


c0 = np.array([  # left
            [ 0, 3, 3, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 2, 2],
            ])
c1 = np.array([  # right
            [ 0, 0, 6, 6],
            [ 8, 8, 8, 8],
            [ 0, 4, 4, 0],
            ])
c2 = np.array([  # floor
            [ 0, 0, 6, 3, ],
            [ 0, 8, 8, 0, ],
            [ 0, 0, 0, 0, ],
            ])
c3 = np.array([  # ceiling
            [ 0, 3, 6, 0, ],
            [ 0, 0, 8, 8, ],
            [ 2, 2, 4, 4, ],
            ])
c4 = np.array([  # back
            [ 0, 0, 0, 0, ],
            [ 0, 0, 8, 8, ],
            [ 0, 2, 4, 0, ],
            ])
c5 = np.array([  # front
            [ 3, 6, 6, 3, ],
            [ 0, 8, 8, 0, ],
            [ 0, 0, 4, 2, ],
            ])

# Strange room (not square and inclined roof)
wall_corners_3D = [c0, c1, c2, c3, c4, c5]

absorptions_3D = [ 0.1, 0.25, 0.25, 0.25, 0.2, 0.15 ]


# Let's describe a pentagonal room with corners :
# (-1,0) (-1,2) (0,3) (2,2) (2,-1)
d0 = np.array([  # side1
            [ -1, -1],
            [ 0, 2],
            ])
d1 = np.array([  # side2
            [ -1, 0],
            [ 2, 3],
            ])
d2 = np.array([  # side3
            [ 0, 2],
            [ 3, 2],
            ])
d3 = np.array([  # side4
            [ 2, 2],
            [ 2, -1],
            ])
d4 = np.array([  # side5
            [ 2, -1],
            [ -1, 0],
            ])
wall_corners_2D = [d0, d1, d2, d3, d4]


# Let's describe a simple non_convex room with corners :
# (0,0) (0,2) (1,1) (2,2) (2,0)
e0 = np.array([  # side1
            [ 0, 0],
            [ 0, 2],
            ])
e1 = np.array([  # side2
            [ 0, 1],
            [ 2, 1],
            ])
e2 = np.array([  # side3
            [ 1, 2],
            [ 1, 2],
            ])
e3 = np.array([  # side4
            [ 2, 2],
            [ 2, 0],
            ])
e4 = np.array([  # side5
            [ 2, 0],
            [ 0, 0],
            ])

wall_corners_2D_non_convex = [e0, e1, e2, e3, e4]
absorptions_2D = [ 0.1, 0.1, 0.1, 0.1, 0.1]


f0 = np.array([  # side1
            [ 0, 0],
            [ 0, 3],
            ])
f1 = np.array([  # side2
            [ 0, 4],
            [ 3, 3],
            ])
f2 = np.array([  # side3
            [ 4, 4],
            [ 3, 0],
            ])
f3 = np.array([  # side4
            [ 4, 0],
            [ 0, 0],
            ])

wall_corners_2D_shoebox = [f0, f1, f2, f3]
absorptions_shoebox = [ 0.1, 0.1, 0.1, 0.1]




class TestRoomWalls(unittest.TestCase):

    def test_max_dist_3D(self):

        walls = [pra.libroom.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
            [1, ],
        ])

        room = pra.libroom.Room(walls, obstructing_walls, microphones)

        eps = 0.001
        result = room.get_max_distance()
        correct = np.sqrt(116)+1
        self.assertTrue(abs(result - correct) < eps)


    def test_max_dist_2D(self):

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D, absorptions_2D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

        eps = 0.001
        result = room.get_max_distance()
        self.assertEqual(result, np.sqrt(25)+1)

    def test_same_wall_true3D(self):
        w1 = pra.libroom.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom.Wall(wall_corners_3D[0], absorptions_3D[0])
        self.assertTrue(w1.same_as(w2))

    def test_same_wall_true2D(self):
        w1 = pra.libroom.Wall2D(wall_corners_2D[0], absorptions_3D[0])
        w2 = pra.libroom.Wall2D(wall_corners_2D[0], absorptions_3D[0])
        self.assertTrue(w1.same_as(w2))

    def test_same_wall_false3D(self):
        w1 = pra.libroom.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom.Wall(wall_corners_3D[1], absorptions_3D[0])
        self.assertTrue(not w1.same_as(w2))

    def test_same_wall_false3D_more_corners(self):

        # Modification of wall_corners_3D[0]: adding a corner => 5 corners wall
        c1 = np.array([
            [0, 3, 3, 1.5, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 1.5, 2]])

        w1 = pra.libroom.Wall(wall_corners_3D[0], absorptions_3D[0])
        w2 = pra.libroom.Wall(c1, absorptions_3D[0])
        self.assertTrue(not w1.same_as(w2))


    def test_next_wall_hit(self):

        walls = [pra.libroom.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
            [1, ],
        ])

        room = pra.libroom.Room(walls, obstructing_walls, microphones)

        eps = 0.001

        # start on one empty space
        start = [-2,4,1]

        # end at the same (x,y) but very high in the sky
        end = [5,-3,1]

        ttuple = np.array(room.next_wall_hit(start, end, False))

        correct_result = np.allclose(ttuple[0], [0,2,1], atol=eps)
        correct_next_wall = ttuple[1] == 4

        self.assertTrue(correct_next_wall and correct_result)


    def test_next_wall_nohit(self):

        walls = [pra.libroom.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
            [1, ],
        ])

        room = pra.libroom.Room(walls, obstructing_walls, microphones)

        eps = 0.001

        # start outside the room
        start = [-1,-1,-1]

        # end outside the room
        end = [-2,-3,-1]

        ttuple = np.array(room.next_wall_hit(start, end, False))

        self.assertTrue(ttuple[1] == -1)


    def test_next_wall_hit2D(self):

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D, absorptions_2D)]
        obstructing_walls = []
        microphones = np.array([
            [1, ],
            [1, ],
            [1, ],
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

        eps = 0.001

        # start on one empty space
        start = [0,-2]

        # end at the same (x,y) but very high in the sky
        end = [0,4]

        ttuple = np.array(room.next_wall_hit(start, end, False))

        correct_result = np.allclose(ttuple[0], [0, -1./3], atol=eps)
        correct_next_wall = ttuple[1] == 4

        self.assertTrue(correct_next_wall and correct_result )


    def test_scat_ray_blocked(self):
        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D_non_convex, absorptions_2D)]
        obstructing_walls = [1, 2]  # index of the 2 possibly obstructing walls
        microphones = np.array([
            [1.5],
            [1.2]
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

        radius = 0.1

        prev_wall = room.get_wall(0)


        prev_last_hit = [0.5, 0.]
        last_hit = [0, 1.9]
        total_dist = 0.

        energy = 1000000.
        energy_thres = 0.001
        scatter_coef = 0.1

        # Very high => will not be reached
        travel_time = 1.
        time_thres = 200.

        sound_speed = 340.

        output = [[[1.,2.]]] #arbitrary initialisation to have the correct shape
        self.assertTrue(not room.scat_ray(energy, scatter_coef, prev_wall, prev_last_hit, last_hit,
                                      radius, total_dist, travel_time, time_thres, energy_thres,
                                      sound_speed, False, output))


    def test_scat_ray_ok(self):

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D_non_convex, absorptions_2D)]
        obstructing_walls = [1,2]
        microphones = np.array([
            [0.5 ],
            [0.2 ]
            ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)
        radius = 0.1

        prev_wall = room.get_wall(0)


        prev_last_hit = [2,0.2]
        last_hit = [0, 1.9]
        total_dist = 0.


        energy = 1000000.
        energy_thres = 0.001
        scatter_coef = 0.1

        # Very high => will not be reached
        travel_time = 100.
        time_thres = 200.

        sound_speed = 340.

        output = [[[1.,2.]]] #arbitrary initialisation to have the correct shape
        self.assertTrue(room.scat_ray(energy, scatter_coef, prev_wall, prev_last_hit, last_hit,
                                      radius, total_dist, travel_time, time_thres, energy_thres,
                                      sound_speed, False, output))


    def test_scat_ray_energy(self):

        # Test energy with the energy / (4*pi*dist) rule

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D_shoebox, absorptions_shoebox)]
        obstructing_walls = []
        microphones = np.array([
            [3.1],
            [1.5 ]
            ])

        mic_pos = microphones[:, 0]
        print(mic_pos)

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)
        radius = 0.1

        prev_wall = room.get_wall(0)

        total_dist = 2. #total to reach last hit
        prev_last_hit = [2,1.5]
        last_hit = [0, 1.5]



        energy = 1.
        scatter_coef = 0.1

        eps = 0.0001
        result = room.compute_scat_energy(energy, scatter_coef, prev_wall, prev_last_hit, last_hit, mic_pos, radius, total_dist, True)
        self.assertTrue(np.allclose(result, np.sqrt(0.1*2*(1-np.sqrt(3.1*3.1-0.1*0.1)/3.1))/(4*np.pi*5), atol=eps))


    def test_contains_2D(self):

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D_non_convex, absorptions_2D)]
        obstructing_walls = []
        microphones = np.array([
            [0.5],
            [0.2]
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

        self.assertTrue(room.contains(microphones[:,0]))


    def test_notcontains_2D(self):

        walls = [pra.libroom.Wall2D(c, a) for c, a in zip(wall_corners_2D_non_convex, absorptions_2D)]
        obstructing_walls = []
        microphones = np.array([
            [1.],
            [1.7]
        ])

        room = pra.libroom.Room2D(walls, obstructing_walls, microphones)

        self.assertTrue(not room.contains(microphones[:,0]))


    def test_contains_3D(self):

        walls = [pra.libroom.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [1.],
            [1.],
            [1.]
        ])

        room = pra.libroom.Room(walls, obstructing_walls, microphones)



        self.assertTrue(room.contains(microphones[:,0]))


    def test_notcontains_3D(self):

        walls = [pra.libroom.Wall(c, a) for c, a in zip(wall_corners_3D, absorptions_3D)]
        obstructing_walls = []
        microphones = np.array([
            [5.],
            [7.],
            [40]
        ])

        room = pra.libroom.Room(walls, obstructing_walls, microphones)



        self.assertTrue(not room.contains(microphones[:,0]))

if __name__ == '__main__':
    unittest.main()


