import numpy as np
import pyroomacoustics as pra

wall_corners = [
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

absorptions = [ 0.1, 0.25, 0.25, 0.25, 0.2, 0.15 ]

def test_room_construct():

    walls = [pra.wall_factory(c, a) for c, a in zip(wall_corners, absorptions)]
    obstructing_walls = []
    microphones = np.array([
        [ 1, ],
        [ 2, ],
        [ 1, ],
        ])

    room = pra.room_factory(walls, obstructing_walls, microphones)

    print(room.max_dist)

    return room

if __name__ == '__main__':
    
    room = test_room_construct()