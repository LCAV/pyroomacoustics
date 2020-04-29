
import numpy as np
import pyroomacoustics as pra

def test_room_volume():

    eps = 0.00001


    # Create the 2D L-shaped room from the floor polygon
    pol = 4 * np.array([[0, 0], [0, 1], [2, 1], [2, 0.5], [1, 0.5], [1, 0]]).T
    r_absor = 0.1
    room = pra.Room.from_corners(pol, fs=16000, max_order=6, absorption=r_absor)

    # Create the 3D room by extruding the 2D by 3 meters
    room.extrude(3., absorption=r_absor)

    assert np.allclose(room.get_volume(), 72, atol=eps)


def test_walls_area():
    eps = 0.00001

    # Create the 2D L-shaped room from the floor polygon
    pol = 4 * np.array([[0, 0], [0, 1], [2, 1], [2, 0.5], [1, 0.5], [1, 0]]).T
    r_absor = 0.1
    room = pra.Room.from_corners(pol, fs=16000, max_order=6, absorption=r_absor)

    # Create the 3D room by extruding the 2D by 3 meters
    room.extrude(3., absorption=r_absor)

    assert np.allclose(room.wall_area(room.walls[0]), 6, atol=eps)
    assert np.allclose(room.wall_area(room.walls[1]), 12, atol=eps)
    assert np.allclose(room.wall_area(room.walls[2]), 6, atol=eps)
    assert np.allclose(room.wall_area(room.walls[3]), 24, atol=eps)
    assert np.allclose(room.wall_area(room.walls[4]), 12, atol=eps)
    assert np.allclose(room.wall_area(room.walls[5]), 12, atol=eps)
    assert np.allclose(room.wall_area(room.walls[6]), 24, atol=eps)
    assert np.allclose(room.wall_area(room.walls[7]), 24, atol=eps)

if __name__ == '__main__':
    test_room_volume()
    test_walls_area()
