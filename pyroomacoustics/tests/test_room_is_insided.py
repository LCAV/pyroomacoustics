
import numpy as np
import pyroomacoustics as pra

def test_room_is_inside():
    # fix the seed for repeatable testing
    np.random.seed(0)

    # This was a problematic case
    # if the source is placed at the same height as one of the corners
    # the test would fail, even though the source is in the room

    floorplan = [ [0, 6, 6, 2, 0],
                  [0, 0, 5, 5, 3] ]

    source_loc = [ 2, 3 ] # same y-coordinate as the corner at [0, 3]

    room = pra.Room.from_corners(floorplan)
    room.add_source(source_loc)

    for i in range(100):
        # because the test is randomized, let's check many times

        assert room.is_inside([0,0], include_borders=True)
        assert not room.is_inside([0,0], include_borders=False)

        assert room.is_inside([3,0], include_borders=True)
        assert not room.is_inside([3,0], include_borders=False)

        assert room.is_inside([1,4], include_borders=True)
        assert not room.is_inside([1,4], include_borders=False)

        assert room.is_inside([0,1], include_borders=True)
        assert not room.is_inside([0,1], include_borders=False)

        assert not room.is_inside([0.5,4], include_borders=False)

    # now test in 3D
    room.extrude(4.)

    for i in range(100):
        # because the test is randomized, let's check many times

        assert room.is_inside([2, 3, 1.7])

        assert not room.is_inside([0.5, 4, 1.8])

        assert not room.is_inside([0.5, 4, 1.8])

        assert room.is_inside([0,0,0], include_borders=True)
        assert not room.is_inside([0,0,0], include_borders=False)

        assert room.is_inside([3,0,0], include_borders=True)
        assert not room.is_inside([3,0,0], include_borders=False)

        assert room.is_inside([0,1,0], include_borders=True)
        assert not room.is_inside([0,1,0], include_borders=False)

        assert room.is_inside([3,2,0], include_borders=True)
        assert not room.is_inside([3,2,0], include_borders=False)

        assert room.is_inside([1,4,3], include_borders=True)
        assert not room.is_inside([1,4,3], include_borders=False)

        assert not room.is_inside([2,2,7])
        assert not room.is_inside([2,2,-7])

if __name__ == '__main__':

    test_room_is_inside()
