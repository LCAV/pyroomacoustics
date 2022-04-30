import warnings

import numpy
import numpy as np
import pyroomacoustics as pra

warnings.filterwarnings(action="ignore", category=UserWarning)


def random_room_ism(max_order=10, eps=1e-6, verbose=False):
    """
    Create a random shoebox room and compute the difference
    """

    # locations of stuff
    room_dim = np.random.randint(1, 101, size=3)
    src_loc = np.random.rand(3) * room_dim
    mic_loc = np.random.rand(3) * room_dim
    # too close is not good
    while np.linalg.norm(mic_loc - src_loc) < 0.05:
        mic_loc = np.random.rand(3) * room_dim

    # random list of materials
    materials = dict(
        zip(
            ["north", "south", "west", "east", "ceiling", "floor"],
            [pra.Material(x) for x in np.random.rand(6)],
        )
    )

    # shoebox room: working correctly
    room = pra.ShoeBox(room_dim, max_order=max_order, materials=materials)
    # general room: not working
    room2 = pra.Room(room.walls, max_order=max_order)

    room.add_source(src_loc)
    room2.add_source(src_loc)

    room.add_microphone(mic_loc)
    room2.add_microphone(mic_loc)

    room.image_source_model()
    room2.image_source_model()

    trans_shoebox = np.sort(room.sources[0].damping)
    trans_general = np.sort(room2.sources[0].damping)

    error = np.linalg.norm(trans_general - trans_shoebox)

    if verbose:
        print("error", np.linalg.norm(trans_shoebox - trans_general))

    assert error < eps


def test_ism_shoebox_vs_general(verbose=False):

    np.random.seed(0)
    n_repeat = 100
    max_order = 10
    eps = 5e-6

    for i in range(n_repeat):
        random_room_ism(max_order=max_order, eps=eps, verbose=verbose)


if __name__ == "__main__":
    test_ism_shoebox_vs_general(verbose=True)
