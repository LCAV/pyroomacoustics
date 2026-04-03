import numpy as np
import pytest

import pyroomacoustics as pra

_FS = 16000
_MAX_ORDER = 3


def test_from_corner_extrude():

    room_dim = [3, 4, 5]
    # This test is sensitive to the selection of this value.
    # If some symmetries are present, for some reasons differences between
    # the two simulated methods happen.
    src_loc = [1.001, 0.999, 1.002]
    mic_loc = [2, 3, 4]
    mat = pra.Material(energy_absorption=0.1)

    room_ref = pra.ShoeBox(room_dim, fs=_FS, max_order=_MAX_ORDER, materials=mat)
    room_ref.add_source(src_loc).add_microphone(mic_loc)
    room_ref.compute_rir()

    # Now construct the same room with the other set of primitives.
    corners = np.array(
        [[0, 0], [room_dim[0], 0], [room_dim[0], room_dim[1]], [0, room_dim[1]]]
    ).T
    room = pra.Room.from_corners(corners, fs=_FS, max_order=_MAX_ORDER, materials=mat)
    room.extrude(height=room_dim[2], materials=mat)
    room.add_source(src_loc).add_microphone(mic_loc)
    room.compute_rir()

    assert np.allclose(room_ref.rir[0][0], room.rir[0][0], rtol=1e-4, atol=1e-4)


def test_from_corner_extrude_different_materials():

    room_dim = [3, 4, 5]
    # This test is sensitive to the selection of this value.
    # If some symmetries are present, for some reasons differences between
    # the two simulated methods happen.
    src_loc = [1.001, 0.999, 1.002]
    mic_loc = [2, 3, 4]
    mat1 = "hard_surface"
    mat2 = 0.1

    materials = pra.make_materials(
        east=mat1, west=mat1, south=mat1, north=mat1, floor=mat2, ceiling=mat2
    )
    room_ref = pra.ShoeBox(room_dim, fs=_FS, max_order=_MAX_ORDER, materials=materials)
    room_ref.add_source(src_loc).add_microphone(mic_loc)
    room_ref.compute_rir()

    # Now construct the same room with the other set of primitives.
    corners = np.array(
        [[0, 0], [room_dim[0], 0], [room_dim[0], room_dim[1]], [0, room_dim[1]]]
    ).T
    room = pra.Room.from_corners(
        corners,
        fs=_FS,
        max_order=_MAX_ORDER,
        materials=pra.Material(energy_absorption=mat1),
    )
    room.extrude(height=room_dim[2], materials=pra.Material(energy_absorption=mat2))
    room.add_source(src_loc).add_microphone(mic_loc)
    room.compute_rir()

    assert np.allclose(room_ref.rir[0][0], room.rir[0][0], rtol=1e-4, atol=1e-4)
