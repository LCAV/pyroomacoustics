import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.fixture
def room():
    fs = 16000
    c = pra.constants.get("c")
    room_dim = np.array([c, c, 2 * c])
    src = np.ones(3) * 0.5 * c
    mic = np.array([0.5, 0.5, 1.5]) * c
    mat = pra.Material(energy_absorption="hard_surface")
    room = (
        pra.ShoeBox(room_dim, fs, max_order=1, materials=mat)
        .add_source(src)
        .add_microphone(mic)
    )
    return room


def test_set_air_absorption_1(room):
    air_abs = 0.5
    room.set_air_absorption([air_abs])

    air_abs_coeffs = room.air_absorption
    assert len(air_abs_coeffs) == 7
    assert all([a == air_abs for a in air_abs_coeffs])


def test_set_air_absorption_2(room):
    air_abs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    room.set_air_absorption(air_abs)

    air_abs_coeffs = room.air_absorption
    assert len(air_abs_coeffs) == 7
    assert all(air_abs == air_abs_coeffs)
