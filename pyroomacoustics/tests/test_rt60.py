import numpy as np

import pyroomacoustics as pra

eps = 1e-10
room_dim = [10, 7.5, 3.5]  # meters
fs = 16000

V = np.prod(room_dim)
S = 2 * (
    room_dim[0] * room_dim[1] + room_dim[1] * room_dim[2] + room_dim[2] * room_dim[0]
)


def test_rt60_theory_single_band():

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    rt60_sabine = pra.rt60_sabine(S, V, e_absorption, 0.0, room.c)
    assert (rt60_sabine - room.rt60_theory(formula="sabine")) < eps

    rt60_eyring = pra.rt60_eyring(S, V, e_absorption, 0.0, room.c)
    assert (rt60_eyring - room.rt60_theory(formula="eyring")) < eps


def test_rt60_theory_multi_band():

    # Create the room
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material("curtains_cotton_0.5"),)

    # run the different rt60 functions
    room.rt60_theory(formula="sabine")
    room.rt60_theory(formula="eyring")


def test_rt60_measure():

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material("curtains_cotton_0.5"), max_order=10,
    )

    # place the source in the room
    room.add_source([2.5, 3.73, 1.76])

    # place a microphone in the room
    room.add_microphone([6.3, 4.87, 1.2])

    room.compute_rir()

    room.measure_rt60()


if __name__ == "__main__":

    test_rt60_theory_single_band()
    test_rt60_theory_multi_band()
    test_rt60_measure()
