# When using the HPF, we expect the RIR to be zero mean.
import numpy as np
import pytest

import pyroomacoustics as pra


@pytest.mark.parametrize("hpf_enable", [(True,), (False,)])
def test_rir_hpf(hpf_enable):
    pra.constants.set("rir_hpf_enable", hpf_enable)
    np.random.seed(0)

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds
    room_dim = [10, 7.5, 3.5]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim,
        fs=16_000,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        use_rand_ism=True,
        air_absorption=True,
    )
    # place the source in the room
    room.add_source([2.5, 3.73, 1.76])

    # finally place the array in the room
    room.add_microphone([6.3, 4.87, 1.2])

    room.compute_rir()
    rir = room.rir[0][0]

    mean = np.mean(rir)

    if hpf_enable:
        assert abs(mean) < 1e-5
    else:
        assert abs(mean) >= 1e-5
