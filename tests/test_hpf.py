import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

# Make sure the high-pass filter is used.
pra.constants.set("rir_hpf_enable", True)


def test_hpf():

    # Define parameters
    room_dims = (3.29, 6.23, 2.58)  # (3.29, 6.23, 2.58) or (5.78, 8.42, 3.86)
    rt60 = 0.827  # 0.827 or 0.932
    source_pos = (2.59, 1.5, 1.31)  # (2.59, 0.83, 1.31) or (4.59, 5.13, 2.45)
    mic_pos = (1.05, 6.14, 1.15)  # (1.05, 6.14, 1.15) or (4.04, 0.79, 2.03)
    fs = 48000

    # from t60 to absorption coefficients
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = (
        pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=pra.Material(energy_absorption=e_absorption),
            max_order=max_order,
        )
        .add_source(source_pos)
        .add_microphone(mic_pos)
    )
    room.compute_rir()
    dc_offset = np.mean(room.rir[0][0])

    eps = 1e-4
    assert abs(dc_offset) < eps, f"Found DC offset {dc_offset} larger than {eps=}."


if __name__ == "__main__":
    test_hpf()
