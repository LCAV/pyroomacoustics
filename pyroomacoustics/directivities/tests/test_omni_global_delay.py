"""
We can simulate an omnidirectional microphone with a MeasuredDirectivity.
In this case, there should be no difference with not using a directivity at all.

Ref: `Issue 398 <https://github.com/LCAV/pyroomacoustics/issues/398>`_
"""

import numpy as np
import pyroomacoustics as pra

# Disable the high-pass filter to keep consistent test result.
pra.constants.set("rir_hpf_enable", False)


def test_omni_delay_analytical_vs_measured():

    fs = 24000
    rir_len = 256

    room_dims = np.array([4.0, 5.0, 6.0])

    # two mics with same location
    mic_array = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )

    sound_source_coords = np.array([3.5, 3.5, 2.0])
    max_order = 0

    analytical_omni = pra.Omnidirectional()

    # simulate omni directivity with measured directivity.
    # # unit impulse, zero padding, grid of size 1 :-)
    ir_grid = pra.doa.GridSphere(spherical_points=np.array([[0.0], [0.0]]))
    rirs = np.zeros((1, rir_len))
    rirs[:, 0] = 1.0
    measured_omni = pra.MeasuredDirectivity(
        orientation=pra.Rotation3D([0.0, 0.0], "yz", degrees=True),
        grid=ir_grid,
        impulse_responses=rirs,
        fs=fs,
    )

    room = pra.ShoeBox(room_dims, fs=fs, max_order=max_order)

    room.add_microphone_array(
        mic_array=mic_array.T, directivity=[measured_omni, analytical_omni]
    )
    room.add_source(sound_source_coords)
    room.compute_rir()

    rir_measured = room.rir[0][0]
    rir_analytical = room.rir[1][0]
    m = min([len(rir_measured), len(rir_analytical)])

    np.testing.assert_allclose(
        rir_measured[:m], rir_analytical[:m], atol=1e-3, rtol=1e-3
    )
