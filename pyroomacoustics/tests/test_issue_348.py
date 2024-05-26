import numpy as np

import pyroomacoustics as pra


def test_circular_microphone_array_xyplane():
    # Create a circular microphone array
    R = 1.0  # radius
    M = 2  # number of microphones
    center = np.array([1.0, 0.0, 0.0])  # center of the array

    mic_array = pra.circular_microphone_array_xyplane(
        center=center, M=M, phi0=0.0, radius=R, fs=16000
    )

    # what it should be
    R_ref = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    assert np.allclose(mic_array.R, R_ref)
