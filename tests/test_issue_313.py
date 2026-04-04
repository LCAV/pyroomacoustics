import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra


def test_issue_313():
    """
    Fixes an issue where the `visibility` attribute of the room is not
    set if there are no visible source or image source in the room.
    """
    # Room
    sigma2 = 5e-4
    fs = 16000

    corners = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 16],
            [0, 16],
            [0, 10],
            [8, 10],
            [8, 6],
            [0, 6],
        ]
    ).T
    room = pra.Room.from_corners(corners, fs=fs, max_order=1, sigma2_awgn=sigma2)

    # Microphones
    def mic_array_at(pos: np.ndarray) -> pra.MicrophoneArray:
        mic_locations = pra.circular_2D_array(center=pos, M=6, phi0=0, radius=37.5e-3)
        mic_locations = np.concatenate(
            (mic_locations, np.array(pos, ndmin=2).T), axis=1
        )
        return pra.MicrophoneArray(mic_locations, room.fs)

    mic = mic_array_at(np.array([3, 3]))
    room.add_microphone_array(mic)

    # Sources
    rng = np.random.RandomState(23)
    duration_samples = int(fs)
    source_location = np.array([3, 13])
    source_signal = rng.randn(duration_samples)
    room.add_source(source_location, signal=source_signal)

    room.image_source_model()

    room.compute_rir()
