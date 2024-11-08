"""
This example creates a free-field simulation of direction of arrival estimation.
"""

import argparse

import numpy as np

import pyroomacoustics as pra

methods = ["MUSIC", "FRIDA", "WAVES", "TOPS", "CSSM", "SRP", "NormMUSIC"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates the direction of arrival of a sound source."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="DOA method to use",
    )
    args = parser.parse_args()

    # we use a white noise signal for the source
    nfft = 256
    fs = 16000
    x = np.random.randn((nfft // 2 + 1) * nfft)

    # create anechoic room
    room = pra.AnechoicRoom(fs=fs)

    # place the source at a 90 degree angle and 5 meters distance from origin
    azimuth_true = np.pi / 2
    room.add_source([5 * np.cos(azimuth_true), 5 * np.sin(azimuth_true), 0], signal=x)

    # place the microphone array
    mic_locs = np.c_[
        [0.1, 0.1, 0],
        [-0.1, 0.1, 0],
        [-0.1, -0.1, 0],
        [0.1, -0.1, 0],
    ]
    room.add_microphone_array(mic_locs)

    # run the simulation
    room.simulate()

    # create frequency-domain input for DOA algorithms
    X = pra.transform.stft.analysis(
        room.mic_array.signals.T, nfft, nfft // 2, win=np.hanning(nfft)
    )
    X = np.swapaxes(X, 2, 0)

    # perform DOA estimation
    doa = pra.doa.algorithms[args.method](mic_locs, fs, nfft)
    doa.locate_sources(X)

    # evaluate result
    print("Source is estimated at:", doa.azimuth_recon)
    print("Real source is at:", azimuth_true)
    print("Error:", pra.doa.circ_dist(azimuth_true, doa.azimuth_recon))
