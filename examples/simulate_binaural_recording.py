# 2023 Robin Scheibler
"""
This scripts simulates a binaural room impulse response

The directivity patterns are loaded from SOFA files.
The SOFA format is described at https://www.sofaconventions.org

Get more SOFA files at https://www.sofaconventions.org/mediawiki/index.php/Files
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
from scipy.signal import fftconvolve

import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    CardioidFamily,
    DirectionVector,
    DirectivityPattern,
)
from pyroomacoustics.open_sofa_interpolate import SOFADirectivityFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a binaural room impulse response"
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default="mit_kemar_normal_pinna.sofa",
        help="Path to HRTF file",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "input_samples/cmu_arctic_us_axb_a0004.wav",
        help="Path to speech or other source audio file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "output_samples/simulate_binaural_recording.wav",
        help="Path to output file",
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=12,
        help="Maximum order to use in the spherical harmonics interpolation. "
        "Setting to -1 will disable interpolation",
    )
    parser.add_argument(
        "--interp-n-points",
        type=int,
        default=1000,
        help="Maximum order to use in the spherical harmonics interpolation",
    )
    args = parser.parse_args()

    interp_order = 12
    interp_n_points = 1000

    fs, speech = wavfile.read(args.source)
    speech = speech * (0.95 / abs(speech).max())
    azimuth_deg = 45.0
    colatitude_deg = 90.0

    hrtf_left = SOFADirectivityFactory(
        path=args.hrtf,
        DIRPAT_pattern_enum=0,
        fs=fs,
        interp_order=args.interp_order,
        interp_n_points=args.interp_n_points,
    )
    dir_left = hrtf_left.create(
        orientation=DirectionVector(
            azimuth=azimuth_deg, colatitude=colatitude_deg, degrees=True
        ),
    )

    hrtf_right = SOFADirectivityFactory(
        path=args.hrtf,
        DIRPAT_pattern_enum=1,
        fs=fs,
        interp_order=args.interp_order,
        interp_n_points=args.interp_n_points,
    )
    dir_right = hrtf_right.create(
        orientation=DirectionVector(
            azimuth=azimuth_deg, colatitude=colatitude_deg, degrees=True
        ),
    )

    room_dim = [6, 6, 2.4]

    all_materials = {
        "east": pra.Material(
            energy_absorption={
                "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
        "west": pra.Material(
            energy_absorption={
                "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
        "north": pra.Material(
            energy_absorption={
                "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
        "south": pra.Material(
            energy_absorption={
                "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
        "ceiling": pra.Material(
            energy_absorption={
                "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
        "floor": pra.Material(
            energy_absorption={
                "coeffs": [0.11, 0.14, 0.37, 0.43, 0.27, 0.25],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000],
            },
            scattering=0.54,
        ),
    }

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        max_order=40,
        # materials=pra.Material(0.5),
        materials=all_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=False,
        use_rand_ism=True,
        max_rand_disp=0.05,
    )

    room.add_source([1.5, 3.01, 1.044], signal=speech)

    room.add_microphone([1.1, 3.01, 1.8], directivity=dir_left)
    room.add_microphone([1.1, 3.01, 1.8], directivity=dir_right)

    room.simulate()

    signals = room.mic_array.signals
    signals *= 0.95 / abs(signals).max()
    signals = (signals * 2**15).astype(np.int16)
    wavfile.write(args.output, fs, signals.T)

    room.plot_rir(FD=True)
    room.plot_rir(FD=False)

    fig = plt.figure()
    for idx, fb in enumerate(range(44)):
        if idx >= 5 * 10:
            break
        ax = fig.add_subplot(5, 10, idx + 1, projection="3d")
        dir_left.plot(freq_bin=fb, ax=ax, depth=True)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        ax.set_title(idx)
    plt.show()
