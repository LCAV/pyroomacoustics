"""
Simulating RIRs with measured directivity pattern from DIRPAT
==============================================================

In this example, we show how we can apply and use class DIRPATRir to open measured
directivity files from the DIRPAT dataset.

The created objects can be directly used to generate RIRs

With DIRPATRir object we can generate RIRs with mics and source having either
frequency independent CARDIOID patterns or
freqeuncy dependent patterns from DIRPAT dataset.

Parameters
--------------------------------------
    orientation :
        class DirectionVector
    path : (string)
        Path towards the DIRPAT sofa file, the ending name of the file should be the same as specified in the DIRPAT dataset

    DIRPAT_pattern_enum : (string)
        Only used to choose the directivity patterns available in the specific files in the DIRPAT dataset

    # AKG_c480_c414_CUBE.sofa DIRPAT file include mic patterns for CARDIOID ,FIGURE_EIGHT,HYPERCARDIOID ,OMNI,SUBCARDIOID
    a)AKG_c480
    b)AKG_c414K
    c)AKG_c414N
    d)AKG_c414S
    e)AKG_c414A

    Eigenmic directivity pattern file "EM32_Directivity.sofa", specify mic name at the end to retrive directivity pattern for that particular mic from the eigenmike
    a)EM_32_* : where * \in [0,31]
    For example EM_32_9 : Will retrive pattern of mic number "10" from the eigenmic.

    # LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa DIRPAT file include source patterns
    a)Genelec_8020
    b)Lambda_labs_CX-1A
    c)HATS_4128C
    d)Tannoy_System_1200
    e)Neumann_KH120A
    f)Yamaha_DXR8
    g)BM_1x12inch_driver_closed_cabinet
    h)BM_1x12inch_driver_open_cabinet
    i)BM_open_stacked_on_closed_withCrossoverNetwork
    j)BM_open_stacked_on_closed_fullrange
    k)Palmer_1x12inch
    l)Vibrolux_2x10inch

    fs : (int)
        Sampling frequency of the filters for interpolation.
        Should be same as the simulator frequency and less than 44100 kHz
    no_points_on_fibo_sphere : (int)
        Number of points on the interpolated Fibonacci sphere.
        if "0" no interpolation will happen.


This implementation shows how we can use both DIRPAT object and frequency independent directivity class CardioidFamily together.
We can also use the objects separately.

~ Prerak SRIVASTAVA, 8/11/2022
"""


import os

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

path_DIRPAT_file = os.path.join(
    os.path.dirname(__file__).replace("examples", ""),
    "pyroomacoustics",
    "data",
    "sofa",
    "AKG_c480_c414_CUBE.sofa",
)
path_Eigenmic_file = os.path.join(
    os.path.dirname(__file__).replace("examples", ""),
    "pyroomacoustics",
    "data",
    "sofa",
    "EM32_Directivity.sofa",
)

akg_c414k = SOFADirectivityFactory(
    path=path_DIRPAT_file, DIRPAT_pattern_enum="AKG_c414K", source=False, fs=16000
)
dir_obj_Dmic = akg_c414k.create(
    orientation=DirectionVector(azimuth=54, colatitude=73, degrees=True),
)

eigenmike = SOFADirectivityFactory(
    path=path_Eigenmic_file, DIRPAT_pattern_enum="EM_32_9", source=False, fs=16000
)
dir_obj_Emic = eigenmike.create(
    orientation=DirectionVector(azimuth=54, colatitude=73, degrees=True)
)

dir_obj_Cmic = CardioidFamily(
    orientation=DirectionVector(azimuth=90, colatitude=123, degrees=True),
    pattern_enum=DirectivityPattern.FIGURE_EIGHT,
)


dir_obj_Csrc = CardioidFamily(
    orientation=DirectionVector(azimuth=56, colatitude=123, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
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
    fs=16000,
    max_order=20,
    materials=pra.Material(0.5),
    air_absorption=True,
    ray_tracing=False,
    min_phase=False,
)

dir_mic = dir_obj_Emic

room.add_source([1.52, 0.883, 1.044], directivity=dir_obj_Csrc)


room.add_microphone([2.31, 1.65, 1.163], directivity=dir_mic)

dir_mic.set_orientation(DirectionVector(54, 73))


room.compute_rir()
room.plot_rir(FD=True)

# print(dir_mic.obj_open_sofa_inter.freq_angles_fft.shape)
# dir_mic.obj_open_sofa_inter.interpolate = False

fig = plt.figure()
for idx, fb in enumerate(range(44)):
    if idx >= 5 * 10:
        break
    ax = fig.add_subplot(5, 10, idx + 1, projection="3d")
    dir_mic.plot(freq_bin=fb, ax=ax, depth=True)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_title(idx)
plt.show()


rir_1_0 = room.rir[0][0]
