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

~ Prerak SRIVASTAVA, 1/09/2022
"""



import pyroomacoustics as pra
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq, fft
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
from scipy.signal import fftconvolve
import os



path_DIRPAT_file=os.path.join(os.path.dirname(__file__).replace("examples",""),"pyroomacoustics","data","AKG_c480_c414_CUBE.sofa")

dir_obj_Dmic = DIRPATRir(
    orientation=DirectionVector(azimuth=54, colatitude=73, degrees=True),
    path=path_DIRPAT_file,
    DIRPAT_pattern_enum="AKG_c414K",
    fs=16000,
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
    max_order=2,
    materials=pra.Material(0.99),
    air_absorption=True,
    ray_tracing=False,
    min_phase=False,
)


room.add_source(
    [1.52, 0.883, 1.044], directivity=dir_obj_Csrc
)


room.add_microphone([2.31, 1.65, 1.163], directivity=dir_obj_Dmic)

dir_obj_Dmic.set_orientation(54, 73)


room.compute_rir()



rir_1_0 = room.rir[0][0]
