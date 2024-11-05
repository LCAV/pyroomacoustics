# 2022 (c) Prerak SRIVASTAVA
# 2024/11/27 Modified by Robin Scheibler
"""
Simulating RIRs with measured directivity patterns from DIRPAT
==============================================================

In this example, we show how we can use measured directivity patterns
from the DIRPAT dataset in a simulation.

The procedure to use the directivity patterns is as follows.

1. Read the files potentially containing multiple measurements.
2. Get a directivity object from the file object with desired orientation.
   The directivities can be accessed by index or label (if existing).
   The same pattern can be used multiple times with different orientations.
3. Provide the directivity pattern object to the microphone object.

The DIRPAT database has three different files.

The ``AKG_c480_c414_CUBE.sofa`` DIRPAT file include mic patterns for CARDIOID,
FIGURE_EIGHT, HYPERCARDIOID, OMNI, SUBCARDIOID.

a)AKG_c480
b)AKG_c414K
c)AKG_c414N
d)AKG_c414S
e)AKG_c414A

The Eigenmic directivity pattern file ``EM32_Directivity.sofa``, specify mic
name at the end to retrive directivity pattern for that particular mic from the
eigenmike. This file contains 32 patterns of the form ``EM_32_*``, where ``*``
is one of 0, 1, ..., 31. For example, ``EM_32_9`` will retrive pattern of mic
number "10" from the eigenmic.

The ``LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa`` DIRPAT file includes
some source patterns.

a) Genelec_8020
b) Lambda_labs_CX-1A
c) HATS_4128C
d) Tannoy_System_1200
e) Neumann_KH120A
f) Yamaha_DXR8
g) BM_1x12inch_driver_closed_cabinet
h) BM_1x12inch_driver_open_cabinet
i) BM_open_stacked_on_closed_withCrossoverNetwork
j) BM_open_stacked_on_closed_fullrange
k) Palmer_1x12inch
l) Vibrolux_2x10inch
"""

import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
    FigureEight,
    MeasuredDirectivityFile,
    Rotation3D,
)

# Reads the file containing the Eigenmike's directivity measurements
eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=16000)
# Reads the file containing the directivity measurements of another microphones
akg = MeasuredDirectivityFile("AKG_c480_c414_CUBE", fs=16000)

# Create a rotation object to orient the microphones.
rot_54_73 = Rotation3D([73, 54], "yz", degrees=True)

# Get the directivity objects from the two files
dir_obj_Dmic = akg.get_mic_directivity("AKG_c414K", orientation=rot_54_73)
dir_obj_Emic = eigenmike.get_mic_directivity("EM_32_9", orientation=rot_54_73)

# Create two analytical directivities for comparison
dir_obj_Cmic = FigureEight(
    orientation=DirectionVector(azimuth=90, colatitude=123, degrees=True),
)
dir_obj_Csrc = Cardioid(
    orientation=DirectionVector(azimuth=56, colatitude=123, degrees=True),
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

dir_mic.set_orientation(Rotation3D([73, 54], rot_order="yz"))


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
    ax.set_title(idx)
plt.show()


rir_1_0 = room.rir[0][0]
