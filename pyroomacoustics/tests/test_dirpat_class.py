import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
import matplotlib.pyplot as plt

from unittest import TestCase
import os 


room_dim = [6, 6, 2.4]

path_DIRPAT_file=os.path.join(os.path.dirname(__file__).replace("tests",""),"data","AKG_c480_c414_CUBE.sofa")

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


# create room
room = pra.ShoeBox(
    room_dim,
    fs=16000,
    max_order=2,
    materials=all_materials,
    air_absorption=True,
    ray_tracing=False,
    min_phase=False,
)  # ,min_phase=False)



# define source with figure_eight directivity
PATTERN_MIC_DIRPAT_ID = "AKG_c480"
ORIENTATION_MIC = DirectionVector(azimuth=90, colatitude=90, degrees=True)
directivity_MIC = DIRPATRir(
    orientation=ORIENTATION_MIC,
    path=path_DIRPAT_file,
    DIRPAT_pattern_enum=PATTERN_MIC_DIRPAT_ID,
    fs=16000,
)

"""
PATTERN_SRC_DIRPAT_ID = "HATS_4128C"
ORIENTATION_SRC = DirectionVector(azimuth=123, colatitude=45, degrees=True)
directivity_SRC = DIRPATRir(
    orientation=ORIENTATION_SRC,
    path="/home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    DIRPAT_pattern_enum=PATTERN_SRC_DIRPAT_ID,
    fs=16000,
)
"""

# add source with figure_eight directivity
room.add_source([1.52, 0.883, 1.044])

# add microphone in its null
room.add_microphone([2.31, 1.65, 1.163], directivity=directivity_MIC)

# Check set different orientation after intailization of the DIRPATRir class
directivity_MIC.set_orientation(np.radians(0), np.radians(0))
# directivity_SRC.set_orientation(np.radians(70), np.radians(34))


room.compute_rir()

rir_1_0 = room.rir[0][0]

plt.clf()
plt.plot(np.arange(rir_1_0.shape[0]), rir_1_0)
plt.show()


# np.save("/home/psrivast/PycharmProjects/axis_2_phd/pyroom_acous_push_1.npy",rir_1_0)