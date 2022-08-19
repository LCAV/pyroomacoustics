import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
import os

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



path_DIRPAT_file=os.path.join(os.path.dirname(__file__).replace("test",""),"data","AKG_c480_c414_CUBE.sofa")

PATTERN_SRC = DirectivityPattern.FIGURE_EIGHT
ORIENTATION_SRC = DirectionVector(azimuth=90, colatitude=90, degrees=True)
directivity_SRC = CardioidFamily(orientation=ORIENTATION_SRC, pattern_enum=PATTERN_SRC)

# define source with figure_eight directivity
PATTERN_MIC_DIRPAT_ID = "AKG_c480"
ORIENTATION_MIC = DirectionVector(azimuth=90, colatitude=90, degrees=True)
directivity_MIC = DIRPATRir(
    orientation=ORIENTATION_MIC,
    path=path_DIRPAT_file,
    DIRPAT_pattern_enum=PATTERN_MIC_DIRPAT_ID,
    fs=16000,
)


# add source with figure_eight directivity
room.add_source([1.52, 0.883, 1.044], directivity=directivity_SRC)

# add microphone in its null
room.add_microphone([2.31, 1.65, 1.163], directivity=directivity_MIC)

# Check set different orientation after intailization of the DIRPATRir class
directivity_MIC.set_orientation(70, 123)
# directivity_SRC.set_orientation(70, 34)


room.compute_rir()

rir_1_0 = room.rir[0][0]

plt.clf()
plt.plot(np.arange(rir_1_0.shape[0]), rir_1_0)
plt.show()

# np.save("/home/psrivast/PycharmProjects/axis_2_phd/pyroom_acous_push_1.npy", rir_1_0)
