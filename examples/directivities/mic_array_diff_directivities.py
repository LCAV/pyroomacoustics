import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra
from pyroomacoustics.directivities import (CardioidFamily, DirectionVector,
                                           DirectivityPattern)

dir_1 = CardioidFamily(
    orientation=DirectionVector(azimuth=180, colatitude=30, degrees=True),
    pattern_enum=DirectivityPattern.HYPERCARDIOID,
)
dir_2 = CardioidFamily(
    orientation=DirectionVector(azimuth=0, colatitude=30, degrees=True),
    pattern_enum=DirectivityPattern.HYPERCARDIOID,
)
# source_dir = None
source_dir = CardioidFamily(
    orientation=DirectionVector(azimuth=45, colatitude=90, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)

# create room
room = pra.ShoeBox(
    p=[7, 7, 3],
    materials=pra.Material(0.4),
    fs=16000,
    max_order=40,
)

# add source
room.add_source([1, 1, 1.7], directivity=source_dir)

# add linear microphone array
M = 2
R = pra.linear_2D_array(center=[5, 5], M=M, phi=0, d=0.7)
R = np.concatenate((R, np.ones((1, M))))
room.add_microphone_array(R, directivity=[dir_1, dir_2])

# plot room
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([-1, 4])

# plot
room.plot_rir()
plt.show()
