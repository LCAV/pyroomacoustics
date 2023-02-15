import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.directivities import (CardioidFamily, DirectionVector,
                                           DirectivityPattern)

three_dim = True  # 2D or 3D

mic_rotation = 0
room_dim = [7, 7]
source_loc = [5, 2.5]
center = [3, 3]
colatitude = None
if three_dim:
    room_dim.append(5)
    source_loc.append(4)
    center.append(2)
    colatitude = 90


# make a room
room = pra.ShoeBox(p=room_dim)

# add source
room.add_source(source_loc)

# add circular microphone array
pattern = DirectivityPattern.CARDIOID
orientation = DirectionVector(azimuth=mic_rotation, colatitude=colatitude, degrees=True)
directivity = CardioidFamily(orientation=orientation, pattern_enum=pattern)
mic_array = pra.beamforming.circular_microphone_array_xyplane(
    center=center,
    M=7,
    phi0=mic_rotation,
    radius=50e-2,
    fs=room.fs,
    directivity=directivity,
)
room.add_microphone_array(mic_array)

# plot everything
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
if three_dim:
    ax.set_zlim([-1, 6])


plt.show()
