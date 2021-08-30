import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectionVector,
    CardioidFamily,
    DirectivityPattern,
)
import matplotlib.pyplot as plt


three_dim = True     # 2D or 3D

mic_rotation = 0
room_dim = [7, 7]
source_loc = [5, 2.5]
colatitude = None
height = None
if three_dim:
    room_dim.append(5)
    source_loc.append(4)
    height = 2
    colatitude = 90


# make a room
room = pra.ShoeBox(p=room_dim)

# add source
room.add_source(source_loc)
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
if three_dim:
    ax.set_zlim([-1, 6])

# plot circular microphone array
pattern = DirectivityPattern.CARDIOID
orientation = DirectionVector(azimuth=mic_rotation, colatitude=colatitude, degrees=True)
directivity = CardioidFamily(orientation=orientation, pattern_enum=pattern)
pra.beamforming.circular_microphone_array_xyplane(
    center=[3, 3],
    M=7,
    phi0=mic_rotation,
    radius=50e-2,
    fs=room.fs,
    height=height,
    directivity=directivity,
    ax=ax,
)


plt.show()
