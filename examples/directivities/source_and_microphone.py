import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

three_dim = True  # 2D or 3D
shoebox = True  # source directivity not supported for non-shoebox!
energy_absorption = 0.4
source_pos = [2, 1.8]
# source_dir = None   # to disable
source_dir = DirectivityPattern.FIGURE_EIGHT
mic_pos = [3.5, 1.8]
# mic_dir = None   # to disable
mic_dir = DirectivityPattern.HYPERCARDIOID


# make 2-D room
if shoebox:
    room = pra.ShoeBox(
        p=[5, 3, 3] if three_dim else [5, 3],
        materials=pra.Material(energy_absorption),
        fs=16000,
        max_order=40,
    )
else:
    corners = np.array([[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]]).T
    room = pra.Room.from_corners(
        corners, materials=pra.Material(energy_absorption), max_order=10, fs=16000
    )
    if three_dim:
        room.extrude(3)
if three_dim:
    colatitude = 90
    source_pos.append(1.8)
    mic_pos.append(1.0)
else:
    colatitude = None

# add source with directivity
if source_dir is not None:
    source_dir = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=colatitude, degrees=True),
        pattern_enum=source_dir,
    )
room.add_source(position=source_pos, directivity=source_dir)

# add microphone with directivity
if mic_dir is not None:
    mic_dir = CardioidFamily(
        orientation=DirectionVector(azimuth=0, colatitude=colatitude, degrees=True),
        pattern_enum=mic_dir,
    )
room.add_microphone(loc=mic_pos, directivity=mic_dir)

# plot room
fig, ax = room.plot()
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 4])
if three_dim:
    ax.set_zlim([-1, 4])

# plot RIR
room.plot_rir()
plt.show()
