import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily

# make a room
room  = pra.ShoeBox(
            p = [7, 7, 5],
            materials=pra.Material(0.07),
            fs=16000,
            max_order=17,
        )

# add source 
room.add_source([5,2.5,4])
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([-1, 6])

# plot circular microphone array
pra.beamforming.circular_microphone_array_helper_xyplane(center=[3,3], M=7, phi0=0, radius=2, fs=16000, height=2, directivity_pattern="FIGURE_EIGHT", plot=True, ax=ax)