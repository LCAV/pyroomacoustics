import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily

# make 2-D room 
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T 
room = pra.Room.from_corners(corners)
fig, ax = room.plot()

# make directivity object
PATTERN = DirectivityPattern.FIGURE_EIGHT
ORIENTATION = DirectionVector(azimuth=60, colatitude=None, degrees=True)
dir_obj = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# plot directivity in the same plot as the room
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
dir_obj.plot_response(azimuth=azimuth, colatitude=None, degrees=True, ax=ax, offset=[2,2])

ax.set_xlim([-1, 6])
ax.set_ylim([-1, 4])
plt.show()