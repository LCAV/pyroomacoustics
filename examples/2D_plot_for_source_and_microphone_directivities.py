import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily

# make 2-D room 
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T 
room = pra.Room.from_corners(corners)
fig, ax = room.plot()

# make directivity object for source
PATTERN = DirectivityPattern.FIGURE_EIGHT
ORIENTATION = DirectionVector(azimuth=90, colatitude=None, degrees=True)
directivity_1 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# make directivity object for microphone
PATTERN = DirectivityPattern.HYPERCARDIOID
ORIENTATION = DirectionVector(azimuth=0, colatitude=None, degrees=True)
directivity_2 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# make 2D plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
ax = directivity_1.plot_response(azimuth=azimuth, colatitude=None, degrees=True, ax=ax, offset=[2,1.8])
directivity_2.plot_response(azimuth=azimuth, colatitude=None, degrees=True, ax=ax, offset=[3.5,1.8])

ax.set_xlim([-1, 6])
ax.set_ylim([-1, 4])
plt.show()