import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

# passing single microphone with single directivity
# create room
corners = np.array([[0,0],[0,10],[10,10],[10,0]]).T
room = pra.Room.from_corners(corners)
room.extrude(5)

# add source
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([2,3,4])

# add microphone with directivity
PATTERN = DirectivityPattern.FIGURE_EIGHT
ORIENTATION = DirectionVector(azimuth=30, colatitude=45, degrees=True)
directivity = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

room.add_microphone([5,6,3], directivity)

# plot
room.plot_rir()
plt.show()
