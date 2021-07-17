import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

# passing microphone array location and direcitivity    
# create room
corners = np.array([[0,0],[0,10],[10,10],[10,0]]).T
room = pra.Room.from_corners(corners)
room.extrude(5)

# add source
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([2,3,4])

# add microphone array with list of directivities
PATTERN = DirectivityPattern.SUBCARDIOID
ORIENTATION = DirectionVector(azimuth=0, colatitude=45, degrees=True)
directivity_1 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

PATTERN = DirectivityPattern.CARDIOID
ORIENTATION = DirectionVector(azimuth=30, colatitude=60, degrees=True)
directivity_2 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

room.add_microphone_array(np.c_[[7,8,3],[8.5,9.5,4.5]], [directivity_1,directivity_2])

# plot
room.plot_rir()
plt.show()
    
