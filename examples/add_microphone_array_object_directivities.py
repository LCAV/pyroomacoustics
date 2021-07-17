import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

# passing MicrophoneArray object and directivity   
# create room
corners = np.array([[0,0],[0,10],[10,10],[10,0]]).T
room = pra.Room.from_corners(corners)
room.extrude(5)

# add source
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([2,3,4])

# add a microphone array with single directivity
PATTERN = DirectivityPattern.HYPERCARDIOID
ORIENTATION = DirectionVector(azimuth=30, colatitude=45, degrees=True)
directivity = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

R = np.array([[5], [5], [3]])
MicrophoneArray_object = pra.MicrophoneArray(R, room.fs)
room.add_microphone_array(MicrophoneArray_object, directivity)

# plot
room.plot_rir()
plt.show()
