import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

# passing MicrophoneArray object and directivity   
# create room
room  = pra.ShoeBox(
            p = [7, 7, 5],
            materials=pra.Material(0.07),
            fs=16000,
            max_order=17,
        )

# add source
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([5,2.5,2.5], signal = audio_sample)
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([-1, 6])

# add a microphone array with single directivity
PATTERN = DirectivityPattern.HYPERCARDIOID
ORIENTATION = DirectionVector(azimuth=60, colatitude=90, degrees=True)
directivity = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

R = np.array([[5], [5], [3]])
MicrophoneArray_object = pra.MicrophoneArray(R, room.fs)
room.add_microphone_array(MicrophoneArray_object, directivity)

# 3-D plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)

directivity.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[5,5,3])
plt.show()

# plot
room.plot_rir()
plt.show()
