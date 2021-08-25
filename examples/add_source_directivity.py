import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

  
# create room
room  = pra.ShoeBox(
            p = [7, 7, 5],
            materials=pra.Material(0.2),
            fs=16000,
            max_order=40,
        )


# define source directivity
PATTERN = DirectivityPattern.FIGURE_EIGHT
ORIENTATION = DirectionVector(azimuth=0, colatitude=90, degrees=True)
directivity_1 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# define microphone directivity
PATTERN = DirectivityPattern.HYPERCARDIOID
ORIENTATION = DirectionVector(azimuth=90, colatitude=0, degrees=True)
directivity_2 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# add source with directivity as directivity_1
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([5,2.5,2.5], signal = audio_sample, directivity = directivity_1)
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([-1, 6])

# add microphone array with directivity as directivity_2
room.add_microphone([2.5,2.5,2.5], directivity_2)

# 3-D plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)

ax = directivity_1.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[5,2.5,2.5])
directivity_2.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[2.5,2.5,2.5])
plt.show()

# plot
room.plot_rir()
plt.show()