import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily

# passing microphone array location and direcitivity    
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

# add microphone array with list of directivities
PATTERN = DirectivityPattern.SUBCARDIOID
ORIENTATION = DirectionVector(azimuth=0, colatitude=45, degrees=True)
directivity_1 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

PATTERN = DirectivityPattern.CARDIOID
ORIENTATION = DirectionVector(azimuth=180, colatitude=135, degrees=True)
directivity_2 = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

room.add_microphone_array(np.c_[[3,3,3],[5,5,4]], [directivity_1,directivity_2])

# 3-D plot
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)

ax = directivity_1.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[3,3,3])
directivity_2.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[5,5,4])
plt.show()

# plot
room.plot_rir()
plt.show()
    
