import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyroomacoustics.directivities import DirectivityPattern, \
    DirectionVector, CardioidFamily
from unittest import TestCase 

  
# create room
room  = pra.ShoeBox(
            p = [7,5,5],
            materials=pra.Material(0.07),
            fs=16000,
            max_order=1,
        )

# define source with figure_eight directivity
PATTERN = DirectivityPattern.FIGURE_EIGHT
ORIENTATION = DirectionVector(azimuth=90, colatitude=90, degrees=True)
directivity = CardioidFamily(orientation=ORIENTATION, pattern_enum=PATTERN)

# add source with figure_eight directivity
audio_sample = wavfile.read("examples/input_samples/cmu_arctic_us_aew_a0001.wav")
room.add_source([2.5,2.5,2.5], signal = audio_sample, directivity = directivity)

# add microphone in its null
room.add_microphone([5,2.5,2.5])

# plot this room
fig, ax = room.plot()
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([-1, 6])
azimuth = np.linspace(start=0, stop=360, num=361, endpoint=True)
colatitude = np.linspace(start=0, stop=180, num=180, endpoint=True)
directivity.plot_response(azimuth=azimuth, colatitude=colatitude, degrees=True, ax=ax, offset=[2.5,2.5,2.5])
plt.show()

# compute rir
room.compute_rir()

class TestSourceDirectivity(TestCase):

    def test_zero_gain(self):
        gain = np.amax(room.rir[0][0])
        self.assertTrue(gain < (10**-10))

def get_error():
    print("-" * 40)
    print("gain")
    print("-" * 40)
    gain = np.amax(room.rir[0][0])
    theoretical_gain = 0
    error = abs(gain-theoretical_gain)
    print()
    print("-" * 40)
    print("The error in gain calculation is: {}".format(error))
    print("-" * 40)


if __name__ == '__main__':
    get_error()

