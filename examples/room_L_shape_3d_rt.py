'''
In this example, we construct an L-shape 3D room. We use the same floor as in
the 2D example and extrude a 3D room from the floor with a given height.  This
is a simple way to create 3D rooms that fits most situations.  Then, we place
one source and two microphones in the room and compute the room impulse
responses.

In this example, we also compare the speed of the C extension module to
that of the pure python code.
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra
from scipy.io import wavfile

# Create the 2D L-shaped room from the floor polygon
pol = np.array([[0,0], [0,10], [10,7.5], [7.5,6], [5,6], [5,0]]).T
r_absor = 0.1
mat = pra.Material.make_freq_flat(0.3, 0.1)
room = pra.Room.from_corners(
        pol,
        fs=16000,
        # absorption=r_absor,
        materials=mat,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
        )

# # Create the 3D room by extruding the 2D by 10 meters
height = 10.
room.extrude(height, materials=mat)

room.set_ray_tracing(receiver_radius=0.5, energy_thres=1e-5)

# # Add a source somewhere in the room
fs, audio_anechoic = wavfile.read('examples/input_samples/cmu_arctic_us_aew_a0001.wav')

room.add_source([1.5, 1.7, 1.6], signal=audio_anechoic)

R = np.array([[3.],
              [2.25],
              [0.6]])

room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

# Use the following function to compute the rir using either 'ism' method, 'rt' method, or 'hybrid' method
chrono = time.time()
room.image_source_model()
room.ray_tracing()
room.compute_rir()
print("Done in", time.time()-chrono, "seconds.")


room.plot_rir()
plt.show()
room.simulate()
audio_reverb = room.mic_array.to_wav('aaa.wav', norm=True, bitdepth=np.int16)

