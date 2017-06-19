'''
In this example, we construct an L-shape 2D room, we place one source and two
microphones in the room and compute the room impulse responses.
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra

room_ll = [-1,-1]
room_ur = [1,1]
src_pos = [0,0]
mic_pos = [0.5, 0.1]

max_order = 10

# Store the corners of the room floor in an array
pol = 3 * np.array([[0,0], [0,1], [2,1], [2,0.5], [1,0.5], [1,0]]).T

# Create the room from its corners
room = pra.Room.from_corners(pol, fs=16000, max_order=max_order, absorption=0.1)

# Add a source somewhere in the room
room.add_source([1.5, 1.2])

# Create a linear array beamformer with 4 microphones
# Place an array of two microphones
R = np.array([[3., 4.2], [2.25, 2.1]])
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

room.compute_rir()

# plot the room and resulting beamformer
room.plot(img_order=6)

# Display the room impulse responses
plt.figure()
room.plot_rir()

plt.show()
