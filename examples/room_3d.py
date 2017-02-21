from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra

room_ll = [-1,-1]
room_ur = [1,1]
src_pos = [0,0]
mic_pos = [0.5, 0.1]

max_order = 7
absorption = 0.85

# Create a 4 by 6 metres shoe box room
pol = 4 * np.array([[0,0], [0,1], [2,1], [2,0.5], [1,0.5], [1,0]]).T
#pol = 3 * np.array([[1,0], [1,0.5], [2,0.5], [2,1], [0,1], [0,0]]).T

# U-room
#pol = 3 * np.array([[0,0], [3,0], [3,2], [2,2], [2,1], [1,1], [1,2], [0,2],]).T


room = pra.Room.shoeBox2D([0, 0], [6, 4], fs=16000, max_order=max_order, absorption=absorption)
room = pra.Room.fromCorners(pol, fs=16000, max_order=max_order, absorption=absorption)

room.extrude(3.)

# Add a source somewhere in the room
room.addSource([1.5, 1.2, 0.5])

# Create a linear array beamformer with 4 microphones
# Place an array of two microphones
R = np.array([[3., 2.2], [2.25, 2.1], [0.6, 0.55]])
room.addMicrophoneArray(pra.MicrophoneArray(R, room.fs))

then = time.time()
room.image_source_model(use_libroom=False)
t_pure_python = time.time() - then

room.plotRIR()
plt.title('Pure python')

then = time.time()
room.image_source_model(use_libroom=True)
t_c = time.time() - then

room.compute_RIR()

room.plotRIR()
plt.title('libroom')

room.plot()

print("Time to compute in Python:", t_pure_python)
print("Time to compute in C:", t_c)
print("Speed-up:", t_pure_python/t_c, "x")

plt.show()
