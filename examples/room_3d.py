from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra

room_ll = [-1,-1]
room_ur = [1,1]
src_pos = [0,0]
mic_pos = [0.5, 0.1]

max_order = 6

# Create a 4 by 6 metres shoe box room
#pol = 3 * np.array([[0,0], [0,1], [2,1], [2,0.5], [1,0.5], [1,0]]).T
#pol = 3 * np.array([[1,0], [1,0.5], [2,0.5], [2,1], [0,1], [0,0]]).T

# U-room
pol = 3 * np.array([[0,0], [3,0], [3,2], [2,2], [2,1], [1,1], [1,2], [0,2],]).T


room = pra.Room.fromCorners(pol, fs=16000, max_order=max_order, absorption=0.9)

room.plot()

room.extrude(1.)

# Add a source somewhere in the room
then = time.time()
room.addSource([1.5, 1.2, 0.5])
t_add_source = time.time() - then

# Create a linear array beamformer with 4 microphones
# Place an array of two microphones
R = np.array([[3., 4.2], [2.25, 2.1], [0.6, 0.55]])
room.addMicrophoneArray(pra.MicrophoneArray(R, room.fs))

then = time.time()
vis1 = room.checkVisibilityForAllImages(room.sources[0], R[:,0], use_libroom=False)
t_pure_python = time.time() - then

then = time.time()
vis2 = room.checkVisibilityForAllImages(room.sources[0], R[:,0], use_libroom=True)
t_c = time.time() - then

print("Is it all right ?", np.all(vis1 == vis2))

print("Time to compute image sources:", t_add_source)
print("Time to compute visibility in Python:", t_pure_python)
print("Time to compute visibility in C:", t_c)
print("Speed-up:", t_pure_python/t_c, "x")

plt.show()
